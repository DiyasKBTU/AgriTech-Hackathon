"""Тесты формул и страниц. Качество скоринга здесь не доказывается — данных о невозвратах нет."""

from pathlib import Path

import pandas as pd
import pytest

from erpanalyzer import credit as C
from erpanalyzer import prices as P

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "stat_gov_kz_prices"
FARMS = ROOT / "data" / "erp_farms"


@pytest.mark.skipif(not (RAW / "2026-08.xlsx").exists(), reason="нет скачанного файла stat.gov.kz")
def test_parse_real_price_file():
    df = P.parse_month(RAW / "2026-08.xlsx")
    nat = df[df.region == P.NATIONAL].set_index("product")["tg_per_kg"]
    assert nat["cattle"] == pytest.approx(1445.429)
    assert nat["barley"] == pytest.approx(81.276)
    assert nat["hay"] == pytest.approx(43.071)


def test_annuity_and_inverse():
    pay = C.annuity(37e6, 84, 0.06)
    assert pay == pytest.approx(540_517, abs=1)
    assert C.max_amount(pay, 84, 0.06) == pytest.approx(37e6)


def test_max_change_counts_calendar_months():
    s = pd.Series([100.0, 110.0, 163.0], index=["2021-04", "2021-05", "2021-07"])
    ch, a, b = C._max_change(s, 3, "up")
    assert (round(ch, 2), a, b) == (0.63, "2021-04", "2021-07")


@pytest.fixture(scope="module")
def data():
    if not (FARMS / "f1" / "farm.json").exists():
        pytest.skip("нет хозяйств-примеров: erpanalyzer all")
    prices = P.load(ROOT / "data")
    return prices, C.load_farms(FARMS)


def test_profile_has_indicators_from_task(data):
    prices, farms = data
    p = C.build_profile(farms["f1"], prices)
    keys = {i.key for i in p.indicators}
    # ТЗ: себестоимость, динамика расходов, поголовье, производство, выручка, маржинальность
    assert {"cost", "costs", "herd", "productivity", "growth", "margin"} <= keys
    assert sum(i.weight for i in p.indicators) == pytest.approx(100)
    assert 0 <= p.score <= 100


def test_dscr_is_cash_flow_over_payments(data):
    prices, farms = data
    p = C.build_profile(farms["f2"], prices)
    assert p.dscr == pytest.approx(p.ebitda / (p.debt12 + p.new_pay12))
    # лимит по потоку: при нём DSCR ровно 1,2
    q = C.build_profile(farms["f2"], prices, loan={"amount": p.limit_cash})
    assert q.dscr == pytest.approx(C.DSCR_MIN, rel=1e-3)


def test_short_history_is_preliminary(data):
    prices, farms = data
    p = C.build_profile(farms["f3"], prices)
    assert p.months < 12
    assert p.klass not in ("A", "B")
    assert any("предварительный" in fl for fl in p.flags)


def test_smaller_loan_does_not_lower_score(data):
    prices, farms = data
    f = farms["f2"]
    a = C.build_profile(f, prices)
    b = C.build_profile(f, prices, loan={"amount": f.card["loan"]["amount"] / 2})
    assert b.score >= a.score


def test_pages_open(data):
    from fastapi.testclient import TestClient
    from erpanalyzer.web import create_app
    c = TestClient(create_app(ROOT / "data", FARMS, ROOT / "data" / "cv_snapshot.json"))
    for u in ["/", "/f1", "/f2", "/f3", "/f1/check", "/f2/check", "/f3/check", "/bank", "/bank/f1", "/bank/f3",
              "/api/profile/f2", "/api/subsidies/f1"]:
        r = c.get(u)
        assert r.status_code == 200, u
    assert c.get("/f9").status_code == 404


def test_subsidy_rules(data):
    from erpanalyzer import subsidies as S
    _, farms = data
    f1 = S.assess(farms["f1"])
    heifers = next(l for l in f1.lines if l.key == "heifers")
    assert heifers.amount == pytest.approx(60 * 260_000)          # норматив приказа № 108
    bulls = next(l for l in f1.lines if l.key == "bulls")
    assert bulls.amount <= bulls.heads_ok * 250 * 300 + 1          # не больше 250 кг × 300 ₸
    assert any("8" in r.text for r in bulls.reasons)                 # слишком рано продавать
    f3 = S.assess(farms["f3"])
    bulls3 = next(l for l in f3.lines if l.key == "bulls")
    assert bulls3.heads_ok == 0                                        # нет племенного отца
    assert bulls3.blocked_amount == 0 and bulls3.lost_amount > 0       # этим бычкам уже не помочь — не «под угрозой»
    assert not any("отец" in x["action"] and x["deadline"] != "до случки" for x in f3.actions)


def test_mvp_flow_data_then_result(data):
    from fastapi.testclient import TestClient
    from erpanalyzer.web import create_app
    c = TestClient(create_app(ROOT / "data", FARMS, ROOT / "data" / "cv_snapshot.json"))
    page = c.get("/f1").text
    assert "ERP · финансы" in page and "ERP · реестр животных" in page and 'href="/f1/check"' in page
    result = c.get("/f1/check").text
    for part in ("Что сделать", "Кредит: как посчитано", "Субсидии: как посчитано", "Залог: как проверено",
                 'href="/bank/f1"'):
        assert part in result, part


def test_subsidy_rows_add_up_to_totals(data):
    """Таблица «по каждой голове» в MVP сходится с итогами: положено + под угрозой + не положено."""
    from erpanalyzer import subsidies as S
    _, farms = data
    for fid in ("f1", "f3"):
        bulls = next(l for l in S.assess(farms[fid]).lines if l.key == "bulls")
        ok = sum(r["pay"] for r in bulls.rows if r["status"] == "проходит")
        bad = sum(r["pay"] for r in bulls.rows if r["status"] != "проходит")
        assert ok == pytest.approx(bulls.amount)
        assert bad == pytest.approx(bulls.blocked_amount + bulls.lost_amount)
        assert len(bulls.rows) == bulls.heads_ok + bulls.heads_risk


def test_presence_check_from_snapshot():
    from erpanalyzer import verify as V
    snap = ROOT / "data" / "cv_snapshot.json"
    if not snap.exists():
        pytest.skip("нет снимка CV-трека")
    pres = V.load_presence(snap, timeout=0.01)
    card = {"monitored": {"place": "x", "tags": {"A": "C01", "B": "C99"}}}
    chk = V.verify(card, pres)
    assert chk["expected"] == 2 and chk["seen"] == 1
    assert V.verify({}, pres) == {"connected": False}
