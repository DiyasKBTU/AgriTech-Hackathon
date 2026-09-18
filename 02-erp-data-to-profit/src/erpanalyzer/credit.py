"""Кредитный профиль хозяйства по данным ERP.

На пальцах: банк не видит, как живёт хозяйство, поэтому просит залог на всю сумму.
В ERP это уже записано: поголовье и падёж, приплод и привес, запас кормов, выручка
и расходы. Модуль считает из этого показатели, которые ТЗ называет для скоринга
(себестоимость, динамика расходов, поголовье, производство, выручка, маржинальность),
добавляет то, на что смотрит банк (покрытие платежей, запас кормов, залог, плохой год),
ставит баллы и класс A–D, считает, какую сумму хозяйство потянет, и что улучшить.

Честно: пороги и веса — из норм и практики кредиторов (источники в README.md),
а не обучены на данных о невозвратах — таких данных у нас нет.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import prices as P

CLASSES = [(80, "A", "надёжный"), (65, "B", "устойчивый"), (50, "C", "с рисками"), (0, "D", "высокий риск")]
DSCR_MIN = 1.2            # минимум у кредиторов 1,15–1,20, хорошо ≥ 1,5 (farmdoc daily, Univ. of Illinois)
COLLATERAL_HAIRCUT = 0.5  # скот в залог — половина стоимости (консервативно; в США до 70%, Southern AgCredit)
MORTALITY_NORM = 0.02     # естественная убыль маточного мясного КРС 2% в год (приказ МСХ РК №3-3/1061)
MFO_RATE = 0.25           # МФО «KMF Агро»: 24,99–38,25% годовых (kmf.kz) — куда идут без профиля

GROUPS = ["Деньги", "Производство", "Запасы и залог", "Плохой год"]


def _f(text: str) -> str:
    """Десятичная запятая, как принято в русском тексте."""
    return text.replace(".", ",")


def _n(x: float) -> str:
    """Целое с пробелами между тысячами."""
    return f"{x:,.0f}".replace(",", " ")


# ---------------------------------------------------------------------------
# данные хозяйства (помесячные сводки разделов ERP)

@dataclass
class Farm:
    fid: str
    card: dict
    herd: pd.DataFrame      # Животные
    feed: pd.DataFrame      # Кормление и склад
    fin: pd.DataFrame       # Финансы
    weigh: pd.DataFrame | None   # Взвешивания
    animals: pd.DataFrame | None = None   # Реестр голов

    @property
    def kind(self) -> str:
        k = self.card["kind"].lower()
        return "feedlot" if "откорм" in k else ("mixed" if "смешан" in k else "cowcalf")


def load_farm(d: Path) -> Farm:
    w, a = d / "weighings.csv", d / "animals.csv"
    return Farm(d.name, json.loads((d / "farm.json").read_text(encoding="utf-8")),
                pd.read_csv(d / "herd.csv", dtype={"month": str}),
                pd.read_csv(d / "feed.csv", dtype={"month": str}),
                pd.read_csv(d / "finance.csv", dtype={"month": str}),
                pd.read_csv(w, dtype={"month": str}) if w.exists() else None,
                pd.read_csv(a, dtype={"born": str, "issue": str}, keep_default_na=False) if a.exists() else None)


def load_farms(root: Path) -> dict[str, Farm]:
    return {d.name: load_farm(d) for d in sorted(root.iterdir()) if (d / "farm.json").exists()}


def revenue(fin: pd.DataFrame) -> pd.Series:
    return fin.filter(like="rev_").sum(axis=1)


def expenses(fin: pd.DataFrame) -> pd.Series:
    return fin.filter(like="exp_").sum(axis=1)


def annuity(amount: float, months: int, rate: float) -> float:
    r = rate / 12
    return amount / months if r == 0 else amount * r / (1 - (1 + r) ** -months)


def max_amount(payment: float, months: int, rate: float) -> float:
    r = rate / 12
    if payment <= 0:
        return 0.0
    return payment * months if r == 0 else payment * (1 - (1 + r) ** -months) / r


# ---------------------------------------------------------------------------
# реальные скачки цен для плохого года

def _max_change(s: pd.Series, months: int, worst: str) -> tuple[float, str, str]:
    """Самый большой рост (worst='up') или падение за `months` календарных месяцев."""
    s = s.dropna()
    no = {m: P._month_no(m) for m in s.index}
    best = (0.0, "", "")
    for a in s.index:
        for b in s.index:
            if no[b] - no[a] != months:
                continue
            ch = s[b] / s[a] - 1
            if (worst == "up" and ch > best[0]) or (worst == "down" and ch < best[0]):
                best = (float(ch), a, b)
    return best


def historical_shocks(prices: pd.DataFrame) -> dict:
    hay = P.series(prices, "hay")
    barley = P.series(prices, "barley")
    cattle = P.series(prices, "cattle")
    hay21 = hay[(hay.index >= "2021-01") & (hay.index <= "2021-12")]
    return {
        "hay_2021": _max_change(hay21, 3, "up"),
        "barley_12m": _max_change(barley, 12, "up"),
        "cattle_1m": _max_change(cattle, 1, "down"),
        "cattle_12m": _max_change(cattle, 12, "down"),
        "range": (cattle.index[0], cattle.index[-1]),
    }


# ---------------------------------------------------------------------------
# показатели

@dataclass
class Indicator:
    key: str
    group: str
    name: str
    value: float | None
    shown: str
    points: float
    weight: float
    norm: str
    source: str             # раздел ERP (или «stat.gov.kz»)


def _scale(v: float, pts: list[tuple[float, float]]) -> float:
    """Кусочно-линейная шкала: pts = [(значение, баллы), ...] по возрастанию значения."""
    xs, ys = zip(*pts)
    return float(np.interp(v, xs, ys))


@dataclass
class Scenario:
    name: str
    source: str
    ebitda: float
    dscr: float


@dataclass
class Profile:
    farm: Farm
    asof: str
    months: int
    indicators: list[Indicator]
    score: float
    klass: str
    klass_name: str
    capped: bool
    ebitda: float
    rev12: float
    exp12: float
    debt12: float
    new_pay12: float
    dscr: float
    limit_cash: float
    herd_value: float
    limit_collateral: float
    limit: float
    scenarios: list[Scenario]
    cost_kg: float
    sale_kg: float
    improve: list[dict] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def grouped(self) -> list[tuple[str, list[Indicator]]]:
        return [(g, [i for i in self.indicators if i.group == g]) for g in GROUPS]


def _klass(score: float) -> tuple[str, str]:
    for lo, k, name in CLASSES:
        if score >= lo:
            return k, name
    return "D", "высокий риск"


def _feed_market_value(f: Farm, months: pd.Series) -> tuple[float, float]:
    """Рыночная стоимость съеденного сена и ячменя (расход склада × цена месяца).
    В засуху своей травы нет — недостающее покупают по новой цене."""
    fd = f.feed[f.feed["month"].isin(months)]
    return (float((fd["hay_use_kg"] * fd["hay_tg_kg"]).sum()),
            float((fd["barley_use_kg"] * fd["barley_tg_kg"]).sum()))


def _growth(fin: pd.DataFrame, series) -> float | None:
    if len(fin) < 24:
        return None
    prev = float(series(fin.iloc[-24:-12]).sum())
    return float(series(fin.tail(12)).sum()) / prev - 1 if prev else None


def build_profile(f: Farm, prices: pd.DataFrame, loan: dict | None = None,
                  overrides: dict | None = None, check: dict | None = None) -> Profile:
    """overrides — «что если»: подменить значение показателя (для подсказок «что улучшить»).
    check — результат verify.verify(): подтверждено ли поголовье системой наблюдения."""
    loan = dict(f.card["loan"], **(loan or {}))
    ov = overrides or {}
    fin, herd = f.fin, f.herd
    tail = fin.tail(12)
    h12 = herd.tail(12)
    n = len(fin)
    scale = 12 / len(tail)                    # истории меньше года — приводим к году
    rev12 = float(revenue(tail).sum()) * scale
    exp12 = float(expenses(tail).sum()) * scale
    ebitda = rev12 - exp12
    debt12 = float(tail["debt_payment"].sum()) * scale
    new_pay12 = annuity(loan["amount"], loan["months"], loan["rate"]) * 12
    pays = debt12 + new_pay12
    dscr = ebitda / pays if pays else 9.9
    region = f.card["region"]
    _, cattle_p = P.latest(prices, "cattle", region)
    _, sheep_p = P.latest(prices, "sheep", region)
    last = herd.iloc[-1]
    herd_value = float(last["cattle_kg"] * cattle_p + last["sheep_kg"] * sheep_p)
    ind: list[Indicator] = []

    def add(key, group, name, value, shown, pts, weight, norm, source):
        ind.append(Indicator(key, group, name, value, shown, pts, weight, norm, source))

    # --- Деньги -------------------------------------------------------------
    g = ov.get("growth", _growth(fin, revenue))
    if g is None:
        add("growth", "Деньги", "Выручка к прошлому году", None, "нет данных", 30, 5,
            "нужно 24 месяца учёта", "Финансы")
    else:
        add("growth", "Деньги", "Выручка к прошлому году", g, f"{g:+.0%}",
            _scale(g, [(-0.3, 0), (-0.1, 50), (0, 80), (0.1, 100)]), 5, "не падает", "Финансы")

    ge = _growth(fin, expenses)
    if g is None or ge is None:
        add("costs", "Деньги", "Расходы к прошлому году", None, "нет данных", 30, 5,
            "нужно 24 месяца учёта", "Финансы")
    else:
        gap = ov.get("costs", g - ge)
        add("costs", "Деньги", "Расходы к прошлому году", gap, _f(f"{ge:+.0%} (выручка {g:+.0%})"),
            _scale(gap, [(-0.25, 0), (-0.1, 50), (0, 90), (0.05, 100)]), 5,
            "растут не быстрее выручки", "Финансы")

    margin = ov.get("margin", ebitda / rev12 if rev12 else 0.0)
    good = 0.08 if f.kind == "feedlot" else 0.20
    add("margin", "Деньги", "Маржинальность (поток ÷ выручка)", margin, _f(f"{margin:.1%}"),
        _scale(margin, [(0, 10), (good / 2, 50), (good, 90), (good * 1.5, 100)]), 8,
        f"≥ {good:.0%} для такого хозяйства", "Финансы")

    # себестоимость 1 кг произведённого живого веса
    live = herd["cattle_kg"] + herd["sheep_kg"]
    start_i = max(0, len(herd) - 13)
    produced = (float(h12["sold_kg"].sum() - h12["bought_kg"].sum())
                + float(live.iloc[-1] - live.iloc[start_i])) * scale
    prod_cost = exp12 - float(tail["exp_animals"].sum()) * scale
    sold_kg = float(h12["sold_kg"].sum())
    cost_kg = prod_cost / produced if produced > 0 else float("nan")
    sale_kg = float(tail["rev_cattle"].sum()) / sold_kg if sold_kg else cattle_p
    buf = ov.get("cost", 1 - cost_kg / sale_kg if produced > 0 else -1.0)
    add("cost", "Деньги", "Себестоимость 1 кг продукции", buf,
        f"{_n(cost_kg)} ₸/кг при цене продажи {_n(sale_kg)} (запас {buf:.0%})"
        if produced > 0 else "не считается",
        _scale(buf, [(0, 0), (0.1, 40), (0.25, 80), (0.35, 100)]), 10,
        "дешевле цены продажи хотя бы на 25%", "Финансы · Животные")

    v = ov.get("dscr", dscr)
    add("dscr", "Деньги", "Покрытие платежей (DSCR)", v, _f(f"{v:.2f}"),
        _scale(v, [(0.8, 0), (1.0, 35), (1.2, 75), (1.5, 100)]), 18,
        "поток ÷ все платежи ≥ 1,2", "Финансы")

    # --- Производство -------------------------------------------------------
    if f.kind == "feedlot":
        occ = ov.get("herd", float(h12["heads"].mean()) / 600)
        add("herd", "Производство", "Поголовье: загрузка площадки", occ, f"{occ:.0%}",
            _scale(occ, [(0.5, 0), (0.7, 50), (0.9, 100)]), 6, "≥ 90% мест", "Животные")
    else:
        start = float(herd["cows"].iloc[max(0, n - 12)])
        ch = ov.get("herd", float(last["cows"]) / start - 1 if start else 0.0)
        add("herd", "Производство", "Поголовье: маточное стадо за год", ch, f"{ch:+.0%}",
            _scale(ch, [(-0.2, 0), (-0.1, 40), (0, 90), (0.05, 100)]), 6, "не сокращается", "Животные")

    if f.kind == "feedlot" and f.weigh is not None:
        w12 = f.weigh.tail(12)
        adg = float((w12["adg"] * w12["heads"]).sum() / w12["heads"].sum())
        heads = float(h12["heads"].mean())
        day_cost = prod_cost / 365 / heads + float(last["avg_kg"]) * cattle_p * loan["rate"] / 365
        breakeven = day_cost / cattle_p
        safety = ov.get("productivity", adg / breakeven)
        add("productivity", "Производство", "Привес к порогу окупаемости", safety,
            _f(f"{adg:.2f} кг/сут при пороге {breakeven:.2f}"),
            _scale(safety, [(0.9, 0), (1.0, 40), (1.15, 75), (1.3, 100)]), 8,
            "выше порога на 30%", "Взвешивания · Финансы")
    else:
        born = float(h12["born_cattle"].sum()) * scale
        rate = ov.get("productivity", born / max(float(h12["cows"].mean()), 1))
        add("productivity", "Производство", "Приплод: телят на 100 коров", rate, f"{rate * 100:.0f}",
            _scale(rate, [(0.5, 20), (0.64, 50), (0.75, 80), (0.85, 100)]), 8,
            "в среднем по РК 64–75", "Животные")

    mort = ov.get("mortality", float(h12["dead"].sum()) * scale / max(float(h12["heads"].mean()), 1))
    add("mortality", "Производство", "Падёж за год", mort, _f(f"{mort:.1%}"),
        _scale(mort, [(0.0, 100), (MORTALITY_NORM, 100), (0.03, 60), (0.06, 0)]), 10,
        "≤ 2% (норма МСХ)", "Животные · Здоровье")

    # --- Запасы и залог -----------------------------------------------------
    fd = f.feed
    if f.kind == "feedlot":
        daily = fd["hay_use_kg"] / 30.4
        days_cover = float((fd["hay_stock_kg"] / daily.replace(0, np.nan)).tail(6).mean())
        cover = ov.get("feed", days_cover / 30)
        shown, norm = f"{days_cover:.0f} дней расхода", "≥ 30 дней"
    else:
        winter = fd[fd["hay_use_kg"] > 0]
        need = float(winter.tail(6)["hay_use_kg"].sum()) if len(winter) else 1.0
        first = winter["month"].iloc[-min(len(winter), 6)] if len(winter) else fd["month"].iloc[-1]
        i = fd.index[fd["month"] == first][0]
        stock = float(fd.loc[max(i - 1, fd.index[0]), "hay_stock_kg"])
        cover = ov.get("feed", stock / need if need else 0.0)
        shown, norm = f"{cover:.0%} потребности на зиму", "100% на стойловый период"
    add("feed", "Запасы и залог", "Запас кормов", cover, shown,
        _scale(cover, [(0.2, 0), (0.5, 40), (0.8, 75), (1.0, 100)]), 10, norm, "Склад · Кормление")

    cov = ov.get("collateral", herd_value * COLLATERAL_HAIRCUT / loan["amount"])
    add("collateral", "Запасы и залог", "Скот как залог", cov,
        _f(f"половина стада = {cov:.1f} суммы"),
        _scale(cov, [(0.3, 0), (0.6, 40), (1.0, 80), (1.3, 100)]), 6,
        "≥ 1,0 суммы кредита", "Животные · stat.gov.kz")

    if check and check.get("connected"):
        share = ov.get("verified", check["share"])
        add("verified", "Запасы и залог", "Поголовье подтверждено наблюдением", share,
            f"{check['seen']} из {check['expected']} голов за сутки",
            _scale(share, [(0.5, 0), (0.8, 50), (0.95, 100)]), 4, "≥ 95% голов под наблюдением на месте",
            "Животные + CV-трек")
    else:
        add("verified", "Запасы и залог", "Поголовье подтверждено наблюдением", ov.get("verified"),
            "не подключено" if "verified" not in ov else f"{ov['verified']:.0%}",
            _scale(ov["verified"], [(0.5, 0), (0.8, 50), (0.95, 100)]) if "verified" in ov else 30, 4,
            "≥ 95% голов под наблюдением на месте", "Животные + CV-трек")

    # --- Плохой год ---------------------------------------------------------
    sh = historical_shocks(prices)
    hay_val, barley_val = (x * scale for x in _feed_market_value(f, tail["month"]))
    cattle_net = float(tail["rev_cattle"].sum() - tail["exp_animals"].sum()) * scale

    def scen(name, src, d_hay=0.0, d_barley=0.0, d_price=0.0, extra_dead=0.0):
        e = ebitda - hay_val * d_hay - barley_val * d_barley + cattle_net * d_price - extra_dead * herd_value
        return Scenario(name, src, e, e / pays if pays else 9.9)

    h21, b12, c12 = sh["hay_2021"], sh["barley_12m"], sh["cattle_12m"]
    reserve = ov.get("hay_reserve", 0.0)
    hay_shock = h21[0] * (1 - reserve)
    drop = min(c12[0], -0.10)
    scenarios = [
        scen("Засуха: сено дорожает", f"сено {h21[1]} → {h21[2]}: {h21[0]:+.0%}"
             + (f"; резерв закрывает {reserve:.0%}" if reserve else ""), d_hay=hay_shock),
        scen("Ячмень дорожает", f"ячмень {b12[1]} → {b12[2]}: {b12[0]:+.0%}", d_barley=b12[0]),
        scen("Цена КРС −10%", f"за 2019–2026 годового падения не было, худший месяц {sh['cattle_1m'][0]:+.0%}",
             d_price=drop),
        scen("Падёж +3% стада", "вспышка болезни (допущение)", extra_dead=0.03),
    ]
    scenarios.append(scen("Всё сразу", "маловероятно; показывает запас прочности",
                          d_hay=hay_shock, d_barley=b12[0], d_price=drop, extra_dead=0.03))
    worst = ov.get("stress", min(s.dscr for s in scenarios[:-1]))
    add("stress", "Плохой год", "Покрытие платежей в худшем сценарии", worst, _f(f"{worst:.2f}"),
        _scale(worst, [(0.6, 0), (0.8, 20), (1.0, 60), (1.2, 100)]), 10,
        "≥ 1,0 — платить сможет и в плохой год", "Финансы · stat.gov.kz")

    score = sum(i.points * i.weight for i in ind) / sum(i.weight for i in ind)
    klass, kname = _klass(score)
    capped = False
    if n < 12 and klass in ("A", "B"):
        klass, kname, capped = "C", "предварительно", True

    limit_cash = max(max_amount((ebitda / DSCR_MIN - debt12) / 12, loan["months"], loan["rate"]), 0.0)
    limit_coll = herd_value * COLLATERAL_HAIRCUT
    flags = [f"Учёт в ERP ведётся {n} мес. — профиль предварительный (A и B не присваиваются)."] if n < 12 else []
    flags += [f"{i.name}: {i.shown}. Норма: {i.norm}." for i in ind if i.points < 40]

    return Profile(f, fin["month"].iloc[-1], n, ind, score, klass, kname, capped, ebitda, rev12, exp12,
                   debt12, new_pay12, dscr, limit_cash, herd_value, limit_coll,
                   max(min(limit_cash, limit_coll), 0.0), scenarios, cost_kg, sale_kg, flags=flags)


# ---------------------------------------------------------------------------
# что исправить

def improvements(f: Farm, prices: pd.DataFrame, p: Profile, top: int = 3,
                 check: dict | None = None) -> list[dict]:
    loan = f.card["loan"]
    ind = {i.key: i for i in p.indicators}
    out = []

    def gain(label, action, money, over=None, loan_over=None, kind="cost"):
        q = build_profile(f, prices, loan=loan_over, overrides=over, check=check)
        out.append({"label": label, "action": action, "money": money, "kind": kind,
                    "delta": q.score - p.score, "klass": q.klass})

    if ind["dscr"].points < 100:
        amount = min(p.limit_cash, loan["amount"])
        if 0 < amount < loan["amount"]:
            gain("Попросить меньше", _f(f"{amount / 1e6:.1f} млн ₸ вместо {loan['amount'] / 1e6:.0f} млн — "
                                        "такой кредит хозяйство потянет"),
                 None, loan_over={"amount": amount})
        longer = min(loan["months"] * 2, loan.get("max_months", 84))
        if longer > loan["months"]:
            gain("Взять кредит на дольше", f"на {longer} мес. вместо {loan['months']} — платёж в месяц станет меньше", None,
                 loan_over={"months": longer})
    if ind["feed"].points < 100:
        if f.kind == "feedlot":
            daily_t = float(f.feed["hay_use_kg"].iloc[-1]) / 30.4 / 1000
            need_t = max(daily_t * 30 - float(f.feed["hay_stock_kg"].iloc[-1]) / 1000, 0)
            gain("Держать запас корма на месяц", f"докупить около {need_t:.0f} т сена заранее",
                 need_t * 1000 * float(f.feed["hay_tg_kg"].iloc[-1]), over={"feed": 1.0})
        else:
            gain("Заготовить сено на всю зиму", "сейчас запаса не хватит на зиму", None,
                 over={"feed": 1.0})
    if ind["mortality"].points < 99:
        saved = max(ind["mortality"].value - MORTALITY_NORM, 0) * float(f.herd.tail(12)["heads"].mean())
        per_head = p.herd_value / max(float(f.herd.iloc[-1]["heads"]), 1)
        gain("Снизить падёж", f"ветеринарный план и прививки — около {saved:.0f} голов в год останутся живы",
             saved * per_head, over={"mortality": MORTALITY_NORM}, kind="gain")
    if ind["stress"].points < 100 and f.kind != "feedlot":
        hay_t = float(f.feed.tail(12)["hay_use_kg"].sum()) / 1000
        gain("Держать запас сена на случай засухи", f"около {hay_t / 2:.0f} т сверх обычного — засуха ударит вдвое слабее",
             hay_t / 2 * 1000 * float(f.feed["hay_tg_kg"].iloc[-1]),
             over={"hay_reserve": 0.5, "feed": max(ind["feed"].value or 0, 1.0)})
    if ind["verified"].points < 100:
        gain("Подключить наблюдение за стадом", "камера или метки — банк увидит, что скот на месте",
             None, over={"verified": 1.0})
    if p.months < 12:
        out.append({"label": f"Вести учёт в ERP ещё {12 - p.months} мес.",
                    "action": "после года учёта оценка станет полной",
                    "money": None, "delta": 0.0, "klass": p.klass})
    out.sort(key=lambda x: -x["delta"])
    return out[:top]
