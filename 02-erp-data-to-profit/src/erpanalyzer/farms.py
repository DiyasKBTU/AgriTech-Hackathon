"""Хозяйства-ПРИМЕРЫ для показа кредитного профиля в мини-ERP.

Честно: сами хозяйства придуманы (поголовье, отёлы, падёж, зарплаты, кредиты).
Реальное в них — цены: выручка и расходы на корм посчитаны по месячным ценам
stat.gov.kz для региона хозяйства (если в регионе нет — по республике).
Качество скоринга этими примерами НЕ доказывается — они показывают экран.

Формат — как разделы ERP, по месяцам:
    farm.json          карточка хозяйства и заявка на кредит
    herd.csv           Животные: поголовье, приплод, покупка, продажа, падёж, средний вес
    feed.csv           Кормление и Склад: расход и остаток корма, закупка
    finance.csv        Финансы: выручка и расходы по статьям, платежи по кредитам
    weighings.csv      Взвешивания: средний привес группы за месяц (для откорма)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import prices as P

MONTHS = pd.period_range("2024-09", "2026-08", freq="M")
DAYS = {m: m.days_in_month for m in MONTHS}
WINTER = {11, 12, 1, 2, 3, 4}          # стойловый период (север и центр)


@dataclass
class Spec:
    fid: str
    name: str
    kind: str
    region: str
    district: str
    since: str                         # с какого месяца хозяйство ведёт учёт в ERP
    loan: dict
    debt_month: float                  # платёж по действующим кредитам, ₸/мес
    staff: int
    salary: float
    notes: list[str] = field(default_factory=list)
    monitored: dict | None = None      # помещение под наблюдением CV-трека: бирка ERP → id коровы в CV
    plan: dict = field(default_factory=dict)   # планы продаж и покупок (для субсидий)


def _price(prices: pd.DataFrame, product: str, region: str) -> pd.Series:
    """Месячный ряд ₸/кг; пропуски региона — по республике, дыры — предыдущим значением."""
    nat = P.series(prices, product)
    reg = P.series(prices, product, region)
    s = reg.combine_first(nat)
    idx = [str(m) for m in MONTHS]
    return s.reindex(sorted(set(s.index) | set(idx))).ffill().reindex(idx)


def _save(out: Path, spec: Spec, herd, feed, fin, weigh=None):
    d = out / spec.fid
    d.mkdir(parents=True, exist_ok=True)
    card = {k: v for k, v in spec.__dict__.items()}
    card["example"] = True
    (d / "farm.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame(herd).to_csv(d / "herd.csv", index=False)
    pd.DataFrame(feed).to_csv(d / "feed.csv", index=False)
    pd.DataFrame(fin).to_csv(d / "finance.csv", index=False)
    if weigh is not None:
        pd.DataFrame(weigh).to_csv(d / "weighings.csv", index=False)


def cow_calf(prices: pd.DataFrame, out: Path, rng: np.random.Generator) -> None:
    """№1: маточное стадо, Акмолинская обл. Дисциплинированный учёт, запас сена на всю зиму."""
    s = Spec("f1", "Хозяйство №1 · маточное стадо", "Маточное стадо + доращивание", "Акмолинская",
             "Целиноградский р-н", "2024-09",
             {"amount": 37_000_000, "months": 84, "max_months": 84, "rate": 0.06,
              "program": "«Игілік» (АКК): до 6%, до 7 лет",
              "purpose": "покупка 60 нетелей (≈43 млн ₸, 15% — свои)"},
             debt_month=500_000, staff=4, salary=260_000,
             notes=["Отёл в марте–апреле, продажа бычков в октябре–ноябре.",
                    "Сено заготавливают сами в июле–августе с запасом 15%."],
             monitored={"place": "Коровник №1 (под наблюдением)",
                        "tags": {f"KZ-11-{i:04d}": f"C{i:02d}" for i in range(1, 11)}},
             plan={"sell_bulls": {"month": "2026-10", "adg": 0.8, "buyer_ok": True,
                                  "buyer": "откормплощадка на 600 мест (Хозяйство №2)"},
                   "buy": {"what": "нетели", "count": 60, "price": 43_000_000, "month": "2026-11",
                           "pedigree": True}})
    cattle = _price(prices, "cattle", s.region)
    hay = _price(prices, "hay", s.region)
    barley = _price(prices, "barley", s.region)
    cows, calves, young = 180, 150, 0      # в сентябре 2024 — телята весеннего отёла
    to_replace = 0
    hay_stock = 470_000.0      # заготовка августа 2024 + остаток
    herd, feed, fin = [], [], []
    for m in MONTHS:
        k, days = str(m), DAYS[m]
        born = round(cows * 0.86 * {3: 0.6, 4: 0.4}.get(m.month, 0))
        dead_c = int(rng.random() < cows * 0.011 / 12 * 1.0)
        dead_y = int(rng.random() < (calves + young) * 0.03 / 12)
        calves += born
        sold_calves = round(calves * 0.75) if m.month == 10 else 0
        kept = (calves - sold_calves) if m.month == 11 else 0
        if calves:
            calves -= dead_y
        else:
            young = max(young - dead_y, 0)
        calves = calves - sold_calves - kept
        young += kept
        culled = round(cows * 0.15) if m.month == 10 else 0
        if m.month == 10:
            to_replace = culled + dead_c
        replaced = min(young, to_replace) if m.month == 11 else 0
        young -= replaced
        cows = cows - culled - dead_c + replaced
        heads = cows + calves + young
        winter = m.month in WINTER
        # норма зимовки мясной коровы — не меньше 12 кг сена в сутки (Палата казахской белоголовой породы)
        hay_use = (cows * 12 + (calves + young) * 4) * days if winter else 0.0
        barley_use = (cows * 1.0) * days if winter else 0.0
        hay_in = 440_000.0 if m.month == 8 else 0.0
        # своё сено: семена, шпагат, перевозка ≈ 35% рынка; топливо и люди — в «прочих» за июль–август
        hay_cost = hay_in * hay[k] * 0.35
        hay_stock = max(hay_stock + hay_in - hay_use, 0.0)
        rev_calves = sold_calves * 215 * cattle[k] * 1.05
        rev_culls = culled * 460 * cattle[k] * 0.85
        subsidy = 12_000 * cows if m.month == 12 else 0.0
        exp = {
            "exp_feed": hay_cost + barley_use * barley[k],
            "exp_salary": s.staff * s.salary,
            "exp_vet": heads * 150.0,
            "exp_pasture": 180_000.0 if not winter else 0.0,
            # топливо, электричество, ремонт, осеменение, налоги; заготовка сена — в июле–августе
            "exp_fuel_other": 900_000.0 + (2_500_000.0 if m.month in (7, 8) else 0.0) + rng.normal(0, 60_000),
        }
        herd.append({"month": k, "heads": heads, "cows": cows, "young": calves + young, "born": born,
                     "born_cattle": born,
                     "bought": 0, "sold": sold_calves + culled, "dead": dead_c + dead_y,
                     "sold_kg": sold_calves * 215 + culled * 460, "bought_kg": 0,
                     "avg_kg": round((cows * 470 + (calves + young) * 160) / max(heads, 1)),
                     "cattle_kg": cows * 470 + (calves + young) * 160, "sheep_kg": 0})
        feed.append({"month": k, "hay_use_kg": round(hay_use), "barley_use_kg": round(barley_use),
                     "hay_stock_kg": round(hay_stock), "hay_bought_kg": round(hay_in),
                     "hay_tg_kg": round(hay[k], 1), "barley_tg_kg": round(barley[k], 1)})
        fin.append({"month": k, "rev_cattle": round(rev_calves + rev_culls), "rev_other": 0,
                    "rev_subsidy": round(subsidy), **{a: round(b) for a, b in exp.items()},
                    "exp_animals": 0, "debt_payment": s.debt_month})
    _save(out, s, herd, feed, fin)


def feedlot(prices: pd.DataFrame, out: Path, rng: np.random.Generator) -> None:
    """№2: откормочная площадка 600 мест, Костанайская обл. Хороший привес, но корм покупают
    «с колёс» (запас 2–4 недели), платежи по кредитам большие."""
    s = Spec("f2", "Хозяйство №2 · откормочная площадка", "Откорм бычков, 600 мест", "Костанайская",
             "Костанайский р-н", "2024-09",
             {"amount": 80_000_000, "months": 36, "max_months": 84, "rate": 0.06,
              "program": "«Береке» (АКК): до 6%, до 7 лет", "purpose": "закуп бычков: расширение до 800 мест"},
             debt_month=1_600_000, staff=9, salary=280_000, plan={"feedlot": True},
             notes=["Закупают бычков по 250 кг каждый месяц, откорм ~150 дней.",
                    "Корм покупают небольшими партиями, запас 2–4 недели."])
    cattle = _price(prices, "cattle", s.region)
    hay = _price(prices, "hay", s.region)
    barley = _price(prices, "barley", s.region)
    cohorts: list[list] = [[250 + 30 * i, 105] for i in range(5)]   # [вес, головы]
    hay_stock, barley_stock = 60_000.0, 20_000.0
    herd, feed, fin, weigh = [], [], [], []
    for m in MONTHS:
        k, days = str(m), DAYS[m]
        adg = float(np.clip(rng.normal(1.02 if m.month not in (1, 2) else 0.9, 0.06), 0.7, 1.3))
        dead = 0
        for c in cohorts:
            c[0] += adg * days
            d = int(rng.binomial(c[1], 0.0013))
            c[1] -= d
            dead += d
        ready = [c for c in cohorts if c[0] >= 400]
        cohorts = [c for c in cohorts if c[0] < 400]
        sold = sum(c[1] for c in ready)
        sold_kg = sum(c[0] * c[1] for c in ready)
        bought = 110 if m.month not in (1, 2) else 90
        cohorts.append([250.0, bought])
        heads = sum(c[1] for c in cohorts)
        hay_use = heads * 9 * days
        barley_use = heads * 3 * days
        hay_in, barley_in = hay_use, barley_use
        cover = rng.uniform(7, 25)                       # на сколько дней хватает корма на складе
        hay_stock = hay_use / days * cover
        barley_stock = barley_use / days * cover
        exp = {
            "exp_feed": hay_in * hay[k] + barley_in * barley[k],
            "exp_salary": s.staff * s.salary,
            "exp_vet": heads * 200.0,
            "exp_pasture": 0.0,
            "exp_fuel_other": 1_100_000.0 + rng.normal(0, 90_000),
        }
        herd.append({"month": k, "heads": heads, "cows": 0, "young": heads, "born": 0, "born_cattle": 0,
                     "bought": bought,
                     "sold": sold, "dead": dead, "sold_kg": round(sold_kg), "bought_kg": bought * 250,
                     "avg_kg": round(sum(c[0] * c[1] for c in cohorts) / max(heads, 1)),
                     "cattle_kg": round(sum(c[0] * c[1] for c in cohorts)), "sheep_kg": 0})
        feed.append({"month": k, "hay_use_kg": round(hay_use), "barley_use_kg": round(barley_use),
                     "hay_stock_kg": round(hay_stock), "barley_stock_kg": round(barley_stock),
                     "hay_bought_kg": round(hay_in), "hay_tg_kg": round(hay[k], 1),
                     "barley_tg_kg": round(barley[k], 1)})
        fin.append({"month": k, "rev_cattle": round(sold_kg * cattle[k]), "rev_other": 0, "rev_subsidy": 0,
                    **{a: round(b) for a, b in exp.items()},
                    "exp_animals": round(bought * 250 * cattle[k] * 1.08), "debt_payment": s.debt_month})
        weigh.append({"month": k, "group": "откорм", "heads": heads, "adg": round(adg, 2)})
    _save(out, s, herd, feed, fin, weigh)


def mixed_small(prices: pd.DataFrame, out: Path, rng: np.random.Generator) -> None:
    """№3: небольшое смешанное хозяйство, Туркестанская обл. Учёт в ERP ведут 10 месяцев,
    взвешивают редко, падёж выше нормы, запаса кормов почти нет."""
    s = Spec("f3", "Хозяйство №3 · КРС и овцы", "Смешанное: 90 КРС + 300 овец", "Туркестанская",
             "Сайрамский р-н", "2025-11",
             {"amount": 15_000_000, "months": 36, "max_months": 84, "rate": 0.06,
              "program": "«Іскер» (АКК): до 6%, животноводство до 84 мес", "purpose": "покупка 150 овцематок"},
             debt_month=250_000, staff=2, salary=180_000,
             plan={"sell_bulls": {"month": "2026-10", "adg": 0.6, "buyer_ok": False, "buyer": "перекупщик на рынке"},
                   "sell_ram_lambs": {"month": "2026-10", "buyer": "откормплощадка на 1000+ голов"}},
             notes=["Учёт в ERP начали в ноябре 2025 г.; до этого — тетрадь.",
                    "Овец продают перед Курбан айтом, КРС — по необходимости."])
    cattle = _price(prices, "cattle", s.region)
    sheep_p = _price(prices, "sheep", s.region)
    hay = _price(prices, "hay", s.region)
    barley = _price(prices, "barley", s.region)
    cattle_n, sheep_n = 90, 300
    hay_stock = 8_000.0
    herd, feed, fin = [], [], []
    for m in MONTHS:
        k, days = str(m), DAYS[m]
        if k < s.since:
            continue
        dead = int(rng.binomial(cattle_n, 0.004)) + int(rng.binomial(sheep_n, 0.005))
        born = round(cattle_n * 0.55) if m.month == 4 else 0      # 55 телят на 100 голов КРС
        lambs = round(sheep_n * 0.9) if m.month == 3 else 0
        sold_sheep = round(sheep_n * 0.35) if m.month == 5 else (8 if m.month in (12, 2) else 0)
        sold_cattle = int(rng.integers(0, 4)) + (15 if m.month == 10 else 0)
        cattle_n = cattle_n + born - sold_cattle - min(dead, 2)
        sheep_n = sheep_n + lambs - sold_sheep - max(dead - 2, 0)
        winter = m.month in {12, 1, 2}
        hay_use = (cattle_n * 7 + sheep_n * 1.2) * days if winter else 0.0
        hay_in = hay_use * 0.9 if winter else (15_000.0 if m.month == 9 else 0.0)
        hay_stock = max(hay_stock + hay_in - hay_use, 0.0)
        rev = sold_cattle * 380 * cattle[k] + sold_sheep * 45 * sheep_p[k]
        exp = {
            "exp_feed": hay_in * hay[k] + (cattle_n * 0.5 * days * barley[k] if winter else 0),
            "exp_salary": s.staff * s.salary,
            "exp_vet": (cattle_n + sheep_n) * 90.0,
            "exp_pasture": 60_000.0,
            "exp_fuel_other": 650_000.0 + rng.normal(0, 80_000),
        }
        herd.append({"month": k, "heads": cattle_n + sheep_n, "cows": cattle_n, "young": sheep_n,
                     "born": born + lambs, "born_cattle": born, "bought": 0,
                     "sold": sold_cattle + sold_sheep, "dead": dead,
                     "sold_kg": sold_cattle * 380 + sold_sheep * 45, "bought_kg": 0,
                     "avg_kg": 0, "cattle_kg": cattle_n * 340, "sheep_kg": sheep_n * 42})
        feed.append({"month": k, "hay_use_kg": round(hay_use), "barley_use_kg": 0,
                     "hay_stock_kg": round(hay_stock), "hay_bought_kg": round(hay_in),
                     "hay_tg_kg": round(hay[k], 1), "barley_tg_kg": round(barley[k], 1)})
        fin.append({"month": k, "rev_cattle": round(rev), "rev_other": 0, "rev_subsidy": 0,
                    **{a: round(b) for a, b in exp.items()}, "exp_animals": 0, "debt_payment": s.debt_month})
    _save(out, s, herd, feed, fin)


def _register(fid: str, groups: list[dict], rng: np.random.Generator) -> pd.DataFrame:
    """Реестр голов (раздел ERP «Животные»): бирка, вид, пол, группа, порода, возраст,
    статус в ИСЖ и замечания. Придуман; доли замечаний заданы на хозяйство."""
    rows = []
    n = 0
    for g in groups:
        for _ in range(g["count"]):
            n += 1
            issue = ""
            r = rng.random()
            if r < g.get("no_isj", 0):
                issue = "нет в ИСЖ"
            elif r < g.get("no_isj", 0) + g.get("no_tag", 0):
                issue = "нет бирки"
            elif r < g.get("no_isj", 0) + g.get("no_tag", 0) + g.get("no_weigh", 0):
                issue = "нет акта взвешивания"
            born = pd.Period("2026-08", "M") - int(rng.integers(g["age"][0], g["age"][1] + 1))
            rows.append({
                "tag": f"KZ-{fid[1:]}{g['code']}-{n:04d}", "species": g["species"], "sex": g["sex"],
                "group": g["group"], "breed": g["breed"], "pedigree": g.get("pedigree", "товарное"),
                "born": str(born), "weight_kg": int(rng.normal(*g["kg"])) if g.get("kg") else "",
                "issue": issue,
            })
    return pd.DataFrame(rows)


REGISTERS = {
    # №1: учёт аккуратный, мелкие замечания
    "f1": [
        {"group": "маточное поголовье", "code": "1", "species": "КРС", "sex": "F", "count": 176,
         "breed": "казахская белоголовая", "pedigree": "товарное", "age": (30, 120), "kg": (470, 30),
         "no_isj": 0.03, "no_tag": 0.02},
        {"group": "быки-производители", "code": "2", "species": "КРС", "sex": "M", "count": 7,
         "breed": "казахская белоголовая", "pedigree": "племенное", "age": (36, 72), "kg": (820, 40)},
        {"group": "бычки на продажу (осень)", "code": "3", "species": "КРС", "sex": "M", "count": 78,
         "breed": "казахская белоголовая", "age": (5, 6), "kg": (215, 20), "no_weigh": 0.10, "no_tag": 0.03},
        {"group": "тёлки", "code": "4", "species": "КРС", "sex": "F", "count": 76,
         "breed": "казахская белоголовая", "age": (5, 6), "kg": (195, 18), "no_tag": 0.03},
    ],
    # №2: откорм, бычки от разных поставщиков
    "f2": [
        {"group": "бычки на откорме", "code": "1", "species": "КРС", "sex": "M", "count": 620,
         "breed": "помесь", "age": (8, 16), "kg": (330, 45), "no_isj": 0.04, "no_weigh": 0.06},
    ],
    # №3: учёт недавно, много пробелов
    "f3": [
        {"group": "маточное поголовье", "code": "1", "species": "КРС", "sex": "F", "count": 62,
         "breed": "аулиекольская", "age": (30, 110), "kg": (420, 40), "no_isj": 0.15, "no_tag": 0.10},
        {"group": "молодняк КРС", "code": "2", "species": "КРС", "sex": "M", "count": 23,
         "breed": "аулиекольская", "age": (4, 16), "kg": (210, 50), "no_tag": 0.15, "no_weigh": 0.3},
        {"group": "овцематки", "code": "3", "species": "МРС", "sex": "F", "count": 250,
         "breed": "едильбаевская", "age": (14, 72), "kg": (62, 6), "no_isj": 0.10, "no_tag": 0.18},
        {"group": "бараны и ягнята", "code": "4", "species": "МРС", "sex": "M", "count": 90,
         "breed": "едильбаевская", "age": (4, 40), "kg": (45, 12), "no_tag": 0.2},
    ],
}


def build(prices_dir: Path, out: Path, seed: int = 7) -> list[str]:
    prices = P.load(prices_dir)
    rng = np.random.default_rng(seed)
    cow_calf(prices, out, rng)
    feedlot(prices, out, rng)
    mixed_small(prices, out, rng)
    for fid, groups in REGISTERS.items():
        _register(fid, groups, np.random.default_rng(seed + int(fid[1:]))).to_csv(out / fid / "animals.csv", index=False)
    return ["f1", "f2", "f3"]
