"""Субсидии без потерь: что хозяйству положено по данным ERP и что мешает получить.

Правила — приказ МСХ РК от 15.03.2019 № 108 «Правила субсидирования развития
племенного животноводства, повышения продуктивности и качества продукции
животноводства», редакция на 01.07.2026 (zakon.uchet.kz). С 12.07.2026 действуют
новые редакции приложений (приказ № 240) — их текст не прочитан, нормативы
могут отличаться. В 2026 г. субсидии на селекционную работу с маточным
поголовьем КРС и МРС отменены (primeminister.kz, 21.05.2026).

Честно: реестр голов и планы продаж — примеры; правила и нормативы — из приказа.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import pandas as pd

from .credit import Farm

TODAY = pd.Period("2026-09", "M")
SRC = "приказ МСХ РК № 108 (ред. 01.07.2026), прил. 1–2"
WINDOW = "приём заявок 20 января – 20 декабря; оплата по очереди подачи"

RULES = {
    "bulls": {"name": "Бычки на откормплощадку", "rate": 300, "unit": "₸ за кг", "min_kg": 180, "cap_kg": 250,
              "age": (8, 15),
              "terms": "рождён в хозяйстве от племенного быка; 8–15 мес.; от 180 кг, в расчёт не более 250 кг; "
                       "площадка от 500 голов, бычок переоформлен на неё в ИСЖ; заявка — 6 мес. после продажи"},
    "heifers": {"name": "Покупка племенных нетелей", "rate": 260_000, "unit": "₸ за голову",
                "terms": "отечественные, 13–26 мес., племенной статус в ИБСПР; не более 50% цены; "
                         "держать не менее 2 лет; заявка — 12 мес. после покупки"},
    "ram_lambs": {"name": "Баранчики на откорм", "rate": 3_000, "unit": "₸ за голову", "age": (4, 12),
                  "terms": "4–12 мес.; откормплощадка от 1000 голов или бойня от 300 голов в сутки"},
    "bull_buy": {"name": "Покупка племенного быка", "rate": 260_000, "unit": "₸ за голову",
                 "terms": "8–26 мес.; на быка 20–30 маток; использовать не менее 18 мес."},
    "insurance": {"name": "Страхование скота", "rate": 0.8, "unit": "доля премии",
                  "terms": "государство платит 80% страховой премии; животные в ИСЖ; оформление на Kezekte.kz"},
}


@dataclass
class Reason:
    text: str
    heads: int
    amount: float
    fix: str
    fixable: bool = True      # False — этим головам уже не помочь (например, отец не племенной)


@dataclass
class Line:
    key: str
    name: str
    rate_text: str
    terms: str
    kind: str                 # due — положено; option — можно получить; supplier — для поставщиков; info
    heads_ok: int = 0
    amount: float = 0.0
    reasons: list[Reason] = field(default_factory=list)
    deadline: str = ""
    note: str = ""
    calc: str = ""            # как считается сумма, простыми словами
    rows: list[dict] = field(default_factory=list)   # проверка по каждой голове

    @property
    def heads_risk(self) -> int:
        return sum(r.heads for r in self.reasons)

    @property
    def blocked_amount(self) -> float:
        """Под угрозой — можно спасти, если исправить данные или сроки."""
        return sum(r.amount for r in self.reasons if r.fixable)

    @property
    def lost_amount(self) -> float:
        """Не положено в этом году — исправить уже нельзя."""
        return sum(r.amount for r in self.reasons if not r.fixable)


@dataclass
class Assessment:
    lines: list[Line]
    total: float
    blocked: float
    options: float
    blocked_reason: str
    tag_issues: int
    register_size: int
    actions: list[dict]
    lost: float = 0.0
    window: str = WINDOW
    source: str = SRC


MONTHS_NOM = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь",
              "октябрь", "ноябрь", "декабрь"]
MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября",
              "октября", "ноября", "декабря"]
MONTHS_PREP = ["январе", "феврале", "марте", "апреле", "мае", "июне", "июле", "августе", "сентябре",
               "октябре", "ноябре", "декабре"]


def mon(p, prep: bool = False, gen: bool = False) -> str:
    p = pd.Period(p, "M")
    names = MONTHS_GEN if gen else (MONTHS_PREP if prep else MONTHS_NOM)
    return f"{names[p.month - 1]} {p.year}"


# замечание в реестре → что это значит для фермера и что сделать
ISSUES = {
    "нет в ИСЖ": ("не внесены в базу ИСЖ, а без этого субсидию не дадут", "внести бычков в базу ИСЖ до продажи"),
    "нет бирки": ("нет бирки, а без неё субсидию не дадут", "поставить бирку и внести в ИСЖ"),
    "нет акта взвешивания": ("нет акта взвешивания, а без него субсидию не дадут",
                             "взвесить бычков и составить акт в день продажи"),
}


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _age(born: str, at: pd.Period) -> int:
    return (at - pd.Period(born, "M")).n


def _rate_text(r: dict) -> str:
    v = r["rate"]
    return f"{v:.0%} премии" if v < 1 else f"{v:,.0f} {r['unit']}".replace(",", " ")


def _bulls(f: Farm, a: pd.DataFrame, plan: dict) -> tuple[Line, list[dict]]:
    r = RULES["bulls"]
    line = Line("bulls", r["name"], _rate_text(r), r["terms"], "due",
                deadline="заявка — в течение 6 мес. после продажи; подать в день продажи (очередь)")
    sale = pd.Period(plan["month"], "M")
    bulls = a[(a["species"] == "КРС") & (a["sex"] == "M") & ~a["group"].str.contains("производител")]
    sire_ok = bool(((a["group"].str.contains("производител")) & (a["pedigree"] == "племенное")).any())
    days = (sale.to_timestamp() - TODAY.to_timestamp()).days + 15
    buckets: dict[str, list] = {}
    for x in bulls.itertuples():
        age = _age(x.born, sale)
        kg = float(x.weight_kg) + plan.get("adg", 0.8) * max(days, 0)
        pay = min(kg, r["cap_kg"]) * r["rate"]
        row = {"tag": x.tag, "born": x.born, "age": age, "kg": round(kg), "pay": pay}
        line.rows.append(row)
        if not sire_ok:
            why = ("отец не племенной бык — этим бычкам субсидия не положена",
                   "со следующего приплода: купить племенного быка")
        elif not plan.get("buyer_ok"):
            why = (f"покупатель — {plan['buyer']}, а нужна площадка от 500 голов",
                   "продавать на откормплощадку от 500 голов")
        elif x.issue:
            why = ISSUES[x.issue]
        elif age < r["age"][0]:
            ok_month = pd.Period(x.born, "M") + r["age"][0]
            why = (f"в {mon(sale, True)} им будет {age} мес., а субсидию дают с 8",
                   f"продать бычков не раньше {mon(ok_month, gen=True)}")
        elif age > r["age"][1]:
            why = (f"в {mon(sale, True)} им будет {age} мес., а субсидию дают до 15", "продать бычков раньше")
        elif kg < r["min_kg"]:
            why = (f"к продаже будут весить около {kg:.0f} кг, а нужно от 180", "докормить бычков до 180 кг")
        else:
            line.heads_ok += 1
            line.amount += pay
            row["status"] = "проходит"
            continue
        row["status"] = why[0]
        buckets.setdefault(why, []).append(pay)
    actions = []
    for (text, fix), pays in sorted(buckets.items(), key=lambda kv: -sum(kv[1])):
        fixable = not text.startswith("отец не племенной")
        line.reasons.append(Reason(text, len(pays), sum(pays), fix, fixable))
        if fixable:
            actions.append({"label": _cap(fix), "action": f"{_cap(text)} — бычков: {len(pays)}",
                            "money": sum(pays), "deadline": mon(plan["month"])})
    line.note = f"план продажи: {mon(plan['month'])}, покупатель — {plan['buyer']}"
    adg = f"{plan.get('adg', 0.8):.1f}".replace(".", ",")
    line.calc = (f"Бычков к продаже: {len(bulls)}. Вес на дату продажи = вес сейчас + привес {adg} кг в день. "
                 "Сумма за бычка = вес (не больше 250 кг) × 300 ₸.")
    return line, actions


def _heifers(plan: dict) -> tuple[Line, list[dict]]:
    r = RULES["heifers"]
    per_head = min(r["rate"], plan["price"] / plan["count"] * 0.5)
    buy = pd.Period(plan["month"], "M")
    line = Line("heifers", r["name"], _rate_text(r), r["terms"], "due",
                deadline=f"покупка — {mon(buy)}, заявка — до: {mon(buy + 12)}")
    if plan.get("pedigree"):
        line.heads_ok, line.amount = plan["count"], per_head * plan["count"]
        line.note = f"кредит на {plan['count']} нетелей: субсидия вернёт {line.amount / plan['price']:.0%} цены"
        each = plan["price"] / plan["count"]
        sp = lambda v: f"{v:,.0f}".replace(",", " ")
        line.calc = (f"Государство возвращает 260 000 ₸ за голову, но не больше половины цены. Одна нетель стоит "
                     f"≈{sp(each)} ₸, половина — {sp(each / 2)} ₸, значит платят {sp(per_head)} ₸. "
                     f"{plan['count']} нетелей × {sp(per_head)} ₸.")
        act = {"label": "Покупать только племенных нетелей", "money": line.amount, "deadline": mon(plan["month"]),
               "action": "государство вернёт 260 000 ₸ за каждую, если у нетели есть племенные документы"}
    else:
        line.reasons.append(Reason("у продавца нет племенного статуса", plan["count"], per_head * plan["count"],
                                   "искать продавца с племенными свидетельствами"))
        act = {"label": "Искать нетелей с племенным статусом", "money": per_head * plan["count"],
               "deadline": mon(plan["month"]), "action": "иначе субсидии не будет"}
    return line, [act]


def _ram_lambs(a: pd.DataFrame, plan: dict) -> tuple[Line, list[dict]]:
    r = RULES["ram_lambs"]
    line = Line("ram_lambs", r["name"], _rate_text(r), r["terms"], "due",
                deadline="заявка — после продажи, до 20 декабря")
    sale = pd.Period(plan["month"], "M")
    rams = a[(a["species"] == "МРС") & (a["sex"] == "M")]
    buckets: dict[tuple, int] = {}
    for x in rams.itertuples():
        age = _age(x.born, sale)
        if not (r["age"][0] <= age <= r["age"][1]):
            continue                      # не баранчик по возрасту — не считаем
        if x.issue:
            key = (x.issue, "поставить баранчикам бирки и внести в ИСЖ" if x.issue != "нет акта взвешивания"
                   else "составить акт взвешивания")
            buckets[key] = buckets.get(key, 0) + 1
        else:
            line.heads_ok += 1
            line.amount += r["rate"]
    line.calc = f"Баранчиков 4–12 мес. к продаже: {line.heads_ok + sum(buckets.values())}. За каждого — 3 000 ₸."
    actions = []
    for (text, fix), n in buckets.items():
        line.reasons.append(Reason(text, n, n * r["rate"], fix))
        actions.append({"label": _cap(fix), "action": f"{_cap(text)}, а без этого субсидию не дадут — баранчиков: {n}",
                        "money": n * r["rate"], "deadline": mon(plan["month"])})
    line.note = f"план продажи: {mon(plan['month'])}, покупатель — {plan['buyer']}"
    return line, actions


def _bull_buy(a: pd.DataFrame, lost_per_year: float = 0.0) -> tuple[Line | None, list[dict]]:
    cows = int(((a["species"] == "КРС") & (a["group"].str.contains("маточн"))).sum())
    has = bool(((a["group"].str.contains("производител")) & (a["pedigree"] == "племенное")).any())
    if has or cows == 0:
        return None, []
    r = RULES["bull_buy"]
    n = max(1, round(cows / 25))
    line = Line("bull_buy", r["name"], _rate_text(r), r["terms"], "option", heads_ok=n, amount=n * r["rate"],
                deadline="заявка — 12 мес. после покупки",
                note=f"{cows} маток → нужно {n} быка; без племенного отца бычки не дают 300 ₸/кг")
    label = "Купить племенного быка" if n == 1 else f"Купить {n} племенных быка"
    now = ("сейчас бычки не получают субсидию, потому что отец не племенной" if lost_per_year
           else "бычки от племенного отца смогут получать субсидию")
    return line, [{"label": label, "money": n * r["rate"], "deadline": "до случки",
                   "action": f"государство вернёт 260 000 ₸ за быка; {now}", "kind": "gain"}]


def _suppliers(a: pd.DataFrame) -> tuple[Line, list[dict]]:
    r = RULES["bulls"]
    bad = a[a["issue"] == "нет в ИСЖ"]
    per = r["cap_kg"] * r["rate"]
    line = Line("suppliers", "Субсидия ваших поставщиков бычков", _rate_text(r), r["terms"], "supplier",
                heads_ok=int((a["issue"] != "нет в ИСЖ").sum()),
                deadline="переоформить в ИСЖ сразу после приёмки",
                note="деньги получает поставщик, но только если площадка переоформила бычка на себя в ИСЖ")
    if len(bad):
        line.reasons.append(Reason("бычки не переоформлены на площадку в ИСЖ", len(bad), len(bad) * per,
                                   "переоформить в ИСЖ"))
    actions = ([{"label": "Переоформить бычков в ИСЖ", "money": len(bad) * per, "deadline": "сразу",
                "action": f"иначе поставщики теряют субсидию и могут перестать возить вам скот — голов: {len(bad)}"}]
               if len(bad) else [])
    return line, actions


def _insurance(a: pd.DataFrame) -> Line:
    r = RULES["insurance"]
    bad = int((a["issue"] == "нет в ИСЖ").sum())
    line = Line("insurance", r["name"], _rate_text(r), r["terms"], "info",
                heads_ok=len(a) - bad, deadline="полис на 3, 6, 9 или 12 мес.",
                note="сумма зависит от тарифа страховой — не рассчитана; банк часто требует страховку залога")
    if bad:
        line.reasons.append(Reason("нет в ИСЖ — такую голову не застраховать", bad, 0.0, "зарегистрировать в ИСЖ"))
    return line


def assess(f: Farm) -> Assessment:
    a = f.animals if f.animals is not None else pd.DataFrame(
        columns=["tag", "species", "sex", "group", "breed", "pedigree", "born", "weight_kg", "issue"])
    plan = f.card.get("plan", {})
    lines, actions = [], []
    if "sell_bulls" in plan:
        l, act = _bulls(f, a, plan["sell_bulls"])
        lines.append(l)
        actions += act
    if "buy" in plan:
        l, act = _heifers(plan["buy"])
        lines.append(l)
        actions += act
    if "sell_ram_lambs" in plan:
        l, act = _ram_lambs(a, plan["sell_ram_lambs"])
        lines.append(l)
        actions += act
    lost_bulls = sum(l.lost_amount for l in lines if l.key == "bulls")
    l, act = _bull_buy(a, lost_bulls)
    if l:
        lines.append(l)
        actions += act
    if plan.get("feedlot"):
        l, act = _suppliers(a)
        lines.append(l)
        actions += act
    lines.append(_insurance(a))

    due = [l for l in lines if l.kind == "due"]
    total = sum(l.amount for l in due)
    blocked = sum(l.blocked_amount for l in due)
    top = max((r for l in due for r in l.reasons if r.fixable), key=lambda r: r.amount, default=None)
    actions.sort(key=lambda x: -(x["money"] or 0))
    return Assessment(
        lines=lines, total=total, blocked=blocked,
        options=sum(l.amount for l in lines if l.kind == "option"),
        blocked_reason=f"{top.heads} гол.: {top.text}" if top else "",
        tag_issues=int((a["issue"] != "").sum()) if len(a) else 0,
        register_size=len(a), actions=actions, lost=sum(l.lost_amount for l in due))


def merge_actions(credit_actions: list[dict], subsidy_actions: list[dict], top: int = 5) -> list[dict]:
    """Общий список «что сделать»: сначала то, что приносит деньги (по сумме),
    затем то, что улучшает кредитный класс (по баллам)."""
    gains = [dict(x, source="субсидии", kind="gain") for x in subsidy_actions]
    gains += [dict(x, source="кредит") for x in credit_actions if x.get("kind") == "gain"]
    gains.sort(key=lambda x: -(x.get("money") or 0))
    rest = [dict(x, source="кредит") for x in credit_actions if x.get("kind") != "gain"]
    rest.sort(key=lambda x: -(x.get("delta") or 0))
    return (gains[:max(top - 1, 1)] + rest)[:top]


def as_dict(s: Assessment) -> dict:
    d = asdict(s)
    for l, src in zip(d["lines"], s.lines):
        l["heads_risk"] = src.heads_risk
        l["blocked_amount"] = src.blocked_amount
        l["lost_amount"] = src.lost_amount
    return d
