"""ERPAnalyzer — MVP. Хозяйство: данные из ERP → «Проверить» → результат, как посчитано, что сделать.
Банк: заявки и справка по каждой. Плюс JSON для системы банка."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, PackageLoader, select_autoescape

from . import concept, credit as C, prices as P, subsidies as S, verify as V

HERE = Path(__file__).parent

# показатель с плохим баллом → как объяснить без терминов
PLAIN = {
    "stress": "в плохой год денег на платежи не хватит",
    "mortality": "высокий падёж",
    "feed": "мало корма в запасе",
    "productivity": "мало приплода",
    "growth": "выручка падает",
    "costs": "расходы растут быстрее выручки",
    "margin": "низкая прибыльность",
    "cost": "себестоимость близка к цене продажи",
    "herd": "стадо сокращается",
    "collateral": "стадо стоит меньше кредита",
    "verified": "скот не подтверждён камерой",
}
NAMES = {
    "growth": "Выручка к прошлому году", "costs": "Расходы к прошлому году",
    "margin": "Сколько остаётся с выручки", "cost": "Себестоимость 1 кг живого веса",
    "dscr": "Во сколько раз деньги больше платежей", "herd": "Поголовье за год",
    "productivity": "Приплод или привес", "mortality": "Падёж за год", "feed": "Запас корма",
    "collateral": "Скот как залог", "verified": "Скот подтверждён камерой",
    "stress": "Во сколько раз деньги больше платежей в плохой год",
}
NORMS = {"dscr": "не меньше 1,2", "stress": "не меньше 1,0"}
SCENARIOS = {"Засуха: сено дорожает": "Засуха — сено дорожает", "Ячмень дорожает": "Ячмень дорожает",
             "Цена КРС −10%": "Скот дешевеет на 10%", "Падёж +3% стада": "Падёж выше на 3% стада"}
KG = {"cattle": "КРС", "sheep": "овцы", "hay": "сено", "barley": "ячмень"}


def mln(v: float) -> str:
    return f"{v / 1e6:.1f}".replace(".", ",").replace("-", "−")


def tg(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ").replace("-", "−")


def amt(v: float) -> str:
    if v == 0:
        return "0 ₸"
    if abs(v) >= 1e6:
        return f"{mln(v)} млн ₸"
    return f"{v / 1000:.0f} тыс. ₸"


def dec(v: float, d: int = 1) -> str:
    return f"{v:,.{d}f}".replace(",", " ").replace(".", ",").replace("-", "−")


def cap(s: str) -> str:
    """Первая буква заглавная, остальное как есть (в отличие от capitalize не портит «ИСЖ»)."""
    return s[:1].upper() + s[1:]


def times(v: float) -> str:
    return f"{v:.2f}".replace(".", ",").replace("-", "−")


def enough(dscr: float) -> str:
    return "хватит с запасом" if dscr >= 1.2 else ("впритык" if dscr >= 1 else "не хватит")


def create_app(prices_dir: Path, farms_dir: Path, cv_snapshot: Path) -> FastAPI:
    app = FastAPI(title=concept.NAME)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    env = Environment(loader=PackageLoader("erpanalyzer", "templates"), autoescape=select_autoescape())
    env.filters.update(mln=mln, tg=tg, dec=dec, amt=amt, times=times, enough=enough, cap=cap)
    prices = P.load(prices_dir)
    farms = C.load_farms(farms_dir)
    shocks = C.historical_shocks(prices)
    cache: dict = {"at": 0.0, "value": None}

    def presence() -> dict:
        if cache["value"] is None or time.monotonic() - cache["at"] > 60:
            cache.update(at=time.monotonic(), value=V.load_presence(cv_snapshot))
        return cache["value"]

    def check(f: C.Farm) -> dict:
        chk = V.verify(f.card, presence())
        pr = C.build_profile(f, prices, check=chk)
        pr.improve = C.improvements(f, prices, pr, check=chk)
        return {"f": f, "chk": chk, "pr": pr, "subs": S.assess(f)}

    def price(product: str, region: str) -> tuple[str, float]:
        return P.latest(prices, product, region)

    # --- экран «Данные из ERP» ---------------------------------------------------
    def data_view(e: dict) -> dict:
        f = e["f"]
        fin = f.fin.tail(12)
        feed = f.feed.tail(12)
        region = f.card["region"]
        a = f.animals
        groups = (a.groupby(["species", "group"]).size().reset_index(name="n").values.tolist()
                  if a is not None else [])
        return {
            "fin": [(r.month, C.revenue(fin.loc[[i]]).iloc[0], C.expenses(fin.loc[[i]]).iloc[0], r.debt_payment)
                    for i, r in zip(fin.index, fin.itertuples())],
            "fin_total": (C.revenue(fin).sum(), C.expenses(fin).sum(), fin["debt_payment"].sum()),
            "feed": list(feed[["month", "hay_use_kg", "hay_stock_kg", "hay_tg_kg"]].itertuples(index=False)),
            "herd": f.herd.iloc[-1],
            "animals": a.to_dict("records") if a is not None else [],
            "groups": groups,
            "issues": int((a["issue"] != "").sum()) if a is not None else 0,
            "prices": [(KG[p], *price(p, region)) for p in ("cattle", "sheep", "hay", "barley")
                       if p != "sheep" or f.herd.iloc[-1]["sheep_kg"] > 0],
        }

    # --- экран «Результат» --------------------------------------------------------
    def results(e: dict) -> list[dict]:
        """Три итога проверки: кредит, субсидии, залог."""
        f, pr, subs, chk = e["f"], e["pr"], e["subs"], e["chk"]
        asked = f.card["loan"]["amount"]
        text = f"Просит {mln(asked)} млн — " + ("в пределах." if pr.limit >= asked else "больше, чем хозяйство потянет.")
        text += f" Оценка: {concept.CLASS_WORDS[pr.klass]}."
        if min(s.dscr for s in pr.scenarios[:-1]) < 1:
            text += " В засуху денег на платежи может не хватить."
        credit = {"id": "credit", "title": "Льготный кредит", "value": f"до {mln(pr.limit)} млн ₸", "text": text}

        supplier = sum(l.blocked_amount for l in subs.lines if l.kind == "supplier")
        if subs.total or subs.blocked:
            text = "положено по данным ERP."
            if subs.blocked:
                text += f" Ещё {amt(subs.blocked)} можно потерять — исправьте пункты из списка «что сделать»."
            if subs.lost:
                text += " Бычкам в этом году субсидия не положена: их отец не племенной."
            sub = {"id": "subs", "title": "Субсидии", "value": amt(subs.total), "text": text}
        elif supplier:
            sub = {"id": "subs", "title": "Субсидии", "value": "своих нет",
                   "text": f"Поставщики бычков теряют {amt(supplier)}: бычки не переоформлены в ИСЖ."}
        else:
            sub = {"id": "subs", "title": "Субсидии", "value": "нет по плану", "text": ""}

        if chk.get("connected"):
            coll = {"id": "coll", "title": "Залог", "value": f"{chk['seen']} из {chk['expected']}",
                    "text": "коров под камерой на месте — банк видит, что скот есть."}
        else:
            coll = {"id": "coll", "title": "Залог", "value": "камеры нет", "text": "банк видит скот только по записям в ERP."}
        return [credit, sub, coll]

    def credit_steps(e: dict) -> list[tuple[str, str, str, str]]:
        """Кредит по шагам: (что считаем, откуда данные, расчёт, результат)."""
        f, pr = e["f"], e["pr"]
        loan = f.card["loan"]
        fin = f.fin.tail(12)
        period = f"{fin['month'].iloc[0]} — {fin['month'].iloc[-1]}"
        if len(fin) < 12:
            period += f" ({len(fin)} мес., пересчитано на год)"
        pay = C.annuity(loan["amount"], loan["months"], loan["rate"])
        pays = pr.debt12 + pr.new_pay12
        room = pr.ebitda / C.DSCR_MIN - pr.debt12
        last = f.herd.iloc[-1]
        m, cp = price("cattle", f.card["region"])
        _, sp = price("sheep", f.card["region"])
        herd = f"живой вес КРС {dec(last['cattle_kg'] / 1000)} т × {tg(cp)} ₸/кг"
        if last["sheep_kg"]:
            herd += f" + овец {dec(last['sheep_kg'] / 1000)} т × {tg(sp)} ₸/кг"
        rate = f"{loan['rate']:.0%}"
        return [
            ("Деньги за год", f"Финансы, {period}",
             f"выручка {mln(pr.rev12)} млн − расходы {mln(pr.exp12)} млн", f"остаётся {mln(pr.ebitda)} млн ₸"),
            ("Платёж по новому кредиту", "Заявка",
             f"{mln(loan['amount'])} млн на {loan['months']} мес. под {rate}", f"{tg(pay)} ₸ в месяц"),
            ("Все платежи за год", "Финансы + заявка",
             f"старые кредиты {mln(pr.debt12)} млн + новый {mln(pr.new_pay12)} млн", f"{mln(pays)} млн ₸"),
            ("Хватит ли денег", "",
             f"{mln(pr.ebitda)} ÷ {mln(pays)} — банку нужно не меньше 1,2 раза",
             f"в {times(pr.dscr)} раза: {enough(pr.dscr)}"),
            ("Можно дать — по деньгам", "",
             f"на платежи можно тратить {mln(pr.ebitda)} ÷ 1,2 − старые {mln(pr.debt12)} = {mln(room)} млн в год; "
             f"такой платёж на {loan['months']} мес. под {rate} — это кредит", f"{mln(pr.limit_cash)} млн ₸"),
            ("Сколько стоит стадо", f"Животные + цены stat.gov.kz, {f.card['region']} обл., {m}",
             herd, f"{mln(pr.herd_value)} млн ₸"),
            ("Можно дать — по залогу", "", "скот берут в залог за половину стоимости", f"{mln(pr.limit_collateral)} млн ₸"),
            ("Итог", "", f"меньшее из {mln(pr.limit_cash)} и {mln(pr.limit_collateral)}", f"до {mln(pr.limit)} млн ₸"),
        ]

    def bad_year(pr: C.Profile) -> list[tuple[str, str, float, float]]:
        pays = pr.debt12 + pr.new_pay12
        rows = [("Обычный год", "как в учёте ERP", pr.ebitda, pr.dscr)]
        rows += [(SCENARIOS.get(s.name, s.name), s.source, s.ebitda, s.dscr) for s in pr.scenarios[:-1]]
        return [(n, src, e, d, pays) for n, src, e, d in rows]

    def indicators(pr: C.Profile) -> list[dict]:
        return [{"name": NAMES.get(i.key, i.name), "shown": i.shown, "norm": NORMS.get(i.key, i.norm),
                 "points": round(i.points), "weight": int(i.weight)} for i in pr.indicators]

    def reasons(pr: C.Profile) -> list[str]:
        out = ["учёт в ERP меньше года"] if pr.months < 12 else []
        if pr.dscr < 1:
            out.append("денег на платежи не хватает уже сейчас")
        for i in sorted(pr.indicators, key=lambda i: i.points):
            if i.points < 40 and i.value is not None and i.key in PLAIN:
                out.append(PLAIN[i.key])
        return out

    def render(name: str, **kw) -> HTMLResponse:
        return HTMLResponse(env.get_template(name).render(k=concept, farms=farms, **kw))

    @app.get("/")
    def home():
        return RedirectResponse(f"/{next(iter(farms))}")

    @app.get("/bank", response_class=HTMLResponse)
    def bank():
        rows = []
        for f in farms.values():
            pr = check(f)["pr"]
            rows.append({"f": f, "pr": pr, "word": concept.CLASS_WORDS[pr.klass],
                         "why": ", ".join(reasons(pr)[:2]) or "замечаний нет"})
        return render("bank.html", role="bank", rows=rows)

    @app.get("/bank/{fid}", response_class=HTMLResponse)
    def bank_farm(fid: str):
        if fid not in farms:
            return HTMLResponse("нет такого хозяйства", status_code=404)
        e = check(farms[fid])
        return render("bank_farm.html", role="bank", fid=fid, results=results(e), why=reasons(e["pr"]),
                      steps=credit_steps(e), bad=bad_year(e["pr"]), **e)

    @app.get("/api/profile/{fid}")
    def api_profile(fid: str):
        if fid not in farms:
            return JSONResponse({"error": "нет такого хозяйства"}, status_code=404)
        e = check(farms[fid])
        pr, chk, subs = e["pr"], e["chk"], e["subs"]
        return JSONResponse({
            "farm": farms[fid].card["name"], "example": True, "months_of_history": pr.months,
            "class": pr.klass, "score": round(pr.score, 1), "dscr": round(pr.dscr, 2),
            "limit_tg": round(pr.limit), "herd_value_tg": round(pr.herd_value),
            "herd_verified": {"connected": chk["connected"], "seen": chk.get("seen"), "expected": chk.get("expected")},
            "subsidies_tg": round(subs.total), "subsidies_at_risk_tg": round(subs.blocked),
            "reasons": reasons(pr),
            "bad_year": [{"name": s.name, "dscr": round(s.dscr, 2)} for s in pr.scenarios],
        })

    @app.get("/api/subsidies/{fid}")
    def api_subsidies(fid: str):
        if fid not in farms:
            return JSONResponse({"error": "нет такого хозяйства"}, status_code=404)
        return JSONResponse(S.as_dict(S.assess(farms[fid])))

    @app.get("/{fid}", response_class=HTMLResponse)
    def farm_data(fid: str):
        if fid not in farms:
            return HTMLResponse("нет такого хозяйства", status_code=404)
        e = check(farms[fid])
        return render("farm_data.html", role="farm", fid=fid, d=data_view(e), shocks=shocks, **e)

    @app.get("/{fid}/check", response_class=HTMLResponse)
    def farm_check(fid: str):
        if fid not in farms:
            return HTMLResponse("нет такого хозяйства", status_code=404)
        e = check(farms[fid])
        todo = S.merge_actions(e["pr"].improve, e["subs"].actions, top=5)
        return render("farm_check.html", role="farm", fid=fid, results=results(e), todo=todo,
                      steps=credit_steps(e), bad=bad_year(e["pr"]), ind=indicators(e["pr"]), **e)

    return app
