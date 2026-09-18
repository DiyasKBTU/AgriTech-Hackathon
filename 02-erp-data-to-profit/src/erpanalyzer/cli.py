"""ERPAnalyzer: проверка данных ERP хозяйства для льготного кредита и субсидий.
Все действия — через одну команду `erpanalyzer`.

    erpanalyzer prices   разобрать скачанные файлы stat.gov.kz → data/prices_kz.csv
    erpanalyzer farms    собрать хозяйства-примеры → data/erp_farms/
    erpanalyzer show     результаты по хозяйствам текстом
    erpanalyzer serve    сайт, http://127.0.0.1:8010
    erpanalyzer deck     презентация: снимки запущенного сайта + docs/presentation/ERPAnalyzer.pptx
    erpanalyzer all      prices + farms + show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
FARMS = DATA / "erp_farms"


def cmd_prices(a):
    from . import prices as P
    df = P.build(RAW / "stat_gov_kz_prices", DATA, RAW / "stat_gov_kz_prices_2018_2021")
    print(f"цены: {len(df)} строк, {df['month'].nunique()} мес., {df['region'].nunique()} регионов")
    for k in ("cattle", "barley", "hay"):
        m, v = P.latest(df, k)
        print(f"  {P.PRODUCT_NAMES[k]:<18} {m}  {v:8.1f} ₸/кг")


def cmd_farms(a):
    from .farms import build
    print(f"хозяйства-примеры: {', '.join(build(DATA, FARMS))} → {FARMS}")


def cmd_show(a):
    from . import credit as C, prices as P, subsidies as S, verify as V
    prices = P.load(DATA)
    presence = V.load_presence(DATA / "cv_snapshot.json")
    for f in C.load_farms(FARMS).values():
        p = C.build_profile(f, prices, check=V.verify(f.card, presence))
        s = S.assess(f)
        print(f"{f.card['name']:<40} класс {p.klass} ({p.score:.0f})  просит {f.card['loan']['amount'] / 1e6:.0f} млн"
              f"  можно {p.limit / 1e6:.1f} млн  DSCR {p.dscr:.2f}")
        print(f"   субсидии: положено {s.total / 1e6:.2f} млн, под угрозой {s.blocked / 1e6:.2f} млн,"
              f" можно получить ещё {s.options / 1e6:.2f} млн")
        for x in S.merge_actions(C.improvements(f, prices, p), s.actions):
            print(f"   что сделать: {x['label']} — {x['action']}")


def cmd_serve(a):
    import uvicorn
    from .web import create_app
    uvicorn.run(create_app(DATA, FARMS, DATA / "cv_snapshot.json"), host=a.host, port=a.port, log_level="info")


def cmd_deck(a):
    from . import deck
    shots = ROOT / "docs" / "presentation" / "shots"
    if not a.no_shots:
        deck.shots(shots)
    out = deck.build(shots, ROOT / "docs" / "presentation" / "ERPAnalyzer.pptx")
    print(f"презентация: {out}")


def cmd_all(a):
    cmd_prices(a)
    cmd_farms(a)
    cmd_show(a)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="erpanalyzer", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("prices", cmd_prices), ("farms", cmd_farms), ("show", cmd_show), ("all", cmd_all)):
        sub.add_parser(name).set_defaults(f=fn)
    d = sub.add_parser("deck")
    d.add_argument("--no-shots", action="store_true", help="не переснимать скриншоты")
    d.set_defaults(f=cmd_deck)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8010)
    s.set_defaults(f=cmd_serve)
    a = ap.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
