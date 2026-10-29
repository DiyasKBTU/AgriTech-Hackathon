"""Детектор отклонений на настоящих болезнях: открытые данные INRAE (Франция).

На MmCows проверено только одно — ложных тревог на здоровом стаде нет:
болезней за время записи там не было. Здесь — 4 фермы, 386 коров, около
98 тысяч коров-суток, и в журнале отмечены хромота, мастит, ацидоз, другие
болезни, вызванное воспаление (LPS — опыт), охота и отёл.

Данные — система определения положения коров в коровнике, а не камера, но
признаки те же, что считает камера: сколько корова ест, лежит и ходит.
Детектор берётся как есть, с прежними настройками: под эти данные ничего
не подбиралось.

Источник: Data on cow activity in relation to health and welfare,
https://doi.org/10.15454/52J8YS, лицензия Etalab 2.0. Файлы — исходные CSV
авторов, MD5 сверены: `data/real/inrae_health/`.

Запуск: `cowid inrae-check` → `reports/inrae/health_check.json`.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

from .anomaly.baseline import PersonalBaseline
from .config import BaselineConfig
from .types import ActivityFeatures

ROOT = Path("data/real/inrae_health")
FARMS = {
    "ферма 1 (28 коров, 2018–2019)": "dataset1-1.csv",
    "ферма 2 (28 коров, 2015)": "dataset2-1.csv",
    "ферма 3 (30 коров, 2013)": "dataset3-1.csv",
    "ферма 4 (300 коров, 2014–2015)": "dataset4-1.csv",
}
REPORT = Path("reports/inrae/health_check.json")

#: Болезни, которые детектор должен замечать, — по-русски.
ILLNESS = {"lameness": "хромота", "mastitis": "мастит", "acidosis": "ацидоз",
           "LPS": "вызванное воспаление (LPS, опыт)", "other_disease": "другие болезни"}
OTHER = {"oestrus": "охота", "calving": "отёл"}
#: Окно «поймал»: от 3 суток до начала эпизода до 2 суток после.
BEFORE, AFTER = 3, 2
#: Сутки вокруг событий, где отметка не считается ложной.
QUIET_BEFORE, QUIET_AFTER = 3, 7

Log = Callable[[str], None]


def load_farm(path: Path, hourly: bool = False
              ) -> tuple[list[ActivityFeatures], dict[tuple[str, date], set[str]]]:
    """Часы → сутки коровы и отметки журнала по суткам.

    Время у корма — EAT, лёжа — REST, «подвижность» — время в проходах
    (IN_ALLEYS): пути в метрах эта система не даёт, а детектору важно не
    единица, а отличие коровы от её же нормы. `hourly` — ещё и разбивка по
    часам суток для признаков «когда ест и лежит».
    """
    rows = []
    flags: dict[tuple[str, date], set[str]] = defaultdict(set)
    last_hour = 0
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                alleys, rest, eat = float(r["IN_ALLEYS"]), float(r["REST"]), float(r["EAT"])
                hour = int(float(r["hour"]))
            except ValueError:
                continue
            key = (r["cow"], date.fromisoformat(r["date"]))
            rows.append((key, hour, alleys, rest, eat))
            last_hour = max(last_hour, hour)
            for name in list(ILLNESS) + list(OTHER):
                if r.get(name) == "1":
                    flags[key].add(name)
    # В части файлов часы 0–23, в части 1–24 (час, который закончился).
    shift = 1 if last_hour == 24 else 0
    days: dict = defaultdict(lambda: {"alleys": 0.0, "feeder": [0.0] * 24, "resting": [0.0] * 24,
                                      "observed": [0.0] * 24})
    for key, hour, alleys, rest, eat in rows:
        d = days[key]
        h = (hour - shift) % 24
        d["alleys"] += alleys
        d["feeder"][h] += eat
        d["resting"][h] += rest
        d["observed"][h] += alleys + rest + eat
    features = [ActivityFeatures(cow_id=cow, day=day, feeder_seconds=sum(d["feeder"]),
                                 resting_seconds=sum(d["resting"]), distance_m=d["alleys"],
                                 observed_seconds=sum(d["observed"]),
                                 hours=({k: d[k] for k in ("feeder", "resting", "observed")}
                                        if hourly else None))
                for (cow, day), d in days.items()]
    return features, flags


def run_detector(features: list[ActivityFeatures], cfg: BaselineConfig) -> dict[tuple[str, date], str]:
    """Статус каждой коровы на каждые сутки — тем же кодом, что на ферме."""
    by_day = defaultdict(list)
    for f in features:
        by_day[f.day].append(f)
    baseline = PersonalBaseline(cfg)
    for day in sorted(by_day):
        baseline.evaluate_day(by_day[day])
    return {(a.cow_id, a.day): a.status for a in baseline.assessments}


def episodes(flags: dict[tuple[str, date], set[str]], kind: str) -> list[tuple[str, date, date]]:
    """Подряд идущие сутки с отметкой (разрыв в одни сутки — тот же эпизод)."""
    by_cow = defaultdict(list)
    for (cow, day), kinds in flags.items():
        if kind in kinds:
            by_cow[cow].append(day)
    out = []
    for cow, days in by_cow.items():
        days.sort()
        start = prev = days[0]
        for d in days[1:]:
            if (d - prev).days > 2:
                out.append((cow, start, prev))
                start = d
            prev = d
        out.append((cow, start, prev))
    return out


FLAGGED = ("alert", "watch")
ASSESSED = ("ok", "watch", "alert", "estrus")


def evaluate(status: dict[tuple[str, date], str], flags: dict[tuple[str, date], set[str]],
             period: tuple[date, date] = (date.min, date.max)) -> dict:
    """Метрики по эпизодам и спокойным суткам, начавшимся в `period` (включительно)."""
    lo_day, hi_day = period
    assessed = {k: s for k, s in status.items() if s in ASSESSED and lo_day <= k[1] <= hi_day}
    flagged_share = sum(s in FLAGGED for s in assessed.values()) / max(1, len(assessed))
    estrus_share = sum(s == "estrus" for s in assessed.values()) / max(1, len(assessed))

    def window(cow, onset, lo, hi):
        return [(onset + timedelta(days=i), assessed.get((cow, onset + timedelta(days=i))))
                for i in range(-lo, hi + 1)]

    out: dict = {"коров-суток оценено": len(assessed),
                 "доля суток с отметкой («проверить» или «на заметку»)": round(flagged_share, 4)}
    illness_rows = {}
    for kind, name in ILLNESS.items():
        rows = []
        for cow, onset, _ in episodes(flags, kind):
            if not lo_day <= onset <= hi_day:
                continue
            w = window(cow, onset, BEFORE, AFTER)
            if not any(s for _, s in w):
                continue                       # норма коровы ещё не набрана — не оценивается
            hits = [d for d, s in w if s in FLAGGED]
            alerts = [d for d, s in w if s == "alert"]
            rows.append({"cow": cow, "onset": onset.isoformat(), "hit": bool(hits),
                         "alert": bool(alerts),
                         "lead_days": (onset - min(hits)).days if hits else None})
        if not rows:
            continue
        n = len(rows)
        hit = sum(r["hit"] for r in rows)
        leads = sorted(r["lead_days"] for r in rows if r["hit"])
        chance = 1 - (1 - flagged_share) ** (BEFORE + AFTER + 1)
        illness_rows[name] = {
            "эпизодов": n, "поймано": hit, "доля": round(hit / n, 3),
            "из них «проверить»": sum(r["alert"] for r in rows),
            "заранее (до отметки в журнале)": sum(1 for r in rows if r["hit"] and r["lead_days"] > 0),
            "медиана, суток до отметки": (leads[len(leads) // 2] if leads else None),
            "случайный контроль: доля": round(chance, 3),
        }
    out["болезни"] = illness_rows

    oe = []
    for cow, onset, _ in episodes(flags, "oestrus"):
        if not lo_day <= onset <= hi_day:
            continue
        w = window(cow, onset, 1, 1)
        if any(s for _, s in w):
            oe.append(any(s == "estrus" for _, s in w))
    if oe:
        out["охота"] = {"эпизодов": len(oe), "поймано": sum(oe), "доля": round(sum(oe) / len(oe), 3),
                        "случайный контроль: доля": round(1 - (1 - estrus_share) ** 3, 3)}

    # Ложные: отметка в сутки, когда у коровы нет события в окне −3…+7.
    near = set()
    for (cow, day), kinds in flags.items():
        if kinds & (set(ILLNESS) | set(OTHER)):
            for i in range(-QUIET_AFTER, QUIET_BEFORE + 1):
                near.add((cow, day + timedelta(days=i)))
    quiet = {k: s for k, s in assessed.items() if k not in near}
    false = sum(s in FLAGGED for s in quiet.values())
    false_alert = sum(s == "alert" for s in quiet.values())
    out["спокойные сутки"] = {
        "коров-суток": len(quiet),
        "отметок («проверить» или «на заметку»)": false,
        "на 100 коров в месяц": round(false / max(1, len(quiet)) * 100 * 30, 1),
        "из них «проверить»": false_alert,
        "«проверить» на 100 коров в месяц": round(false_alert / max(1, len(quiet)) * 100 * 30, 1),
    }
    return out


#: Часть для подбора `max_episode_days` — ферма 4 до июля 2015; всё остальное —
#: проверка, считается один раз.
DEV_FARM = "ферма 4 (300 коров, 2014–2015)"
DEV_PERIOD = (date(2014, 12, 1), date(2015, 6, 30))
TEST_FROM = date(2015, 7, 1)


#: Версии детектора на проверочной части. «после» — текущие настройки.
#: Признаки по часам суток (BLOCKS) на части подбора оказались хуже — отметок в
#: 3–4 раза больше при той же поимке (`reports/inrae/tuning_dev.json`), — сюда не взяты.
VERSIONS = (
    ("как было", {"max_episode_days": 0, "cusum_h": 2.5}),
    ("новая норма", {"max_episode_days": 5, "cusum_h": 2.5}),
    ("после", {}),
)


def run(cfg: BaselineConfig, root: Path = ROOT, log: Log = print) -> dict:
    """Версии детектора на проверочной части: как было, новая норма, текущие настройки."""
    report = {"источник": "INRAE, Data on cow activity in relation to health and welfare, "
                          "doi:10.15454/52J8YS, Etalab 2.0",
              "правила": {"поймал": f"«проверить» или «на заметку» за {BEFORE} сут до начала эпизода "
                                    f"… {AFTER} сут после",
                          "охота": "«признаки охоты» за ±1 сут",
                          "ложное": f"отметка, когда у коровы нет болезни, охоты, отёла за "
                                    f"{QUIET_BEFORE} сут до … {QUIET_AFTER} сут после",
                          "подвижность": "время в проходах (пути в метрах в данных нет)",
                          "подбор": f"max_episode_days и cusum_h выбраны на {DEV_FARM}, "
                                    f"{DEV_PERIOD[0]:%d.%m.%Y}–{DEV_PERIOD[1]:%d.%m.%Y}; "
                                    "остальное — проверка. Проверочная часть уже раз смотрелась "
                                    "для версии «новая норма» — для «после» она уже не совсем новая",
                          "остальные настройки": "как на ферме, под эти данные не подбирались"},
              "max_episode_days": cfg.max_episode_days, "cusum_h": cfg.cusum_h,
              "версии": {label: {"настройки": upd or {"max_episode_days": cfg.max_episode_days,
                                                       "cusum_h": cfg.cusum_h},
                                 "подбор": {}, "проверка": {}} for label, upd in VERSIONS}}
    for name, filename in FARMS.items():
        path = root / filename
        if not path.exists():
            log(f"{name}: нет файла {path}")
            continue
        features, flags = load_farm(path)
        test = (TEST_FROM, date.max) if name == DEV_FARM else (date.min, date.max)
        for label, upd in VERSIONS:
            status = run_detector(features, cfg.model_copy(update=upd))
            report["версии"][label]["проверка"][name] = evaluate(status, flags, test)
            if name == DEV_FARM:
                report["версии"][label]["подбор"][name] = evaluate(status, flags, DEV_PERIOD)
        log(f"{name}: {len(features)} коров-суток")
    report["всего"] = {label: _total(report["версии"][label]["проверка"]) for label, _ in VERSIONS}
    report["summary_text"] = summary_text(report)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def _total(farms: dict) -> dict:
    out: dict = {"болезни": {}}
    for farm in farms.values():
        for name, r in farm.get("болезни", {}).items():
            t = out["болезни"].setdefault(name, {"эпизодов": 0, "поймано": 0, "заранее": 0,
                                                 "случайно ожидалось": 0.0})
            t["эпизодов"] += r["эпизодов"]
            t["поймано"] += r["поймано"]
            t["заранее"] += r["заранее (до отметки в журнале)"]
            t["случайно ожидалось"] += r["эпизодов"] * r["случайный контроль: доля"]
    for t in out["болезни"].values():
        t["доля"] = round(t["поймано"] / max(1, t["эпизодов"]), 3)
        t["случайно ожидалось"] = round(t["случайно ожидалось"], 1)
    q = [f["спокойные сутки"] for f in farms.values()]
    days = sum(x["коров-суток"] for x in q)
    out["ложные на 100 коров в месяц"] = round(sum(x["отметок («проверить» или «на заметку»)"] for x in q)
                                               / max(1, days) * 100 * 30, 1)
    out["ложные «проверить» на 100 коров в месяц"] = round(sum(x["из них «проверить»"] for x in q)
                                                          / max(1, days) * 100 * 30, 1)
    oe = [f["охота"] for f in farms.values() if "охота" in f]
    if oe:
        out["охота"] = {"эпизодов": sum(x["эпизодов"] for x in oe), "поймано": sum(x["поймано"] for x in oe)}
    return out


def summary_text(r: dict) -> str:
    lines = ["Детектор отклонений на настоящих болезнях (INRAE, проверочная часть: "
             "ферма 4 с июля 2015 и фермы 1–3 целиком):"]
    for label in r["всего"]:
        t = r["всего"][label]
        lines.append(f" {label} {r['версии'][label]['настройки']}:")
        for name, x in t["болезни"].items():
            lines.append(f"  {name}: поймано {x['поймано']} из {x['эпизодов']} ({x['доля']:.0%}), "
                         f"заранее {x['заранее']}; наугад с той же частотой — {x['случайно ожидалось']}")
        if "охота" in t:
            lines.append(f"  охота: {t['охота']['поймано']} из {t['охота']['эпизодов']}")
        lines.append(f"  отметок в спокойные сутки: {t['ложные на 100 коров в месяц']} на 100 коров в месяц, "
                     f"из них «проверить» — {t['ложные «проверить» на 100 коров в месяц']}")
    return "\n".join(lines)
