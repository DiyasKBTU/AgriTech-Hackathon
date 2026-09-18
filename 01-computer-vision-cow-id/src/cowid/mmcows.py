"""MmCows: реальные суточные показатели коров и проверка детектора отклонений.

Что берём из датасета (Purdue, 16 голштинов, 21 июля – 4 августа 2023):

* **лежит** — датчик на ноге, отметка раз в минуту (коровы 1–10);
* **у кормового стола / у поилки** — радиометка положения на шее, раз в 15 с.
  Зоны найдены по размеченным суткам 25 июля: где стоит корова, когда
  человек отметил «ест» и «пьёт». Камера считает время у корма так же —
  по зоне, поэтому здесь заодно видно, насколько зонный способ точен;
* **путь** — по той же радиометке (шумный, см. `ZONE_NOTE`);
* **удой** — из доильного зала, раз в сутки.

Это не камера, а датчики. Но детектору отклонений всё равно, откуда пришли
минуты: он сравнивает корову с её собственной нормой. Поэтому на этих данных
честно видно главное, чего не могли показать заданные сценарии:

* насколько на самом деле гуляют сутки здоровой коровы (реальный разброс);
* сколько тревог детектор поднимает на коровах, у которых за время записи
  в ветеринарном журнале нет диагнозов.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .anomaly.baseline import METRICS, PersonalBaseline
from .config import BaselineConfig
from .store.db import Store
from .types import ActivityFeatures

ROOT = Path("data/real/mmcows_sensor/sensor_data")
CAMERA_ID = "mmcows-sensors"
SENSOR_COWS = range(1, 11)
#: Полные сутки записи (21.07 начинается в 12:30, 04.08 кончается в 7:00).
FULL_DAYS = ["0722", "0723", "0724", "0725", "0726", "0727", "0728",
             "0729", "0730", "0731", "0801", "0802", "0803"]
UWB_STEP_S = 15.0

#: Зоны в координатах радиометок, см. Найдены по разметке 25.07.
FEED_Y_MAX = -600.0
DRINK_ABS_X_MIN = 750.0
DRINK_ABS_Y_MAX = 200.0
#: Шаг радиометки меньше этого считается шумом, м.
STEP_NOISE_M = 0.6

Log = Callable[[str], None]


def cow_name(n: int) -> str:
    return f"C{n:02d}"


def _day(code: str) -> date:
    return date(2023, int(code[:2]), int(code[2:]))


def _read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _num(v: str) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else x


def zone_of(x: float, y: float) -> str:
    if y < FEED_Y_MAX:
        return "feeder"
    if abs(x) > DRINK_ABS_X_MIN and abs(y) < DRINK_ABS_Y_MAX:
        return "drinker"
    return "other"


def _uwb(cow: int, day: str, root: Path) -> np.ndarray:
    rows = _read_csv(root / "main_data" / "uwb" / f"T{cow:02d}" / f"T{cow:02d}_{day}.csv")
    pts = [(_num(r["timestamp"]), _num(r["coord_x_cm"]), _num(r["coord_y_cm"])) for r in rows]
    return np.asarray([p for p in pts if None not in p], dtype=np.float64)


def _milk(cow: int, root: Path) -> dict[date, float]:
    out = {}
    for r in _read_csv(root / "main_data" / "milk" / f"{cow_name(cow)}.csv"):
        ts, kg = _num(r["timestamp"]), _num(r["milk_weight_kg"])
        if ts is None or kg is None:
            continue
        # Отметка в 17:00 UTC = полдень по местному времени фермы (CDT).
        out[datetime.fromtimestamp(ts - 5 * 3600, tz=timezone.utc).date()] = kg
    return out


def daily_features(cow: int, day: str, root: Path = ROOT,
                   milk: Optional[dict[date, float]] = None) -> Optional[ActivityFeatures]:
    ankle = _read_csv(root / "main_data" / "ankle" / cow_name(cow) / f"{cow_name(cow)}_{day}.csv")
    lying = [_num(r["lying"]) for r in ankle]
    lying = [v for v in lying if v is not None]
    if not lying:
        return None
    observed = len(lying) * 60.0
    uwb = _uwb(cow, day, root)
    if len(uwb) < 100:
        return None
    x, y = uwb[:, 1], uwb[:, 2]
    zones = np.array([zone_of(a, b) for a, b in zip(x, y)])
    share = lambda name: float((zones == name).mean())       # noqa: E731
    xm, ym = x / 100.0, y / 100.0
    steps = np.hypot(np.diff(xm), np.diff(ym))
    path = float(steps[steps > STEP_NOISE_M].sum())
    covered = len(uwb) * UWB_STEP_S
    return ActivityFeatures(
        cow_id=cow_name(cow),
        day=_day(day),
        feeder_seconds=share("feeder") * observed,
        drinker_seconds=share("drinker") * observed,
        resting_seconds=float(np.sum(lying)) * 60.0,
        distance_m=path * observed / covered,
        observed_seconds=observed,
        milk_kg=(milk or {}).get(_day(day)),
    )


HOURLY_NORMS = Path("var/mmcows_hourly_norms.json")


def _local_hour(ts: float) -> int:
    return int(((ts - 5 * 3600) % 86400) // 3600)


def hourly_norms(root: Path = ROOT, days: list[str] = FULL_DAYS) -> dict[str, dict[str, list]]:
    """Обычные часы каждой коровы по датчикам: доля времени лёжа, у корма и
    у поилки в каждый час суток — медиана по суткам записи.

    Зачем по часам: днём корова лежит иначе, чем ночью. Камера на показе
    видит несколько часов, и сравнивать их надо с теми же часами её обычного
    дня, а не со средним за сутки.
    """
    out = {}
    for cow in SENSOR_COWS:
        per = {k: defaultdict(list) for k in ("lying", "feeder", "drinker")}
        for day in days:
            try:
                ankle = _read_csv(root / "main_data" / "ankle" / cow_name(cow) / f"{cow_name(cow)}_{day}.csv")
                uwb = _uwb(cow, day, root)
            except FileNotFoundError:
                continue
            hours = defaultdict(list)
            for r in ankle:
                ts, v = _num(r["timestamp"]), _num(r["lying"])
                if ts is not None and v is not None:
                    hours[_local_hour(ts)].append(v)
            for h, vals in hours.items():
                if len(vals) >= 30:
                    per["lying"][h].append(float(np.mean(vals)))
            zones = defaultdict(list)
            for ts, x, y in uwb:
                zones[_local_hour(ts)].append(zone_of(x, y))
            for h, vals in zones.items():
                if len(vals) >= 60:
                    arr = np.asarray(vals)
                    per["feeder"][h].append(float((arr == "feeder").mean()))
                    per["drinker"][h].append(float((arr == "drinker").mean()))
        out[cow_name(cow)] = {k: [round(float(np.median(v[h])), 4) if v.get(h) else None
                                  for h in range(24)] for k, v in per.items()}
    return out


def load_hourly_norms(path: Path = HOURLY_NORMS) -> dict:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(hourly_norms(), ensure_ascii=False), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def load_all(root: Path = ROOT, log: Log = print) -> list[ActivityFeatures]:
    out = []
    for cow in SENSOR_COWS:
        milk = _milk(cow, root)
        for day in FULL_DAYS:
            f = daily_features(cow, day, root, milk)
            if f is not None:
                out.append(f)
        log(f"  {cow_name(cow)}: суток {sum(1 for f in out if f.cow_id == cow_name(cow))}")
    return out


# --------------------------------------------------------------------------
# Разметка 25.07: насколько зона совпадает с поведением
# --------------------------------------------------------------------------

def zone_agreement(root: Path = ROOT) -> dict:
    """Сравнение «в зоне» с разметкой человека за 25.07 (коровы 1–10).

    Показывает, насколько «время у кормового стола» и «время у поилки»
    отражают реальную еду и питьё — тот же вопрос стоит и перед камерой.
    Оговорка: зоны подобраны по этим же суткам.
    """
    eat_in = eat_all = feed_all = 0
    drink_in = drink_all = water_all = 0
    for cow in SENSOR_COWS:
        uwb = _uwb(cow, "0725", root)
        labels = {int(float(r["timestamp"])): int(float(r["behavior"]))
                  for r in _read_csv(root / "behavior_labels" / "individual" /
                                     f"{cow_name(cow)}_0725.csv")}
        for ts, x, y in uwb:
            b = labels.get(int(ts), 0)
            if b == 0:
                continue
            z = zone_of(x, y)
            eating, drinking = b in (3, 4), b == 6
            eat_all += eating
            drink_all += drinking
            feed_all += z == "feeder"
            water_all += z == "drinker"
            eat_in += eating and z == "feeder"
            drink_in += drinking and z == "drinker"
    return {
        "feed_recall": eat_in / max(eat_all, 1),
        "feed_precision": eat_in / max(feed_all, 1),
        "drink_recall": drink_in / max(drink_all, 1),
        "drink_precision": drink_in / max(water_all, 1),
    }


def labelled_vs_sensor(root: Path = ROOT) -> dict:
    """Лежание за 25.07: датчик на ноге против разметки человека.

    Сравниваются одни и те же минуты — те, где человек поставил метку
    (около 4 часов в сутки коровы на дойке без разметки).
    """
    rows = []
    agree = total = 0
    for cow in SENSOR_COWS:
        labels = {int(float(r["timestamp"])): int(float(r["behavior"])) for r in _read_csv(
            root / "behavior_labels" / "individual" / f"{cow_name(cow)}_0725.csv")}
        human = sensor = n = 0
        for r in _read_csv(root / "main_data" / "ankle" / cow_name(cow) /
                           f"{cow_name(cow)}_0725.csv"):
            ts, lying = _num(r["timestamp"]), _num(r["lying"])
            b = labels.get(int(ts), 0) if ts is not None else 0
            if lying is None or b == 0:
                continue
            n += 1
            human += b == 7
            sensor += lying >= 0.5
            agree += (b == 7) == (lying >= 0.5)
            total += 1
        rows.append({"cow": cow_name(cow), "minutes": n,
                     "human_h": round(human / 60, 1), "ankle_h": round(sensor / 60, 1)})
    diff = [abs(r["ankle_h"] - r["human_h"]) for r in rows]
    return {"rows": rows, "minute_agreement": round(agree / max(total, 1), 3),
            "mean_abs_diff_h": round(float(np.mean(diff)), 2)}


# --------------------------------------------------------------------------
# Журнал фермы
# --------------------------------------------------------------------------

@dataclass
class FarmNote:
    cow_id: str
    day: date
    kind: str
    text: str


def farm_notes(root: Path = ROOT) -> list[FarmNote]:
    """Записи ветеринарного журнала за время съёмки и неделю до неё."""
    out = []
    for cow in range(1, 17):
        for r in _read_csv(root / "sub_data" / "health_records" / f"{cow_name(cow)}.csv"):
            d = datetime.strptime(r["Date"][:10], "%Y-%m-%d").date()
            if date(2023, 7, 14) <= d <= date(2023, 8, 4) and r["Event"] not in ("Other", "Move"):
                out.append(FarmNote(cow_name(cow), d, r["Event"],
                                    f"{r['Specific event']}: {r['Description']}".strip()))
    return out


# --------------------------------------------------------------------------
# Импорт и проверка
# --------------------------------------------------------------------------

def relative_spread(features: list[ActivityFeatures]) -> dict[str, float]:
    """Насколько гуляют сутки у одной коровы: медиана по коровам (MAD/медиана)."""
    from .anomaly.baseline import derived_metrics, mad_sigma

    by_cow = defaultdict(list)
    for f in features:
        by_cow[f.cow_id].append(derived_metrics(f))
    out = {}
    for metric in METRICS:
        rel = []
        for rows in by_cow.values():
            vals = np.asarray([m[metric] for m in rows if m.get(metric) is not None])
            if len(vals) < 5:
                continue
            med, sigma = mad_sigma(vals)
            if med > 0:
                rel.append(sigma / med)
        if rel:
            out[metric] = float(np.median(rel))
    return out


def import_and_check(store: Store, cfg: BaselineConfig, root: Path = ROOT,
                     log: Log = print) -> dict:
    features = load_all(root, log)
    store.save_features(CAMERA_ID, features, {f.cow_id: "sensor" for f in features})

    by_day = defaultdict(list)
    for f in features:
        by_day[f.day].append(f)
    baseline = PersonalBaseline(cfg)
    events = []
    for day in sorted(by_day):
        found = baseline.evaluate_day(by_day[day])
        store.save_events(found)
        events.extend(found)

    notes = farm_notes(root)
    handling = {(n.cow_id, n.day) for n in notes}
    assessed = [a for a in baseline.assessments if a.status not in ("learning", "insufficient")]
    herd_events = [e for e in events if e.cow_id == "стадо"]
    cow_events = [e for e in events if e.cow_id != "стадо"]
    alerts = [e for e in cow_events if e.severity == "alert"]
    watch = [e for e in cow_events if e.severity == "warning"]
    estrus = [e for e in cow_events if e.severity == "info"]
    per_100_month = lambda n: n / max(len(assessed), 1) * 100 * 30   # noqa: E731

    report = {
        "cows": len({f.cow_id for f in features}),
        "cow_days": len(features),
        "cow_days_assessed": len(assessed),
        "alerts": len(alerts),
        "watch": len(watch),
        "estrus": len(estrus),
        "herd_events": len(herd_events),
        "days": len(by_day),
        "herd_adjust": cfg.herd_adjust,
        "alerts_per_100_cows_month": round(per_100_month(len(alerts)), 1),
        "alerts_or_watch_per_100_cows_month": round(per_100_month(len(alerts) + len(watch)), 1),
        "events": [{
            "cow": e.cow_id, "day": e.day.isoformat(), "severity": e.severity,
            "title": e.title,
            "farm_note_same_day": next((n.text for n in notes
                                        if n.cow_id == e.cow_id and n.day == e.day), ""),
        } for e in events],
        "farm_notes": [{"cow": n.cow_id, "day": n.day.isoformat(), "kind": n.kind,
                        "text": n.text} for n in notes],
        "relative_spread": {k: round(v, 3) for k, v in relative_spread(features).items()},
        "zones_vs_labels_0725": {k: round(v, 3) for k, v in zone_agreement(root).items()},
        "lying_sensor_vs_labels_0725": labelled_vs_sensor(root),
        "handling_days_with_event": sum((e.cow_id, e.day) in handling for e in events),
    }
    report["summary_text"] = summary_text(report)
    return report


def summary_text(r: dict) -> str:
    spread = r["relative_spread"]
    zones = r["zones_vs_labels_0725"]
    lines = [
        f"Коров {r['cows']}, суток {r['cow_days']}, оценено суток (после 5 дней на норму) "
        f"{r['cow_days_assessed']}.",
        f"По отдельным коровам: «проверить» {r['alerts']}, «на заметку» {r['watch']}, "
        f"«признаки охоты» {r['estrus']}. События «Всё стадо»: {r['herd_events']} из "
        f"{r['days']} суток.",
        f"В пересчёте: «проверить» {r['alerts_per_100_cows_month']} на 100 голов в месяц, "
        f"«проверить» + «на заметку» {r['alerts_or_watch_per_100_cows_month']}.",
        "Реальный разброс суток у одной коровы (MAD/медиана): " + ", ".join(
            f"{k} {v:.0%}" for k, v in spread.items()) + ".",
        f"Зона кормового стола ловит {zones['feed_recall']:.0%} еды, из времени в зоне "
        f"{zones['feed_precision']:.0%} — еда. Зона поилки ловит {zones['drink_recall']:.0%} "
        f"питья, но питьё — лишь {zones['drink_precision']:.0%} времени у поилки.",
        f"Лежание: датчик на ноге против разметки человека 25.07 — совпадение по минутам "
        f"{r['lying_sensor_vs_labels_0725']['minute_agreement']:.0%}, расхождение в среднем "
        f"{r['lying_sensor_vs_labels_0725']['mean_abs_diff_h']} ч на корову.",
    ]
    return "\n".join(lines)


def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
