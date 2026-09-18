"""Что показывать о стаде и о корове: признаки, её норма, статус по дням.

Система не называет болезни. Она показывает **признаки** — что изменилось
у коровы по сравнению с её собственной нормой — и подсказывает, что осмотреть.
Диагноз ставит ветврач.

Норма в базе не хранится, поэтому здесь история прогоняется через
`PersonalBaseline` заново, по дням, — ровно так же, как при ночной обработке.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Optional

from .anomaly.baseline import (
    ESTRUS_DIRECTION, ILLNESS_DIRECTION, METRIC_MEANING, METRICS, PersonalBaseline,
    derived_metrics,
)
from .config import BaselineConfig
from .store.db import Store


@dataclass(frozen=True)
class Indicator:
    key: str
    name: str
    what: str
    unit: str
    source: str
    status: str            # "работает" | "не проверено" | "не сделано"
    look_at: str = ""


#: Каталог признаков. Статус честный: что считается сейчас, что нет.
INDICATORS = [
    Indicator("feeder_share", "Ест", "доля времени у кормового стола", "%",
              "камера над кормовым столом", "работает",
              "аппетит, жвачка, наполнение рубца, вымя, копыта"),
    Indicator("resting_share", "Лежит", "доля времени без движения вне корма и воды", "%",
              "камера над загоном", "работает",
              "встаёт ли охотно, ноги и копыта, температура"),
    Indicator("drinker_share", "Пьёт", "доля времени у поилки", "%",
              "камера над поилкой", "работает",
              "поилка исправна, жара в коровнике, температура"),
    Indicator("activity_rate", "Двигается", "пройденный путь за час наблюдения", "м/ч",
              "любая камера с калибровкой", "работает",
              "походка, копыта; резкий рост — признак охоты"),
    Indicator("lameness_behaviour", "Возможна хромота", "косвенно: лежит больше своей нормы "
              "и ест или двигается меньше; плюс хромота в реестре фермы", "признак",
              "те же камеры + реестр", "косвенно",
              "ноги, копыта, суставы"),
    Indicator("lameness", "Хромает (по походке)", "оценка походки в проходе", "балл",
              "камера сбоку или сверху над проходом", "не сделано",
              "копыта, суставы"),
    Indicator("body_condition", "Исхудала", "упитанность по силуэту сверху", "балл",
              "камера сверху в проходе", "не сделано",
              "рацион, зубы, паразиты"),
    Indicator("milk_kg", "Удой", "суточный удой", "кг",
              "доильная установка или ERP (импорт)", "работает",
              "вымя, аппетит, рацион"),
    Indicator("isolation", "Держится отдельно", "время вдали от стада", "%",
              "камера над загоном", "не сделано",
              "общее состояние"),
]

STATUS_LABEL = {
    "ok": "в норме",
    "watch": "на заметку",
    "alert": "проверить",
    "estrus": "признаки охоты",
    "insufficient": "мало данных",
    "learning": "набирается норма",
}
STATUS_ORDER = {"alert": 0, "watch": 1, "estrus": 2, "insufficient": 3, "learning": 4, "ok": 5}

#: Признак так, как его скажет зоотехник: (показатель, сторона) → название.
#: Порядок — порядок строк на доске признаков: сначала то, что чаще говорит
#: о нездоровье. «Удой выше» признаком не считается.
SIGN_NAMES = {
    ("feeder_share", "down"): "Ест меньше",
    ("resting_share", "up"): "Дольше лежит",
    ("activity_rate", "down"): "Меньше двигается",
    ("milk_kg", "down"): "Удой ниже",
    ("drinker_share", "down"): "Пьёт меньше",
    ("drinker_share", "up"): "Чаще у поилки",
    ("activity_rate", "up"): "Больше двигается",
    ("resting_share", "down"): "Почти не ложится",
    ("feeder_share", "up"): "Дольше у корма",
}
#: То же для интерфейса: ключ «показатель:сторона».
SIGN_NAMES_API = {f"{m}:{s}": name for (m, s), name in SIGN_NAMES.items()}


def _look_at(metric: str, side: str) -> str:
    meaning = METRIC_MEANING.get((metric, side), "")
    look = meaning.split(" — ", 1)[1] if " — " in meaning else meaning
    return look.removeprefix("осмотреть: ")


def sign_level(delta_pct: float, z: float, cfg: BaselineConfig) -> Optional[str]:
    """Насколько далеко показатель ушёл от нормы коровы.

    Те же пороги, что у детектора: изменение меньше `min_effect_pct` фермеру
    ничего не говорит; «заметно» — вдвое дальше обычного колебания коровы,
    «сильно» — вчетверо (с этого одно отклонение уже даёт «на заметку»).
    """
    if abs(delta_pct) < cfg.min_effect_pct:
        return None
    if abs(z) >= cfg.z_single:
        return "strong"
    if abs(z) >= cfg.z_warning:
        return "notable"
    return None


def _in_score(d, status: str, cfg: BaselineConfig) -> bool:
    """Отклонение вошло в сводный признак, по которому корова уже отмечена.

    Корову отмечают и по сумме нескольких умеренных отклонений. Каждое из них
    по отдельности меньше «заметно», но на доске их надо показать — иначе
    корова «на заметку» есть в списке, а причины на доске нет.
    """
    if status not in ("alert", "watch", "estrus") or abs(d.delta_pct) < cfg.min_effect_pct:
        return False
    direction = ESTRUS_DIRECTION if status == "estrus" else ILLNESS_DIRECTION
    return direction.get(d.metric, 0) * d.robust_z >= 1.0


def _indicators(metrics: dict, assessment, cfg: BaselineConfig) -> tuple[dict, list[dict]]:
    """Показатели суток с нормой коровы и список признаков простыми словами."""
    devs = {d.metric: d for d in assessment.deviations}
    out, signs = {}, []
    for m in METRICS:
        row = {"value": metrics[m]}
        d = devs.get(m)
        if d is not None:
            level = sign_level(d.delta_pct, d.robust_z, cfg)
            if level is None and _in_score(d, assessment.status, cfg):
                level = "notable"
            row.update({
                "norm": d.baseline_median, "scale": d.baseline_scale,
                "delta_pct": round(d.delta_pct, 1), "z": round(d.robust_z, 2),
                "norm_days": d.n_baseline_days, "level": level,
            })
            side = "down" if d.delta_pct < 0 else "up"
            if level and (m, side) in SIGN_NAMES:
                signs.append({
                    "metric": m, "side": side, "name": SIGN_NAMES[(m, side)],
                    "look": _look_at(m, side), "level": level,
                    "delta_pct": row["delta_pct"], "z": row["z"],
                    "value": d.value, "norm": d.baseline_median,
                })
        out[m] = row
    signs.sort(key=lambda s: (s["level"] != "strong", -abs(s["z"])))
    return out, signs


def _replay(store: Store, cfg: BaselineConfig):
    history = store.features_before(date.max)
    by_day: dict[date, list] = defaultdict(list)
    for f in history:
        by_day[f.day].append(f)
    baseline = PersonalBaseline(cfg)
    for day in sorted(by_day):
        baseline.evaluate_day(by_day[day])
    return baseline, history


def _thresholds(cfg: BaselineConfig) -> dict:
    """Пороги детектора — чтобы интерфейс объяснял их, а не держал свои копии."""
    return {"min_effect_pct": cfg.min_effect_pct, "z_notable": cfg.z_warning,
            "z_strong": cfg.z_single, "cusum_h": cfg.cusum_h,
            "min_observed_h": cfg.min_observed_seconds / 3600}


def _day_row(a, f, cfg: BaselineConfig) -> dict:
    indicators, signs = _indicators(derived_metrics(f), a, cfg)
    return {
        "day": a.day.isoformat(),
        "status": a.status,
        "status_label": STATUS_LABEL.get(a.status, a.status),
        "observed_h": round(f.observed_seconds / 3600, 1),
        "cusum": round(a.cusum, 2),
        "episode_days": a.episode_days,
        "indicators": indicators,
        "signs": signs,
    }


LAME_WITH = {("feeder_share", "down"), ("activity_rate", "down")}


def _lameness_row(cows: list[dict]) -> dict:
    """Косвенный признак хромоты: дольше лежит и при этом ест или ходит меньше.

    Хромота прямо камерой здесь не меряется (нужна оценка походки). Но хромая
    корова лежит дольше и меньше ест — так и в исследованиях, и на реальных
    данных MmCows (три недавно хромавшие коровы лежали 15.2 ч против 12.1 ч).
    """
    hits = []
    for c in cows:
        keys = {(s["metric"], s["side"]) for s in c["signs"]}
        if ("resting_share", "up") in keys and keys & LAME_WITH:
            rest = next(s for s in c["signs"] if s["metric"] == "resting_share")
            hits.append({"cow_id": c["cow_id"], "status": c["status"], **rest})
    return {"metric": "lameness", "side": "", "name": "Возможна хромота",
            "look": "ноги, копыта, суставы — косвенный признак: дольше лежит и ест или ходит меньше",
            "strong": [h for h in hits if h["level"] == "strong"],
            "notable": [h for h in hits if h["level"] != "strong"]}


def herd_summary(store: Store, cfg: BaselineConfig, day: Optional[date] = None) -> dict:
    """Стадо за одни сутки (по умолчанию — последние): статусы, признаки, доска.

    Доска признаков — ответ на вопрос «что сегодня не так в стаде»: по каждому
    признаку (ест меньше, дольше лежит…) — какие коровы и насколько далеко от
    своей нормы. Признаки, которые камера пока не считает, стоят там же
    с честной пометкой.
    """
    baseline, history = _replay(store, cfg)
    by_day: dict[date, dict] = defaultdict(dict)
    for a in baseline.assessments:
        by_day[a.day][a.cow_id] = a
    days = sorted(by_day)
    if day is not None and day not in by_day:
        raise KeyError(f"нет данных за {day.isoformat()}")
    target = day or (days[-1] if days else None)

    features = {(f.cow_id, f.day): f for f in history}
    open_events = Counter(e["cow_id"] for e in store.events(status="new", limit=10000))

    cows = []
    for cow_id, a in sorted(by_day.get(target, {}).items()):
        row = _day_row(a, features[(cow_id, a.day)], cfg)
        cows.append({"cow_id": cow_id, "open_events": open_events.get(cow_id, 0), **row})
    cows.sort(key=lambda c: (STATUS_ORDER.get(c["status"], 9), c["cow_id"]))

    board = [_lameness_row(cows)]
    for (metric, side), name in SIGN_NAMES.items():
        hits = [{"cow_id": c["cow_id"], "status": c["status"], **s}
                for c in cows for s in c["signs"]
                if s["metric"] == metric and s["side"] == side]
        hits.sort(key=lambda h: -abs(h["z"]))
        board.append({
            "metric": metric, "side": side, "name": name, "look": _look_at(metric, side),
            "strong": [h for h in hits if h["level"] == "strong"],
            "notable": [h for h in hits if h["level"] == "notable"],
        })
    not_done = [i.__dict__ for i in INDICATORS if i.status == "не сделано"]

    herd_events = [e for e in store.events(limit=100000)
                   if e["cow_id"] == "стадо" and target and e["day"] == target.isoformat()]
    return {
        "day": target.isoformat() if target else None,
        "days": [d.isoformat() for d in days],
        "counts": dict(Counter(c["status"] for c in cows)),
        "cows": cows,
        "board": board,
        "not_measured": not_done,
        "herd_events": herd_events,
        "indicators": [i.__dict__ for i in INDICATORS],
        "status_label": STATUS_LABEL,
        "sign_names": SIGN_NAMES_API,
        "thresholds": _thresholds(cfg),
    }


def cow_detail(store: Store, cfg: BaselineConfig, cow_id: str) -> dict:
    baseline, history = _replay(store, cfg)
    features = {f.day: f for f in history if f.cow_id == cow_id}
    days = [_day_row(a, features[a.day], cfg)
            for a in baseline.assessments if a.cow_id == cow_id]
    events = [e for e in store.events(limit=100000) if e["cow_id"] == cow_id]
    for e in events:
        e["evidence"] = store.evidence(cow_id, e["day"]) or {}
    source = next((a["id_source"] for a in store.animals() if a["cow_id"] == cow_id), None)
    return {
        "cow_id": cow_id,
        "id_source": source,
        "days": days,
        "events": events,
        "indicators": [i.__dict__ for i in INDICATORS],
        "status_label": STATUS_LABEL,
        "sign_names": SIGN_NAMES_API,
        "thresholds": _thresholds(cfg),
    }
