"""Базовые типы данных, которыми обмениваются ступени конвейера.

Конвейер: кадр -> Detection -> Track -> TrackletSummary -> IdentityDecision
          -> ActivityFeatures -> Deviation -> Event
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------
# Геометрия
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BBox:
    """Прямоугольник в пикселях, формат xyxy."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Точка опоры животного. Для позиции на плоскости загона она честнее центра:
        центр прямоугольника «плавает» вверх-вниз, когда животное опускает голову."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    @property
    def aspect(self) -> float:
        """Отношение ширины к высоте. Лежащее животное шире и ниже стоящего."""
        h = max(self.height, 1e-6)
        return self.width / h

    def iou(self, other: "BBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def clip(self, width: int, height: int) -> "BBox":
        return BBox(
            max(0.0, min(self.x1, width - 1)),
            max(0.0, min(self.y1, height - 1)),
            max(0.0, min(self.x2, width - 1)),
            max(0.0, min(self.y2, height - 1)),
        )


# --------------------------------------------------------------------------
# Ступень 1: детекция
# --------------------------------------------------------------------------

@dataclass
class Detection:
    """Одно обнаруженное животное на одном кадре."""

    bbox: BBox
    score: float
    frame_idx: int
    label: str = "cow"
    #: Четыре угла повёрнутой рамки туловища (x, y), если детектор OBB.
    #: По ним корова вырезается и кладётся горизонтально перед узнаванием.
    corners: Optional[tuple[tuple[float, float], ...]] = None
    #: Прямоугольник бирки, если детектор её нашёл (опционально).
    tag_bbox: Optional[BBox] = None


# --------------------------------------------------------------------------
# Ступень 2: трекинг
# --------------------------------------------------------------------------

@dataclass
class TrackObservation:
    """Положение трека на одном кадре."""

    frame_idx: int
    bbox: BBox
    score: float
    corners: Optional[tuple[tuple[float, float], ...]] = None
    #: Класс детектора: "cow" или поза "standing" / "lying", если детектор её различает.
    label: str = "cow"


@dataclass
class Tracklet:
    """Непрерывный отрезок наблюдения одного животного.

    `track_id` временный: он живёт, пока идёт трек, и после разрыва животное
    получит новый. Постоянный `cow_id` присваивается на ступени идентификации.
    """

    track_id: int
    observations: list[TrackObservation] = field(default_factory=list)
    #: Эмбеддинги внешности, накопленные по кадрам трека.
    embeddings: list[np.ndarray] = field(default_factory=list)
    #: Номера, распознанные с бирки на отдельных кадрах (могут быть разными из-за ошибок OCR).
    tag_reads: list[str] = field(default_factory=list)
    #: Номера кадров, на которых эти номера были прочитаны. Нужны, чтобы понять,
    #: не сменилось ли животное посередине трека.
    tag_read_frames: list[int] = field(default_factory=list)
    #: Итог идентификации, проставляется ступенью Identifier.
    cow_id: Optional[str] = None
    id_source: Optional[str] = None  # "tag" | "biometric" | "unknown"
    id_confidence: float = 0.0

    @property
    def length(self) -> int:
        return len(self.observations)

    @property
    def first_frame(self) -> int:
        return self.observations[0].frame_idx if self.observations else -1

    @property
    def last_frame(self) -> int:
        return self.observations[-1].frame_idx if self.observations else -1


# --------------------------------------------------------------------------
# Ступень 3: идентификация
# --------------------------------------------------------------------------

@dataclass
class IdentityDecision:
    """Результат идентификации трека."""

    cow_id: Optional[str]          # None означает «не знаю» (open-set)
    source: str                    # "tag" | "biometric" | "unknown"
    confidence: float
    #: Сколько кадров трека проголосовало за победивший вариант.
    votes: int = 0
    total_votes: int = 0
    #: Расстояние до ближайшего соседа в галерее (для биометрии).
    distance: Optional[float] = None


# --------------------------------------------------------------------------
# Ступень 4: активность
# --------------------------------------------------------------------------

@dataclass
class ActivityFeatures:
    """Показатели активности одного животного за один день."""

    cow_id: str
    day: date
    feeder_seconds: float = 0.0
    drinker_seconds: float = 0.0
    drinker_visits: int = 0
    resting_seconds: float = 0.0
    standing_seconds: float = 0.0
    distance_m: float = 0.0
    observed_seconds: float = 0.0   # сколько всего животное было видно
    tracks_count: int = 0
    #: Суточный удой из доильной установки или ERP. Камера его не меряет;
    #: если данных нет — None, и признак просто не оценивается.
    milk_kg: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "cow_id": self.cow_id,
            "day": self.day.isoformat(),
            "feeder_seconds": round(self.feeder_seconds, 1),
            "drinker_seconds": round(self.drinker_seconds, 1),
            "drinker_visits": self.drinker_visits,
            "resting_seconds": round(self.resting_seconds, 1),
            "standing_seconds": round(self.standing_seconds, 1),
            "distance_m": round(self.distance_m, 1),
            "observed_seconds": round(self.observed_seconds, 1),
            "tracks_count": self.tracks_count,
            "milk_kg": None if self.milk_kg is None else round(self.milk_kg, 2),
        }


# --------------------------------------------------------------------------
# Ступень 5: отклонение и событие
# --------------------------------------------------------------------------

@dataclass
class Deviation:
    """Отклонение одного показателя от персональной нормы животного."""

    metric: str
    value: float
    baseline_median: float
    baseline_scale: float     # робастный разброс (MAD, приведённый к сигме)
    robust_z: float
    delta_pct: float
    n_baseline_days: int


@dataclass
class Event:
    """Событие для зоотехника — конечный продукт всего конвейера."""

    cow_id: str
    day: date
    severity: str              # "info" | "warning" | "alert"
    title: str
    detail: str
    deviations: list[Deviation] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def as_dict(self) -> dict:
        return {
            "cow_id": self.cow_id,
            "day": self.day.isoformat(),
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "created_at": self.created_at.isoformat(timespec="seconds"),
            "deviations": [
                {
                    "metric": d.metric,
                    # Доли времени — числа меньше единицы: округление до
                    # десятых превратило бы 4% в ноль.
                    "value": round(d.value, 4),
                    "baseline": round(d.baseline_median, 4),
                    "robust_z": round(d.robust_z, 2),
                    # Две цифры: интерфейс округляет до целых так же, как заголовок
                    # события; при одной цифре 11,46 → 11,5 → «12%» против «11%».
                    "delta_pct": round(d.delta_pct, 2),
                }
                for d in self.deviations
            ],
        }
