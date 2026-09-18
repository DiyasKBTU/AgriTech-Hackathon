"""Сценарии для проверки детектора отклонений.

Что это и чем отличается от симуляции видео. Здесь не рисуются коровы и не
изображается ферма. Детектор отклонений работает с таблицей «животное — сутки —
показатели», поэтому и проверяется таблицами. Каждый сценарий — это история,
которую зоотехник может прочитать и сказать, правдоподобна ли она:

    корова 14 дней ела нормально, потом три дня подряд ест на 30% меньше
    и больше лежит — заметит ли система, и на какие сутки?

Откуда берётся разброс. Здоровое животное не ест одинаково каждый день.
Суточная изменчивость пищевого поведения у одной и той же коровы в
исследованиях составляет порядка 10–15%. Сюда же добавляется ошибка измерения
камерой. Обе величины сведены в один параметр `daily_cv`, и результаты всегда
показываются для нескольких его значений — чтобы было видно, как качество
детектора зависит от качества измерения.

Что эта проверка доказывает и что нет. Она показывает, КАКИЕ отклонения
детектор способен поймать при заданной точности измерения, за сколько суток
и сколько ложных тревог будет у здоровых животных. Она НЕ доказывает, что
камера на конкретной ферме меряет с такой точностью — это проверяется
отдельно, на реальном видео с разметкой.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np

from ..config import BaselineConfig
from ..types import ActivityFeatures
from .baseline import PersonalBaseline


@dataclass
class Scenario:
    """История одного отклонения."""

    name: str
    description: str
    #: Во сколько раз меняется показатель в дни эпизода: 0.7 = на 30% меньше.
    feeder: float = 1.0
    drinker: float = 1.0
    resting: float = 1.0
    activity: float = 1.0
    #: Сколько суток подряд длится эпизод.
    days: int = 3
    #: Должна ли система поднять тревогу. Для «одного плохого дня» — нет.
    should_alert: bool = True
    #: Какой статус считается правильной реакцией.
    expected_status: str = "alert"
    #: Если задано — в дни эпизода животное почти не видно в кадре.
    observed_hours: Optional[float] = None


SCENARIOS: list[Scenario] = [
    Scenario(
        "Ацидоз",
        "ест на 30% меньше, отдыхает на 15% больше, ходит на 20% меньше, 3 суток",
        feeder=0.70, resting=1.15, activity=0.80, days=3,
    ),
    Scenario(
        "Хромота",
        "ходит на 40% меньше, лежит на 25% больше, ест на 15% меньше, 4 суток",
        feeder=0.85, resting=1.25, activity=0.60, days=4,
    ),
    Scenario(
        "Тяжёлый мастит",
        "ест вдвое меньше, пьёт на 30% меньше, 2 суток",
        feeder=0.50, drinker=0.70, resting=1.20, activity=0.75, days=2,
    ),
    Scenario(
        "Лёгкое недомогание",
        "ест на 15% меньше, 2 суток — на грани различимого",
        feeder=0.85, resting=1.05, days=2,
    ),
    Scenario(
        "Охота",
        "ходит в 2.5 раза больше, отдыхает на 30% меньше, 1 сутки",
        activity=2.5, resting=0.70, days=1,
        should_alert=False, expected_status="estrus",
    ),
    Scenario(
        "Один плохой день",
        "ест на 40% меньше ровно одни сутки, потом всё в норме — это не болезнь",
        feeder=0.60, days=1, should_alert=False, expected_status="not_alert",
    ),
    Scenario(
        "Животное почти не видно",
        "камера видела корову 20 минут — судить не о чем",
        feeder=0.40, days=1, observed_hours=0.33,
        should_alert=False, expected_status="insufficient",
    ),
]


@dataclass
class ScenarioResult:
    scenario: Scenario
    daily_cv: float
    herd: int
    #: Доля животных, у которых система отреагировала правильно.
    correct_rate: float
    #: На какие сутки эпизода поднялась тревога (медиана), если поднялась.
    median_detection_day: Optional[float]
    detections: list[int] = field(default_factory=list)


@dataclass
class HealthyResult:
    daily_cv: float
    herd: int
    days: int
    false_alerts: int

    @property
    def per_100_cows_month(self) -> float:
        return self.false_alerts / (self.herd * self.days) * 100 * 30


def _cow_profile(rng: np.random.Generator) -> dict[str, float]:
    """Индивидуальная норма животного — у каждой коровы своя."""
    return {
        "feeder_share": rng.uniform(0.15, 0.30),
        "drinker_share": rng.uniform(0.02, 0.05),
        "resting_share": rng.uniform(0.40, 0.55),
        "activity_rate": rng.uniform(150.0, 350.0),
    }


def _day(
    cow_id: str, day: date, base: dict[str, float], cv: float,
    rng: np.random.Generator, factors: Optional[Scenario] = None,
    observed_hours: Optional[float] = None,
) -> ActivityFeatures:
    """Сутки одного животного: норма x эффект эпизода x суточный разброс.

    Время в кадре тоже гуляет от суток к суткам (8–20 часов): животное уходит
    из поля зрения камеры. Детектор обязан с этим справляться, поэтому разброс
    покрытия заложен в каждые сутки, а не только в сценарий «почти не видно».
    """
    hours = observed_hours if observed_hours is not None else rng.uniform(8.0, 20.0)
    observed = hours * 3600.0

    def noisy(value: float) -> float:
        return max(0.0, value * (1.0 + rng.normal(0.0, cv)))

    f = factors
    feeder = noisy(base["feeder_share"] * (f.feeder if f else 1.0))
    drinker = noisy(base["drinker_share"] * (f.drinker if f else 1.0))
    resting = noisy(base["resting_share"] * (f.resting if f else 1.0))
    activity = noisy(base["activity_rate"] * (f.activity if f else 1.0))

    return ActivityFeatures(
        cow_id=cow_id, day=day,
        feeder_seconds=feeder * observed,
        drinker_seconds=drinker * observed,
        resting_seconds=min(resting, 1.0) * observed,
        distance_m=activity * hours,
        observed_seconds=observed,
        tracks_count=10,
    )


def run_scenario(
    scenario: Scenario, cfg: BaselineConfig, daily_cv: float = 0.12,
    herd: int = 200, calm_days: int = 14, sick_share: float = 0.2, seed: int = 0,
) -> ScenarioResult:
    """Прогоняет сценарий: заболевшие животные внутри обычного стада.

    Детектор один на всё стадо — как на ферме. Это важно: разброс «здорового
    дня» оценивается по стаду, и если прогонять больную корову отдельно,
    детектору не на что опереться, и проверка получается заниженной.
    Эпизод случается у части животных (`sick_share`), остальные — здоровый фон.
    """
    rng = np.random.default_rng(seed)
    start = date(2026, 1, 1)
    detector = PersonalBaseline(cfg)
    cows = {f"C{i:04d}": _cow_profile(rng) for i in range(herd)}
    n_sick = max(1, int(herd * sick_share))
    sick = set(list(cows)[:n_sick])

    def day_batch(day_index: int, episode: bool) -> list[ActivityFeatures]:
        day = start + timedelta(days=day_index)
        batch = []
        for cow, base in cows.items():
            if episode and cow in sick:
                batch.append(_day(cow, day, base, daily_cv, rng, scenario, scenario.observed_hours))
            else:
                batch.append(_day(cow, day, base, daily_cv, rng))
        return batch

    for d in range(calm_days):
        detector.evaluate_day(day_batch(d, episode=False))

    statuses: dict[str, list[str]] = {c: [] for c in sick}
    first_alert: dict[str, int] = {}
    for k in range(scenario.days):
        detector.evaluate_day(day_batch(calm_days + k, episode=True))
        for a in detector.assessments[-herd:]:
            if a.cow_id in sick:
                statuses[a.cow_id].append(a.status)
                if a.status == "alert" and a.cow_id not in first_alert:
                    first_alert[a.cow_id] = k + 1

    # После эпизода — ещё двое спокойных суток: тревога, продолжающаяся после
    # выздоровления, для «одного плохого дня» тоже ошибка.
    for k in range(2):
        detector.evaluate_day(day_batch(calm_days + scenario.days + k, episode=False))
        if not scenario.should_alert:
            for a in detector.assessments[-herd:]:
                if a.cow_id in sick and a.status == "alert":
                    statuses[a.cow_id].append("alert")

    correct = 0
    for cow in sick:
        st = statuses[cow]
        if scenario.expected_status == "alert":
            ok = cow in first_alert
        elif scenario.expected_status == "estrus":
            ok = "estrus" in st and "alert" not in st
        elif scenario.expected_status == "insufficient":
            ok = all(s == "insufficient" for s in st[: scenario.days])
        else:  # "not_alert"
            ok = "alert" not in st
        correct += int(ok)

    detections = list(first_alert.values())
    median_day = float(np.median(detections)) if detections else None
    return ScenarioResult(
        scenario=scenario, daily_cv=daily_cv, herd=n_sick,
        correct_rate=correct / n_sick, median_detection_day=median_day,
        detections=detections,
    )


def run_healthy(
    cfg: BaselineConfig, daily_cv: float = 0.12, herd: int = 200,
    days: int = 60, seed: int = 1,
) -> HealthyResult:
    """Здоровое стадо на протяжении двух месяцев: сколько будет ложных тревог."""
    rng = np.random.default_rng(seed)
    start = date(2026, 1, 1)
    detector = PersonalBaseline(cfg)
    bases = {f"H{i:04d}": _cow_profile(rng) for i in range(herd)}
    false_alerts = 0
    for d in range(days):
        day = start + timedelta(days=d)
        events = detector.evaluate_day([
            _day(cow, day, base, daily_cv, rng) for cow, base in bases.items()
        ])
        false_alerts += sum(1 for e in events if e.severity == "alert")
    return HealthyResult(daily_cv=daily_cv, herd=herd, days=days, false_alerts=false_alerts)
