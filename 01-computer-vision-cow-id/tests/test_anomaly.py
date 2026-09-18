"""Детектор отклонений: зоотехнические сценарии.

Каждый тест — это история, которую зоотехник может прочитать и оценить,
правдоподобна ли реакция системы. Разброс «здорового дня» задан 12% —
это порядок суточной изменчивости пищевого поведения коровы плюс ошибка камеры.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from cowid.anomaly.baseline import PersonalBaseline, derived_metrics, mad_sigma
from cowid.anomaly.scenarios import SCENARIOS, run_healthy, run_scenario
from cowid.config import BaselineConfig
from cowid.types import ActivityFeatures

SCENARIO = {s.name: s for s in SCENARIOS}
CFG = BaselineConfig()


def features(cow: str, day: date, feeder_h: float, observed_h: float = 16.0,
             resting_h: float = 8.0, distance_m: float = 3500.0) -> ActivityFeatures:
    return ActivityFeatures(
        cow_id=cow, day=day, feeder_seconds=feeder_h * 3600,
        resting_seconds=resting_h * 3600, drinker_seconds=0.5 * 3600,
        distance_m=distance_m, observed_seconds=observed_h * 3600, tracks_count=5,
    )


# --------------------------------------------------------------------------
# Сценарии: что система должна заметить
# --------------------------------------------------------------------------

def test_severe_mastitis_is_caught_on_first_day():
    r = run_scenario(SCENARIO["Тяжёлый мастит"], CFG, daily_cv=0.12, herd=100)
    assert r.correct_rate >= 0.95
    assert r.median_detection_day == 1


def test_lameness_is_caught_within_two_days():
    r = run_scenario(SCENARIO["Хромота"], CFG, daily_cv=0.12, herd=100)
    assert r.correct_rate >= 0.85
    assert r.median_detection_day <= 2


# --------------------------------------------------------------------------
# Сценарии: на что система НЕ должна поднимать тревогу
# --------------------------------------------------------------------------

def test_single_bad_day_does_not_raise_alert():
    """Корова один день поела хуже и вернулась к норме — это не болезнь."""
    r = run_scenario(SCENARIO["Один плохой день"], CFG, daily_cv=0.12, herd=100)
    assert r.correct_rate >= 0.90


def test_estrus_is_reported_as_estrus_not_illness():
    """Охота — рост активности. У зоотехника на неё другое действие, чем на болезнь."""
    r = run_scenario(SCENARIO["Охота"], CFG, daily_cv=0.12, herd=100)
    assert r.correct_rate >= 0.90


def test_animal_barely_seen_is_not_judged():
    """Камера видела корову 20 минут — о ней за сутки ничего утверждать нельзя."""
    r = run_scenario(SCENARIO["Животное почти не видно"], CFG, daily_cv=0.12, herd=50)
    assert r.correct_rate == 1.0


def test_healthy_herd_has_few_false_alerts():
    """Главное условие того, что системой будут пользоваться: здоровых не дёргать."""
    h = run_healthy(CFG, daily_cv=0.12, herd=120, days=45)
    assert h.per_100_cows_month < 40


# --------------------------------------------------------------------------
# Механика нормы
# --------------------------------------------------------------------------

def test_shares_do_not_depend_on_how_long_animal_was_seen():
    """Увидели корову на 6 часов или на 18 — доля времени у корма одна и та же.

    Без этого любой день, когда животное ушло из кадра, выглядел бы
    как «ест втрое меньше».
    """
    short = derived_metrics(features("A", date(2026, 1, 1), feeder_h=1.5, observed_h=6))
    long = derived_metrics(features("A", date(2026, 1, 1), feeder_h=4.5, observed_h=18))
    assert short["feeder_share"] == pytest.approx(long["feeder_share"])


def test_personal_norm_flags_only_the_animal_that_changed():
    """Спокойная и активная корова с разной нормой. Порог по стаду поймал бы не ту."""
    det = PersonalBaseline(CFG)
    start = date(2026, 1, 1)
    rng = np.random.default_rng(0)
    for d in range(14):
        day = start + timedelta(days=d)
        det.evaluate_day([
            features("QUIET", day, feeder_h=2.0 * (1 + rng.normal(0, 0.05))),
            features("ACTIVE", day, feeder_h=5.0 * (1 + rng.normal(0, 0.05))),
        ])
    for d in range(14, 17):
        day = start + timedelta(days=d)
        # Активная корова упала до 2.8 часа: это всё ещё больше нормы спокойной,
        # но для неё самой — падение почти вдвое, и так три дня подряд.
        det.evaluate_day([
            features("QUIET", day, feeder_h=2.0),
            features("ACTIVE", day, feeder_h=2.8, resting_h=10.5, distance_m=2400),
        ])
    last = {a.cow_id: a.status for a in det.assessments[-2:]}
    assert last["ACTIVE"] == "alert"
    assert last["QUIET"] != "alert"


def test_episode_days_do_not_leak_into_the_norm():
    """Пока идёт эпизод болезни, его дни не должны попадать в норму.

    Иначе норма сползает к болезни, и на третьи сутки система её «не видит».
    """
    det = PersonalBaseline(CFG)
    start = date(2026, 1, 1)
    for d in range(10):
        det.evaluate_day([features("A", start + timedelta(days=d), feeder_h=4.0)])
    before = len(det._cows["A"].history)
    for d in range(10, 14):
        det.evaluate_day([features("A", start + timedelta(days=d), feeder_h=1.5,
                                   resting_h=13, distance_m=1500)])
    assert len(det._cows["A"].history) == before


def test_small_sample_spread_is_not_underestimated():
    """По малой выборке разброс занижается; поправка должна это компенсировать."""
    rng = np.random.default_rng(3)
    estimates = [mad_sigma(rng.normal(10, 1.0, size=7))[1] for _ in range(4000)]
    assert np.mean(estimates) == pytest.approx(1.0, abs=0.08)


def test_new_animal_leans_on_herd_spread():
    """Новая корова с идеально ровными первыми днями не должна считаться «сверхстабильной».

    Иначе первое же обычное колебание у неё выглядело бы как тревога.
    """
    det = PersonalBaseline(CFG)
    start = date(2026, 1, 1)
    rng = np.random.default_rng(5)
    for d in range(10):
        day = start + timedelta(days=d)
        herd = [features(f"H{i}", day, feeder_h=4 * (1 + rng.normal(0, 0.12)))
                for i in range(20)]
        new = features("NEW", day, feeder_h=4.0) if d >= 4 else None
        det.evaluate_day(herd + ([new] if new else []))

    det._spread_cache = None
    stat = det.baseline("NEW", "feeder_share")
    # Собственный разброс у NEW нулевой; итоговый должен опираться на стадо.
    assert stat.scale / stat.median > 0.05


# --------------------------------------------------------------------------
# Всё стадо сразу
# --------------------------------------------------------------------------

def test_whole_herd_shift_is_one_herd_event_not_many_cow_events():
    """После жары всё стадо лежит больше — это событие стада, а не 12 коров."""
    rng = np.random.default_rng(3)
    detector = PersonalBaseline(CFG)
    start = date(2026, 5, 1)
    cows = [f"K{i}" for i in range(12)]
    for d in range(10):
        detector.evaluate_day([
            features(c, start + timedelta(days=d), feeder_h=3.0 * (1 + rng.normal(0, 0.05)),
                     resting_h=8.0 * (1 + rng.normal(0, 0.05)))
            for c in cows
        ])
    hot = detector.evaluate_day([
        features(c, start + timedelta(days=10), feeder_h=3.0, resting_h=8.0 * 1.3)
        for c in cows
    ])
    assert [e.cow_id for e in hot] == ["стадо"]
    assert "лёжа" in hot[0].title

    # Та же картина, но у одной коровы: это её событие, стадо ни при чём.
    one = detector.evaluate_day([
        features(c, start + timedelta(days=11), feeder_h=3.0 * (0.4 if c == "K0" else 1.0),
                 resting_h=8.0)
        for c in cows
    ])
    assert {e.cow_id for e in one} == {"K0"}
