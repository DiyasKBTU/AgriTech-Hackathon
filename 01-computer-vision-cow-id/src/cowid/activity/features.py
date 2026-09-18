"""Ступень 4б: суточные показатели активности конкретного животного.

Считаем ровно те величины, на которые зоотехник реально реагирует:

* время у кормушки — падает за 12–48 часов до клинических признаков ацидоза,
  мастита, метрита; самый ранний и самый дешёвый маркер из доступных камере;
* визиты к поилке — снижение потребления воды опережает падение аппетита,
  рост числа подходов летом означает тепловой стресс;
* пройденный путь — резкий рост это охота, резкое падение это боль или болезнь;
* время лёжа — больное животное ложится больше, хромое встаёт неохотно.

Каждая величина считается **на животное**, а не на стадо. Это не деталь
реализации, а суть подхода: одна корова ест 60 минут в сутки, другая 110,
и среднее по стаду не скажет ничего ни про одну из них.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Iterable, Optional

import numpy as np

from ..config import ActivityConfig
from ..types import ActivityFeatures, Tracklet
from .zones import Calibration, Zone


class ActivityExtractor:
    """Превращает треки с присвоенными ID в суточные показатели животных."""

    def __init__(
        self,
        cfg: ActivityConfig,
        zones: list[Zone],
        calibration: Calibration,
        seconds_per_frame: float,
        day_scale: float = 1.0,
    ):
        self.cfg = cfg
        self.zones = {z.name: z for z in zones}
        self.calibration = calibration
        self.seconds_per_frame = seconds_per_frame
        #: Наблюдение короче суток. Доли времени экстраполируются на 24 часа —
        #: так же, как это делает выборочное наблюдение в зоотехнии. Эталон
        #: симулятора применяет тот же множитель, сравнение остаётся корректным.
        self.day_scale = day_scale
        #: Сколько «суточных секунд» приходится на один отсчёт.
        self.unit = seconds_per_frame * day_scale

    # -- публичный вход ---------------------------------------------------

    def extract(self, tracklets: Iterable[Tracklet], day: date) -> list[ActivityFeatures]:
        """Собирает показатели за сутки, объединяя все треки одного животного.

        Одно животное за день даёт десятки треков: оно уходит из кадра, его
        закрывают другие коровы, трек рвётся. Показатели надо суммировать
        по всем трекам с одним cow_id — иначе получим не сутки, а один эпизод.
        """
        by_cow: dict[str, list[Tracklet]] = defaultdict(list)
        for t in tracklets:
            if t.cow_id:                      # «не знаю» в статистику не идёт
                by_cow[t.cow_id].append(t)

        out: list[ActivityFeatures] = []
        for cow_id, tracks in by_cow.items():
            out.append(self._features_for_cow(cow_id, tracks, day))
        return out

    # -- расчёт по одному животному ---------------------------------------

    def _features_for_cow(
        self, cow_id: str, tracks: list[Tracklet], day: date
    ) -> ActivityFeatures:
        feat = ActivityFeatures(cow_id=cow_id, day=day, tracks_count=len(tracks))
        feeder = self.zones.get("feeder")
        drinker = self.zones.get("drinker")

        #: Длительность текущего визита к поилке, в секундах НАБЛЮДЕНИЯ.
        drinker_run_observed = 0.0

        for track in tracks:
            points = self._smoothed_track(track)
            feat.observed_seconds += track.length * self.unit

            prev_point: Optional[tuple[float, float]] = None
            in_drinker = False

            for obs, point in zip(track.observations, points):
                dt = self.unit

                # Смещение между соседними отсчётами. Порог в пикселях отсекает
                # дрожание рамки детектора: без него стоящее животное «набегает»
                # километры за сутки на одном только шуме.
                step_px = 0.0
                moving = False
                if prev_point is not None:
                    step_px = float(
                        np.hypot(point[0] - prev_point[0], point[1] - prev_point[1])
                    )
                    moving = step_px > self.cfg.still_move_px
                    if moving:
                        feat.distance_m += (
                            self.calibration.distance_m(prev_point, point) * self.day_scale
                        )
                prev_point = point

                # Состояние животного. Определяем по зоне и скорости, а не по
                # форме рамки: при съёмке сверху соотношение сторон рамки зависит
                # от того, куда животное повёрнуто, а не от того, лежит оно или стоит.
                in_feeder = feeder is not None and feeder.contains(point)
                now_in_drinker = drinker is not None and drinker.contains(point)

                if in_feeder:
                    feat.feeder_seconds += dt
                elif now_in_drinker:
                    feat.drinker_seconds += dt
                elif not moving:
                    # Вне кормовых зон и почти не двигается — животное отдыхает.
                    feat.resting_seconds += dt
                else:
                    feat.standing_seconds += dt

                if drinker is not None:
                    if now_in_drinker:
                        drinker_run_observed += self.seconds_per_frame
                        in_drinker = True
                    elif in_drinker:
                        # Визит засчитываем только если животное задержалось:
                        # проход мимо поилки — это не питьё. Порог сравнивается
                        # с секундами НАБЛЮДЕНИЯ, а не с экстраполированными
                        # суточными: иначе один отсчёт в зоне сразу превысит
                        # любой разумный порог.
                        if drinker_run_observed >= drinker.min_visit_seconds:
                            feat.drinker_visits += 1
                        drinker_run_observed = 0.0
                        in_drinker = False

            if in_drinker and drinker_run_observed >= (
                drinker.min_visit_seconds if drinker else 0
            ):
                feat.drinker_visits += 1

        return feat

    def _smoothed_track(self, track: Tracklet) -> list[tuple[float, float]]:
        """Сглаживает траекторию скользящим средним.

        Рамка детектора дрожит от кадра к кадру на несколько пикселей. Без
        сглаживания это дрожание накапливается в «пройденный путь» и полностью
        забивает полезный сигнал.
        """
        if self.cfg.position_point == "bottom":
            points = [obs.bbox.bottom_center for obs in track.observations]
        else:
            points = [obs.bbox.center for obs in track.observations]
        window = max(1, int(self.cfg.smooth_window))
        if window <= 1 or len(points) <= window:
            return points

        arr = np.asarray(points, dtype=np.float64)
        kernel = np.ones(window) / window
        xs = np.convolve(arr[:, 0], kernel, mode="same")
        ys = np.convolve(arr[:, 1], kernel, mode="same")
        # Края свёртки искажены — оставляем там исходные значения.
        half = window // 2
        xs[:half], ys[:half] = arr[:half, 0], arr[:half, 1]
        xs[-half:], ys[-half:] = arr[-half:, 0], arr[-half:, 1]
        return list(zip(xs.tolist(), ys.tolist()))
