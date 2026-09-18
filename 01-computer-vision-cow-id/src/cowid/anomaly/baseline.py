"""Персональная норма животного и выявление отклонений от неё.

Главный принцип: **животное — это своя собственная норма.** Спокойная корова
проводит у кормового стола час, активная — два, и обе здоровы. Порог по стаду
завалит первую ложными тревогами и пропустит болезнь у второй.

Как устроено решение — пять шагов, каждый закрывает конкретную ошибку.

1. **Доли вместо абсолютного времени.** Камера видит животное не всё время:
   оно уходит из кадра, его закрывают другие. Если сегодня корову видели
   6 часов, а вчера 20, её «время у кормушки» упадёт втрое без всякой болезни.
   Поэтому сравниваются доли: какую часть наблюдаемого времени животное
   провело у корма, у воды, лёжа, и сколько метров прошло за час наблюдения.

2. **Отсев дней с плохим наблюдением.** Если животное было в кадре меньше
   порога, о нём за этот день ничего нельзя утверждать. Такой день не
   оценивается и не входит в норму — система честно говорит «мало данных».

3. **Робастная норма.** Медиана и MAD вместо среднего и сигмы: день вакцинации
   или перегруппировки не должен сдвигать норму.

4. **Сводный признак нездоровья.** Больное животное меняет несколько вещей
   согласованно — меньше ест, меньше ходит, больше лежит. Шум измерения двигает
   показатели независимо. Сумма отклонений «в сторону болезни» различает одно
   от другого лучше любого отдельного показателя.

5. **Накопление свидетельств по дням (CUSUM).** Болезнь развивается днями.
   Одиночный плохой день — обычно шум. Поэтому признаки копятся от суток
   к суткам, а случайный выброс сам собой гаснет. Пока идёт такое накопление,
   дни не добавляются в норму — иначе норма «заражается» болезнью и перестаёт
   её замечать.

Отдельно распознаётся охота: рост подвижности при сокращении отдыха.
Это не болезнь, и смешивать её с тревогой нельзя — у зоотехника на неё
другое действие.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Optional

import numpy as np

from ..config import BaselineConfig
from ..types import ActivityFeatures, Deviation, Event

#: Множитель, приводящий MAD к масштабу сигмы нормального распределения.
MAD_TO_SIGMA = 0.6745


def mad_sigma(values: np.ndarray) -> tuple[float, float]:
    """Медиана и робастная оценка сигмы с поправкой на малую выборку.

    По 5–14 значениям медианное абсолютное отклонение систематически занижает
    разброс — на 10–20%. Заниженный разброс делает любое обычное колебание
    «большим», и детектор начинает тревожить здоровых животных. Поправочный
    множитель n/(n-0.8) — стандартное приближение для малых выборок.
    """
    n = len(values)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    correction = n / (n - 0.8) if n > 1 else 1.0
    return median, mad / MAD_TO_SIGMA * correction

#: Производные показатели. Все нормированы на время наблюдения.
METRICS = ("feeder_share", "drinker_share", "resting_share", "activity_rate", "milk_kg")

#: В какую сторону показатель сдвигается у нездорового животного.
ILLNESS_DIRECTION = {
    "feeder_share": -1,      # ест меньше
    "drinker_share": -1,     # пьёт меньше
    "resting_share": +1,     # больше лежит
    "activity_rate": -1,     # меньше ходит
    "milk_kg": -1,           # меньше молока
}

#: В какую сторону — у животного в охоте.
ESTRUS_DIRECTION = {
    "activity_rate": +1,     # резко больше ходит
    "resting_share": -1,     # почти не ложится
}

METRIC_LABELS = {
    "feeder_share": "время у кормового стола",
    "drinker_share": "время у поилки",
    "resting_share": "время лёжа и без движения",
    "activity_rate": "подвижность",
    "milk_kg": "удой",
}

#: Признак простыми словами и что осмотреть. Диагноз система не ставит —
#: она говорит, что изменилось, а причину определяет ветврач.
METRIC_MEANING = {
    ("feeder_share", "down"): "ест меньше обычного — осмотреть: аппетит, жвачка, рубец, вымя, копыта",
    ("feeder_share", "up"): "дольше обычного у корма — проверить, хватает ли корма и места у стола",
    ("drinker_share", "down"): "пьёт меньше обычного — осмотреть корову, проверить поилку",
    ("drinker_share", "up"): "чаще у поилки — проверить жару в коровнике и температуру коровы",
    ("resting_share", "up"): "больше лежит и стоит без движения — посмотреть, охотно ли встаёт, ноги",
    ("resting_share", "down"): "почти не ложится — посмотреть подстилку, ноги, беспокойство",
    ("activity_rate", "down"): "меньше двигается — посмотреть походку и копыта",
    ("activity_rate", "up"): "двигается намного больше — признак охоты или беспокойства",
    ("milk_kg", "down"): "удой ниже обычного — осмотреть вымя, проверить аппетит и рацион",
}


# --------------------------------------------------------------------------
# Производные показатели
# --------------------------------------------------------------------------

def derived_metrics(f: ActivityFeatures) -> dict[str, float]:
    """Переводит суточные показатели в доли от времени наблюдения."""
    observed = max(f.observed_seconds, 1e-6)
    hours = observed / 3600.0
    return {
        "feeder_share": f.feeder_seconds / observed,
        "drinker_share": f.drinker_seconds / observed,
        "resting_share": f.resting_seconds / observed,
        "activity_rate": f.distance_m / hours if hours > 0 else 0.0,
        "milk_kg": f.milk_kg,
    }


def format_metric(metric: str, value: float) -> str:
    if metric.endswith("_share"):
        return f"{value * 100:.0f}% времени"
    if metric == "activity_rate":
        return f"{value:.0f} м/ч"
    if metric == "milk_kg":
        return f"{value:.1f} кг"
    return f"{value:.2f}"


# --------------------------------------------------------------------------
# Норма
# --------------------------------------------------------------------------

@dataclass
class BaselineStat:
    metric: str
    median: float
    #: Итоговый разброс в масштабе сигмы, уже с учётом разброса стада.
    scale: float
    n_days: int


@dataclass
class CowState:
    """Всё, что система помнит об одном животном."""

    #: Чистые дни, из которых строится норма: (дата, показатели).
    history: list[tuple[date, dict[str, float]]] = field(default_factory=list)
    #: Накопленный признак нездоровья (CUSUM).
    cusum: float = 0.0
    #: Сколько суток подряд длится текущий эпизод.
    episode_days: int = 0


@dataclass
class DayAssessment:
    """Оценка одних суток одного животного — для интерфейса и отладки."""

    cow_id: str
    day: date
    status: str                   # "ok" | "watch" | "alert" | "estrus" | "insufficient" | "learning"
    illness_score: float = 0.0
    estrus_score: float = 0.0
    cusum: float = 0.0
    episode_days: int = 0
    deviations: list[Deviation] = field(default_factory=list)


class PersonalBaseline:
    """Персональные нормы по всем животным и выявление отклонений."""

    def __init__(self, cfg: BaselineConfig):
        self.cfg = cfg
        self._cows: dict[str, CowState] = defaultdict(CowState)
        self.assessments: list[DayAssessment] = []
        #: Разброс стада, посчитанный один раз на сутки. Без кэша он
        #: пересчитывался бы для каждой коровы заново — на стаде в тысячу голов
        #: это миллион медиан в сутки вместо тысячи.
        self._spread_cache: Optional[dict[str, float]] = None
        #: Сдвиг стада за текущие сутки: метрика -> (сдвиг, значение, норма).
        self._herd_shift: dict[str, tuple[float, float, float]] = {}
        self._herd_n: dict[str, int] = {}

    # -- норма ------------------------------------------------------------

    def herd_relative_spread(self, metric: str) -> float:
        """Насколько обычно гуляют сутки здорового животного — по всему стаду.

        Зачем. По пяти-семи дням собственной истории разброс оценивается очень
        неточно и часто получается почти нулевым. Тогда любое обычное колебание
        выглядит огромным отклонением — первые прогоны давали сотни ложных тревог
        на здоровом стаде именно поэтому.

        При этом относительная изменчивость у животных одного стада похожа:
        её задают физиология и точность камеры, а не характер коровы. Поэтому
        «насколько гуляет день» берётся по стаду, а центр нормы — у каждой
        коровы свой.
        """
        rel: list[float] = []
        for state in self._cows.values():
            rows = state.history[-self.cfg.window_days:]
            values = np.asarray([m[metric] for _, m in rows if m.get(metric) is not None],
                                dtype=np.float64)
            if len(values) < self.cfg.min_days:
                continue
            med, sigma = mad_sigma(values)
            if med <= 1e-9:
                continue
            rel.append(sigma / med)
        if len(rel) < 3:
            return self.cfg.prior_relative_spread
        return float(np.median(rel))

    def baseline(self, cow_id: str, metric: str) -> Optional[BaselineStat]:
        rows = self._cows[cow_id].history[-self.cfg.window_days:]
        values = np.asarray([m[metric] for _, m in rows if m.get(metric) is not None],
                            dtype=np.float64)
        if len(values) < self.cfg.min_days:
            return None
        median, own = mad_sigma(values)
        spread = (
            self._spread_cache[metric]
            if self._spread_cache is not None
            else self.herd_relative_spread(metric)
        )
        herd = spread * abs(median)

        # Взвешиваем собственный разброс и разброс стада. Пока истории мало,
        # животное опирается на стадо; чем больше своих спокойных суток,
        # тем больше на себя. prior_weight — «сколько суток стоит мнение стада».
        n, n0 = len(values), self.cfg.prior_weight_days
        scale = float(np.sqrt((n * own ** 2 + n0 * herd ** 2) / (n + n0)))
        scale = max(scale, abs(median) * 0.03, 1e-6)
        return BaselineStat(metric=metric, median=median, scale=scale, n_days=n)

    def _deviations(self, cow_id: str, metrics: dict[str, float]) -> list[Deviation]:
        out: list[Deviation] = []
        for metric in METRICS:
            value = metrics.get(metric)
            if value is None:
                continue
            stat = self.baseline(cow_id, metric)
            if stat is None:
                continue
            # Норма на сегодня — с поправкой на то, как сдвинулось всё стадо.
            shift = self._herd_shift.get(metric, (0.0, 0.0, 0.0))[0]
            norm = stat.median * (1.0 + shift)
            z = (value - norm) / stat.scale
            delta = ((value - norm) / norm * 100.0) if norm else 0.0
            out.append(Deviation(
                metric=metric, value=value, baseline_median=norm,
                baseline_scale=stat.scale, robust_z=float(z),
                delta_pct=float(delta), n_baseline_days=stat.n_days,
            ))
        return out

    def _directional_score(self, deviations: list[Deviation], direction: dict[str, int]) -> float:
        """Сумма отклонений в заданную сторону, нормированная на корень из числа.

        Нормировка держит шкалу постоянной: сумма независимых шумов растёт как
        корень из их количества, и без деления порог пришлось бы подбирать
        заново при каждом изменении набора показателей.

        Отклонения меньше `min_effect_pct` не учитываются: на стабильном животном
        даже крошечный сдвиг даёт большой z, но фермеру изменение на 5% ни о чём
        не говорит.
        """
        parts: list[float] = []
        for dev in deviations:
            sign = direction.get(dev.metric)
            if not sign:
                continue
            if abs(dev.delta_pct) < self.cfg.min_effect_pct:
                parts.append(0.0)
                continue
            parts.append(max(0.0, dev.robust_z * sign))
        if not parts:
            return 0.0
        return float(np.sum(parts) / np.sqrt(len(parts)))

    # -- оценка суток -----------------------------------------------------

    def assess(self, f: ActivityFeatures) -> DayAssessment:
        state = self._cows[f.cow_id]

        # Шаг 2: мало наблюдений — ничего не утверждаем и в норму не берём.
        if f.observed_seconds < self.cfg.min_observed_seconds:
            return DayAssessment(cow_id=f.cow_id, day=f.day, status="insufficient",
                                 cusum=state.cusum, episode_days=state.episode_days)

        metrics = derived_metrics(f)
        deviations = self._deviations(f.cow_id, metrics)

        # Нормы ещё нет — копим историю.
        if not deviations:
            state.history.append((f.day, metrics))
            return DayAssessment(cow_id=f.cow_id, day=f.day, status="learning")

        illness = self._directional_score(deviations, ILLNESS_DIRECTION)
        estrus = self._directional_score(deviations, ESTRUS_DIRECTION)

        # Шаг 5: накопление свидетельств. Слагаемое k вычитается каждые сутки,
        # поэтому одиночный выброс гаснет, а устойчивый сдвиг копится.
        state.cusum = max(0.0, state.cusum + illness - self.cfg.cusum_k)
        state.episode_days = state.episode_days + 1 if state.cusum > 0.0 else 0
        # «Эпизод» — это заметное накопление, а не любое положительное значение:
        # под обычным шумом накопитель то и дело чуть выше нуля.
        developing = state.cusum >= self.cfg.cusum_h * 0.5

        single = any(
            ILLNESS_DIRECTION.get(d.metric, 0) * d.robust_z >= self.cfg.z_single
            and abs(d.delta_pct) >= self.cfg.single_min_effect_pct
            for d in deviations
        )

        if state.cusum >= self.cfg.cusum_h or illness >= self.cfg.z_strong:
            status = "alert"
        elif estrus >= self.cfg.z_estrus and illness < self.cfg.z_warning:
            status = "estrus"
        elif illness >= self.cfg.z_warning or developing or single:
            status = "watch"
        else:
            status = "ok"

        # Что брать в норму — самое тонкое место всего модуля.
        #
        # Не брать дни болезни: иначе норма сползает к болезни и перестаёт её
        # видеть. Но и брать только идеально спокойные дни нельзя — мы на этом
        # обожглись. Тогда в норму попадают лишь дни с малым отклонением, разброс
        # оценивается заниженным, любое обычное колебание начинает казаться
        # большим, такие дни снова не берутся — и норма застывает навсегда.
        # На здоровом стаде это давало сотни ложных тревог.
        #
        # Поэтому исключаются только сутки с настоящими признаками эпизода:
        # тревога, охота или заметное накопление. Одиночный шумный день
        # в норму идёт — это и есть обычная жизнь животного.
        if status not in ("alert", "estrus") and not developing:
            state.history.append((f.day, metrics))

        return DayAssessment(
            cow_id=f.cow_id, day=f.day, status=status,
            illness_score=illness, estrus_score=estrus,
            cusum=state.cusum, episode_days=state.episode_days,
            deviations=sorted(deviations, key=lambda d: abs(d.robust_z), reverse=True),
        )

    def evaluate_day(self, items: Iterable[ActivityFeatures]) -> list[Event]:
        """Оценивает сутки по всем животным и возвращает события для зоотехника."""
        items = list(items)
        events: list[Event] = []
        # Разброс стада фиксируется на начало суток: оценки одного дня не должны
        # зависеть от того, в каком порядке обработаны животные.
        self._spread_cache = {m: self.herd_relative_spread(m) for m in METRICS}
        # Поправка — только на заметный сдвиг всего стада. Мелкий сдвиг медианы
        # бывает и от нескольких больных коров; поправлять на него нельзя,
        # иначе детектор теряет чувствительность к ним же.
        shift = self._herd_shift_for(items) if self.cfg.herd_adjust else {}
        self._herd_shift = {m: v for m, v in shift.items() if self._noticeable(m, v[0])}
        herd_event = self._herd_event(items)
        if herd_event is not None:
            events.append(herd_event)
        for f in items:
            a = self.assess(f)
            self.assessments.append(a)
            event = self._to_event(f, a)
            if event is not None:
                events.append(event)
        self._spread_cache = None
        self._herd_shift = {}
        return events

    # -- стадо целиком ------------------------------------------------------

    def _herd_shift_for(self, items: list[ActivityFeatures]) -> dict[str, tuple[float, float, float]]:
        """Насколько сегодня сдвинулось большинство коров относительно своих норм.

        Медиана, а не среднее: несколько больных коров не должны сдвигать
        «погоду» для всех остальных.
        """
        rel: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        for f in items:
            if f.observed_seconds < self.cfg.min_observed_seconds:
                continue
            metrics = derived_metrics(f)
            for metric in METRICS:
                value = metrics.get(metric)
                if value is None:
                    continue
                stat = self.baseline(f.cow_id, metric)
                if stat is None or stat.median <= 0:
                    continue
                rel[metric].append((value / stat.median - 1.0, value, stat.median))
        out = {}
        self._herd_n = {}
        for metric, rows in rel.items():
            if len(rows) < self.cfg.herd_min_cows:
                continue
            arr = np.asarray(rows)
            out[metric] = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])),
                           float(np.median(arr[:, 2])))
            self._herd_n[metric] = len(rows)
        return out

    def _noticeable(self, metric: str, shift: float) -> bool:
        """Сдвиг стада больше трёх случайных колебаний медианы и не меньше порога."""
        spread = (self._spread_cache or {}).get(metric, self.cfg.prior_relative_spread)
        n = self._herd_n.get(metric, 1)
        limit = max(self.cfg.herd_event_pct / 100, 3 * 1.2533 * spread / np.sqrt(n))
        return abs(shift) >= limit

    def _herd_event(self, items: list[ActivityFeatures]) -> Optional[Event]:
        """Событие «Всё стадо», если сдвиг не объясняется случайностью.

        Медиана по n коровам сама гуляет примерно на 1.25·разброс/√n. Шумный
        признак (время у поилки гуляет на ~30% в сутки) иначе давал бы событие
        стада почти каждый день. Порог — три таких колебания, но не меньше
        `herd_event_pct`.
        """
        big = dict(self._herd_shift)
        if not big:
            return None
        day = items[0].day
        devs = [Deviation(metric=m, value=v[1], baseline_median=v[2], baseline_scale=0.0,
                          robust_z=0.0, delta_pct=v[0] * 100, n_baseline_days=0)
                for m, v in sorted(big.items(), key=lambda kv: -abs(kv[1][0]))]
        top = devs[0]
        label = METRIC_LABELS.get(top.metric, top.metric)
        direction = "рост" if top.delta_pct > 0 else "снижение"
        lines = [
            f"{METRIC_LABELS.get(d.metric, d.metric)}: у большинства коров "
            f"{d.delta_pct:+.0f}% к их обычному — медиана {format_metric(d.metric, d.value)} "
            f"при обычных {format_metric(d.metric, d.baseline_median)}"
            for d in devs
        ]
        lines.append("Меняется всё стадо сразу — проверить погоду и вентиляцию, корм, воду, "
                     "не было ли перегруппировки или обработки. Нормы отдельных коров на эти "
                     "сутки поправлены на этот сдвиг.")
        return Event(cow_id="стадо", day=day, severity="warning",
                     title=f"Всё стадо: {label} — {direction} на {abs(top.delta_pct):.0f}%",
                     detail="\n".join(lines), deviations=devs)

    # -- события ----------------------------------------------------------

    def _to_event(self, f: ActivityFeatures, a: DayAssessment) -> Optional[Event]:
        if a.status not in ("alert", "watch", "estrus"):
            return None

        notable = [d for d in a.deviations if abs(d.delta_pct) >= self.cfg.min_effect_pct]
        if not notable:
            if a.status != "alert":
                return None
            notable = a.deviations[:2]

        if a.status == "estrus":
            severity, title = "info", "Признаки охоты: двигается больше, лежит меньше обычного"
        elif a.status == "alert":
            severity = "alert"
            title = self._title(notable[0], a.episode_days)
        else:
            severity = "warning"
            title = self._title(notable[0], a.episode_days)

        return Event(
            cow_id=f.cow_id, day=f.day, severity=severity, title=title,
            detail=self._detail(f, a, notable), deviations=notable,
        )

    @staticmethod
    def _title(dev: Deviation, episode_days: int) -> str:
        label = METRIC_LABELS.get(dev.metric, dev.metric)
        direction = "снижение" if dev.delta_pct < 0 else "рост"
        title = f"{label.capitalize()}: {direction} на {abs(dev.delta_pct):.0f}% от нормы"
        if episode_days >= 2:
            title += f", {episode_days}-е сутки подряд"
        return title

    @staticmethod
    def _detail(f: ActivityFeatures, a: DayAssessment, devs: list[Deviation]) -> str:
        lines: list[str] = []
        for dev in devs:
            side = "down" if dev.delta_pct < 0 else "up"
            label = METRIC_LABELS.get(dev.metric, dev.metric)
            line = (
                f"{label}: {format_metric(dev.metric, dev.value)} при норме "
                f"{format_metric(dev.metric, dev.baseline_median)} "
                f"({dev.delta_pct:+.0f}%, по {dev.n_baseline_days} спокойным суткам)"
            )
            meaning = METRIC_MEANING.get((dev.metric, side))
            if meaning:
                line += f" — {meaning}"
            lines.append(line)
        lines.append(
            f"Сводный признак нездоровья {a.illness_score:.1f}, накоплено за эпизод "
            f"{a.cusum:.1f}. Животное было в кадре {f.observed_seconds / 3600:.1f} ч."
        )
        return "\n".join(lines)
