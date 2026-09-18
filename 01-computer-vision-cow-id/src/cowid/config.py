"""Конфигурация. Читается из YAML, проверяется pydantic.

Все пороги вынесены сюда намеренно: на защите спрашивают «почему такой порог»,
и отвечать надо файлом конфигурации с пояснением, а не константой в коде.
Одна конфигурация описывает одну камеру: её зоны, калибровку и режим обработки.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class ZoneConfig(BaseModel):
    """Функциональная зона кадра: кормовой стол, поилка. Полигон в пикселях."""

    name: str
    polygon: list[tuple[float, float]]
    #: Сколько секунд животное должно пробыть в зоне, чтобы засчитать визит.
    min_visit_seconds: float = 20.0


class CalibrationConfig(BaseModel):
    """Перевод пикселей в метры.

    Для камеры, смотрящей сверху, достаточно одного масштаба. Для наклонной
    камеры нужна гомография: матрица 3x3, построенная по четырём точкам
    с известными координатами (углы кормового стола, разметка прохода).
    """

    pixels_per_meter: float = 40.0
    homography: Optional[list[list[float]]] = None


class VideoConfig(BaseModel):
    #: Обрабатывать каждый N-й кадр. Для учёта активности 25 кадров в секунду
    #: не нужны: животное за 40 мс не успевает сдвинуться. Прореживание
    #: в 5 раз почти не меняет результат и в 5 раз ускоряет обработку.
    frame_stride: int = 5
    #: Частота кадров, если её не удалось прочитать из файла.
    fallback_fps: float = 25.0


class DetectorConfig(BaseModel):
    #: Детектор, обученный `cowid train-detector` на кадрах сверху (один класс).
    #: Запасной вариант для камеры сбоку — готовая COCO-модель `yolo26n.pt`
    #: с `class_ids: [19]` (в COCO корова — класс 19).
    model: str = "models/detector/cow_obb.pt"
    #: Пустой список — брать все классы модели.
    class_ids: list[int] = Field(default_factory=list)
    conf: float = 0.35
    iou: float = 0.5
    imgsz: int = 640
    device: str = "auto"
    #: Резать кадр на n×n плиток (кадр из нескольких камер). 1 — не резать.
    tiles: int = 1


class TrackerConfig(BaseModel):
    #: Порог перекрытия рамок для сопоставления.
    match_iou: float = 0.3
    #: Вес внешности в основном сопоставлении. Ноль — чистая геометрия.
    appearance_weight: float = 0.2
    #: Порог сходства для восстановления трека после перекрытия. Жёстче, чем
    #: у галереи: ошибка здесь склеивает двух разных животных.
    appearance_gate: float = 0.08
    #: Физический предел смещения животного за обработанный кадр, пиксели.
    max_reassoc_px_per_frame: float = 30.0
    #: Допуск сопоставления по центру, в долях размера животного.
    center_gate_ratio: float = 0.35
    #: Сколько кадров трек живёт без детекций.
    max_age: int = 30
    #: Сколько попаданий нужно, чтобы трек считался подтверждённым.
    min_hits: int = 3


class EmbedderConfig(BaseModel):
    #: "learned" — модель, обученная `cowid train-reid` на реальном датасете. Основной вариант.
    #: "coatpattern" — ручной дескриптор рисунка шкуры, без обучения и без torch.
    #:   Запасной: работает где угодно, но только на пятнистых породах.
    kind: Literal["learned", "coatpattern"] = "learned"
    weights: Optional[str] = "models/reid/encoder.pt"
    grid: tuple[int, int] = (6, 3)
    device: str = "auto"


class TagOCRConfig(BaseModel):
    #: "easyocr" — для реальных бирок; "digits" — упрощённая OCR без нейросетей;
    #: "none" — бирки не читаем, только биометрия.
    kind: Literal["easyocr", "digits", "none"] = "none"
    min_confidence: float = 0.6
    #: Полный ИНЖ — 3 буквы и 9 цифр (KZ + область + вид + номер); камера часто
    #: читает только цифры, поэтому буквы необязательны.
    pattern: str = r"^[A-Z]{0,3}-?\d{3,12}$"


class GalleryConfig(BaseModel):
    #: Порог косинусного расстояния, выше которого ответ «не знаю».
    #: Взят из `cowid eval-reid` на Cows2021: при нём чужая корова принимается
    #: за свою не чаще 1 раза из 100. Для другой модели — перепроверить.
    unknown_distance: float = 0.30
    max_per_cow: int = 64
    #: Сколько кадров трека должно прочитать один номер, чтобы зарегистрировать животное.
    min_tag_reads_to_enroll: int = 3
    #: Доля голосов, нужная для решения по треку.
    vote_ratio: float = 0.5
    #: Сколько последних голосов дорожки решают номер (0 — все). При редких
    #: кадрах трекер иногда перескакивает на соседку — старые голоса тогда
    #: не должны тянуть чужой номер (подобрано на MmCows, `cowid farm-chain`).
    vote_window: int = 0


class ActivityConfig(BaseModel):
    #: "center" — камера сверху; "bottom" — наклонная камера сбоку.
    position_point: Literal["center", "bottom"] = "center"
    #: Порог движения — смещение центра между обработанными кадрами, в пикселях.
    #: В пикселях, а не в м/с: при прореживании кадров «скорость» теряет смысл,
    #: а вопрос «сдвинулось ли животное сильнее, чем дрожит рамка» — нет.
    still_move_px: float = 1.5
    smooth_window: int = 5


class BaselineConfig(BaseModel):
    """Пороги персональной нормы и выявления отклонений.

    Значения выбраны по таблице чувствительности (`cowid scenarios`),
    а не на глаз.
    """

    #: Сколько спокойных суток нужно, прежде чем строить норму.
    min_days: int = 5
    #: Сколько последних спокойных суток берётся в норму.
    window_days: int = 14
    #: Сутки, когда животное было в кадре меньше этого, не оцениваются.
    min_observed_seconds: float = 3600.0
    #: Отклонения меньше этого процента не учитываются: для фермера
    #: изменение на несколько процентов ничего не значит.
    min_effect_pct: float = 12.0
    #: Сводный признак, при котором сутки берутся «на заметку».
    z_warning: float = 2.0
    #: Сводный признак, при котором тревога поднимается сразу, без накопления.
    z_strong: float = 5.0
    #: Порог признака охоты.
    z_estrus: float = 3.0
    #: Одно очень сильное отклонение в сторону болезни (например, «ест на 44%
    #: меньше») — «на заметку», даже если остальные признаки в норме. На
    #: реальных данных MmCows без этого правила такой день получал «в норме».
    z_single: float = 4.0
    single_min_effect_pct: float = 25.0
    #: CUSUM: сколько «гасится» каждые сутки и сколько нужно накопить для тревоги.
    #: k ставится посередине между типичным сводным признаком здорового
    #: животного (~0.6) и больного (~2.5): тогда шум гаснет, а болезнь копится.
    #:
    #: h — главная ручка «чуткость против спокойствия». При h=2.5 и разбросе
    #: 12% (`cowid scenarios`): ложных ~7 на 100 голов в месяц, хромота 95%,
    #: мастит 100%, ацидоз 65%. При разбросе 18% ложных уже ~23 — поэтому
    #: точность измерения камерой важнее, чем подбор h. Ферма может сдвинуть
    #: h в любую сторону: выше — спокойнее, но больше пропусков.
    cusum_k: float = 1.5
    cusum_h: float = 2.5
    #: Разброс здорового дня, если стадо ещё слишком мало, чтобы его оценить.
    prior_relative_spread: float = 0.15
    #: Сколько суток собственной истории «весит» разброс стада.
    prior_weight_days: float = 7.0
    #: Поправка на всё стадо. На реальных данных MmCows после жары всё стадо
    #: два дня лежало на 10–15% больше, и детектор отмечал отдельных коров.
    #: Если большинство коров сдвинулось в одну сторону, норма каждой коровы
    #: на этот день сдвигается так же, а сам сдвиг идёт отдельным событием.
    herd_adjust: bool = True
    #: Сколько коров с готовой нормой нужно, чтобы судить о сдвиге стада.
    herd_min_cows: int = 5
    #: Сдвиг стада, при котором поднимается событие «Всё стадо», в процентах.
    herd_event_pct: float = 8.0


class StorageConfig(BaseModel):
    database: str = "var/cowid.sqlite"
    evidence_dir: str = "var/evidence"


class FarmConfig(BaseModel):
    """Что известно о коровах помимо камеры: реестр фермы и обычные часы."""

    #: Реестр (карточки коров): "mmcows" — реальные записи фермы MmCows.
    registry: Optional[str] = None
    #: На какую дату строить карточку (день съёмки).
    as_of: Optional[str] = None
    #: Обычные часы коров по датчикам: "mmcows" — 13 суток MmCows.
    hourly_norms: Optional[str] = None


class PipelineConfig(BaseModel):
    camera_id: str = "camera-1"
    farm: FarmConfig = Field(default_factory=FarmConfig)
    #: Зоны сеткой для кадра из нескольких камер (`activity.zones.MaskZones`).
    zone_masks: Optional[str] = None
    video: VideoConfig = Field(default_factory=VideoConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    tag_ocr: TagOCRConfig = Field(default_factory=TagOCRConfig)
    gallery: GalleryConfig = Field(default_factory=GalleryConfig)
    activity: ActivityConfig = Field(default_factory=ActivityConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    zones: list[ZoneConfig] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw or {})

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            yaml.safe_dump(self.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
