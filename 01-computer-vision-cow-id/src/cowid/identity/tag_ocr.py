"""Чтение номера с ушной бирки.

Это первый и главный канал идентификации. Его ценность в том, что он даёт
**номер, который уже существует в ИСЖ и в ERP хозяйства**, — без ручной
регистрации животных и без сопоставления «визуальный ID -> реальный номер».

Три реализации одного контракта:

* ``DigitTemplateOCR`` — собственная мини-OCR на сопоставлении с шаблонами цифр.
  Не требует ни torch, ни внешних моделей. Работает на чётких биркax
  (синтетика, крупный план на проходе).
* ``EasyOCRTagReader`` — обёртка над EasyOCR для реального видео с грязными,
  потёртыми и повёрнутыми бирками.
* ``NullTagReader`` — заглушка: конвейер работает только на биометрии.

В любом случае ответ проверяется по шаблону номера: строка, не похожая на номер
животного, отбрасывается. Это дешёвая защита от мусора OCR, который иначе
загрязнит галерею неверными метками.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Protocol

import cv2
import numpy as np

from ..config import TagOCRConfig
from ..types import BBox


@dataclass
class TagRead:
    """Результат чтения бирки на одном кадре."""

    text: str
    confidence: float
    bbox: Optional[BBox] = None


class TagReader(Protocol):
    def read(self, frame: np.ndarray, bbox: BBox) -> Optional[TagRead]:
        ...


def _normalised_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Нормированная корреляция двух изображений одинакового размера.

    Заменяет cv2.matchTemplate: эталон и глиф уже приведены к общему размеру,
    скользить окном не нужно, а прямой расчёт заметно быстрее и не зависит
    от требования «шаблон не больше изображения».
    """
    x = a.astype(np.float32).ravel()
    y = b.astype(np.float32).ravel()
    x -= x.mean()
    y -= y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom < 1e-6:
        return 0.0
    return float(np.dot(x, y) / denom)


def _split_merged_glyphs(
    patch: np.ndarray, x0: int, y0: int, w: int, h: int
) -> list[tuple[int, int, int, int]]:
    """Режет слипшиеся цифры по минимумам вертикальной проекции.

    На мелкой бирке соседние цифры часто соединяются одним пикселем и попадают
    в один связный компонент. Оценив, сколько цифр примерно уместилось по ширине,
    ищем столько же разрезов в самых «тонких» местах.
    """
    expected = max(1, round(w / max(h * 0.62, 1)))
    if expected <= 1:
        return [(x0, y0, w, h)]

    projection = (patch > 0).sum(axis=0).astype(np.float32)
    cuts: list[int] = []
    slice_w = w / expected
    for k in range(1, expected):
        target = int(k * slice_w)
        lo = max(1, target - int(slice_w * 0.3))
        hi = min(w - 1, target + int(slice_w * 0.3))
        if hi <= lo:
            continue
        cuts.append(lo + int(np.argmin(projection[lo:hi])))

    bounds = [0] + sorted(set(cuts)) + [w]
    out: list[tuple[int, int, int, int]] = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a < 2:
            continue
        out.append((x0 + a, y0, b - a, h))
    return out or [(x0, y0, w, h)]


class NullTagReader:
    """Бирки не читаем — только биометрия."""

    def read(self, frame: np.ndarray, bbox: BBox) -> Optional[TagRead]:
        return None


class DigitTemplateOCR:
    """Мини-OCR: находит жёлтую бирку и распознаёт цифры сопоставлением с шаблонами.

    Алгоритм осознанно простой и полностью объяснимый:
      1. Ищем область бирки по цвету в HSV (бирки КРС выпускаются яркими:
         жёлтые, оранжевые, зелёные — это сделано именно для читаемости).
      2. Бинаризуем, находим связные компоненты — кандидаты в цифры.
      3. Каждую цифру сопоставляем с отрендеренными шаблонами 0–9 по нормированной
         корреляции, берём лучший вариант.
      4. Склеиваем цифры слева направо и проверяем результат по шаблону номера.

    Уверенность = средняя корреляция по цифрам. Порог отсекает случайный шум.
    """

    #: Диапазоны цвета бирки в HSV. Жёлто-оранжевый — самый распространённый.
    TAG_HSV_RANGES = [
        ((15, 90, 120), (40, 255, 255)),    # жёлтый / оранжевый
        ((40, 70, 90), (80, 255, 255)),     # зелёный
    ]

    def __init__(self, cfg: TagOCRConfig):
        self.cfg = cfg
        self.pattern = re.compile(cfg.pattern)
        self._templates = self._build_digit_templates()

    #: Размер, к которому приводятся и эталоны, и найденные глифы (ширина, высота).
    GLYPH_SIZE = (14, 20)

    @staticmethod
    def _build_digit_templates(size: tuple[int, int] = GLYPH_SIZE) -> dict[str, np.ndarray]:
        """Рендерит эталоны цифр тем же шрифтом, которым они наносятся в кадре."""
        templates: dict[str, np.ndarray] = {}
        for digit in "0123456789":
            canvas = np.zeros((32, 24), dtype=np.uint8)
            cv2.putText(canvas, digit, (3, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, 255, 2, cv2.LINE_AA)
            ys, xs = np.nonzero(canvas)
            if len(xs) == 0:
                continue
            cropped = canvas[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            templates[digit] = cv2.resize(cropped, size, interpolation=cv2.INTER_AREA)
        return templates

    def _find_tag_region(self, crop: np.ndarray) -> Optional[np.ndarray]:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in self.TAG_HSV_RANGES:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_area = None, 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            # Бирка — небольшой вытянутый по горизонтали прямоугольник.
            if area < 120 or w < 12 or h < 6:
                continue
            aspect = w / max(h, 1)
            if not (1.2 <= aspect <= 6.0):
                continue
            if area > best_area:
                best, best_area = (x, y, w, h), area
        if best is None:
            return None
        x, y, w, h = best
        # Обрезаем внутрь, а не наружу. Бирка обведена тёмным контуром, и если
        # он попадёт в кадр, при бинаризации он станет передним планом, сольётся
        # с цифрами в один компонент и распознавание развалится.
        inset_x = max(1, int(w * 0.06))
        inset_y = max(1, int(h * 0.12))
        y0, y1 = y + inset_y, y + h - inset_y
        x0, x1 = x + inset_x, x + w - inset_x
        if y1 - y0 < 6 or x1 - x0 < 8:
            return None
        return crop[y0:y1, x0:x1]

    def _recognise_digits(self, tag: np.ndarray) -> tuple[str, float]:
        # Бирка в кадре мелкая. Увеличиваем её до рабочего размера, иначе штрихи
        # цифр занимают один-два пикселя и любой шум их съедает.
        scale = max(2, int(80 / max(tag.shape[0], 1)))
        tag = cv2.resize(tag, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(tag, cv2.COLOR_BGR2GRAY)
        # Цифры тёмные на светлой бирке — инвертируем, чтобы они стали передним планом.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        boxes: list[tuple[int, int, int, int]] = []
        img_h = binary.shape[0]
        for i in range(1, n_labels):
            x, y, w, h, area = stats[i]
            if h < img_h * 0.30 or h > img_h * 0.99 or area < 12:
                continue
            if w > h * 1.1:
                # Компонент слишком широкий: соседние цифры слиплись. Разрезаем
                # его по провалам вертикальной проекции, а не выбрасываем —
                # иначе из «1000» получится «1».
                boxes.extend(_split_merged_glyphs(binary[y:y + h, x:x + w], x, y, w, h))
            else:
                boxes.append((x, y, w, h))
        if not boxes:
            return "", 0.0

        boxes.sort(key=lambda b: b[0])
        digits, scores = [], []
        for (x, y, w, h) in boxes:
            glyph = binary[y:y + h, x:x + w]
            glyph = cv2.resize(glyph, self.GLYPH_SIZE, interpolation=cv2.INTER_AREA)
            best_digit, best_score = None, -1.0
            for digit, template in self._templates.items():
                score = _normalised_correlation(glyph, template)
                if score > best_score:
                    best_digit, best_score = digit, score
            if best_digit is not None:
                digits.append(best_digit)
                scores.append(best_score)

        if not digits:
            return "", 0.0
        return "".join(digits), float(np.mean(scores))

    def read(self, frame: np.ndarray, bbox: BBox) -> Optional[TagRead]:
        h, w = frame.shape[:2]
        b = bbox.clip(w, h)
        if b.width < 10 or b.height < 10:
            return None
        crop = frame[int(b.y1):int(b.y2), int(b.x1):int(b.x2)]
        if crop.size == 0:
            return None

        tag = self._find_tag_region(crop)
        if tag is None:
            return None

        text, confidence = self._recognise_digits(tag)
        if not text or confidence < self.cfg.min_confidence:
            return None
        normalised = self._normalise(text)
        if normalised is None:
            return None
        return TagRead(text=normalised, confidence=confidence)

    def _normalise(self, text: str) -> Optional[str]:
        cleaned = re.sub(r"[^A-Z0-9-]", "", text.upper())
        if not cleaned:
            return None
        # Номер из ИСЖ в хозяйстве хранится с префиксом страны; в кадре виден только хвост.
        candidate = cleaned if cleaned.startswith("KZ") else cleaned
        if not self.pattern.match(candidate):
            return None
        return candidate


class EasyOCRTagReader:
    """Чтение бирок EasyOCR — вариант для реального видео."""

    def __init__(self, cfg: TagOCRConfig, languages: tuple[str, ...] = ("en",)):
        import easyocr

        self.cfg = cfg
        self.pattern = re.compile(cfg.pattern)
        self.reader = easyocr.Reader(list(languages), verbose=False)

    def read(self, frame: np.ndarray, bbox: BBox) -> Optional[TagRead]:
        h, w = frame.shape[:2]
        b = bbox.clip(w, h)
        crop = frame[int(b.y1):int(b.y2), int(b.x1):int(b.x2)]
        if crop.size == 0:
            return None
        results = self.reader.readtext(crop, allowlist="0123456789")
        best, best_conf = None, 0.0
        for _, text, conf in results:
            cleaned = re.sub(r"[^0-9]", "", text)
            if cleaned and conf > best_conf and self.pattern.match(cleaned):
                best, best_conf = cleaned, float(conf)
        if best is None or best_conf < self.cfg.min_confidence:
            return None
        return TagRead(text=best, confidence=best_conf)


def build_tag_reader(cfg: TagOCRConfig) -> TagReader:
    if cfg.kind == "none":
        return NullTagReader()
    if cfg.kind == "easyocr":
        return EasyOCRTagReader(cfg)
    return DigitTemplateOCR(cfg)
