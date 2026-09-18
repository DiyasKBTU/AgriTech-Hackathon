"""Биометрический «портрет» животного — вектор, по которому его узнают.

Два варианта с одним контрактом.

* ``LearnedEmbedder`` — **основной**. Сеть, обученная командой `cowid train-reid`
  на реальном датасете с размеченными животными. Её качество измерено на
  животных, которых она ни разу не видела при обучении.
* ``CoatPatternEmbedder`` — **запасной, без обучения и без torch**. Описывает,
  где на теле расположены тёмные пятна. Работает на любом компьютере и не
  требует весов, но годится только для пятнистых пород и вида сверху.

Что выяснилось по дороге и стоит помнить. Сеть, предобученная на ImageNet
и не дообученная на коровах, на сырых кадрах узнавала животных хуже ручного
дескриптора: признаки ImageNet не инвариантны к повороту, а корова сверху
повёрнута как угодно. Нормализация позы давала больше, чем выбор архитектуры.
Поэтому такой вариант из проекта убран: без дообучения он не нужен,
а с дообучением это уже ``LearnedEmbedder``.

Оба варианта нормируют вектор на единичную длину, поэтому близость двух
портретов считается простым скалярным произведением.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol

import cv2
import numpy as np

from ..config import EmbedderConfig
from ..types import BBox


Corners = Optional[tuple[tuple[float, float], ...]]


class Embedder(Protocol):
    def embed(self, frame: np.ndarray, bbox: BBox,
              corners: Corners = None) -> Optional[np.ndarray]:
        ...


def spot_threshold(gray: np.ndarray, animal: np.ndarray) -> float:
    """Граница между пятном и светлой шерстью — по самому животному.

    Раньше здесь был фиксированный перцентиль яркости (45-й). Это ломалось на
    животных, у которых пятна занимают меньше 45% шкуры, — а у голштинов так
    бывает часто: порог попадал на белую шерсть, всё тело считалось «пятном»,
    и портреты разных коров становились одинаковыми.

    Порог Оцу ищет естественную границу между двумя группами яркости, поэтому
    не зависит от того, какую долю тела занимают пятна.
    """
    values = gray[animal].astype(np.uint8)
    if values.size < 10 or int(values.max()) - int(values.min()) < 20:
        # Однотонное животное: пятен нет, различать по рисунку нечего.
        return -1.0
    threshold, _ = cv2.threshold(values.reshape(-1, 1), 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(threshold)


def _crop(frame: np.ndarray, bbox: BBox, pad: float = 0.0) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    bw, bh = bbox.width, bbox.height
    x1 = int(max(0, bbox.x1 - pad * bw))
    y1 = int(max(0, bbox.y1 - pad * bh))
    x2 = int(min(w, bbox.x2 + pad * bw))
    y2 = int(min(h, bbox.y2 + pad * bh))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return frame[y1:y2, x1:x2]


def rectify(frame: np.ndarray, corners) -> Optional[np.ndarray]:
    """Вырезает корову по повёрнутой рамке и кладёт горизонтально.

    Так нарезаны снимки Cows2021, на которых училась модель узнавания:
    длинная сторона туловища — по горизонтали, края за кадром — чёрные.
    """
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    # Первым ребром должна идти длинная сторона.
    if np.linalg.norm(pts[1] - pts[0]) < np.linalg.norm(pts[2] - pts[1]):
        pts = np.roll(pts, -1, axis=0)
    w = int(round(float(np.linalg.norm(pts[1] - pts[0]))))
    h = int(round(float(np.linalg.norm(pts[2] - pts[1]))))
    if w < 16 or h < 8:
        return None
    dst = np.array([[0, 0], [w, 0], [w, h]], dtype=np.float32)
    matrix = cv2.getAffineTransform(pts[:3], dst)
    return cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


class CoatPatternEmbedder:
    """Дескриптор рисунка шкуры: где именно на теле расположены тёмные пятна.

    Наивная версия («нарезать рамку сеткой и посчитать гистограммы») не работает,
    и это стоит понимать до того, как строить на ней решение. Мы её измерили:
    расстояние между кадрами ОДНОГО животного получалось не меньше, чем между
    РАЗНЫМИ животными. Две причины:

    1. **Поворот.** Животное в кадре сверху повёрнуто произвольно. Одна и та же
       корова, идущая влево и вправо, давала совершенно разные векторы.
    2. **Фон.** Внутри прямоугольной рамки половина пикселей — подстилка, и
       дескриптор описывал в основном её.

    Поэтому здесь делается четыре вещи:

    * силуэт отделяется от фона маской, и всё считается только по пикселям животного;
    * силуэт разворачивается по главной оси (PCA), чтобы курс перестал влиять;
    * ориентация приводится к каноническому виду — PCA задаёт ось, но не
      направление, поэтому «голова слева» и «голова справа» надо развести явно,
      иначе остаётся неоднозначность в 180 градусов;
    * в каждой ячейке сетки считается доля тёмного ОТ ПЛОЩАДИ ЖИВОТНОГО в этой
      ячейке, а не от площади ячейки.

    Ограничение называем честно: дескриптор рассчитан на пятнистых животных
    (голштин) и вид сверху. Для однотонных пород (ангус, казахская белоголовая)
    рисунка нет, и работать будет только канал бирки или обучаемая модель.
    """

    def __init__(self, cfg: EmbedderConfig):
        self.grid = tuple(cfg.grid)
        self.size = (96, 64)

    def embed(self, frame: np.ndarray, bbox: BBox,
              corners: Corners = None) -> Optional[np.ndarray]:
        # Поворот дескриптор нормализует сам, поэтому углы рамки ему не нужны.
        crop = _crop(frame, bbox, pad=0.02)
        if crop is None:
            return None

        aligned = self._normalise_pose(crop)
        if aligned is None:
            return None
        patch, mask = aligned

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
        animal = mask > 0
        if animal.sum() < 50:
            return None

        # Порог «пятна» — по самому животному: освещение за сутки меняется,
        # и фиксированное значение яркости поплывёт.
        spots = animal & (gray <= spot_threshold(gray, animal))

        gw, gh = self.grid
        cell_h = patch.shape[0] / gh
        cell_w = patch.shape[1] / gw
        parts: list[float] = []

        for row in range(gh):
            for col in range(gw):
                y0, y1 = int(row * cell_h), int((row + 1) * cell_h)
                x0, x1 = int(col * cell_w), int((col + 1) * cell_w)
                cell_animal = animal[y0:y1, x0:x1]
                n_animal = float(cell_animal.sum())
                if n_animal < 4:
                    parts.extend([0.0, 0.0])
                    continue
                spot_share = float(spots[y0:y1, x0:x1].sum()) / n_animal
                # Вторая координата — насколько ячейка вообще занята животным:
                # это кодирует форму силуэта и помогает различать позы.
                fill = n_animal / max(1.0, (y1 - y0) * (x1 - x0))
                parts.extend([spot_share, fill])

        vec = np.asarray(parts, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    # -- нормализация позы -------------------------------------------------

    def _normalise_pose(self, crop: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Отделяет животное от фона, разворачивает по оси и канонизирует направление."""
        mask = self._segment(crop)
        if mask is None:
            return None

        angle = self._principal_angle(mask)
        h, w = crop.shape[:2]
        # Разворачиваем в холсте с запасом, чтобы углы силуэта не срезались.
        diag = int(np.hypot(h, w)) + 4
        pad_y, pad_x = (diag - h) // 2, (diag - w) // 2
        big = cv2.copyMakeBorder(crop, pad_y, diag - h - pad_y, pad_x, diag - w - pad_x,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
        big_mask = cv2.copyMakeBorder(mask, pad_y, diag - h - pad_y, pad_x, diag - w - pad_x,
                                      cv2.BORDER_CONSTANT, value=0)
        matrix = cv2.getRotationMatrix2D((diag / 2, diag / 2), angle, 1.0)
        rot = cv2.warpAffine(big, matrix, (diag, diag), flags=cv2.INTER_LINEAR)
        rot_mask = cv2.warpAffine(big_mask, matrix, (diag, diag), flags=cv2.INTER_NEAREST)

        ys, xs = np.nonzero(rot_mask)
        if len(xs) < 30:
            return None
        patch = rot[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        patch_mask = rot_mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

        patch = cv2.resize(patch, self.size, interpolation=cv2.INTER_AREA)
        patch_mask = cv2.resize(patch_mask, self.size, interpolation=cv2.INTER_NEAREST)
        patch, patch_mask = self._canonical_flip(patch, patch_mask)
        return patch, patch_mask

    @staticmethod
    def _segment(crop: np.ndarray) -> Optional[np.ndarray]:
        """Маска животного. Корова светлее подстилки, поэтому берём Оцу и
        оставляем крупнейший связный компонент — так в маску не попадают
        соседние животные, случайно задетые рамкой."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n <= 1:
            return None
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)
        if mask.sum() / 255 < 50:
            return None
        # Заполняем дыры: тёмные пятна на шкуре не должны выпадать из силуэта.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
        return mask

    @staticmethod
    def _principal_angle(mask: np.ndarray) -> float:
        ys, xs = np.nonzero(mask)
        coords = np.stack([xs, ys]).astype(np.float32)
        centered = coords - coords.mean(axis=1, keepdims=True)
        cov = centered @ centered.T / max(1, centered.shape[1] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal = eigvecs[:, int(np.argmax(eigvals))]
        return float(np.degrees(np.arctan2(principal[1], principal[0])))

    @staticmethod
    def _canonical_flip(patch: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Убирает неоднозначность направления, оставшуюся после PCA.

        Главная ось задаёт линию, но не говорит, где голова. Без этого шага одна
        и та же корова, идущая в противоположные стороны, даёт зеркальные векторы
        и перестаёт узнавать сама себя. Разворачиваем так, чтобы более «тяжёлая»
        по тёмному пигменту половина всегда оказывалась слева и сверху.
        """
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        body = mask > 0
        dark = (body & (gray <= spot_threshold(gray, body))).astype(np.float32)
        h, w = dark.shape

        if dark[:, : w // 2].sum() < dark[:, w // 2:].sum():
            patch = cv2.flip(patch, 1)
            mask = cv2.flip(mask, 1)
            dark = cv2.flip(dark, 1)
        if dark[: h // 2, :].sum() < dark[h // 2:, :].sum():
            patch = cv2.flip(patch, 0)
            mask = cv2.flip(mask, 0)
        return patch, mask


class LearnedEmbedder:
    """Эмбеддер на ОБУЧЕННОЙ модели — то, что реально стоит ставить на ферму.

    Загружает веса, полученные командой `cowid train-reid` на реальном датасете.
    В отличие от двух вариантов выше, эта сеть училась именно различать
    конкретных животных, а не классифицировать объекты ImageNet.

    Модель училась на снимках Cows2021, где туловище вырезано по повёрнутой
    рамке и положено горизонтально. Если детектор дал углы рамки, корова
    вырезается так же. Без углов (детектор COCO) берётся обычный
    прямоугольник — это хуже: в него попадают пол и соседи.
    """

    def __init__(self, cfg: EmbedderConfig, weights: str | Path):
        import torch
        from torchvision import transforms

        from .train import TrainConfig, _build_model

        self.torch = torch
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        train_cfg = TrainConfig(**checkpoint["config"])
        self.metrics = checkpoint.get("metrics", {})

        self.device = self._resolve_device(cfg.device)
        encoder, _ = _build_model(train_cfg, n_classes=1)
        encoder.load_state_dict(checkpoint["state_dict"])
        self.encoder = encoder.eval().to(self.device)

        size = train_cfg.image_size
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((size, size), antialias=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        return "cuda" if self.torch.cuda.is_available() else "cpu"

    def embed(self, frame: np.ndarray, bbox: BBox,
              corners: Corners = None) -> Optional[np.ndarray]:
        crop = rectify(frame, corners) if corners else _crop(frame, bbox, pad=0.05)
        if crop is None:
            return None
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.preprocess(rgb).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            vec = self.encoder(tensor).squeeze(0).cpu().numpy()
        return vec.astype(np.float32)


def build_embedder(cfg: EmbedderConfig) -> Embedder:
    if cfg.kind == "coatpattern":
        return CoatPatternEmbedder(cfg)
    weights = Path(cfg.weights) if cfg.weights else None
    if weights is None or not weights.exists():
        raise FileNotFoundError(
            f"Нет весов модели идентификации: {weights}. Обучите модель "
            f"(`cowid train-reid --data <датасет>`) или временно укажите "
            f"`embedder.kind: coatpattern` в конфигурации камеры."
        )
    return LearnedEmbedder(cfg, weights)
