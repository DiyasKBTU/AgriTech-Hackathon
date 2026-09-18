"""Ступень 4а: зоны загона и перевод пикселей в метры.

Здесь намеренно нет нейросетей. Когда животное уже опознано и у него есть
устойчивый трек, «время у кормушки» и «пройденный путь» — это геометрия.
Это не упрощение, а проектное решение: объяснимые и проверяемые показатели
вызывают у фермера доверие, а у жюри — меньше вопросов «а почему модель так решила».
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import CalibrationConfig, ZoneConfig


@dataclass
class Zone:
    """Функциональная зона: кормушка, поилка, зона отдыха, проход."""

    name: str
    polygon: np.ndarray            # (N, 2), пиксели
    min_visit_seconds: float = 20.0

    @classmethod
    def from_config(cls, cfg: ZoneConfig) -> "Zone":
        return cls(
            name=cfg.name,
            polygon=np.asarray(cfg.polygon, dtype=np.float32),
            min_visit_seconds=cfg.min_visit_seconds,
        )

    def contains(self, point: tuple[float, float]) -> bool:
        """Точка внутри полигона (алгоритм трассировки луча)."""
        x, y = point
        poly = self.polygon
        inside = False
        n = len(poly)
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            intersects = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            )
            if intersects:
                inside = not inside
            j = i
        return inside


class MaskZones:
    """Зоны сеткой клеток — для кадра из нескольких камер (MmCows, 2×2).

    Полигон на каждой камере обводят вручную; здесь клетки получены по
    разметке «ест/пьёт» на утренних кадрах (`cowid farm-chain`). Кадр режется
    на плитки так же, как у детектора, и точка переводится в клетку своей камеры.
    """

    def __init__(self, path: str, tiles: int = 2):
        data = np.load(path)
        self.tiles = tiles
        self.grid = int(data["grid"])
        self.size = (int(data["width"]), int(data["height"]))
        self.masks = {name: data[name] for name in ("feeder", "drinker") if name in data.files}

    def zone_at(self, point: tuple[float, float], frame_shape: tuple[int, ...]) -> Optional[str]:
        h, w = frame_shape[:2]
        tw, th = w / self.tiles, h / self.tiles
        col, row = min(self.tiles - 1, int(point[0] // tw)), min(self.tiles - 1, int(point[1] // th))
        x = (point[0] - col * tw) * self.size[0] / tw
        y = (point[1] - row * th) * self.size[1] / th
        r, c = int(y // self.grid), int(x // self.grid)
        cam = row * self.tiles + col
        for name, mask in self.masks.items():
            if 0 <= r < mask.shape[1] and 0 <= c < mask.shape[2] and mask[cam, r, c]:
                return name
        return None


class Calibration:
    """Перевод пиксельных координат в метры.

    Простой режим — один масштаб пикселей на метр: годится для камеры,
    смотрящей сверху вниз, где перспективные искажения малы.

    Точный режим — гомография: матрица 3x3, переводящая плоскость изображения
    в плоскость загона. Нужна для наклонной камеры; строится по четырём точкам
    с известными координатами (углы кормового стола, разметка прохода).
    """

    def __init__(self, cfg: CalibrationConfig):
        self.pixels_per_meter = cfg.pixels_per_meter
        self.homography: Optional[np.ndarray] = (
            np.asarray(cfg.homography, dtype=np.float64) if cfg.homography else None
        )

    def to_meters(self, point: tuple[float, float]) -> tuple[float, float]:
        if self.homography is None:
            return (point[0] / self.pixels_per_meter, point[1] / self.pixels_per_meter)
        vec = np.array([point[0], point[1], 1.0], dtype=np.float64)
        out = self.homography @ vec
        if abs(out[2]) < 1e-9:
            return (0.0, 0.0)
        return (float(out[0] / out[2]), float(out[1] / out[2]))

    def distance_m(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        a = self.to_meters(p1)
        b = self.to_meters(p2)
        return float(np.hypot(b[0] - a[0], b[1] - a[1]))


def build_zones(configs: list[ZoneConfig]) -> list[Zone]:
    return [Zone.from_config(c) for c in configs]
