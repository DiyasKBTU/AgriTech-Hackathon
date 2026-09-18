"""Ступень 1: находим животных в кадре.

Основной вариант — детектор, обученный командой `cowid train-detector` на
размеченных кадрах Cows2021 (камера сверху). Он выдаёт повёрнутую рамку
туловища (OBB): по ней корова вырезается и разворачивается так же, как снимки,
на которых училась модель узнавания.

Запасной вариант — готовая YOLO на COCO (класс 19 «cow»), обычные
прямоугольники. Годится для наклонной камеры сбоку; сверху находит коров
заметно хуже — сравнение в `models/detector/evaluation.json`.

Детектор отвечает только на вопрос «где в кадре животные». Какое это животное,
он не знает — это решают следующие ступени.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..config import DetectorConfig
from ..types import BBox, Detection


class Detector(Protocol):
    def detect(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        ...


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class YoloDetector:
    def __init__(self, cfg: DetectorConfig):
        from ultralytics import YOLO  # импорт внутри: остальной код работает и без torch

        self.cfg = cfg
        self.model = YOLO(cfg.model)
        self.device = resolve_device(cfg.device)

    def detect(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        n = max(1, self.cfg.tiles)
        if n == 1:
            return self._detect(frame, frame_idx)
        # Кадр из нескольких камер (или очень крупный) режется на плитки:
        # иначе коровы на уменьшенном кадре слишком мелкие для детектора.
        h, w = frame.shape[:2]
        out: list[Detection] = []
        for row in range(n):
            for col in range(n):
                y0, x0 = row * h // n, col * w // n
                tile = frame[y0:(row + 1) * h // n, x0:(col + 1) * w // n]
                for d in self._detect(tile, frame_idx):
                    b = d.bbox
                    d.bbox = BBox(b.x1 + x0, b.y1 + y0, b.x2 + x0, b.y2 + y0)
                    if d.corners:
                        d.corners = tuple((x + x0, y + y0) for x, y in d.corners)
                    out.append(d)
        return out

    def _detect(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        results = self.model.predict(
            frame,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            classes=self.cfg.class_ids or None,
            device=self.device,
            verbose=False,
        )
        out: list[Detection] = []
        for res in results:
            if res.obb is not None:
                # Повёрнутые рамки: трекеру нужен описанный прямоугольник,
                # узнаванию — сами углы.
                xyxy = res.obb.xyxy.cpu().numpy()
                conf = res.obb.conf.cpu().numpy()
                polys = res.obb.xyxyxyxy.cpu().numpy()
            elif res.boxes is not None:
                xyxy = res.boxes.xyxy.cpu().numpy()
                conf = res.boxes.conf.cpu().numpy()
                polys = [None] * len(xyxy)
            else:
                continue
            classes = (res.boxes.cls.cpu().numpy().astype(int)
                       if res.obb is None and res.boxes is not None else [0] * len(xyxy))
            for (x1, y1, x2, y2), score, poly, cls in zip(xyxy, conf, polys, classes):
                name = str(res.names.get(int(cls), "cow"))
                out.append(Detection(
                    bbox=BBox(float(x1), float(y1), float(x2), float(y2)),
                    score=float(score),
                    frame_idx=frame_idx,
                    label=name if name in ("standing", "lying") else "cow",
                    corners=None if poly is None else tuple(
                        (float(x), float(y)) for x, y in poly),
                ))
        return out


def build_detector(cfg: DetectorConfig) -> Detector:
    return YoloDetector(cfg)
