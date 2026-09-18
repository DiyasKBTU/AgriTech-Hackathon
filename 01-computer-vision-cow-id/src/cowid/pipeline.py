"""Конвейер: видео с камеры -> события для зоотехника.

    КАМЕРА -> ДЕТЕКЦИЯ -> ТРЕКИНГ -> ИДЕНТИФИКАЦИЯ -> АКТИВНОСТЬ -> ОТКЛОНЕНИЕ -> СОБЫТИЕ

Единица обработки — одна видеозапись одной камеры за одни сутки. Сутки —
естественный шаг: персональная норма строится по дням, и зоотехник получает
события утром по итогам прошедших суток, а не каждые пять минут.

Состояние между сутками живёт в двух местах: галерея портретов животных
(файл) и суточные показатели (база). Из показателей перед оценкой новых суток
заново строится персональная норма каждого животного.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Iterator, Optional

import cv2
import numpy as np

from .activity.features import ActivityExtractor
from .activity.zones import Calibration, build_zones
from .anomaly.baseline import PersonalBaseline
from .config import PipelineConfig
from .detect.detectors import Detector, build_detector
from .identity.embedder import Embedder, build_embedder
from .identity.gallery import BiometricGallery
from .identity.identifier import TagTaughtIdentifier
from .identity.tag_ocr import TagReader, build_tag_reader
from .store.db import Store
from .track.split import split_tracklets_by_tag
from .track.tracker import CowTracker
from .types import ActivityFeatures, Detection, Event, Tracklet


@dataclass
class DayResult:
    day: date
    video: str
    tracklets: list[Tracklet] = field(default_factory=list)
    features: list[ActivityFeatures] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    frames_read: int = 0
    frames_processed: int = 0
    seconds_elapsed: float = 0.0
    identifier_stats: dict = field(default_factory=dict)
    gallery_size: int = 0

    @property
    def identified(self) -> int:
        return sum(1 for t in self.tracklets if t.cow_id)

    @property
    def processing_fps(self) -> float:
        return self.frames_processed / self.seconds_elapsed if self.seconds_elapsed else 0.0


class CowIdPipeline:
    def __init__(
        self,
        cfg: PipelineConfig,
        store: Optional[Store] = None,
        detector: Optional[Detector] = None,
        embedder: Optional[Embedder] = None,
        tag_reader: Optional[TagReader] = None,
        gallery_path: Optional[str | Path] = None,
    ):
        """Компоненты можно подставить снаружи — так конвейер тестируется без
        тяжёлых моделей, а на ферме меняется модель без правки кода."""
        self.cfg = cfg
        self.store = store
        self.detector = detector or build_detector(cfg.detector)
        self.embedder = embedder or build_embedder(cfg.embedder)
        self.tag_reader = tag_reader or build_tag_reader(cfg.tag_ocr)

        self.gallery_path = Path(gallery_path) if gallery_path else None
        if self.gallery_path and self.gallery_path.exists():
            self.gallery = BiometricGallery.load(self.gallery_path, cfg.gallery)
        else:
            self.gallery = BiometricGallery(cfg.gallery)
        self.identifier = TagTaughtIdentifier(self.gallery, cfg.gallery)

        self.zones = build_zones(cfg.zones)
        self.calibration = Calibration(cfg.calibration)

    # -- основной вход ----------------------------------------------------

    def process_video(
        self,
        video_path: str | Path,
        day: date,
        progress: Optional[Callable[[int, int], None]] = None,
        max_frames: Optional[int] = None,
    ) -> DayResult:
        started = time.perf_counter()
        video_path = Path(video_path)
        fps, total = self._video_meta(video_path)
        run_id = (self.store.start_run(self.cfg.camera_id, str(video_path), day)
                  if self.store else None)
        stats_before = self.identifier.stats.snapshot()

        stride = max(1, self.cfg.video.frame_stride)
        seconds_per_sample = stride / fps

        tracker = CowTracker(self.cfg.tracker)
        frames_read = frames_processed = 0

        for frame_idx, frame in self._frames(video_path, stride, max_frames):
            frames_read = frame_idx + 1
            detections = self.detector.detect(frame, frame_idx)
            embeddings = [self.embedder.embed(frame, d.bbox, d.corners) for d in detections]
            reads = [self.tag_reader.read(frame, d.bbox) for d in detections]
            active = tracker.update(detections, frame_idx, embeddings)
            self._attach_tag_reads(active, detections, [r.text if r else None for r in reads],
                                   frame_idx)
            frames_processed += 1
            if progress and frames_processed % 50 == 0:
                progress(frames_read, total)

        # Трекер при скученности иногда передаёт трек соседнему животному.
        # Бирка это видит: номер на кадрах трека устойчиво меняется. Такие
        # треки разрезаются до идентификации.
        tracklets = split_tracklets_by_tag(tracker.finalize())

        # Длинные треки первыми: у них больше шансов на читаемую бирку, и они
        # первыми наполняют галерею, по которой затем узнаются короткие.
        tracklets.sort(key=lambda t: t.length, reverse=True)
        for t in tracklets:
            self.identifier.identify(t)

        extractor = ActivityExtractor(self.cfg.activity, self.zones, self.calibration,
                                      seconds_per_frame=seconds_per_sample)
        features = extractor.extract(tracklets, day)
        events = self._evaluate(day, features)

        result = DayResult(
            day=day, video=str(video_path), tracklets=tracklets, features=features,
            events=events, frames_read=frames_read, frames_processed=frames_processed,
            seconds_elapsed=time.perf_counter() - started,
            identifier_stats=self.identifier.stats.since(stats_before).as_dict(),
            gallery_size=self.gallery.size(),
        )
        self._persist(result, run_id)
        return result

    # -- оценка суток -----------------------------------------------------

    def _evaluate(self, day: date, features: list[ActivityFeatures]) -> list[Event]:
        """Строит персональные нормы по истории из базы и оценивает новые сутки.

        Норма не хранится отдельно, а каждый раз воспроизводится по фактам.
        Так невозможна ситуация, когда норма в памяти разошлась с тем, что
        записано в базе, — например, после исправления показателей за прошлый день.
        """
        baseline = PersonalBaseline(self.cfg.baseline)
        if self.store is not None:
            history = self.store.features_before(day)
            for past_day in sorted({f.day for f in history}):
                baseline.evaluate_day([f for f in history if f.day == past_day])
        return baseline.evaluate_day(features)

    def _persist(self, result: DayResult, run_id: Optional[int]) -> None:
        if self.gallery_path is not None:
            self.gallery.save(self.gallery_path)
        if self.store is None:
            return
        sources = {t.cow_id: t.id_source for t in result.tracklets if t.cow_id}
        self.store.save_features(self.cfg.camera_id, result.features, sources)
        self.store.save_events(result.events)
        if run_id is not None:
            self.store.finish_run(
                run_id, frames=result.frames_processed, tracklets=len(result.tracklets),
                identified=result.identified,
                stats={**result.identifier_stats, "gallery": result.gallery_size,
                       "fps": round(result.processing_fps, 1)},
            )

    # -- видео ------------------------------------------------------------

    def _video_meta(self, path: Path) -> tuple[float, int]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Не удалось открыть видео: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or self.cfg.video.fallback_fps
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return float(fps), total

    @staticmethod
    def _frames(path: Path, stride: int,
                max_frames: Optional[int]) -> Iterator[tuple[int, np.ndarray]]:
        """Кадры с прореживанием. Пропускаемые кадры не декодируются целиком
        (grab без retrieve) — это в несколько раз быстрее, чем читать все."""
        cap = cv2.VideoCapture(str(path))
        idx = 0
        try:
            while True:
                if max_frames is not None and idx >= max_frames:
                    break
                if idx % stride == 0:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    yield idx, frame
                elif not cap.grab():
                    break
                idx += 1
        finally:
            cap.release()

    @staticmethod
    def _attach_tag_reads(active: list[Tracklet], detections: list[Detection],
                          texts: list[Optional[str]], frame_idx: int) -> None:
        """Прочитанный номер привязывается к треку, который только что обновлён
        этой же детекцией: их рамки совпадают."""
        for det, text in zip(detections, texts):
            if not text:
                continue
            best, best_iou = None, 0.0
            for t in active:
                if not t.observations or t.observations[-1].frame_idx != frame_idx:
                    continue
                iou = t.observations[-1].bbox.iou(det.bbox)
                if iou > best_iou:
                    best, best_iou = t, iou
            if best is not None and best_iou > 0.5:
                best.tag_reads.append(text)
                best.tag_read_frames.append(frame_idx)
