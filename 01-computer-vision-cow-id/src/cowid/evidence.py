"""Сбор доказательств: чем система подтверждает каждое своё утверждение.

Зачем это нужно. Событие «корова KZ-1234 провела у кормушки на 42% меньше нормы»
само по себе непроверяемо. Зоотехник не станет доверять числу, которое неоткуда
перепроверить, и будет прав. Поэтому к каждому событию система обязана приложить:

* **кадр животного** — вот кто это, посмотрите сами;
* **кадр с биркой** — вот откуда взялся номер, и вот что на нём прочитано;
* **траекторию за сутки** — вот где животное было и сколько времени;
* **ленту суток** — когда именно оно ело, пило и отдыхало.

Это не украшение интерфейса. Это то, что отличает систему, которой пользуются,
от системы, которую выключают на второй неделе.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import PipelineConfig
from .activity.zones import build_zones
from .types import BBox, Tracklet


@dataclass
class CowEvidence:
    """Доказательный материал по одному животному за одни сутки."""

    cow_id: str
    day: str
    #: Путь к кадру с животным, относительно каталога доказательств.
    animal_crop: Optional[str] = None
    #: Путь к кадру, на котором была прочитана бирка.
    tag_crop: Optional[str] = None
    #: Что именно прочитано с бирки и на скольких кадрах.
    tag_text: Optional[str] = None
    tag_votes: int = 0
    tag_total: int = 0
    #: Картинка траектории за сутки.
    track_image: Optional[str] = None
    #: Лента состояний: доли времени по порядку кадров, для полоски в интерфейсе.
    timeline: list[str] = field(default_factory=list)
    id_source: Optional[str] = None
    id_confidence: float = 0.0
    frames_seen: int = 0

    def as_dict(self) -> dict:
        return {
            "cow_id": self.cow_id,
            "day": self.day,
            "animal_crop": self.animal_crop,
            "tag_crop": self.tag_crop,
            "tag_text": self.tag_text,
            "tag_votes": self.tag_votes,
            "tag_total": self.tag_total,
            "track_image": self.track_image,
            "timeline": self.timeline,
            "id_source": self.id_source,
            "id_confidence": round(self.id_confidence, 3),
            "frames_seen": self.frames_seen,
        }


class EvidenceCollector:
    """Второй проход по видео: вырезает кадры и рисует траектории.

    Отдельный проход, а не сбор по ходу конвейера, сделан намеренно: к моменту,
    когда известно, какое животное нас интересует и какой у него номер, видео уже
    прочитано до конца. Хранить в памяти все кадры дороже, чем перечитать файл.
    """

    def __init__(self, cfg: PipelineConfig, out_dir: str | Path):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Тот же читатель бирок, что и в конвейере: для крупного плана нужно
        # показать ровно ту область, в которой система прочитала номер,
        # а не «примерно там, где ухо».
        from .identity.tag_ocr import build_tag_reader

        self._tag_reader = build_tag_reader(cfg.tag_ocr)
        self._frame_size = (1280, 720)

    def collect(
        self,
        tracklets: list[Tracklet],
        day: date,
        video_path: str | Path,
        cow_ids: Optional[set[str]] = None,
    ) -> dict[str, CowEvidence]:
        """Собирает доказательства по указанным животным за одни сутки."""
        by_cow: dict[str, list[Tracklet]] = {}
        for t in tracklets:
            if not t.cow_id:
                continue
            if cow_ids is not None and t.cow_id not in cow_ids:
                continue
            by_cow.setdefault(t.cow_id, []).append(t)
        if not by_cow:
            return {}

        # Какие кадры вообще нужно достать из видео — собираем заранее,
        # чтобы прочитать файл ровно один раз.
        wanted: dict[int, list[tuple[str, BBox, str]]] = {}
        evidence: dict[str, CowEvidence] = {}

        for cow_id, tracks in by_cow.items():
            best = max(tracks, key=lambda t: t.length)
            ev = CowEvidence(
                cow_id=cow_id,
                day=day.isoformat(),
                id_source=best.id_source,
                id_confidence=best.id_confidence,
                frames_seen=sum(t.length for t in tracks),
                tag_total=sum(len(t.tag_reads) for t in tracks),
            )
            reads = [r for t in tracks for r in t.tag_reads if r]
            if reads:
                ev.tag_text = max(set(reads), key=reads.count)
                ev.tag_votes = reads.count(ev.tag_text)

            # Кадр животного берём из середины самого длинного трека: там
            # животное уже уверенно отслеживается и обычно хорошо видно.
            mid = best.observations[len(best.observations) // 2]
            wanted.setdefault(mid.frame_idx, []).append((cow_id, mid.bbox, "animal"))

            # Кадр с биркой. Номер читается далеко не на каждом кадре трека,
            # поэтому предлагаем несколько кандидатов и оставляем первый, на
            # котором область бирки действительно нашлась.
            tag_track = next((t for t in tracks if t.tag_reads), None)
            if tag_track is not None and tag_track.observations:
                n = len(tag_track.observations)
                for frac in (0.5, 0.25, 0.75, 0.1, 0.9, 0.35, 0.6):
                    obs = tag_track.observations[min(n - 1, int(n * frac))]
                    wanted.setdefault(obs.frame_idx, []).append((cow_id, obs.bbox, "tag"))

            evidence[cow_id] = ev

        self._extract_crops(video_path, wanted, evidence, day)

        for cow_id, tracks in by_cow.items():
            evidence[cow_id].track_image = self._draw_track(cow_id, tracks, day)
            evidence[cow_id].timeline = self._timeline(tracks)

        return evidence

    # -- кадры -------------------------------------------------------------

    def _extract_crops(
        self,
        video_path: str | Path,
        wanted: dict[int, list[tuple[str, BBox, str]]],
        evidence: dict[str, CowEvidence],
        day: date,
    ) -> None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return
        self._frame_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280,
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720,
        )
        try:
            frame_idx = 0
            remaining = dict(wanted)
            while remaining:
                ok, frame = cap.read()
                if not ok:
                    break
                items = remaining.pop(frame_idx, None)
                if items:
                    for cow_id, bbox, kind in items:
                        if kind == "tag" and evidence[cow_id].tag_crop:
                            continue          # крупный план бирки уже получен
                        path = self._save_crop(frame, bbox, cow_id, day, kind)
                        if path is None:
                            continue
                        if kind == "animal":
                            evidence[cow_id].animal_crop = path
                        else:
                            evidence[cow_id].tag_crop = path
                frame_idx += 1
        finally:
            cap.release()

    def _save_crop(
        self, frame: np.ndarray, bbox: BBox, cow_id: str, day: date, kind: str
    ) -> Optional[str]:
        """Два разных кадра для двух разных вопросов.

        «animal» отвечает на вопрос «кто это и где он был» — берём общий план
        с запасом, чтобы животное было видно в контексте загона.
        «tag» отвечает на вопрос «откуда взялся номер» — вырезаем ровно ту
        область, где OCR прочитал цифры, и увеличиваем её так, чтобы номер мог
        прочитать человек. Если он не читается глазами, ему нельзя верить.
        """
        h, w = frame.shape[:2]
        b = bbox.clip(w, h)

        if kind == "tag":
            shown = self._crop_tag(frame, b)
            if shown is None:
                return None
        else:
            side = max(b.width, b.height) * 1.6
            cx, cy = b.center
            # Берём область 4:3 вокруг животного — привычная пропорция кадра.
            half_w, half_h = side * 0.72, side * 0.54
            x1 = int(max(0, cx - half_w)); y1 = int(max(0, cy - half_h))
            x2 = int(min(w, cx + half_w)); y2 = int(min(h, cy + half_h))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            shown = crop.copy()
            cv2.rectangle(
                shown,
                (int(b.x1 - x1), int(b.y1 - y1)),
                (int(b.x2 - x1), int(b.y2 - y1)),
                (230, 160, 60), 2,
            )
            scale = max(1.0, 300.0 / max(1, shown.shape[0]))
            if scale > 1.0:
                shown = cv2.resize(shown, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_CUBIC)

        rel = f"{cow_id}/{day.isoformat()}_{kind}.jpg"
        path = self.out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), shown, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return rel

    def _crop_tag(self, frame: np.ndarray, b: BBox) -> Optional[np.ndarray]:
        """Крупный план ушной бирки — той области, где OCR нашёл цифры."""
        crop = frame[int(b.y1):int(b.y2), int(b.x1):int(b.x2)]
        if crop.size == 0:
            return None

        region = None
        finder = getattr(self._tag_reader, "_find_tag_region", None)
        if finder is not None:
            try:
                region = finder(crop)
            except Exception:
                region = None
        if region is None or region.size == 0:
            return None

        # Увеличиваем до читаемого размера с запасом по краям.
        pad = 6
        padded = cv2.copyMakeBorder(region, pad, pad, pad, pad,
                                    cv2.BORDER_CONSTANT, value=(250, 250, 250))
        scale = max(1.0, 110.0 / max(1, padded.shape[0]))
        big = cv2.resize(padded, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(big, (0, 0), (big.shape[1] - 1, big.shape[0] - 1), (60, 190, 90), 3)
        return big

    # -- траектория --------------------------------------------------------

    def _draw_track(self, cow_id: str, tracks: list[Tracklet], day: date) -> Optional[str]:
        """Рисует путь животного по загону с зонами кормушки и поилки."""
        width, height = 640, 360
        canvas = np.full((height, width, 3), 250, dtype=np.uint8)
        frame_w, frame_h = self._frame_size
        sx, sy = width / frame_w, height / frame_h

        for zone in self.cfg.zones:
            poly = np.asarray(
                [[int(x * sx), int(y * sy)] for x, y in zone.polygon], dtype=np.int32
            )
            # BGR: кормушка зеленоватая, поилка голубая — как в легенде.
            shade = (226, 240, 228) if zone.name == "feeder" else (238, 226, 210)
            cv2.fillPoly(canvas, [poly], shade)
            cv2.polylines(canvas, [poly], True, (170, 175, 182), 1)
            cv2.putText(canvas, zone.name, tuple(poly[0] + np.array([6, 16])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 126, 134), 1, cv2.LINE_AA)

        cv2.rectangle(canvas, (2, 2), (width - 3, height - 3), (196, 200, 206), 1)

        points: list[tuple[int, int]] = []
        for track in sorted(tracks, key=lambda t: t.first_frame):
            for obs in track.observations:
                cx, cy = obs.bbox.center
                points.append((int(cx * sx), int(cy * sy)))

        # Путь рисуем градиентом от начала суток к концу: видно направление
        # и то, что животное реально перемещалось, а не стояло точкой.
        for i in range(1, len(points)):
            t = i / max(1, len(points) - 1)
            colour = (int(190 - 120 * t), int(120 + 40 * t), int(60 + 40 * t))
            cv2.line(canvas, points[i - 1], points[i], colour, 1, cv2.LINE_AA)
        if points:
            cv2.circle(canvas, points[0], 4, (120, 170, 120), -1)
            cv2.circle(canvas, points[-1], 4, (70, 90, 200), -1)

        rel = f"{cow_id}/{day.isoformat()}_track.png"
        path = self.out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), canvas)
        return rel

    # -- лента суток -------------------------------------------------------

    def _timeline(self, tracks: list[Tracklet], buckets: int = 96) -> list[str]:
        """Где животное было в каждый отрезок наблюдения: полоска на весь день.

        Возвращает список меток зон по отрезкам — интерфейс рисует из них
        цветную ленту, по которой сразу видно, когда корова подходила к корму.
        """
        zones = {z.name: z for z in build_zones(self.cfg.zones)}

        frames: dict[int, str] = {}
        for track in tracks:
            for obs in track.observations:
                point = obs.bbox.center
                label = "other"
                for name, zone in zones.items():
                    if zone.contains(point):
                        label = name
                        break
                frames[obs.frame_idx] = label
        if not frames:
            return []

        last = max(frames)
        size = max(1, (last + 1) / buckets)
        timeline: list[str] = []
        for b in range(buckets):
            lo, hi = int(b * size), int((b + 1) * size)
            labels = [frames[f] for f in range(lo, hi) if f in frames]
            if not labels:
                timeline.append("gap")
            else:
                timeline.append(max(set(labels), key=labels.count))
        return timeline
