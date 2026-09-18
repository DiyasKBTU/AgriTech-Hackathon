"""Сквозной прогон на реальном видео, где неизвестно, какая корова в кадре.

В Cows2021 есть 301 ролик (8–11 марта 2020) без ответов «кто есть кто».
Размеченные снимки 181 коровы сняты раньше — 5 февраля…7 марта. Поэтому
коровы регистрируются по февральским снимкам (`cowid enroll`), а конвейер
прогоняется на мартовских роликах. Подглядеть в ответы нельзя: их нет.

Как тогда понять, что узнавание работает. Два способа без подгонки:

* **Одна корова в двух местах сразу.** Если в одном ролике две дорожки
  одновременно получили один и тот же номер, одна из них точно ошибочна.
  Это нижняя граница ошибок: ошибку, где чужой номер не совпал ни с кем
  в кадре, так не поймать.
* **Сверка глазами.** Для случайных дорожек рядом кладутся корова из видео
  и февральские фото той коровы, которую назвала система. У голштинов рисунок
  пятен уникален, совпадение видно без подготовки.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .config import PipelineConfig
from .identity.embedder import _crop, rectify
from .identity.reid_dataset import IMAGE_SUFFIXES
from .pipeline import CowIdPipeline
from .types import Tracklet

Log = Callable[[str], None]
CROP_SIZE = (260, 120)


@dataclass
class TrackRow:
    clip: str
    track_id: int
    first_frame: int
    last_frame: int
    length: int
    cow_id: Optional[str]
    source: Optional[str]
    confidence: float
    crop: str


def _clip_day(clip: Path) -> date:
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", str(clip))
    return date.fromisoformat(m.group(1)) if m else date.today()


def _read_frame(video: Path, index: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _track_crop(video: Path, t: Tracklet) -> Optional[np.ndarray]:
    obs = t.observations[len(t.observations) // 2]
    frame = _read_frame(video, obs.frame_idx)
    if frame is None:
        return None
    crop = rectify(frame, obs.corners) if obs.corners else _crop(frame, obs.bbox)
    return None if crop is None else cv2.resize(crop, CROP_SIZE)


def _conflicts(rows: list[TrackRow], min_overlap: int) -> list[tuple[TrackRow, TrackRow]]:
    """Пары дорожек одного ролика с одним номером, идущие одновременно."""
    found = []
    named = [r for r in rows if r.cow_id]
    for i, a in enumerate(named):
        for b in named[i + 1:]:
            if a.clip != b.clip or a.cow_id != b.cow_id:
                continue
            overlap = min(a.last_frame, b.last_frame) - max(a.first_frame, b.first_frame)
            if overlap >= min_overlap:
                found.append((a, b))
    return found


def run(cfg: PipelineConfig, clips: list[Path], gallery: Path, out_dir: Path,
        min_track: int = 8, log: Log = print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(exist_ok=True)
    # Галерея копируется: прогон не должен менять рабочую галерею площадки.
    work_gallery = out_dir / "gallery.json"
    shutil.copy2(gallery, work_gallery)
    pipeline = CowIdPipeline(cfg, store=None, gallery_path=work_gallery)

    rows: list[TrackRow] = []
    frames = 0
    started = time.perf_counter()
    for n, clip in enumerate(clips, 1):
        result = pipeline.process_video(clip, _clip_day(clip))
        frames += result.frames_processed
        name = clip.parent.name if clip.stem == "RGB" else clip.stem
        for t in result.tracklets:
            if t.length < min_track:
                continue
            crop = _track_crop(clip, t)
            crop_path = crops_dir / f"{name}_{t.track_id}.jpg"
            if crop is not None:
                cv2.imwrite(str(crop_path), crop)
            rows.append(TrackRow(name, t.track_id, t.first_frame, t.last_frame, t.length,
                                 t.cow_id, t.id_source, round(t.id_confidence, 3),
                                 str(crop_path) if crop is not None else ""))
        if n % 20 == 0 or n == len(clips):
            log(f"  роликов {n}/{len(clips)}, дорожек {len(rows)}")
    seconds = time.perf_counter() - started

    stride = max(1, cfg.video.frame_stride)
    conflicts = _conflicts(rows, min_overlap=3 * stride)
    identified = [r for r in rows if r.cow_id]
    report = {
        "clips": len(clips),
        "frames_processed": frames,
        "seconds": round(seconds, 1),
        "fps": round(frames / seconds, 1) if seconds else 0.0,
        "tracks": len(rows),
        "identified": len(identified),
        "unknown": len(rows) - len(identified),
        "cows_named": len({r.cow_id for r in identified}),
        "conflicts": len(conflicts),
        "tracks_in_conflict": len({(r.clip, r.track_id) for pair in conflicts for r in pair}),
        "min_track_frames": min_track,
    }
    (out_dir / "tracks.json").write_text(
        json.dumps([r.__dict__ for r in rows], ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _photos(root: Path, cow_id: str, k: int) -> list[np.ndarray]:
    files = sorted(p for p in (root / cow_id).glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        return []
    picks = [files[i] for i in np.linspace(0, len(files) - 1, min(k, len(files))).astype(int)]
    out = []
    for p in picks:
        img = cv2.imread(str(p))
        if img is not None:
            out.append(cv2.resize(img, CROP_SIZE))
    return out


def contact_sheets(out_dir: Path, photos_root: Path, n_tracks: int = 40,
                   per_page: int = 10, seed: int = 7) -> list[Path]:
    """Листы для сверки глазами: слева корова из видео, справа фото из галереи.

    Дорожки выбираются случайно, но не больше одной на корову — иначе лист
    заполнят несколько самых частых коров.
    """
    rows = json.loads((out_dir / "tracks.json").read_text(encoding="utf-8"))
    rng = random.Random(seed)
    rng.shuffle(rows)
    picked, seen = [], set()
    for r in rows:
        if r["cow_id"] and r["crop"] and r["cow_id"] not in seen:
            picked.append(r)
            seen.add(r["cow_id"])
        if len(picked) == n_tracks:
            break

    pages = []
    w, h = CROP_SIZE
    for page in range(0, len(picked), per_page):
        lines = []
        for i, r in enumerate(picked[page:page + per_page], start=page + 1):
            video = cv2.imread(r["crop"])
            refs = _photos(photos_root, r["cow_id"], 2)
            if video is None or not refs:
                continue
            gap = np.full((h, 12, 3), 255, np.uint8)
            line = np.hstack([video, gap, *refs])
            label = np.full((22, line.shape[1], 3), 255, np.uint8)
            cv2.putText(label, f"#{i}  video {r['clip']} track {r['track_id']}  ->  cow {r['cow_id']}"
                               f"  (votes {r['confidence']:.0%})",
                        (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            lines.append(np.vstack([label, line]))
        if not lines:
            continue
        width = max(x.shape[1] for x in lines)
        lines = [np.hstack([x, np.full((x.shape[0], width - x.shape[1], 3), 255, np.uint8)])
                 for x in lines]
        path = out_dir / f"sheet_{page // per_page + 1}.jpg"
        cv2.imwrite(str(path), np.vstack(lines))
        pages.append(path)
    (out_dir / "sheet_tracks.json").write_text(
        json.dumps(picked, ensure_ascii=False, indent=1), encoding="utf-8")
    return pages
