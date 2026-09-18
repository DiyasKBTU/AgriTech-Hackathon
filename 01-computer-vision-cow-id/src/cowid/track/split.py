"""Разрезание треков, внутри которых сменилось животное.

Проблема, которую это решает. При скученности у кормового стола трекер иногда
передаёт трек соседнему животному: рамки перекрываются почти одинаково, и
геометрия не может выбрать правильную. Дальше трек живёт долго и всё это время
ведёт чужую корову — её показатели уходят в чужую карточку. Мы это измерили:
на сутки приходилось тринадцать длинных треков на двенадцать животных и при
этом двести подмен внутри них. Трекер выглядел отличным по числу треков и был
при этом бесполезным.

Дескриптор внешности здесь не спасает: у рисунка шкуры слишком мал запас между
«то же животное» и «другое животное», чтобы разрешать спор в момент пересечения.

Зато есть канал, которому мы доверяем, — ушная бирка. Если первую половину
трека OCR устойчиво читал «1001», а вторую так же устойчиво «1007», то это
не ошибка распознавания, а смена животного, и место разреза известно с точностью
до кадра. Геометрия этого не видит, а бирка видит.

Важно: разрезаем только при УСТОЙЧИВОЙ смене. Одиночное неверное чтение — это
обычная ошибка OCR, и её надо игнорировать, иначе мы покрошим здоровые треки.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from ..types import Tracklet


def _stable_segments(
    reads: list[tuple[int, str]], min_reads: int
) -> list[tuple[int, str]]:
    """Находит участки устойчивого чтения номера: (кадр начала, номер).

    Идём по прочитанным номерам подряд. Смена засчитывается, только если новый
    номер повторился подряд не меньше `min_reads` раз — так одиночные ошибки
    OCR не порождают ложных разрезов.
    """
    segments: list[tuple[int, str]] = []
    current: Optional[str] = None
    run_text: Optional[str] = None
    run: list[tuple[int, str]] = []

    for frame, text in reads:
        if text == run_text:
            run.append((frame, text))
        else:
            run_text, run = text, [(frame, text)]

        if len(run) >= min_reads and run_text != current:
            current = run_text
            segments.append((run[0][0], run_text))
    return segments


def split_tracklet_by_tag(
    tracklet: Tracklet, min_reads: int = 3, min_segment_frames: int = 30
) -> list[Tracklet]:
    """Разрезает один трек по точкам устойчивой смены номера на бирке."""
    if len(tracklet.tag_reads) != len(tracklet.tag_read_frames):
        return [tracklet]

    reads = sorted(zip(tracklet.tag_read_frames, tracklet.tag_reads))
    if len({t for _, t in reads}) < 2:
        return [tracklet]

    segments = _stable_segments(reads, min_reads)
    if len(segments) < 2:
        return [tracklet]

    boundaries = [frame for frame, _ in segments[1:]]
    pieces: list[Tracklet] = []
    start = tracklet.first_frame
    edges = [start] + boundaries + [tracklet.last_frame + 1]

    for lo, hi in zip(edges, edges[1:]):
        if hi - lo < min_segment_frames:
            continue
        piece = Tracklet(track_id=tracklet.track_id * 1000 + len(pieces))
        for obs, emb in zip(
            tracklet.observations,
            tracklet.embeddings or [None] * len(tracklet.observations),
        ):
            if lo <= obs.frame_idx < hi:
                piece.observations.append(obs)
                if emb is not None:
                    piece.embeddings.append(emb)
        for frame, text in reads:
            if lo <= frame < hi:
                piece.tag_reads.append(text)
                piece.tag_read_frames.append(frame)
        if piece.observations:
            pieces.append(piece)

    # Если разрезать не получилось (например, все куски слишком короткие),
    # честнее вернуть исходный трек, чем потерять наблюдения.
    return pieces if len(pieces) >= 2 else [tracklet]


def split_tracklets_by_tag(
    tracklets: list[Tracklet], min_reads: int = 3, min_segment_frames: int = 30
) -> list[Tracklet]:
    out: list[Tracklet] = []
    for t in tracklets:
        out.extend(split_tracklet_by_tag(t, min_reads, min_segment_frames))
    return out


def count_tag_conflicts(tracklets: list[Tracklet]) -> int:
    """Сколько треков содержат больше одного устойчиво прочитанного номера.

    Диагностический показатель: если после разрезания он не падает почти до нуля,
    значит трекер подменяет животных чаще, чем бирка успевает это заметить.
    """
    conflicts = 0
    for t in tracklets:
        if len(t.tag_reads) < 2:
            continue
        counts = Counter(t.tag_reads)
        strong = [text for text, n in counts.items() if n >= 3]
        if len(strong) > 1:
            conflicts += 1
    return conflicts
