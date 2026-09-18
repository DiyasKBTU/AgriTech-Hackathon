"""Общие вспомогательные функции для тестов."""

from __future__ import annotations

import numpy as np
import pytest

from cowid.types import BBox, Tracklet, TrackObservation


def unit(vec) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def make_tracklet(track_id: int, tag_reads: list[str], embeddings: list[np.ndarray],
                  cow_id=None) -> Tracklet:
    t = Tracklet(track_id=track_id, tag_reads=list(tag_reads), embeddings=list(embeddings),
                 cow_id=cow_id)
    t.tag_read_frames = list(range(len(tag_reads)))
    t.observations = [
        TrackObservation(frame_idx=i, bbox=BBox(0, 0, 10, 10), score=1.0)
        for i in range(max(len(embeddings), len(tag_reads)))
    ]
    return t


@pytest.fixture
def tmp_db(tmp_path):
    from cowid.store.db import Store

    return Store(tmp_path / "cowid.sqlite")
