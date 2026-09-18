"""Трекинг и активность: геометрия, окклюзии, разрезание треков по бирке."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from cowid.activity.features import ActivityExtractor
from cowid.activity.zones import Calibration, Zone
from cowid.config import ActivityConfig, CalibrationConfig, TrackerConfig
from cowid.track.split import count_tag_conflicts, split_tracklet_by_tag
from cowid.track.tracker import CowTracker
from cowid.types import BBox, Detection, Tracklet, TrackObservation

from .conftest import unit


def test_bbox_iou_and_centre():
    a, b = BBox(0, 0, 10, 10), BBox(5, 0, 15, 10)
    assert a.iou(b) == pytest.approx(1 / 3, abs=1e-6)
    assert BBox(0, 0, 10, 20).bottom_center == (5.0, 20.0)


def test_zone_contains_point():
    zone = Zone(name="feeder", polygon=np.array([(0, 0), (100, 0), (100, 50), (0, 50)]))
    assert zone.contains((50, 25))
    assert not zone.contains((150, 25))


# --------------------------------------------------------------------------
# Трекер
# --------------------------------------------------------------------------

def test_one_animal_gives_one_track():
    tracker = CowTracker(TrackerConfig(min_hits=1))
    for f in range(10):
        tracker.update([Detection(BBox(f * 5, 0, f * 5 + 40, 40), 0.9, f)], f)
    tracks = tracker.finalize()
    assert len(tracks) == 1 and tracks[0].length == 10


def test_track_survives_occlusion():
    """Животное закрыли на восемь кадров, потом оно вышло — номер тот же."""
    tracker = CowTracker(TrackerConfig(min_hits=1, max_age=30))
    emb = unit([1, 0, 0])
    for f in range(6):
        tracker.update([Detection(BBox(f * 6, 0, f * 6 + 40, 40), 0.9, f)], f, [emb])
    for f in range(6, 14):
        tracker.update([], f, [])
    tracker.update([Detection(BBox(84, 0, 124, 40), 0.9, 14)], 14, [emb])
    assert len(tracker.finalize()) == 1


def test_noisy_boxes_do_not_make_the_prediction_drift():
    """Регрессия на реальную ошибку: скорость считалась от ПРЕДСКАЗАННОГО
    положения, ошибка накапливалась, и неподвижное животное рассыпалось на
    сотни обрывков."""
    rng = np.random.default_rng(0)
    tracker = CowTracker(TrackerConfig(min_hits=1))
    for f in range(200):
        j = rng.normal(0, 3, 4)
        tracker.update([Detection(BBox(100 + j[0], 100 + j[1], 190 + j[2], 160 + j[3]), 0.9, f)],
                       f, [None])
    tracks = tracker.finalize()
    assert len(tracks) == 1 and tracks[0].length == 200


def test_two_crossing_animals_stay_two_tracks():
    tracker = CowTracker(TrackerConfig(min_hits=1, max_age=20))
    ea, eb = unit([1, 0, 0]), unit([0, 1, 0])
    for f in range(8):
        a = Detection(BBox(f * 8, 0, f * 8 + 40, 40), 0.9, f)
        b = Detection(BBox(200 - f * 8, 0, 240 - f * 8, 40), 0.9, f)
        tracker.update([a, b], f, [ea, eb])
    for f in range(8, 16):
        s = f - 8
        a = Detection(BBox(64 - s * 8, 0, 104 - s * 8, 40), 0.9, f)
        b = Detection(BBox(136 + s * 8, 0, 176 + s * 8, 40), 0.9, f)
        tracker.update([a, b], f, [ea, eb])
    assert len(tracker.finalize()) == 2


# --------------------------------------------------------------------------
# Разрезание трека по бирке
# --------------------------------------------------------------------------

def _track_with_reads(reads: list[tuple[int, str]], length: int = 120) -> Tracklet:
    t = Tracklet(track_id=7)
    t.observations = [TrackObservation(i, BBox(0, 0, 10, 10), 1.0) for i in range(length)]
    t.tag_read_frames = [f for f, _ in reads]
    t.tag_reads = [r for _, r in reads]
    return t


def test_track_is_cut_where_the_tag_number_changes():
    """Первую половину трек вёл корову 1001, вторую — 1007. Это два животных."""
    reads = [(f, "1001") for f in range(5, 50, 5)] + [(f, "1007") for f in range(65, 115, 5)]
    pieces = split_tracklet_by_tag(_track_with_reads(reads))
    assert len(pieces) == 2
    assert set(pieces[0].tag_reads) == {"1001"}
    assert set(pieces[1].tag_reads) == {"1007"}


def test_single_misread_does_not_cut_a_healthy_track():
    reads = [(f, "1001") for f in range(5, 115, 5)]
    reads[10] = (reads[10][0], "1091")
    assert len(split_tracklet_by_tag(_track_with_reads(reads))) == 1


def test_conflict_counter():
    reads = [(f, "1001") for f in range(0, 20, 5)] + [(f, "1007") for f in range(60, 80, 5)]
    assert count_tag_conflicts([_track_with_reads(reads)]) == 1


# --------------------------------------------------------------------------
# Активность
# --------------------------------------------------------------------------

def test_time_in_feeder_zone_is_counted():
    zones = [Zone(name="feeder", polygon=np.array([(0, 0), (200, 0), (200, 100), (0, 100)]))]
    ex = ActivityExtractor(ActivityConfig(smooth_window=1), zones,
                           Calibration(CalibrationConfig(pixels_per_meter=10)),
                           seconds_per_frame=60.0)
    t = Tracklet(track_id=1, cow_id="KZ-1001")
    t.observations = (
        [TrackObservation(i, BBox(50, 40, 90, 80), 1.0) for i in range(5)]
        + [TrackObservation(i, BBox(50, 300, 90, 340), 1.0) for i in range(5, 10)]
    )
    feats = ex.extract([t], date(2026, 1, 1))
    assert feats[0].feeder_seconds == pytest.approx(300.0)
    assert feats[0].observed_seconds == pytest.approx(600.0)


def test_unidentified_tracks_do_not_enter_statistics():
    ex = ActivityExtractor(ActivityConfig(), [], Calibration(CalibrationConfig()), 1.0)
    t = Tracklet(track_id=1, cow_id=None)
    t.observations = [TrackObservation(0, BBox(0, 0, 10, 10), 1.0)]
    assert ex.extract([t], date(2026, 1, 1)) == []
