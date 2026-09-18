"""Цепочка на ферме MmCows: проверяем «трубы» метрик, а не качество моделей."""

import numpy as np

from cowid import mmcows_chain as fc


def test_pairs_and_occlusion():
    a = np.array([0, 0, 100, 100], np.float32)
    b = np.array([90, 0, 190, 100], np.float32)      # чуть задевает a
    far = np.array([500, 500, 600, 600], np.float32)
    assert fc._occluded([a, b, far]) == [False, False, False]
    c = np.array([50, 0, 150, 100], np.float32)      # IoU с a = 1/3
    assert fc._occluded([a, c]) == [True, True]
    # Разметка a, far; найдено far и сдвинутая a — пары по IoU ≥ 0.5.
    shifted = a + np.array([10, 0, 10, 0], np.float32)
    assert sorted(fc._pairs([a, far], [far, shifted])) == [(0, 1), (1, 0)]
    assert fc._pairs([a], [b]) == []


def test_idf1_counts_identity_swap():
    """Две коровы меняются номерами на полпути: IDF1 падает, подмены считаются."""
    mm = fc._motmetrics()
    boxes = fc._xywh([[0, 0, 10, 10], [50, 0, 60, 10]])
    acc = mm.MOTAccumulator(auto_id=True)
    ids = fc.HypIds()
    for step in range(10):
        hyp = ["C01", "C02"] if step < 5 else ["C02", "C01"]
        acc.update([1, 2], [ids.cow(h) for h in hyp],
                   mm.distances.iou_matrix(boxes, boxes, max_iou=0.5))
    assert ids.cow("C02") == 2 and ids.unknown("x") != ids.unknown("y")
    r = mm.metrics.create().compute(acc, metrics=["idf1", "num_switches", "mota"], name="t")
    assert abs(float(r["idf1"].iloc[0]) - 0.5) < 1e-9
    assert int(r["num_switches"].iloc[0]) == 2


def test_hour_and_bucket():
    assert abs(fc._hour("1690331831_19-37-11") - (19 + 37 / 60 + 11 / 3600)) < 1e-9
    assert fc._bucket(15.0).startswith("день")
    assert fc._bucket(22.5).startswith("ночь")
    assert fc.cow_name(7) == "C07"


def test_crop_pads_and_skips_tiny():
    frame = np.zeros((1000, 1600, 3), np.uint8)
    piece = fc.crop(frame, np.array([100, 100, 300, 200], np.float32))
    assert piece.shape[:2] == (110, 220)
    assert fc.crop(frame, np.array([10, 10, 30, 30], np.float32)) is None
