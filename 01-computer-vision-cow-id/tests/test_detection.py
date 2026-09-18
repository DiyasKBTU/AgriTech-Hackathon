"""Геометрия повёрнутых рамок: разметка, сопоставление, вырезка коровы.

Проверяется код, а не качество детектора — качество меряет `cowid eval-detector`
на реальных кадрах Cows2021.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from cowid.detect.train_detector import (
    match_counts, obb_corners, polygon_iou, prepare_dataset,
)
from cowid.identity.embedder import rectify

XML = """<annotation>
  <size><width>200</width><height>100</height><depth>3</depth></size>
  <object><type>bndbox</type><name>head</name>
    <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>10</xmax><ymax>10</ymax></bndbox></object>
  <object><type>robndbox</type><name>cow</name>
    <robndbox><cx>100</cx><cy>50</cy><w>80</w><h>40</h><angle>0</angle></robndbox></object>
</annotation>"""


def test_corners_without_rotation_are_the_plain_box():
    c = obb_corners(100, 50, 80, 40, 0.0)
    assert np.allclose(c, [[60, 30], [140, 30], [140, 70], [60, 70]])


def test_quarter_turn_makes_the_body_vertical():
    c = obb_corners(100, 100, 80, 20, math.pi / 2)
    width = c[:, 0].max() - c[:, 0].min()
    height = c[:, 1].max() - c[:, 1].min()
    assert math.isclose(width, 20, abs_tol=1e-3)
    assert math.isclose(height, 80, abs_tol=1e-3)


def test_polygon_iou_and_greedy_matching():
    a = obb_corners(50, 50, 40, 20, 0.3)
    far = obb_corners(300, 300, 40, 20, 0.3)
    assert math.isclose(polygon_iou(a, a), 1.0, abs_tol=1e-6)
    assert polygon_iou(a, far) == 0.0
    # одна рамка попала, одна мимо; одна корова найдена, одна пропущена
    assert match_counts([a, far], [a, obb_corners(150, 50, 40, 20, 0)]) == (1, 1, 1)


def test_rectify_lays_a_diagonal_cow_flat():
    frame = np.full((300, 300, 3), 60, dtype=np.uint8)
    corners = obb_corners(150, 150, 120, 40, math.radians(35))
    cv2.fillPoly(frame, [corners.astype(np.int32)], (255, 255, 255))
    crop = rectify(frame, corners)
    assert crop.shape[:2] == (40, 120)
    assert crop[4:-4, 4:-4].mean() > 240


def test_rectify_puts_the_long_side_first_whatever_the_corner_order():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    corners = np.roll(obb_corners(100, 100, 90, 30, 0.2), 1, axis=0)
    assert rectify(frame, corners).shape[:2] == (30, 90)


def test_prepare_dataset_writes_normalised_obb_labels(tmp_path):
    src = tmp_path / "src"
    for rel in ("Train/images/train", "Train/images/val", "Test/images/val"):
        d = src / rel
        d.mkdir(parents=True)
        (d / "00001.xml").write_text(XML, encoding="utf-8")
        cv2.imwrite(str(d / "00001.jpg"), np.zeros((100, 200, 3), dtype=np.uint8))

    counts = prepare_dataset(src, tmp_path / "out", log=lambda m: None)

    assert counts["train"] == {"images": 1, "cows": 1}   # голова не считается коровой
    label = (tmp_path / "out/labels/test/00001.txt").read_text().split()
    assert label[0] == "0"
    assert np.allclose([float(v) for v in label[1:]],
                       [0.3, 0.3, 0.7, 0.3, 0.7, 0.7, 0.3, 0.7])
    assert (tmp_path / "out/images/val/00001.jpg").exists()
