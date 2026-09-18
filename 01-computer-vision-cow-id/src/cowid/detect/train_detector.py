"""Обучение детектора коров «вид сверху» на реальной разметке Cows2021.

Зачем обучать, если в COCO уже есть класс «корова». COCO снимали сбоку:
корова на лугу, на дороге. Камера в коровнике смотрит сверху, и сверху корова
выглядит как пятнистый овал без ног и морды. Насколько готовая модель с этим
справляется, считает `evaluate_coco_baseline` на тех же проверочных кадрах.

Почему повёрнутая рамка (OBB), а не обычный прямоугольник. Корова сверху
идёт под любым углом. Обычный прямоугольник вокруг диагональной коровы
наполовину состоит из пола и соседей. Повёрнутая рамка облегает туловище,
и по ней корова вырезается и разворачивается горизонтально — ровно так,
как нарезаны снимки, на которых училась модель узнавания.

Данные. В Cows2021 10 402 кадра 1280x720 с камеры над проходом, у каждой
коровы размечена повёрнутая рамка туловища. Авторы разделили кадры по времени:

    train  5–29 февраля 2020   7 248 кадров
    val    29 фев – 4 марта    1 023 кадра   подбор эпохи
    test   4–11 марта          2 131 кадр    итоговая проверка, модель их не видела

Формат угла проверен на кадрах: рамка из разметки поворачивается на +angle
в координатах изображения (ось y вниз). Обратный знак промахивается мимо
туловища — это было видно и на глаз, и по яркости внутри рамки
(378 коров из 429 за этот вариант).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

COWS2021_DETECTION = Path(
    "data/real/cows2021/4vnrca7qw1642qlwxjadp87h7/Sub-levels/Detection_and_localisation"
)
SPLITS = {
    "train": "Train/images/train",
    "val": "Train/images/val",
    "test": "Test/images/val",
}
PREPARED = Path("data/prepared/cows2021_obb")
#: Порог совпадения найденной рамки с размеченной — общепринятый.
MATCH_IOU = 0.5


# --------------------------------------------------------------------------
# Разметка
# --------------------------------------------------------------------------

def obb_corners(cx: float, cy: float, w: float, h: float, angle: float) -> np.ndarray:
    """Четыре угла повёрнутой рамки, по часовой стрелке от левого верхнего.
    Первое ребро (углы 0→1) идёт вдоль ширины — у коровы это длина туловища."""
    c, s = math.cos(angle), math.sin(angle)
    pts = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        pts.append((cx + c * dx - s * dy, cy + s * dx + c * dy))
    return np.asarray(pts, dtype=np.float32)


def read_cows(xml_path: Path) -> list[np.ndarray]:
    """Рамки коров из файла разметки (формат roLabelImg)."""
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        rb = obj.find("robndbox")
        if obj.findtext("name") != "cow" or rb is None:
            continue
        values = [float(rb.findtext(k)) for k in ("cx", "cy", "w", "h", "angle")]
        boxes.append(obb_corners(*values))
    return boxes


def prepare_dataset(src: Path = COWS2021_DETECTION, out: Path = PREPARED,
                    log: Callable[[str], None] = print) -> dict:
    """Переводит разметку в формат YOLO-OBB.

    Кадры не копируются, а связываются жёсткими ссылками: на диске они лежат
    один раз, место не тратится. Если ссылку создать нельзя (другой диск),
    кадр копируется.
    """
    counts = {}
    for split, rel in SPLITS.items():
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        n_img = n_cows = 0
        for xml_path in sorted((src / rel).glob("*.xml")):
            jpg = xml_path.with_suffix(".jpg")
            if not jpg.exists():
                continue
            root = ET.parse(xml_path).getroot()
            width = float(root.findtext("size/width"))
            height = float(root.findtext("size/height"))
            lines = []
            for corners in read_cows(xml_path):
                norm = corners / np.array([width, height], dtype=np.float32)
                norm = np.clip(norm, 0.0, 1.0)
                lines.append("0 " + " ".join(f"{v:.6f}" for v in norm.reshape(-1)))
            (lbl_dir / f"{jpg.stem}.txt").write_text("\n".join(lines) + "\n")
            target = img_dir / jpg.name
            if not target.exists():
                try:
                    os.link(jpg, target)
                except OSError:
                    shutil.copy2(jpg, target)
            n_img += 1
            n_cows += len(lines)
        counts[split] = {"images": n_img, "cows": n_cows}
        log(f"{split:5s}: кадров {n_img}, коров {n_cows}")

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: cow\n",
        encoding="utf-8",
    )
    counts["data_yaml"] = str(data_yaml)
    return counts


# --------------------------------------------------------------------------
# Обучение
# --------------------------------------------------------------------------

def train_detector(data_yaml: Path, out_dir: Path, base_model: str, epochs: int,
                   imgsz: int, batch: int, workers: int, device: str,
                   log: Callable[[str], None] = print) -> Path:
    """Дообучает предобученную OBB-модель на коровах. Возвращает путь к весам.

    Базовая модель предобучена на DOTA — аэрофотоснимках, где объекты тоже
    видны сверху и повёрнуты. Это ближе к нашей камере, чем COCO.
    """
    from ultralytics import YOLO

    pretrained = Path("models/pretrained") / base_model
    pretrained.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(pretrained))
    runs = Path("reports/detector_runs").resolve()
    model.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch,
        workers=workers, device=device, project=str(runs), name="cows2021_obb",
        exist_ok=True, seed=42, plots=True, verbose=False,
    )
    best = runs / "cows2021_obb" / "weights" / "best.pt"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "cow_obb.pt"
    shutil.copy2(best, target)
    log(f"Веса: {target}")
    return target


# --------------------------------------------------------------------------
# Проверка
# --------------------------------------------------------------------------

def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return float(inter / union) if union > 0 else 0.0


def axis_box(corners: np.ndarray) -> np.ndarray:
    x1, y1 = corners.min(axis=0)
    x2, y2 = corners.max(axis=0)
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def match_counts(pred: list[np.ndarray], truth: list[np.ndarray]) -> tuple[int, int, int]:
    """Жадное сопоставление по IoU. Возвращает (найдено, пропущено, лишних)."""
    pairs = sorted(
        ((polygon_iou(p, t), i, j) for i, p in enumerate(pred) for j, t in enumerate(truth)),
        reverse=True,
    )
    used_p: set[int] = set()
    used_t: set[int] = set()
    for iou, i, j in pairs:
        if iou < MATCH_IOU:
            break
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
    found = len(used_t)
    return found, len(truth) - found, len(pred) - len(used_p)


def _summary(found: int, missed: int, extra: int, n_images: int) -> dict:
    total = found + missed
    return {
        "images": n_images,
        "cows": total,
        "found": found,
        "missed": missed,
        "extra": extra,
        "recall": found / total if total else 0.0,
        "precision": found / (found + extra) if found + extra else 0.0,
    }


def _test_items(src: Path) -> list[tuple[Path, list[np.ndarray]]]:
    return [(x.with_suffix(".jpg"), read_cows(x))
            for x in sorted((src / SPLITS["test"]).glob("*.xml"))]


def evaluate_detector(weights: Path, data_yaml: Path, conf: float, device: str,
                      src: Path = COWS2021_DETECTION) -> dict:
    """Проверка на кадрах 4–11 марта, которых модель не видела.

    Две оценки. Стандартная (mAP) — для сравнения со статьями. Простая —
    «из N коров на кадрах найдено столько-то, лишних рамок столько-то» при том
    пороге уверенности, с которым детектор работает в конвейере.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), split="test", device=device,
                        plots=False, verbose=False,
                        project=str(Path("reports/detector_runs").resolve()),
                        name="test_eval", exist_ok=True)
    report: dict = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }

    obb = [0, 0, 0]
    axis = [0, 0, 0]
    items = _test_items(src)
    for jpg, truth in items:
        res = model.predict(str(jpg), conf=conf, device=device, verbose=False)[0]
        pred = [p for p in res.obb.xyxyxyxy.cpu().numpy()] if res.obb is not None else []
        for acc, (p, t) in ((obb, (pred, truth)),
                            (axis, ([axis_box(x) for x in pred], [axis_box(x) for x in truth]))):
            f, m, e = match_counts(p, t)
            acc[0] += f
            acc[1] += m
            acc[2] += e
    report["oriented"] = _summary(*obb, len(items))
    report["axis_aligned"] = _summary(*axis, len(items))
    report["conf"] = conf
    return report


def evaluate_coco_baseline(model_name: str, conf: float, device: str,
                           src: Path = COWS2021_DETECTION) -> dict:
    """Готовая модель COCO (класс 19 «cow») на тех же проверочных кадрах.

    COCO даёт только обычные прямоугольники, поэтому сравнение идёт
    по прямоугольникам, описанным вокруг размеченных туловищ.
    Порог совпадения тот же — IoU 0.5.
    """
    from ultralytics import YOLO

    pretrained = Path("models/pretrained") / model_name
    pretrained.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(pretrained))
    acc = [0, 0, 0]
    loose = [0, 0, 0]
    items = _test_items(src)
    for jpg, truth in items:
        res = model.predict(str(jpg), conf=conf, classes=[19], device=device, verbose=False)[0]
        pred = []
        for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy():
            pred.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32))
        t_axis = [axis_box(t) for t in truth]
        f, m, e = match_counts(pred, t_axis)
        acc[0] += f
        acc[1] += m
        acc[2] += e
        # Мягкий счёт: COCO-рамка захватывает голову и ноги, которых нет
        # в разметке туловища, поэтому IoU у неё ниже при верном попадании.
        # Здесь достаточно, чтобы центр размеченной коровы лежал внутри рамки.
        hit_t: set[int] = set()
        hit_p: set[int] = set()
        for i, p in enumerate(pred):
            for j, t in enumerate(truth):
                if j in hit_t:
                    continue
                cx, cy = t.mean(axis=0)
                if p[0, 0] <= cx <= p[1, 0] and p[0, 1] <= cy <= p[2, 1]:
                    hit_t.add(j)
                    hit_p.add(i)
                    break
        loose[0] += len(hit_t)
        loose[1] += len(truth) - len(hit_t)
        loose[2] += len(pred) - len(hit_p)
    return {
        "model": model_name,
        "conf": conf,
        "iou50": _summary(*acc, len(items)),
        "center_inside": _summary(*loose, len(items)),
    }


def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

