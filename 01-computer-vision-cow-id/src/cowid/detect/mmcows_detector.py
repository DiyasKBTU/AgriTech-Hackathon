"""Детектор коров на другой ферме (MmCows): до и после дообучения.

Вопрос, который задают на любой ферме: «а на наших коровах и нашей камере
заработает?». Здесь это проверяется на чужой ферме с другим ракурсом:

1. Детектор «вид сверху» (Cows2021) и готовая модель COCO — без дообучения.
2. Дообучение на кадрах этой фермы за 00:00–12:00 25 июля, два класса:
   корова стоит / корова лежит.
3. Проверка на кадрах 14:00–24:00 того же дня, которых модель не видела.
4. Лежание: сколько часов камера насчитала бы каждой корове против разметки
   человека — при условии, что номер коровы известен.

Данные: `visual_data` — 4 камеры, кадр раз в 15 с, 4480x2800, рамки коров
с номерами, раздельно стоящие и лежащие.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .train_detector import MATCH_IOU, polygon_iou

SRC = Path("data/real/mmcows_visual/visual_data")
PREPARED = Path("data/prepared/mmcows_det")
CAMS = ("cam_1", "cam_2", "cam_3", "cam_4")
WIDTH = 1600
#: Кадр раз в минуту: соседние кадры через 15 с почти одинаковы.
EVERY = 4

Log = Callable[[str], None]


def _hour(name: str) -> float:
    hh, mm, ss = name.split("_")[1].split(".")[0].split("-")
    return int(hh) + int(mm) / 60 + int(ss) / 3600


def split_of(name: str) -> str:
    h = _hour(name)
    if h < 12:
        return "train"
    if h < 14:
        return "val"
    return "test"


def read_labels(cam: str, stem: str, src: Path = SRC) -> list[tuple[int, int, np.ndarray]]:
    """[(номер коровы, класс, рамка xyxy в долях кадра)]."""
    out = []
    for cls, sub in ((0, "standing"), (1, "lying")):
        path = src / "labels" / sub / "0725" / cam / f"{stem}.txt"
        if not path.exists():
            continue
        for row in path.read_text().splitlines():
            parts = row.split()
            if len(parts) != 5:
                continue
            cow = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:])
            out.append((cow, cls, np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])))
    return out


def _shrink(args: tuple[str, str]) -> None:
    src, dst = args
    if os.path.exists(dst):
        return
    img = cv2.imread(src)
    if img is None:
        return
    h, w = img.shape[:2]
    img = cv2.resize(img, (WIDTH, int(h * WIDTH / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def prepare(src: Path = SRC, out: Path = PREPARED, log: Log = print) -> dict:
    jobs = []
    counts = defaultdict(lambda: [0, 0, 0])
    for cam in CAMS:
        names = sorted((src / "images" / "0725" / cam).glob("*.jpg"))
        for i, img in enumerate(names):
            if i % EVERY:
                continue
            split = split_of(img.name)
            stem = img.stem
            labels = read_labels(cam, stem, src)
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            name = f"{cam}_{stem}"
            lines = []
            for cow, cls, (x1, y1, x2, y2) in labels:
                lines.append(f"{cls} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} "
                             f"{x2 - x1:.6f} {y2 - y1:.6f}")
                counts[split][cls] += 1
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n")
            jobs.append((str(img), str(out / "images" / split / f"{name}.jpg")))
            counts[split][2] += 1
    log(f"кадров к уменьшению: {len(jobs)}")
    with ProcessPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) - 2)) as pool:
        for n, _ in enumerate(pool.map(_shrink, jobs, chunksize=16), 1):
            if n % 1000 == 0:
                log(f"  уменьшено {n}/{len(jobs)}")
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        "test: images/test\nnames:\n  0: standing\n  1: lying\n", encoding="utf-8")
    summary = {s: {"images": c[2], "standing": c[0], "lying": c[1]} for s, c in counts.items()}
    log(json.dumps(summary, ensure_ascii=False))
    return summary


def _box(xyxy) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _match(pred: list[np.ndarray], truth: list[np.ndarray]) -> dict[int, int]:
    pairs = sorted(((polygon_iou(p, t), i, j) for i, p in enumerate(pred)
                    for j, t in enumerate(truth)), reverse=True)
    used_p, used_t, out = set(), set(), {}
    for iou, i, j in pairs:
        if iou < MATCH_IOU:
            break
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        out[j] = i
    return out


def evaluate(model_path: str, classes, imgsz: int, conf: float, device: str,
             out: Path = PREPARED, name: str = "", is_obb: bool = False,
             trained: bool = False, log: Log = print) -> dict:
    """Полнота и точность на тестовых кадрах, отдельно по стоящим и лежащим.

    Для обученной модели дополнительно — верно ли она назвала «стоит/лежит»
    у найденных коров, и сколько часов лежания насчитала бы каждой корове.
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    stats = {0: [0, 0], 1: [0, 0]}      # класс -> [найдено, всего]
    extra = total_pred = 0
    class_right = class_total = 0
    lying_true = defaultdict(float)
    lying_pred = defaultdict(float)
    seen = defaultdict(float)
    images = sorted((out / "images" / "test").glob("*.jpg"))
    for n, img_path in enumerate(images, 1):
        cam = "_".join(img_path.stem.split("_")[:2])
        stem = "_".join(img_path.stem.split("_")[2:])
        labels = read_labels(cam, stem)
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        res = model.predict(img, imgsz=imgsz, conf=conf, classes=classes, device=device,
                            verbose=False)[0]
        if is_obb:
            boxes = res.obb.xyxy.cpu().numpy() if res.obb is not None else np.zeros((0, 4))
            pcls = np.zeros(len(boxes), dtype=int)
        else:
            boxes = res.boxes.xyxy.cpu().numpy()
            pcls = res.boxes.cls.cpu().numpy().astype(int)
        pred = [_box(b) for b in boxes]
        truth = [_box(b * [w, h, w, h]) for _, _, b in labels]
        matched = _match(pred, truth)
        total_pred += len(pred)
        extra += len(pred) - len(matched)
        for j, (cow, cls, _) in enumerate(labels):
            stats[cls][1] += 1
            if j in matched:
                stats[cls][0] += 1
                if trained:
                    ok = int(pcls[matched[j]]) == cls
                    class_right += ok
                    class_total += 1
            # Лежание по корове: минута кадра засчитывается, если корову нашли.
            if trained and j in matched:
                seen[cow] += 1
                lying_true[cow] += cls == 1
                lying_pred[cow] += int(pcls[matched[j]]) == 1
        if n % 500 == 0:
            log(f"  {name}: {n}/{len(images)}")
    found = stats[0][0] + stats[1][0]
    report = {
        "model": name,
        "images": len(images),
        "standing_recall": stats[0][0] / max(stats[0][1], 1),
        "lying_recall": stats[1][0] / max(stats[1][1], 1),
        "recall": found / max(stats[0][1] + stats[1][1], 1),
        "precision": found / max(total_pred, 1),
        "cows_standing": stats[0][1],
        "cows_lying": stats[1][1],
        "extra": extra,
    }
    if trained:
        report["class_accuracy"] = class_right / max(class_total, 1)
        rows = []
        for cow in sorted(seen):
            k = seen[cow]
            rows.append({"cow": f"C{cow:02d}", "frames": int(k),
                         "lying_true_share": lying_true[cow] / k,
                         "lying_pred_share": lying_pred[cow] / k})
        report["lying_per_cow"] = rows
        diffs = [abs(r["lying_true_share"] - r["lying_pred_share"]) * 24 for r in rows]
        report["lying_mean_abs_diff_h_per_day"] = float(np.mean(diffs)) if diffs else None
    return report


def train(base: str, epochs: int, imgsz: int, batch: int, device: str,
          out: Path = PREPARED, log: Log = print) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(Path("models/pretrained") / base))
    runs = Path("reports/detector_runs").resolve()
    model.train(data=str(out / "data.yaml"), epochs=epochs, imgsz=imgsz, batch=batch,
                workers=4, device=device, project=str(runs), name="mmcows_det",
                exist_ok=True, seed=42, plots=True, verbose=False)
    target = Path("models/detector_mmcows/cow_lying.pt")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(runs / "mmcows_det" / "weights" / "best.pt", target)
    log(f"Веса: {target}")
    return target


def summary_text(r: dict) -> str:
    lines = ["Другая ферма (MmCows), кадры 14:00–24:00 25.07, "
             f"{r['before'][0]['images']} кадров, коров стоя {r['before'][0]['cows_standing']}, "
             f"лёжа {r['before'][0]['cows_lying']}."]
    for x in r["before"] + [r["after"]]:
        lines.append(f"{x['model']}: найдено стоящих {x['standing_recall']:.0%}, лежащих "
                     f"{x['lying_recall']:.0%}, точность {x['precision']:.0%}")
    a = r["after"]
    lines.append(f"Обученная модель верно назвала «стоит/лежит» у {a['class_accuracy']:.0%} "
                 f"найденных коров; расхождение доли лежания с разметкой — в среднем "
                 f"{a['lying_mean_abs_diff_h_per_day']:.1f} ч в пересчёте на сутки.")
    return "\n".join(lines)
