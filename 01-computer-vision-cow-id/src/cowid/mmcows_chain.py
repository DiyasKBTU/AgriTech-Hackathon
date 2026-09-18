"""Вся цепочка ТЗ на одной реальной ферме с ответами (MmCows, 25.07.2023).

    камера → найти корову → узнать, какая это → вести дорожку →
    активность конкретной коровы → сравнить с разметкой людей

Почему MmCows. У 16 коров этой фермы номер в разметке сохраняется весь день
на всех четырёх камерах (кадр раз в 15 секунд), а поведение каждой коровы
размечено людьми посекундно. Это единственные найденные открытые данные, где
можно честно посчитать метрики всех звеньев сразу: узнавание, tracking
(IDF1, подмены номера) и активность.

Деление по времени того же дня — как у детектора (`detect/mmcows_detector.py`):

* 03:00–12:00 — обучение: здесь «метка» (номер из разметки) учит биометрию,
  как на ферме её учила бы бирка;
* 12:00–14:00 — выбор эпохи и настроек трекера;
* 14:00–24:00 — проверка, один раз. Сюда входят вечер и ночь.

Честные оговорки (они же в отчёте):

* это те же 16 коров в тот же день: проверяется «узнать корову через
  несколько часов при другом свете и в другой позе», а не «через месяц»;
* кадр раз в 15 секунд — для трекинга тяжелее, чем обычная камера
  (25 кадров в секунду), поэтому цифры tracking скорее занижены;
* лежащая корова часто занимает одно и то же место — модель могла бы узнавать
  место, а не корову. Контроль: закрытый центр снимка.

Запуск: `cowid farm-chain` (шаги можно запускать по отдельности).
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .detect.mmcows_detector import CAMS, SRC, read_labels, split_of

DET_IMAGES = Path("data/prepared/mmcows_det/images")   # 1600 px, кадр раз в минуту
CROPS = Path("data/prepared/mmcows_reid")
MODEL_DIR = Path("models/reid_mmcows")
BASE_ENCODER = Path("models/reid/encoder.pt")          # узнавание по спине, Cows2021
CACHE = Path("var/mmcows_chain")
GALLERY = Path("var/gallery_mmcows.json")
REPORT = Path("reports/mmcows/chain.json")
WIDTH, HEIGHT = 1600, 1000
PAD = 0.05
MIN_SIDE = 32
SLOT_S = 15
#: Часы проверки по освещённости (по часам в именах файлов датасета).
#: Ночью в коровнике горят лампы — яркость кадра почти как днём, меняются
#: цвет и тени.
BUCKETS = (("день, 14–18 ч", 14, 18), ("вечер, 18–21 ч", 18, 21), ("ночь при лампах, 21–24 ч", 21, 24))
#: Корова «перекрыта», если её рамка перекрывается с соседней хотя бы на столько.
OCCLUDED_IOU = 0.10
MIN_VOTES = 3

Log = Callable[[str], None]


def cow_name(n: int) -> str:
    return f"C{n:02d}"


def _hour(stem: str) -> float:
    hh, mm, ss = stem.split("_")[1].split("-")
    return int(hh) + int(mm) / 60 + int(ss) / 3600


def _bucket(hour: float) -> str:
    return next((name for name, a, b in BUCKETS if a <= hour < b), "другое")


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _occluded(boxes: list[np.ndarray]) -> list[bool]:
    return [any(_iou(b, o) >= OCCLUDED_IOU for j, o in enumerate(boxes) if j != i)
            for i, b in enumerate(boxes)]


def crop(frame: np.ndarray, box: np.ndarray, pad: float = PAD) -> Optional[np.ndarray]:
    """Прямоугольная вырезка с полями — одинаково при обучении и в работе."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    x2, y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
    if min(x2 - x1, y2 - y1) < MIN_SIDE:
        return None
    return frame[y1:y2, x1:x2]


def gt(cam: str, stem: str) -> tuple[list[int], list[int], list[np.ndarray]]:
    """Номера, позы (1 — лежит) и рамки из разметки, в пикселях кадра 1600×1000."""
    rows = read_labels(cam, stem)
    scale = np.array([WIDTH, HEIGHT, WIDTH, HEIGHT], dtype=np.float32)
    return ([r[0] for r in rows], [r[1] for r in rows],
            [r[2].astype(np.float32) * scale for r in rows])


# --------------------------------------------------------------------------
# 1. Вырезки коров для узнавания
# --------------------------------------------------------------------------

def _crop_image(path: Path, split: str, out: Path) -> Counter:
    # cam_1_1690271846_02-57-26.jpg
    parts = path.stem.split("_")
    cam, stem = "_".join(parts[:2]), "_".join(parts[2:])
    ids, poses, boxes = gt(cam, stem)
    counts: Counter = Counter()
    if not ids:
        return counts
    frame = cv2.imread(str(path))
    if frame is None:
        return counts
    for cow, pose, box, occ in zip(ids, poses, boxes, _occluded(boxes)):
        piece = crop(frame, box)
        if piece is None:
            counts["мелкие"] += 1
            continue
        tag = ("L" if pose else "S") + ("o" if occ else "")
        dst = out / split / cow_name(cow) / f"{cam}_{stem}_{tag}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), piece, [cv2.IMWRITE_JPEG_QUALITY, 92])
        counts[split] += 1
    return counts


def prepare_crops(out: Path = CROPS, log: Log = print) -> dict:
    """Вырезки по разметке из кадров раз в минуту (те же, что у детектора)."""
    jobs = [(p, split) for split in ("train", "val", "test")
            for p in sorted((DET_IMAGES / split).glob("*.jpg"))]
    if not jobs:
        raise FileNotFoundError(f"Нет кадров {DET_IMAGES}: сначала `cowid mmcows-detector`")
    total: Counter = Counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for n, c in enumerate(pool.map(lambda j: _crop_image(j[0], j[1], out), jobs), 1):
            total.update(c)
            if n % 2000 == 0:
                log(f"  кадров {n}/{len(jobs)}")
    log(f"вырезок: {dict(total)}")
    return dict(total)


@dataclass
class CropInfo:
    path: Path
    cow: str
    cam: str
    hour: float
    lying: bool
    occluded: bool


def list_crops(split: str, root: Path = CROPS) -> list[CropInfo]:
    items = []
    for p in sorted((root / split).glob("C*/*.jpg")):
        parts = p.stem.split("_")          # cam, N, unix, HH-MM-SS, tag
        tag = parts[-1]
        items.append(CropInfo(p, p.parent.name, "_".join(parts[:2]),
                              _hour("_".join(parts[2:4])), tag.startswith("L"), "o" in tag))
    return items


def _thin(items: list, per_cow: int, seed: int = 7) -> list:
    """Равномерно по времени не больше `per_cow` вырезок на корову."""
    by_cow = defaultdict(list)
    for it in items:
        by_cow[it.cow].append(it)
    out = []
    for cow, rows in sorted(by_cow.items()):
        rows.sort(key=lambda r: (r.hour, r.cam))
        if len(rows) > per_cow:
            step = len(rows) / per_cow
            rows = [rows[int(i * step)] for i in range(per_cow)]
        out.extend(rows)
    random.Random(seed).shuffle(out)
    return out


# --------------------------------------------------------------------------
# 2. Дообучение узнавания на 16 коровах фермы
# --------------------------------------------------------------------------

def _splits(train_per_cow: int = 1200, gallery_per_cow: int = 150, val_per_cow: int = 300):
    from .identity.reid_dataset import ReidSplit, Sample

    def samples(items):
        return [Sample(path=i.path, identity=i.cow) for i in items]

    train = list_crops("train")
    gallery = _thin(train, gallery_per_cow)
    fit_split = ReidSplit(train=samples(_thin(train, train_per_cow)),
                          gallery=samples(gallery), query=samples(list_crops("test")))
    select = ReidSplit(gallery=samples(gallery),
                       query=samples(_thin(list_crops("val"), val_per_cow)))
    return fit_split, select


def train_reid(epochs: int = 8, log: Log = print) -> dict:
    """Дообучение кодировщика Cows2021 на утренних вырезках этой фермы."""
    from .identity.train import TrainConfig, fit

    split, select = _splits()
    log(f"обучение: {len(split.train)} вырезок, выбор эпохи: {len(select.query)}, "
        f"проверка: {len(split.query)} запросов против {len(split.gallery)} в галерее")
    cfg = TrainConfig(data_root=str(CROPS), out_dir=str(MODEL_DIR), epochs=epochs,
                      lr=2e-4, batch_size=48, num_workers=4, split_by_time=False,
                      max_images_per_identity=None)
    init = str(BASE_ENCODER) if BASE_ENCODER.exists() else None
    report = fit(cfg, split, log=log, init_weights=init, select=select)
    report["note"] = ("Закрытое множество: те же 16 коров. Обучение 03–12 ч, выбор эпохи 12–14 ч, "
                      "проверка 14–24 ч. Галерея — вырезки 03–12 ч.")
    (MODEL_DIR / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _embed_paths(encoder, paths: list[Path], device: str, mask: float = 0.0,
                 batch: int = 128) -> np.ndarray:
    import torch
    from PIL import Image

    from .identity.train import _build_transforms

    tf = _build_transforms(224, train=False, mask_center=mask)
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            imgs = []
            for p in paths[i:i + batch]:
                with Image.open(p) as im:
                    imgs.append(tf(im.convert("RGB")))
            out.append(encoder(torch.stack(imgs).to(device)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def _backbone_only(encoder):
    import torch

    class Plain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = encoder.backbone

        def forward(self, x):
            return torch.nn.functional.normalize(self.net(x), dim=1)

    return Plain().eval()


def _rank1_by(q_emb, q_items, g_emb, g_ids) -> dict:
    """Rank-1 в разрезе: поза, перекрытие, время суток, камера."""
    best = np.asarray(g_ids)[np.argmax(q_emb @ g_emb.T, axis=1)]
    ok = best == np.asarray([i.cow for i in q_items])
    groups = {
        "лежит": [i.lying for i in q_items],
        "стоит": [not i.lying for i in q_items],
        "перекрыта соседней": [i.occluded for i in q_items],
        "не перекрыта": [not i.occluded for i in q_items],
        **{name: [a <= i.hour < b for i in q_items] for name, a, b in BUCKETS},
    }
    out = {"все": {"rank1": round(float(ok.mean()), 4), "n": int(len(ok))}}
    for name, mask in groups.items():
        m = np.asarray(mask)
        if m.any():
            out[name] = {"rank1": round(float(ok[m].mean()), 4), "n": int(m.sum())}
    return out


def reid_checks(log: Log = print) -> dict:
    """Узнавание на проверке 14–24 ч: без дообучения, после, и контроли."""
    import torch

    from .identity.train import _build_model, evaluate_reid, load_encoder, TrainConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = list_crops("train")
    gallery = _thin(train, 150)
    query = list_crops("test")
    g_ids = [i.cow for i in gallery]
    q_ids = [i.cow for i in query]

    def run(name: str, encoder, mask: float = 0.0, breakdown: bool = False) -> dict:
        g = _embed_paths(encoder, [i.path for i in gallery], device, mask)
        q = _embed_paths(encoder, [i.path for i in query], device, mask)
        r = evaluate_reid(q, q_ids, g, g_ids)
        if breakdown:
            r["разрез"] = _rank1_by(q, query, g, g_ids)
        log(f"  {name}: Rank-1 {r['rank1']:.3f}, mAP {r['mAP']:.3f}")
        return r

    report = {"gallery": len(gallery), "query": len(query), "cows": len(set(q_ids))}
    tuned, _ = load_encoder(MODEL_DIR / "encoder.pt", device)
    report["дообученная"] = run("дообученная", tuned, breakdown=True)
    report["дообученная, центр закрыт"] = run("дообученная, центр закрыт", tuned, mask=0.6)
    if BASE_ENCODER.exists():
        base, _ = load_encoder(BASE_ENCODER, device)
        report["Cows2021 без дообучения"] = run("Cows2021 без дообучения", base)
    plain, _ = _build_model(TrainConfig(data_root=""), n_classes=1)
    report["необученная сеть (ImageNet)"] = run("необученная сеть", _backbone_only(plain).to(device))
    return report


def unknown_threshold() -> float:
    """Порог «не знаю» — по части выбора эпохи (12–14 ч): чужая корова
    принимается за свою не чаще 1 раза из 100."""
    report = json.loads((MODEL_DIR / "training_report.json").read_text(encoding="utf-8"))
    return float(report["best"].get("unknown_distance_at_far1", 0.3))


def enroll_gallery(threshold: float, per_cow: int = 300, log: Log = print) -> Path:
    """Галерея 16 коров из утренних вырезок — для проверки и живого показа."""
    import torch

    from .config import GalleryConfig
    from .identity.gallery import BiometricGallery
    from .identity.train import load_encoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, _ = load_encoder(MODEL_DIR / "encoder.pt", device)
    items = _thin(list_crops("train"), per_cow)
    emb = _embed_paths(encoder, [i.path for i in items], device)
    gallery = BiometricGallery(GalleryConfig(unknown_distance=threshold, max_per_cow=per_cow))
    by_cow = defaultdict(list)
    for it, e in zip(items, emb):
        by_cow[it.cow].append(e)
    for cow, vectors in by_cow.items():
        gallery.enroll(cow, vectors, source="tag")
    gallery.save(GALLERY)
    log(f"галерея: {gallery.size()} коров, {gallery.total_embeddings()} портретов, "
        f"порог «не знаю» {threshold:.3f} → {GALLERY}")
    return GALLERY


# --------------------------------------------------------------------------
# 3. Кадры проверки: детектор + отпечатки внешности (кэш)
# --------------------------------------------------------------------------

def _frames(cam: str, windows: tuple[str, ...]) -> list[Path]:
    return [p for p in sorted((SRC / "images" / "0725" / cam).glob("*.jpg"))
            if split_of(p.name) in windows]


def _load(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_2)
    if img is None:
        return None
    return cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)


def _prefetch(paths: list[Path], pool: ThreadPoolExecutor, ahead: int = 24):
    """Кадры по порядку, но не больше `ahead` декодированных наперёд:
    `pool.map` декодировал бы всю камеру сразу — это 14 ГБ памяти."""
    from collections import deque

    queue: deque = deque()
    for path in paths:
        queue.append((path, pool.submit(_load, path)))
        if len(queue) >= ahead:
            p, fut = queue.popleft()
            yield p, fut.result()
    while queue:
        p, fut = queue.popleft()
        yield p, fut.result()


def run_frames(windows: tuple[str, ...] = ("val", "test"), log: Log = print) -> None:
    """Каждый кадр раз в 15 секунд: рамки, поза, отпечаток — в кэш по камерам."""
    import torch
    from PIL import Image

    from .config import PipelineConfig
    from .detect.detectors import build_detector
    from .identity.train import _build_transforms, load_encoder

    cfg = PipelineConfig.load(Path("configs/mmcows.yaml"))
    det_cfg = cfg.detector.model_copy(update={"tiles": 1})
    detector = build_detector(det_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, _ = load_encoder(MODEL_DIR / "encoder.pt", device)
    tf = _build_transforms(224, train=False)
    CACHE.mkdir(parents=True, exist_ok=True)

    for cam in CAMS:
        target = CACHE / f"{cam}.npz"
        if target.exists():
            log(f"{cam}: уже посчитано")
            continue
        paths = _frames(cam, windows)
        stems, counts, boxes, scores, lying, embs, bright = [], [], [], [], [], [], []
        with ThreadPoolExecutor(max_workers=6) as pool:
            for n, (path, frame) in enumerate(_prefetch(paths, pool), 1):
                if frame is None:
                    continue
                dets = detector.detect(frame, n)
                pieces, keep = [], []
                for d in dets:
                    b = np.array([d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2], np.float32)
                    piece = crop(frame, b)
                    if piece is None:
                        continue
                    pieces.append(tf(Image.fromarray(cv2.cvtColor(piece, cv2.COLOR_BGR2RGB))))
                    keep.append((b, d.score, d.label == "lying"))
                if pieces:
                    with torch.no_grad():
                        vec = encoder(torch.stack(pieces).to(device)).float().cpu().numpy()
                    embs.append(vec.astype(np.float16))
                stems.append(path.stem)
                counts.append(len(keep))
                for b, s, lie in keep:
                    boxes.append(b)
                    scores.append(s)
                    lying.append(lie)
                bright.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
                if n % 500 == 0:
                    log(f"  {cam}: {n}/{len(paths)}")
        np.savez_compressed(
            target, stems=np.array(stems), counts=np.array(counts, np.int32),
            boxes=np.array(boxes, np.float32).reshape(-1, 4), scores=np.array(scores, np.float32),
            lying=np.array(lying, bool), brightness=np.array(bright, np.float32),
            emb=(np.concatenate(embs) if embs else np.zeros((0, 512), np.float16)))
        log(f"{cam}: {len(stems)} кадров, {len(boxes)} коров → {target}")


@dataclass
class FrameData:
    stem: str
    hour: float
    slot: int
    boxes: np.ndarray
    scores: np.ndarray
    lying: np.ndarray
    emb: np.ndarray
    brightness: float


@lru_cache(maxsize=8)
def load_cache(cam: str, window: str) -> tuple[FrameData, ...]:
    # Каждое обращение data["…"] распаковывает массив из архива заново —
    # поэтому массивы читаются один раз.
    with np.load(CACHE / f"{cam}.npz") as data:
        arr = {k: data[k] for k in data.files}
    emb = arr["emb"].astype(np.float32)
    out, pos = [], 0
    for stem, n, br in zip(arr["stems"], arr["counts"], arr["brightness"]):
        sl = slice(pos, pos + int(n))
        pos += int(n)
        stem = str(stem)
        if split_of(stem + ".jpg") != window:
            continue
        unix = int(stem.split("_")[0])
        out.append(FrameData(stem, _hour(stem), round(unix / SLOT_S), arr["boxes"][sl],
                             arr["scores"][sl], arr["lying"][sl], emb[sl], float(br)))
    return tuple(out)


# --------------------------------------------------------------------------
# 4. Tracking и номер коровы во времени: IDF1, MOTA, подмены номера
# --------------------------------------------------------------------------

def _motmetrics():
    # motmetrics 1.4 ещё вызывает np.asfarray, которую убрали в NumPy 2.
    if not hasattr(np, "asfarray"):
        np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
    import motmetrics as mm

    return mm


class HypIds:
    """Числовые номера для motmetrics (с pandas 3 он принимает только числа).

    Корова C05 → 5, как в разметке. Всё, что система не назвала, получает
    свой уникальный номер: «не знаю» не должно совпасть ни с одной коровой.
    """

    def __init__(self):
        self._unknown: dict[str, int] = {}

    @staticmethod
    def cow(name: str) -> int:
        return int(name[1:])

    def unknown(self, key: str) -> int:
        return self._unknown.setdefault(key, 1_000_000 + len(self._unknown))


def _xywh(boxes) -> np.ndarray:
    b = np.asarray(boxes, np.float64).reshape(-1, 4)
    return np.column_stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]])


def _pairs(gt_boxes: list[np.ndarray], hyp_boxes: list[np.ndarray]) -> list[tuple[int, int]]:
    """Сопоставление разметки и найденного по IoU ≥ 0.5 (венгерский алгоритм)."""
    from scipy.optimize import linear_sum_assignment

    if not gt_boxes or not hyp_boxes:
        return []
    cost = np.array([[1 - _iou(g, h) for h in hyp_boxes] for g in gt_boxes])
    rows, cols = linear_sum_assignment(cost)
    return [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= 0.5]


@dataclass
class Variant:
    name: str
    use_appearance: bool
    use_tracking: bool = True
    #: Вес голоса кадра, где корова лежит: лежащую корову за перилами стойла
    #: модель узнаёт заметно хуже, чем стоящую (разрез на 12–14 ч).
    lying_weight: float = 1.0
    #: Сколько последних голосов дорожки учитывать (0 — все): если трекер
    #: перескочил на соседнюю корову, старые голоса не тянут чужой номер.
    window: int = 0
    #: Настройки трекера поверх configs/mmcows.yaml, пары (имя, значение).
    tracker: tuple = ()


CAREFUL = "трекинг + внешность, осторожный порог «не знаю»"

#: Настройки выбраны на 12–14 ч (перебор в `tune_tracking`): кадр раз в 15 с —
#: корова успевает сместиться, поэтому внешность весит больше, а номер
#: решают последние 8 голосов (2 минуты).
TRACKER = (("max_age", 4), ("match_iou", 0.35), ("appearance_weight", 0.8))
WINDOW = 8

VARIANTS = (
    Variant("без трекинга: номер по одному кадру", True, use_tracking=False),
    Variant("трекинг только по положению", False, window=WINDOW, tracker=TRACKER),
    Variant("трекинг + внешность (наш)", True, window=WINDOW, tracker=TRACKER),
)


def _gallery_matrix(gallery) -> tuple[np.ndarray, np.ndarray]:
    gallery.query(np.zeros(512, np.float32))        # собрать матрицу галереи
    return gallery._matrix, np.asarray(gallery._row_ids)


def _match_all(emb: np.ndarray, matrix: np.ndarray, row_ids: np.ndarray,
               unknown_distance: float) -> list[Optional[str]]:
    """То же, что `BiometricGallery.match`, но для всех рамок кадра сразу:
    ближайший портрет → его корова; дальше порога — «не знаю»."""
    if len(emb) == 0:
        return []
    sims = emb @ matrix.T
    best = sims.argmax(axis=1)
    dist = 1.0 - sims[np.arange(len(emb)), best]
    return [str(row_ids[b]) if d <= unknown_distance else None for b, d in zip(best, dist)]


def _decide(votes) -> Optional[str]:
    """Номер дорожки по взвешенным голосам: не меньше MIN_VOTES полных
    голосов (или все, если дорожка короче) и больше половины веса."""
    total = sum(w for _, w in votes)
    if total <= 0:
        return None
    tally: Counter = Counter()
    for cow, w in votes:
        if cow is not None:
            tally[cow] += w
    if not tally:
        return None
    cow, best = tally.most_common(1)[0]
    if best >= min(MIN_VOTES, total) and best / total >= 0.5:
        return cow
    return None


def _track_camera(frames: list[FrameData], gallery, variant: Variant, cfg_tracker):
    """Прогон одной камеры. Для каждого кадра: [(рамка, номер трека, номер коровы, лежит)]."""
    from .track.tracker import CowTracker
    from .types import BBox, Detection

    from collections import deque

    tracker = CowTracker(cfg_tracker.model_copy(update=dict(variant.tracker)))
    votes: dict[int, deque] = defaultdict(lambda: deque(maxlen=variant.window or None))
    out = []
    matrix, row_ids = _gallery_matrix(gallery)
    for idx, fr in enumerate(frames):
        guess = _match_all(fr.emb, matrix, row_ids, gallery.cfg.unknown_distance)
        if not variant.use_tracking:
            out.append([(fr.boxes[i], None, guess[i], bool(fr.lying[i]))
                        for i in range(len(fr.boxes))])
            continue
        dets = [Detection(bbox=BBox(*map(float, b)), score=float(s), frame_idx=idx,
                          label="lying" if lie else "standing")
                for b, s, lie in zip(fr.boxes, fr.scores, fr.lying)]
        embs = [e for e in fr.emb] if variant.use_appearance else None
        # Какой детекции какой голос: трекер хранит наблюдения, по рамке и находим.
        by_box = {tuple(np.round(b, 1)): i for i, b in enumerate(fr.boxes)}
        rows = []
        for t in tracker.update(dets, idx, embs):
            obs = t.observations[-1]
            if obs.frame_idx != idx:
                continue
            b = obs.bbox
            box = np.array([b.x1, b.y1, b.x2, b.y2], np.float32)
            i = by_box.get(tuple(np.round(box, 1)))
            if i is not None:
                weight = variant.lying_weight if fr.lying[i] else 1.0
                votes[t.track_id].append((guess[i], weight))
            cow = _decide(votes[t.track_id])
            rows.append((box, t.track_id, cow, bool(fr.lying[i]) if i is not None else False))
        out.append(rows)
    return out


MOT_COUNTS = ["num_frames", "idtp", "idfp", "idfn", "num_switches", "num_misses",
              "num_false_positives", "num_objects", "num_predictions", "mostly_tracked",
              "mostly_lost", "num_unique_objects", "num_fragmentations"]


def _overall(df) -> dict:
    """Итог по камерам из сумм: номера на разных камерах не сопоставляются,
    поэтому IDF1 и MOTA складываются из счётчиков так же, как в motmetrics."""
    s = {k: int(df[k].sum()) for k in MOT_COUNTS}
    idtp, idfp, idfn = s["idtp"], s["idfp"], s["idfn"]
    return {
        **s,
        "idf1": round(2 * idtp / max(1, 2 * idtp + idfp + idfn), 4),
        "idp": round(idtp / max(1, idtp + idfp), 4),
        "idr": round(idtp / max(1, idtp + idfn), 4),
        "mota": round(1 - (s["num_misses"] + s["num_false_positives"] + s["num_switches"])
                      / max(1, s["num_objects"]), 4),
        "precision": round(1 - s["num_false_positives"] / max(1, s["num_predictions"]), 4),
        "recall": round(1 - s["num_misses"] / max(1, s["num_objects"]), 4),
    }


def evaluate_tracking(window: str = "test", unknown_distance: Optional[float] = None,
                      variants: tuple[Variant, ...] = VARIANTS,
                      log: Log = print) -> tuple[dict, dict]:
    """Метрики MOT (motmetrics) и точность номера по кадрам, по вариантам.

    Два уровня:
    * дорожки — сохраняет ли трекер одну дорожку на корову (номер трека);
    * номер коровы — правильный ли номер C01…C16 у рамки во времени. Рамка,
      которой система не дала номер («не знаю»), здесь считается пропуском:
      номер не назван — значит, минуты этой коровы не записаны.
    """
    from .config import PipelineConfig
    from .identity.gallery import BiometricGallery

    mm = _motmetrics()
    cfg = PipelineConfig.load(Path("configs/mmcows.yaml"))
    tcfg = cfg.tracker
    gallery = BiometricGallery.load(GALLERY, cfg.gallery.model_copy(update={
        "unknown_distance": unknown_distance if unknown_distance is not None else unknown_threshold(),
        "max_per_cow": 300}))
    zones = feed_zones()
    mh = mm.metrics.create()

    report, per_slot = {}, {}
    for variant in variants:
        accs_track, accs_id = [], []
        idstats: Counter = Counter()
        by_group: dict[str, Counter] = defaultdict(Counter)
        slots: dict[tuple[int, str], list] = defaultdict(list)
        for cam in CAMS:
            frames = load_cache(cam, window)
            rows_all = _track_camera(frames, gallery, variant, tcfg)
            acc_t = mm.MOTAccumulator(auto_id=True) if variant.use_tracking else None
            acc_i = mm.MOTAccumulator(auto_id=True)
            for fr, rows in zip(frames, rows_all):
                ids, poses, gboxes = gt(cam, fr.stem)
                hyp_boxes = [r[0] for r in rows]
                if acc_t is not None:
                    acc_t.update(list(ids), [r[1] for r in rows],
                                 mm.distances.iou_matrix(_xywh(gboxes), _xywh(hyp_boxes), max_iou=0.5))
                named = [r for r in rows if r[2] is not None]
                acc_i.update(list(ids), [HypIds.cow(r[2]) for r in named],
                             mm.distances.iou_matrix(_xywh(gboxes), _xywh([r[0] for r in named]),
                                                     max_iou=0.5))
                occ = _occluded(gboxes)
                for gi, hi in _pairs(gboxes, hyp_boxes):
                    truth, said = cow_name(ids[gi]), rows[hi][2]
                    verdict = "верно" if said == truth else ("не знаю" if said is None else "ошибка")
                    idstats[verdict] += 1
                    for group in (_bucket(fr.hour),
                                  "перекрыта соседней" if occ[gi] else "не перекрыта",
                                  "лежит" if poses[gi] else "стоит"):
                        by_group[group][verdict] += 1
                # Активность — по всему, что система назвала, как было бы на ферме
                # (без подсказки разметки, какие рамки настоящие).
                for box, _, said, lying in named:
                    r, c = _cell(box)
                    slots[(fr.slot, said)].append((lying, bool(zones[cam][r, c])))
            if acc_t is not None:
                accs_track.append(acc_t)
            accs_id.append(acc_i)
        total = sum(idstats.values()) or 1
        summary_i = _overall(mh.compute_many(accs_id, metrics=MOT_COUNTS))
        summary_t = (_overall(mh.compute_many(accs_track, metrics=MOT_COUNTS))
                     if accs_track else None)
        report[variant.name] = {
            "дорожки (номер трека)": summary_t,
            "номер коровы во времени": summary_i,
            "номер в кадре": {k: round(v / total, 4) for k, v in idstats.items()} | {"n": total},
            "разрез": {g: {k: round(v / max(1, sum(c.values())), 4) for k, v in c.items()}
                       | {"n": sum(c.values())} for g, c in sorted(by_group.items())},
        }
        per_slot[variant.name] = slots
        r = report[variant.name]
        log(f"  {variant.name}: IDF1 номера {summary_i['idf1']:.3f}, "
            f"подмен {summary_i['num_switches']}, "
            f"номер в кадре верно {r['номер в кадре'].get('верно', 0):.3f}, "
            f"ошибка {r['номер в кадре'].get('ошибка', 0):.3f}"
            + (f"; дорожки IDF1 {summary_t['idf1']:.3f}, MOTA {summary_t['mota']:.3f}" if summary_t else ""))
    return report, per_slot


# --------------------------------------------------------------------------
# 5. Активность конкретной коровы: часы лёжа и у корма против разметки людей
# --------------------------------------------------------------------------

GRID = 40            # клетка зоны, пикселей кадра 1600×1000
EATING = (3, 4)      # коды разметки «ест» (голова вверх / вниз)
LYING = 7


def _cell(box: np.ndarray) -> tuple[int, int]:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return int(min(HEIGHT - 1, max(0, cy)) // GRID), int(min(WIDTH - 1, max(0, cx)) // GRID)


ZONES_FILE = MODEL_DIR / "zones.npz"
DRINKING = (6,)


@lru_cache(maxsize=2)
def feed_zones(min_rows: int = 5) -> dict[str, np.ndarray]:
    return zone_masks(EATING, min_rows)


@lru_cache(maxsize=4)
def zone_masks(codes: tuple, min_rows: int = 5) -> dict[str, np.ndarray]:
    """Зона кормового стола на каждой камере — по утренним кадрам (03–12 ч).

    На ферме зону один раз обводят при установке камеры. Здесь её «обводит»
    разметка: клетка кадра входит в зону, если коровы в ней большую часть
    времени едят. Проверка — на других часах (14–24 ч).
    """
    behavior = {n: _behavior(cow_name(n)) for n in range(1, 17)}
    zones = {}
    for cam in CAMS:
        eat = np.zeros((HEIGHT // GRID, WIDTH // GRID))
        seen = np.zeros_like(eat)
        for p in sorted((DET_IMAGES / "train").glob(f"{cam}_*.jpg")):
            stem = "_".join(p.stem.split("_")[2:])
            slot = round(int(stem.split("_")[0]) / SLOT_S)
            ids, _, boxes = gt(cam, stem)
            for cow, box in zip(ids, boxes):
                code = behavior[cow].get(slot, 0)
                if code == 0:
                    continue
                r, c = _cell(box)
                seen[r, c] += 1
                eat[r, c] += code in codes
        zones[cam] = (seen >= min_rows) & (eat > 0.5 * np.maximum(seen, 1))
    return zones


def save_zones(log: Log = print) -> Path:
    """Зоны кормового стола и поилки для живого режима (кадр из 4 камер)."""
    feed, water = zone_masks(EATING, 5), zone_masks(DRINKING, 2)
    ZONES_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ZONES_FILE, cams=np.array(CAMS), grid=GRID, width=WIDTH, height=HEIGHT,
                        feeder=np.stack([feed[c] for c in CAMS]),
                        drinker=np.stack([water[c] for c in CAMS]))
    log(f"зоны: у корма {int(sum(m.sum() for m in feed.values()))} клеток, у поилки "
        f"{int(sum(m.sum() for m in water.values()))} клеток → {ZONES_FILE}")
    return ZONES_FILE


def _behavior(cow: str) -> dict[int, int]:
    """Посекундная разметка поведения → поведение на 15-секундный слот."""
    path = SRC / "behavior_labels" / "individual" / f"{cow}_0725.csv"
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 3:
            continue
        if not parts[2].strip():
            continue
        ts, code = int(float(parts[0])), int(float(parts[2]))   # в части файлов «7.0»
        if ts % SLOT_S == 0:              # секунда в середине слота
            out[ts // SLOT_S] = code
    return out


def evaluate_activity(slots: dict[tuple[int, str], list], window: str = "test") -> dict:
    """Часы лёжа и у корма за окно проверки: камера с нашим номером против людей.

    Камера видит корову не всё время, поэтому доля считается от слотов, где
    корова узнана, и переводится в часы окна — ровно так же, как норма коровы
    в `anomaly/baseline.py` считается от времени наблюдения. Если корову в один
    слот видят две камеры, берётся большинство.
    """
    hours = 10.0 if window == "test" else 2.0
    all_slots = sorted({f.slot for cam in CAMS for f in load_cache(cam, window)})
    lo, hi = all_slots[0], all_slots[-1]
    per_cow = []
    for n in range(1, 17):
        cow = cow_name(n)
        labels = {s: c for s, c in _behavior(cow).items() if lo <= s <= hi and c != 0}
        if not labels:
            continue
        seen = {s: v for (s, c), v in slots.items() if c == cow}
        row = {"cow": cow, "слотов с разметкой": len(labels),
               "корова узнана, доля времени": round(len(seen) / len(all_slots), 3)}
        for key, truth_codes, pick in (("лёжа", (LYING,), 0), ("у корма", EATING, 1)):
            truth = sum(1 for c in labels.values() if c in truth_codes) / len(labels)
            row[f"люди, ч {key}"] = round(truth * hours, 2)
            if seen:
                votes = [sum(v[pick] for v in obs) > len(obs) / 2 for obs in seen.values()]
                share = sum(votes) / len(votes)
                row[f"камера, ч {key}"] = round(share * hours, 2)
                row[f"ошибка, ч {key}"] = round((share - truth) * hours, 2)
        per_cow.append(row)

    def stats(key: str) -> dict:
        errs = [abs(r[f"ошибка, ч {key}"]) for r in per_cow if f"ошибка, ч {key}" in r]
        truth = [r[f"люди, ч {key}"] for r in per_cow if f"ошибка, ч {key}" in r]
        cam = [r[f"камера, ч {key}"] for r in per_cow if f"ошибка, ч {key}" in r]
        return {
            "средняя ошибка, ч": round(float(np.mean(errs)), 2) if errs else None,
            "наибольшая ошибка, ч": round(float(np.max(errs)), 2) if errs else None,
            "разброс между коровами у людей, ч": round(float(np.std(truth)), 2) if truth else None,
            "связь камера—люди (r)": (round(float(np.corrcoef(cam, truth)[0, 1]), 3)
                                     if len(cam) > 2 and np.std(cam) > 0 else None),
        }

    return {"окно, ч": hours, "коров": len(per_cow),
            "лёжа": stats("лёжа"), "у корма": stats("у корма"), "коровы": per_cow}


def perfect_id_slots(window: str = "test") -> dict[tuple[int, str], list]:
    """Те же признаки, но номер коровы взят из разметки — предел точности позы
    и зоны без ошибок узнавания. Разница с нашим номером — цена узнавания."""
    zones = feed_zones()
    slots = defaultdict(list)
    for cam in CAMS:
        for fr in load_cache(cam, window):
            ids, _, gboxes = gt(cam, fr.stem)
            for gi, hi in _pairs(gboxes, list(fr.boxes)):
                r, c = _cell(fr.boxes[hi])
                slots[(fr.slot, cow_name(ids[gi]))].append((bool(fr.lying[hi]), bool(zones[cam][r, c])))
    return slots


def brightness_by_bucket(window: str = "test") -> dict:
    vals = defaultdict(list)
    for cam in CAMS:
        for f in load_cache(cam, window):
            vals[_bucket(f.hour)].append(f.brightness)
    return {k: round(float(np.mean(v))) for k, v in vals.items()}


def summary_text(r: dict) -> str:
    reid, trk, act = r.get("узнавание", {}), r.get("tracking", {}), r.get("активность", {})
    lines = ["Вся цепочка на ферме MmCows: 16 коров, 4 камеры, кадр раз в 15 с. "
             "Обучение 03–12 ч, проверка 14–24 ч (вечер и ночь при лампах входят)."]
    if reid:
        lines.append("Узнавание по одной вырезке, Rank-1: " + ", ".join(
            f"{k} {v['rank1'] * 100:.1f}%" for k, v in reid.items() if isinstance(v, dict) and "rank1" in v))
    for name, v in trk.items():
        t, i, f = v["дорожки (номер трека)"], v["номер коровы во времени"], v["номер в кадре"]
        lines.append(
            f"{name}: номер в кадре верно {f.get('верно', 0) * 100:.1f}%, «не знаю» "
            f"{f.get('не знаю', 0) * 100:.1f}%, ошибка {f.get('ошибка', 0) * 100:.1f}%; "
            f"IDF1 номера {i['idf1'] * 100:.1f}%, подмен номера {i['num_switches']}"
            + (f"; дорожки: IDF1 {t['idf1'] * 100:.1f}%, MOTA {t['mota'] * 100:.1f}%, "
               f"разрывов {t['num_switches']}" if t else ""))
    if act:
        for key in ("лёжа", "у корма"):
            s = act[key]
            lines.append(f"Часы {key} по камере с нашим номером за {act['окно, ч']:.0f} ч: "
                         f"средняя ошибка {s['средняя ошибка, ч']} ч, наибольшая {s['наибольшая ошибка, ч']} ч "
                         f"(коровы различаются между собой на ±{s['разброс между коровами у людей, ч']} ч).")
    return "\n".join(lines)


#: Пороги «не знаю» (косинусное расстояние), из которых выбирается лучший
#: на части 12–14 ч. Первый — строгий, по доле чужих 1% на одной вырезке.
THRESHOLDS = (None, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 2.0)


def tune_threshold(log: Log = print) -> tuple[float, list[dict]]:
    """Порог «не знаю» — по IDF1 номера на 12–14 ч, проверка его не видит.

    Порог по одной вырезке слишком строгий для дорожки: голосование по многим
    кадрам само гасит одиночные ошибки, и лишние «не знаю» только мешают.
    """
    ours = (VARIANTS[-1],)
    rows = []
    for thr in THRESHOLDS:
        value = unknown_threshold() if thr is None else thr
        rep, _ = evaluate_tracking("val", unknown_distance=value, variants=ours, log=lambda *_: None)
        r = rep[ours[0].name]
        rows.append({"порог": round(value, 4), "IDF1 номера": r["номер коровы во времени"]["idf1"],
                     "верно": r["номер в кадре"].get("верно", 0),
                     "ошибка": r["номер в кадре"].get("ошибка", 0)})
        log(f"  порог {value:.3f}: IDF1 номера {rows[-1]['IDF1 номера']:.3f}, "
            f"верно {rows[-1]['верно']:.3f}, ошибка {rows[-1]['ошибка']:.3f}")
    best = max(rows, key=lambda r: r["IDF1 номера"])
    return best["порог"], rows


def tune_tracking(threshold: float, log: Log = print) -> list[dict]:
    """Перебор голосования и трекера на 12–14 ч (то, чем выбраны TRACKER и WINDOW)."""
    import itertools

    rows = []
    for window, age, iou, weight in itertools.product((0, 8, 20), (2, 4, 8), (0.2, 0.35), (0.2, 0.8)):
        v = Variant("перебор", True, window=window,
                    tracker=(("max_age", age), ("match_iou", iou), ("appearance_weight", weight)))
        rep, _ = evaluate_tracking("val", unknown_distance=threshold, variants=(v,), log=lambda *_: None)
        r = rep[v.name]
        rows.append({"окно голосов": window, "max_age": age, "match_iou": iou,
                     "вес внешности": weight, "IDF1 номера": r["номер коровы во времени"]["idf1"],
                     "ошибка": r["номер в кадре"].get("ошибка", 0)})
    rows.sort(key=lambda r: -r["IDF1 номера"])
    log(f"  лучшее: {rows[0]}")
    return rows


def run_all(epochs: int = 8, force: bool = False, log: Log = print) -> dict:
    if force or not any(CROPS.glob("train/C*/*.jpg")):
        prepare_crops(log=log)
    if force or not (MODEL_DIR / "encoder.pt").exists():
        train_reid(epochs=epochs, log=log)
    training = json.loads((MODEL_DIR / "training_report.json").read_text(encoding="utf-8"))
    if force or not GALLERY.exists():
        enroll_gallery(unknown_threshold(), log=log)
    checks = MODEL_DIR / "checks.json"
    if force or not checks.exists():
        log("Узнавание на проверке 14–24 ч:")
        checks.write_text(json.dumps(reid_checks(log=log), ensure_ascii=False, indent=2),
                          encoding="utf-8")
    reid = json.loads(checks.read_text(encoding="utf-8"))
    run_frames(log=log)
    log("Порог «не знаю» на 12–14 ч:")
    threshold, grid = tune_threshold(log=log)
    cautious = unknown_threshold()
    enroll_gallery(threshold, log=log)
    log(f"Tracking на проверке 14–24 ч, порог {threshold:.3f} (лучший IDF1 на 12–14 ч):")
    tracking, slots = evaluate_tracking("test", unknown_distance=threshold, log=log)
    careful = Variant(CAREFUL, True, window=WINDOW, tracker=TRACKER)
    log(f"Осторожный порог {cautious:.3f}:")
    extra, _ = evaluate_tracking("test", unknown_distance=cautious, variants=(careful,), log=log)
    tracking.update(extra)
    ours = VARIANTS[-1].name
    report = {
        "данные": "MmCows 25.07.2023: 16 коров, 4 камеры, кадр раз в 15 с; разметка номеров и поведения людьми",
        "деление": {"обучение": "03–12 ч", "выбор эпохи и порога": "12–14 ч", "проверка": "14–24 ч"},
        "яркость кадра (0–255)": brightness_by_bucket(),
        "обучение узнавания": {"эпоха": training["best"].get("epoch"),
                               "порог «не знаю»": threshold,
                               "осторожный порог": cautious,
                               "подбор порога на 12–14 ч": grid,
                               "проверка": training.get("test")},
        "узнавание": reid,
        "tracking": tracking,
        "активность": evaluate_activity(slots[ours]),
        "активность при номере из разметки": evaluate_activity(perfect_id_slots()),
    }
    report["summary_text"] = summary_text(report)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(report["summary_text"])
    return report
