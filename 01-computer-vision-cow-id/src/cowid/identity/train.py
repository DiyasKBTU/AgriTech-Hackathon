"""Обучение модели идентификации животных на реальных данных.

Что именно обучается. Не классификатор «какая это из 186 коров» — такая модель
бесполезна на ферме, где завтра привезут новую партию. Обучается **функция
похожести**: сеть превращает снимок животного в вектор так, чтобы у одного и
того же животного векторы с разных снимков были рядом, а у разных животных —
далеко. Это называется metric learning.

Как её заставить этому научиться. Во время обучения сверху ставится голова
ArcFace. От обычного классификатора она отличается тем, что требует не просто
угадать класс, а угадать его **с запасом по углу**: между вектором животного и
«его» направлением должен быть зазор. Из-за этого требования сеть вынуждена
разносить разных животных дальше друг от друга, чем нужно для простого
угадывания. После обучения голова выбрасывается, остаётся только сеть-кодировщик.

Как проверяется. Только на животных, которых модель не видела ни разу
(см. `reid_dataset.py`). Считается:

* **Rank-1** — доля запросов, для которых ближайший сосед в галерее оказался
  тем же животным;
* **mAP** — учитывает не только первого соседа, а весь список;
* **TAR@FAR** — сколько правильных срабатываний при заданной доле ложных.
  Это и есть метрика режима «не знаю»: она отвечает на вопрос, можно ли
  подобрать порог, при котором система почти не путает чужих животных.

Запуск:
    cowid train-reid --data <папка, где подпапка = корова>
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .reid_dataset import ReidSplit, Sample, index_by_folder, save_split, split_by_identity


@dataclass
class TrainConfig:
    data_root: str
    out_dir: str = "models/reid"
    backbone: str = "resnet50"
    embedding_dim: int = 512
    image_size: int = 224
    batch_size: int = 48
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 5e-4
    #: Параметры ArcFace: масштаб логитов и угловой зазор.
    arcface_scale: float = 30.0
    arcface_margin: float = 0.30
    #: Доля ЖИВОТНЫХ (не снимков), уходящих в проверку.
    test_identity_ratio: float = 0.3
    min_images_per_identity: int = 4
    max_images_per_identity: Optional[int] = 60
    num_workers: int = 4
    #: Проверочные животные: галерея из ранних дней, запросы из поздних.
    split_by_time: bool = True
    seed: int = 42
    device: str = "auto"


# --------------------------------------------------------------------------
# Данные
# --------------------------------------------------------------------------

class MaskCenter:
    """Закрывает центральную часть снимка (после нормализации это «средний серый»).

    Контрольная проверка: если модель узнаёт животное и без центра снимка, где
    находится само животное, значит, она опирается на фон — место и время съёмки,
    — и её цифрам доверять нельзя.
    """

    def __init__(self, share: float):
        self.share = share

    def __call__(self, tensor):
        _, h, w = tensor.shape
        dh, dw = int(h * self.share / 2), int(w * self.share / 2)
        tensor[:, h // 2 - dh:h // 2 + dh, w // 2 - dw:w // 2 + dw] = 0.0
        return tensor


def _build_transforms(image_size: int, train: bool, mask_center: float = 0.0):
    from torchvision import transforms

    if not train:
        steps = [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        if mask_center > 0:
            steps.append(MaskCenter(mask_center))
        return transforms.Compose(steps)

    # Аугментации подобраны под то, что реально меняется на ферме: свет в течение
    # суток, ракурс камеры, грязь на шкуре, размытие от движения. Отражение по
    # горизонтали НЕ применяем: рисунок шкуры несимметричен, и зеркальный снимок
    # для модели — это другое животное.
    return transforms.Compose([
        transforms.Resize((int(image_size * 1.15), int(image_size * 1.15)), antialias=True),
        transforms.RandomRotation(12),
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), antialias=True),
        transforms.ColorJitter(brightness=0.35, contrast=0.3, saturation=0.25, hue=0.03),
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.8))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.12)),
    ])


class ReidImages:
    """Снимки животных для DataLoader.

    Объявлен на уровне модуля намеренно: на Windows рабочие процессы загрузчика
    запускаются заново (spawn), и класс, объявленный внутри функции, в них не
    передаётся — обучение падало бы при num_workers > 0.
    """

    def __init__(self, samples: list[Sample], label_map: dict[str, int], transform):
        self.samples = samples
        self.label_map = label_map
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image

        s = self.samples[idx]
        with Image.open(s.path) as img:
            img = img.convert("RGB")
        return self.transform(img), self.label_map.get(s.identity, -1)


def _make_dataset(samples: list[Sample], label_map: dict[str, int], image_size: int, train: bool,
                  mask_center: float = 0.0):
    return ReidImages(samples, label_map, _build_transforms(image_size, train, mask_center))


# --------------------------------------------------------------------------
# Модель
# --------------------------------------------------------------------------

def _build_model(cfg: TrainConfig, n_classes: int):
    import torch
    from torch import nn
    from torchvision import models

    builder = getattr(models, cfg.backbone)
    net = builder(weights="DEFAULT")
    feature_dim = net.fc.in_features
    net.fc = nn.Identity()

    class Encoder(nn.Module):
        """Сеть-кодировщик: снимок -> вектор единичной длины."""

        def __init__(self):
            super().__init__()
            self.backbone = net
            self.neck = nn.Sequential(
                nn.BatchNorm1d(feature_dim),
                nn.Linear(feature_dim, cfg.embedding_dim),
                nn.BatchNorm1d(cfg.embedding_dim),
            )

        def forward(self, x):
            feat = self.neck(self.backbone(x))
            return nn.functional.normalize(feat, dim=1)

    class ArcFace(nn.Module):
        """Голова с угловым зазором. Нужна только при обучении."""

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(n_classes, cfg.embedding_dim) * 0.01)

        def forward(self, embeddings, labels):
            w = nn.functional.normalize(self.weight, dim=1)
            cosine = embeddings @ w.t()
            theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
            # Зазор добавляется только к «своему» классу: сеть обязана попасть
            # в него с запасом, а не просто оказаться ближе остальных.
            one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1.0)
            target = torch.cos(theta + cfg.arcface_margin)
            logits = torch.where(one_hot.bool(), target, cosine) * cfg.arcface_scale
            return logits

    return Encoder(), ArcFace()


# --------------------------------------------------------------------------
# Метрики
# --------------------------------------------------------------------------

def evaluate_reid(
    query_emb: np.ndarray,
    query_ids: list[str],
    gallery_emb: np.ndarray,
    gallery_ids: list[str],
) -> dict:
    """Rank-1, Rank-5, mAP и TAR@FAR на животных, которых модель не видела."""
    if len(query_emb) == 0 or len(gallery_emb) == 0:
        return {}

    similarity = query_emb @ gallery_emb.T          # векторы нормированы
    gallery_arr = np.asarray(gallery_ids)

    order = np.argsort(-similarity, axis=1)
    matches = gallery_arr[order] == np.asarray(query_ids)[:, None]

    rank1 = float(matches[:, 0].mean())
    rank5 = float(matches[:, :5].any(axis=1).mean())

    # mAP: усреднённая точность по всем правильным ответам каждого запроса.
    aps = []
    for row in matches:
        positives = np.flatnonzero(row)
        if len(positives) == 0:
            aps.append(0.0)
            continue
        precision = (np.arange(len(positives)) + 1) / (positives + 1)
        aps.append(float(precision.mean()))
    mean_ap = float(np.mean(aps))

    # Открытое множество: режим «не знаю».
    #
    # Вопрос: если бы этой коровы в базе НЕ было, с какой уверенностью система
    # спутала бы её с кем-то другим? Для каждого запроса берём лучшее сходство
    # со снимками ЧУЖИХ коров — это «самозванец». Лучшее сходство со снимками
    # своей коровы — «свой». Порог «не знаю» ставится так, чтобы самозванцев
    # пропускать не чаще заданной доли.
    #
    # Раньше самозванцами считались только запросы, где модель ошиблась.
    # При точности 98% их два десятка, и порог по двадцати числам получался
    # случайным — метрика скакала от эпохи к эпохе. Здесь самозванец есть
    # у каждого запроса, и оценка устойчива.
    q_ids = np.asarray(query_ids)
    same = gallery_arr[None, :] == q_ids[:, None]
    genuine = np.where(same, similarity, -np.inf).max(axis=1)
    impostor = np.where(~same, similarity, -np.inf).max(axis=1)
    has_own = np.isfinite(genuine)
    top1_correct = matches[:, 0]

    tar_at_far = {}
    thresholds = {}
    for far in (0.01, 0.05, 0.10):
        threshold = float(np.quantile(impostor, 1.0 - far))
        thresholds[far] = threshold
        # Узнана верно И уверенно: первая по сходству — своя корова,
        # и сходство выше порога «не знаю».
        detected = has_own & top1_correct & (genuine >= threshold)
        tar_at_far[f"tar@far{far:g}"] = float(detected.mean())

    return {
        "rank1": round(rank1, 4),
        "rank5": round(rank5, 4),
        "mAP": round(mean_ap, 4),
        "n_query": len(query_ids),
        "n_gallery": len(gallery_ids),
        "n_test_identities": len(set(query_ids)),
        **{k: round(v, 4) for k, v in tar_at_far.items()},
        # Порог «не знаю» в терминах косинусного расстояния, как его ждёт галерея:
        # при таком пороге чужое животное принимается за своё не чаще 1 раза из 100.
        "unknown_distance_at_far1": round(1.0 - thresholds[0.01], 4),
    }


# --------------------------------------------------------------------------
# Обучение
# --------------------------------------------------------------------------

def _extract(encoder, samples: list[Sample], cfg: TrainConfig, device,
             mask_center: float = 0.0) -> tuple[np.ndarray, list[str]]:
    import torch
    from torch.utils.data import DataLoader

    if not samples:
        return np.zeros((0, cfg.embedding_dim), dtype=np.float32), []
    dataset = _make_dataset(samples, {}, cfg.image_size, train=False, mask_center=mask_center)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers)
    vectors = []
    encoder.eval()
    with torch.no_grad():
        for images, _ in loader:
            vectors.append(encoder(images.to(device)).cpu().numpy())
    return np.concatenate(vectors), [s.identity for s in samples]


def _device(cfg: TrainConfig) -> str:
    import torch

    if cfg.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.device


def train(cfg: TrainConfig, log=print) -> dict:
    """Открытое множество: проверка на животных, которых модель не видела."""
    samples = index_by_folder(
        cfg.data_root,
        min_images_per_identity=cfg.min_images_per_identity,
        max_images_per_identity=cfg.max_images_per_identity,
    )
    if not samples:
        raise RuntimeError(f"В {cfg.data_root} не найдено снимков, разложенных по папкам-животным")

    split = split_by_identity(samples, test_identity_ratio=cfg.test_identity_ratio,
                              seed=cfg.seed, by_time=cfg.split_by_time)
    summary = split.summary()
    log(f"Разбиение по животным: {summary}")
    if summary["identity_overlap"] != 0:
        raise RuntimeError("Животные из обучения попали в проверку — метрики были бы завышены")
    return fit(cfg, split, log=log)


def load_encoder(weights: str | Path, device: str = "cpu"):
    """Кодировщик из сохранённых весов и настройки, с которыми он обучался."""
    import torch

    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**checkpoint["config"])
    encoder, _ = _build_model(cfg, n_classes=1)
    encoder.load_state_dict(checkpoint["state_dict"])
    return encoder.to(device).eval(), cfg


def fit(cfg: TrainConfig, split: ReidSplit, log=print, init_weights: Optional[str] = None,
        select: Optional[ReidSplit] = None) -> dict:
    """Обучение на готовом разбиении.

    `init_weights` — начать с уже обученного кодировщика (дообучение под ферму).
    `select` — отдельная часть (галерея + запросы), по которой выбирается лучшая
    эпоха. Без неё эпоха выбирается по самой проверке — это чуть завышает цифру,
    и об этом надо говорить. С ней проверка (`split.query`) остаётся нетронутой
    до конца и считается один раз, на выбранных весах.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    torch.manual_seed(cfg.seed)
    device = _device(cfg)
    log(f"Устройство: {device}")
    summary = split.summary()

    label_map = {identity: i for i, identity in enumerate(sorted(split.train_identities))}
    train_ds = _make_dataset(split.train, label_map, cfg.image_size, train=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=True)

    encoder, head = _build_model(cfg, n_classes=len(label_map))
    if init_weights:
        state = torch.load(init_weights, map_location="cpu", weights_only=False)["state_dict"]
        encoder.load_state_dict(state)
        log(f"Начальные веса: {init_weights}")
    encoder, head = encoder.to(device), head.to(device)
    chooser = select or split
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=max(1, cfg.epochs * len(train_loader)),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_split(split, out_dir / "split.json")

    history = []
    best = {"rank1": -1.0}
    for epoch in range(1, cfg.epochs + 1):
        encoder.train()
        head.train()
        started = time.perf_counter()
        total_loss, total_correct, total_seen = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            embeddings = encoder(images)
            logits = head(embeddings, labels)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.detach()) * len(labels)
            total_correct += int((logits.argmax(1) == labels).sum())
            total_seen += len(labels)

        q_emb, q_ids = _extract(encoder, chooser.query, cfg, device)
        g_emb, g_ids = _extract(encoder, chooser.gallery, cfg, device)
        metrics = evaluate_reid(q_emb, q_ids, g_emb, g_ids)

        row = {
            "epoch": epoch,
            "loss": round(total_loss / max(1, total_seen), 4),
            "train_acc": round(total_correct / max(1, total_seen), 4),
            "seconds": round(time.perf_counter() - started, 1),
            **metrics,
        }
        history.append(row)
        log(
            f"эпоха {epoch:2d}/{cfg.epochs}  loss {row['loss']:.3f}  "
            f"Rank-1 {metrics.get('rank1', 0):.3f}  mAP {metrics.get('mAP', 0):.3f}  "
            f"TAR@FAR1% {metrics.get('tar@far0.01', 0):.3f}  ({row['seconds']}с)"
        )

        if metrics.get("rank1", 0) > best["rank1"]:
            best = dict(metrics)
            best["epoch"] = epoch
            torch.save(
                {
                    "state_dict": encoder.state_dict(),
                    "config": asdict(cfg),
                    "metrics": metrics,
                    "epoch": epoch,
                },
                out_dir / "encoder.pt",
            )

    report = {
        "config": asdict(cfg),
        "split": summary,
        "best": best,
        "history": history,
        "weights": str(out_dir / "encoder.pt"),
    }
    if select is not None:
        # Проверка — один раз, на весах лучшей по отдельной части эпохи.
        encoder.load_state_dict(torch.load(out_dir / "encoder.pt", map_location="cpu",
                                           weights_only=False)["state_dict"])
        encoder = encoder.to(device)
        q_emb, q_ids = _extract(encoder, split.query, cfg, device)
        g_emb, g_ids = _extract(encoder, split.gallery, cfg, device)
        report["selected_on"] = "отдельная часть (select)"
        report["test"] = evaluate_reid(q_emb, q_ids, g_emb, g_ids)
        log(f"Проверка на нетронутой части: Rank-1 {report['test'].get('rank1', 0):.3f}")
    (out_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"Лучший результат: Rank-1 {best.get('rank1', 0):.3f} на эпохе {best.get('epoch')}")
    log(f"Веса сохранены: {out_dir / 'encoder.pt'}")
    return report


def evaluate_saved_model(model_dir: str | Path, mask_center: float = 0.0) -> dict:
    """Перепроверяет сохранённую модель на её же проверочной части.

    Берёт разбиение из split.json, записанного при обучении, — поэтому результат
    воспроизводим и в проверку гарантированно не попадают животные из обучения.
    `mask_center` — доля снимка по центру, которая закрывается (контроль фона).
    """
    import torch

    model_dir = Path(model_dir)
    checkpoint = torch.load(model_dir / "encoder.pt", map_location="cpu", weights_only=False)
    cfg = TrainConfig(**checkpoint["config"])
    split_data = json.loads((model_dir / "split.json").read_text(encoding="utf-8"))

    def samples(key: str) -> list[Sample]:
        return [Sample(path=Path(r["path"]), identity=r["identity"]) for r in split_data[key]]

    device = "cuda" if cfg.device == "auto" and torch.cuda.is_available() else (
        "cpu" if cfg.device == "auto" else cfg.device)
    encoder, _ = _build_model(cfg, n_classes=1)
    encoder.load_state_dict(checkpoint["state_dict"])
    encoder = encoder.to(device).eval()

    q_emb, q_ids = _extract(encoder, samples("query"), cfg, device, mask_center)
    g_emb, g_ids = _extract(encoder, samples("gallery"), cfg, device, mask_center)
    report = evaluate_reid(q_emb, q_ids, g_emb, g_ids)
    report["mask_center"] = mask_center
    report["split"] = split_data.get("summary", {})
    report["trained_epoch"] = checkpoint.get("epoch")
    return report


def control_checks(model_dir: str | Path, mask_share: float = 0.6) -> dict:
    """Не завышена ли цифра узнавания. Четыре прогона на той же проверке:

    * обученная модель — как в `evaluate_saved_model`;
    * необученная сеть (признаки ImageNet, коров не видела). Если она почти
      так же хороша, проверка слишком лёгкая и ничего не доказывает;
    * обе — с закрытым центром снимка. Если результат почти не падает,
      модель узнаёт по фону, а не по животному.
    """
    import torch
    from torch.utils.data import DataLoader

    model_dir = Path(model_dir)
    checkpoint = torch.load(model_dir / "encoder.pt", map_location="cpu", weights_only=False)
    cfg = TrainConfig(**checkpoint["config"])
    split_data = json.loads((model_dir / "split.json").read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def samples(key: str) -> list[Sample]:
        return [Sample(path=Path(r["path"]), identity=r["identity"]) for r in split_data[key]]

    encoder, _ = _build_model(cfg, n_classes=1)
    backbone = encoder.backbone.to(device).eval()

    def untrained(items: list[Sample], mask: float):
        ds = _make_dataset(items, {}, cfg.image_size, train=False, mask_center=mask)
        vectors = []
        with torch.no_grad():
            for images, _ in DataLoader(ds, batch_size=cfg.batch_size, num_workers=0):
                feats = backbone(images.to(device))
                vectors.append(torch.nn.functional.normalize(feats, dim=1).cpu().numpy())
        return np.concatenate(vectors), [s.identity for s in items]

    report = {}
    for mask in (0.0, mask_share):
        q, qi = untrained(samples("query"), mask)
        g, gi = untrained(samples("gallery"), mask)
        report[f"untrained_mask{int(mask * 100)}"] = evaluate_reid(q, qi, g, gi)
        report[f"trained_mask{int(mask * 100)}"] = evaluate_saved_model(model_dir, mask)
    (model_dir / "controls.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
