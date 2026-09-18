"""Индексация реального датасета для обучения идентификации животных.

Главное здесь — не загрузка картинок, а **как делятся данные**. Именно здесь
чаще всего и подделывают результаты, обычно не нарочно.

Неправильно: перемешать все снимки и отрезать 20% на тест. Тогда в тесте
окажутся снимки тех же самых животных, что и в обучении. Модель просто
запомнила их, и высокая точность ничего не говорит о работе на ферме, где
завтра приедет новая партия скота.

Правильно (и так сделано здесь): делить **по животным**. Одни особи целиком
уходят в обучение, другие целиком в проверку. Модель на проверке видит животных,
которых не видела никогда. Это называется open-set, и это то, что происходит
в реальной эксплуатации.

Внутри проверочной части снимки каждого животного ещё раз делятся на «галерею»
(то, что система запомнила) и «запросы» (то, что она пытается узнать).
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


DATE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})")


@dataclass
class Sample:
    path: Path
    identity: str

    @property
    def date(self) -> str:
        """Дата съёмки из имени файла, если она там есть (Cows2021 так и называет)."""
        m = DATE_RE.search(Path(self.path).name)
        return m.group(1) if m else ""

    def as_dict(self) -> dict:
        return {"path": str(self.path), "identity": self.identity}


@dataclass
class ReidSplit:
    """Готовое разбиение датасета."""

    train: list[Sample] = field(default_factory=list)
    #: Проверочная часть: животные, которых модель не видела при обучении.
    gallery: list[Sample] = field(default_factory=list)
    query: list[Sample] = field(default_factory=list)

    @property
    def train_identities(self) -> set[str]:
        return {s.identity for s in self.train}

    @property
    def test_identities(self) -> set[str]:
        return {s.identity for s in self.gallery} | {s.identity for s in self.query}

    def summary(self) -> dict:
        overlap = self.train_identities & self.test_identities
        return {
            "train_images": len(self.train),
            "train_identities": len(self.train_identities),
            "gallery_images": len(self.gallery),
            "query_images": len(self.query),
            "test_identities": len(self.test_identities),
            # Пересечение обязано быть пустым. Если оно не пусто, все метрики
            # завышены, и на защите это первый вопрос, который стоит задать.
            "identity_overlap": len(overlap),
        }


def index_by_folder(
    root: str | Path,
    min_images_per_identity: int = 4,
    max_images_per_identity: Optional[int] = None,
    identity_from: str = "parent",
) -> list[Sample]:
    """Собирает снимки, считая именем животного название папки.

    Оба используемых датасета разложены именно так: одна папка — одно животное.
    Папки с малым числом снимков отбрасываются: по одному кадру нельзя ни
    обучать, ни делить на галерею и запрос.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Каталог датасета не найден: {root}")

    by_identity: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
            continue
        if identity_from == "parent":
            identity = path.parent.name
        else:
            identity = path.parent.parent.name
        by_identity[identity].append(path)

    samples: list[Sample] = []
    for identity, paths in sorted(by_identity.items()):
        if len(paths) < min_images_per_identity:
            continue
        paths = sorted(paths)
        if max_images_per_identity is not None and len(paths) > max_images_per_identity:
            # Прореживаем равномерно, а не берём первые N: подряд идущие кадры
            # почти одинаковы и дают ложное ощущение большого датасета.
            step = len(paths) / max_images_per_identity
            paths = [paths[int(i * step)] for i in range(max_images_per_identity)]
        samples.extend(Sample(path=p, identity=identity) for p in paths)
    return samples


def split_by_identity(
    samples: Iterable[Sample],
    test_identity_ratio: float = 0.3,
    query_ratio: float = 0.5,
    seed: int = 42,
    by_time: bool = True,
) -> ReidSplit:
    """Делит по ЖИВОТНЫМ, а не по снимкам. См. пояснение в заголовке модуля.

    `by_time` делает проверку ещё ближе к ферме: у каждого проверочного
    животного «галерея» собирается из более ранних дней съёмки, а «запросы» —
    из более поздних. Так модель проверяется на задаче «запомнили корову на
    прошлой неделе — узнайте её сегодня», где меняются свет, грязь на шкуре
    и поза. Случайное деление снимков внутри животного эту задачу упрощает:
    соседние кадры одной съёмки почти одинаковы.
    """
    rng = random.Random(seed)
    by_identity: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_identity[s.identity].append(s)

    identities = sorted(by_identity)
    rng.shuffle(identities)
    n_test = max(1, int(len(identities) * test_identity_ratio))
    test_ids = set(identities[:n_test])
    train_ids = set(identities[n_test:])

    split = ReidSplit()
    for identity in train_ids:
        split.train.extend(by_identity[identity])

    for identity in test_ids:
        items = list(by_identity[identity])
        dates = sorted({s.date for s in items if s.date})
        if by_time and len(dates) >= 2:
            # Ранние дни — в галерею, поздние — в запросы.
            cut = dates[max(1, int(len(dates) * (1 - query_ratio)))]
            gallery = [s for s in items if s.date < cut]
            query = [s for s in items if s.date >= cut]
            if gallery and query:
                split.gallery.extend(gallery)
                split.query.extend(query)
                continue
        rng.shuffle(items)
        n_query = max(1, int(len(items) * query_ratio))
        # Хотя бы один снимок обязан остаться в галерее, иначе животное
        # невозможно узнать в принципе.
        n_query = min(n_query, len(items) - 1)
        split.query.extend(items[:n_query])
        split.gallery.extend(items[n_query:])
    return split


def save_split(split: ReidSplit, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(
            {
                "summary": split.summary(),
                "train": [s.as_dict() for s in split.train],
                "gallery": [s.as_dict() for s in split.gallery],
                "query": [s.as_dict() for s in split.query],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
