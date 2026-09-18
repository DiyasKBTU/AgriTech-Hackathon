"""Галерея биометрических портретов: хранение, поиск и открытое множество.

Галерея — это база «как выглядит каждое животное». Её принципиальное отличие от
обычного классификатора в том, что она **открытая**: список животных не фиксирован,
новые особи добавляются на ходу, а на незнакомое животное система обязана ответить
«не знаю», а не назначить чужой номер.

Почему «не знаю» важнее точности. Если система уверенно назовёт новую корову
номером KZ-1007, то в карточку KZ-1007 польются чужие показатели активности,
её персональная норма поедет, и зоотехник получит ложное событие про здоровое
животное. Одна такая ошибка убивает доверие к системе быстрее, чем десять пропусков.

Поиск реализован обычным матричным умножением numpy: векторы нормированы, поэтому
косинусная близость — это скалярное произведение. Для стада в тысячи голов с
десятками портретов на голову это доли миллисекунды, внешний индекс (FAISS)
не нужен и не добавлен намеренно — меньше зависимостей, меньше риска на защите.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import GalleryConfig


@dataclass
class GalleryMatch:
    cow_id: Optional[str]
    distance: float
    is_known: bool


@dataclass
class _CowEntry:
    cow_id: str
    embeddings: deque = field(default_factory=deque)
    #: Как животное попало в галерею: по бирке (надёжно) или вручную.
    source: str = "tag"
    n_enrollments: int = 0


class BiometricGallery:
    """Хранилище портретов животных с поиском ближайшего и порогом «не знаю»."""

    def __init__(self, cfg: GalleryConfig):
        self.cfg = cfg
        self._entries: dict[str, _CowEntry] = {}
        #: Плоские матрицы для быстрого поиска, перестраиваются лениво.
        self._matrix: Optional[np.ndarray] = None
        self._row_ids: list[str] = []
        self._dirty = True

    # -- наполнение -------------------------------------------------------

    def enroll(self, cow_id: str, embeddings: list[np.ndarray], source: str = "tag") -> None:
        """Добавляет портреты животного. Это и есть автоматическая регистрация.

        Вызывается, когда трек получил номер с бирки: все кадры трека
        подписываются этим номером и отправляются сюда. Человек не участвует.
        """
        if not embeddings:
            return
        entry = self._entries.get(cow_id)
        if entry is None:
            entry = _CowEntry(cow_id=cow_id, embeddings=deque(maxlen=self.cfg.max_per_cow),
                              source=source)
            self._entries[cow_id] = entry
        for emb in embeddings:
            entry.embeddings.append(np.asarray(emb, dtype=np.float32))
        entry.n_enrollments += 1
        self._dirty = True

    def known_ids(self) -> list[str]:
        return sorted(self._entries.keys())

    def size(self) -> int:
        return len(self._entries)

    def total_embeddings(self) -> int:
        return sum(len(e.embeddings) for e in self._entries.values())

    # -- поиск ------------------------------------------------------------

    def _rebuild(self) -> None:
        rows, ids = [], []
        for entry in self._entries.values():
            for emb in entry.embeddings:
                rows.append(emb)
                ids.append(entry.cow_id)
        self._matrix = np.stack(rows) if rows else None
        self._row_ids = ids
        self._dirty = False

    def query(self, embedding: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """Возвращает ближайшие портреты как (cow_id, косинусное расстояние)."""
        if self._dirty:
            self._rebuild()
        if self._matrix is None or len(self._row_ids) == 0:
            return []
        sims = self._matrix @ np.asarray(embedding, dtype=np.float32)
        distances = 1.0 - sims
        order = np.argsort(distances)[: max(top_k, 1) * 4]

        # Сворачиваем до лучшего расстояния на животное.
        best: dict[str, float] = {}
        for idx in order:
            cow_id = self._row_ids[idx]
            d = float(distances[idx])
            if cow_id not in best or d < best[cow_id]:
                best[cow_id] = d
        return sorted(best.items(), key=lambda kv: kv[1])[:top_k]

    def match(self, embedding: np.ndarray) -> GalleryMatch:
        """Ближайшее животное с решением об открытом множестве."""
        hits = self.query(embedding, top_k=1)
        if not hits:
            return GalleryMatch(cow_id=None, distance=float("inf"), is_known=False)
        cow_id, distance = hits[0]
        if distance > self.cfg.unknown_distance:
            return GalleryMatch(cow_id=None, distance=distance, is_known=False)
        return GalleryMatch(cow_id=cow_id, distance=distance, is_known=True)

    # -- сохранение -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cows": [
                {
                    "cow_id": e.cow_id,
                    "source": e.source,
                    "n_enrollments": e.n_enrollments,
                    "embeddings": [emb.tolist() for emb in e.embeddings],
                }
                for e in self._entries.values()
            ]
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, cfg: GalleryConfig) -> "BiometricGallery":
        gallery = cls(cfg)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for row in data.get("cows", []):
            embs = [np.asarray(e, dtype=np.float32) for e in row["embeddings"]]
            gallery.enroll(row["cow_id"], embs, source=row.get("source", "tag"))
        return gallery

    # -- диагностика ------------------------------------------------------

    def stats(self) -> dict:
        return {
            "cows": self.size(),
            "embeddings": self.total_embeddings(),
            "unknown_distance": self.cfg.unknown_distance,
            "per_cow": {
                cow_id: len(entry.embeddings) for cow_id, entry in sorted(self._entries.items())
            },
        }
