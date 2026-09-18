"""Ступень 3: идентификация трека. Здесь живёт основная идея решения.

    Бирка учит биометрию.

Проблема, которую это решает. Биометрическая идентификация животных точна, но
требует регистрации: каждое животное надо один раз снять и связать его портрет
с номером. На 500 голов это дни ручной работы, а на откорме, где партии меняются
каждые пару месяцев, работа повторяется бесконечно. Именно этот барьер, а не
точность моделей, мешает внедрению камер на фермах.

Решение — замкнуть цикл самообучения:

    трек животного (десятки кадров)
      -> хотя бы на одном кадре ухо повёрнуто удачно и OCR читает номер с бирки
      -> этим номером автоматически подписываются ВСЕ кадры трека
      -> их эмбеддинги уходят в галерею под этим номером
      -> система сама себе разметила обучающую выборку

Дальше, когда бирку не видно вообще (ухо отвёрнуто, грязь, бирка потеряна),
животное узнаётся по внешности — по портрету, который система собрала сама,
без единого человеко-часа.

Три следствия, каждое из которых ценно само по себе:

1. **Регистрация стоит ноль.** Барьер внедрения снят.
2. **Номер сразу совпадает с ИСЖ/ERP.** Биркование в Казахстане обязательно,
   значит номер уже нанесён на каждое животное — сопоставлять вручную нечего.
3. **Потеря бирки обнаруживается автоматически:** биометрия узнаёт животное,
   а бирка не читается много дней подряд -> отдельное событие для хозяйства.

И главный бонус для защиты: OCR бирки даёт **бесплатный эталон** для проверки
биометрии. Мы можем честно измерить Rank-1 и FAR без ручной разметки.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import GalleryConfig
from ..types import IdentityDecision, Tracklet
from .gallery import BiometricGallery

#: Индивидуальный номер животного по Правилам идентификации РК (приказ МСХ
#: № 7-1/68): KZ, буква области, цифра вида, 8 цифр порядкового номера.
INZH = re.compile(r"^KZ[A-Z]\d{9}$")


@dataclass
class IdentifierStats:
    """Счётчики для отчёта: сколько треков как опознано."""

    tracks_total: int = 0
    by_tag: int = 0
    by_biometric: int = 0
    unknown: int = 0
    auto_enrolled: int = 0
    #: Треки, где бирку прочитать не удалось, но биометрия справилась.
    biometric_rescue: int = 0
    #: Треки, где OCR дал противоречивые номера (голосование разрешило конфликт).
    tag_conflicts: int = 0

    def as_dict(self) -> dict:
        return {
            "tracks_total": self.tracks_total,
            "by_tag": self.by_tag,
            "by_biometric": self.by_biometric,
            "unknown": self.unknown,
            "auto_enrolled": self.auto_enrolled,
            "biometric_rescue": self.biometric_rescue,
            "tag_conflicts": self.tag_conflicts,
        }

    def snapshot(self) -> "IdentifierStats":
        return IdentifierStats(**self.as_dict())

    def since(self, earlier: "IdentifierStats") -> "IdentifierStats":
        """Разница с более ранним снимком — счётчики за одни сутки, а не за всё время."""
        return IdentifierStats(
            **{k: v - getattr(earlier, k) for k, v in self.as_dict().items()}
        )


class TagTaughtIdentifier:
    """Идентификатор «бирка учит биометрию» с голосованием по треку."""

    def __init__(self, gallery: BiometricGallery, cfg: GalleryConfig, tag_prefix: str = "KZ-"):
        self.gallery = gallery
        self.cfg = cfg
        self.tag_prefix = tag_prefix
        self.stats = IdentifierStats()
        #: Сколько дней подряд у животного не читалась бирка — для события «бирка потеряна».
        self.tag_miss_streak: dict[str, int] = {}

    # -- основной вход ----------------------------------------------------

    def identify(self, tracklet: Tracklet) -> IdentityDecision:
        """Определяет животное по треку целиком, а не по одному кадру.

        Порядок каналов не случаен: бирка приоритетнее биометрии, потому что
        она даёт номер напрямую и без предварительного обучения. Биометрия —
        резервный канал и одновременно то, что бирка обучает.
        """
        self.stats.tracks_total += 1

        decision = self._decide_by_tag(tracklet)
        if decision is not None:
            self._auto_enroll(tracklet, decision.cow_id)
            tracklet.cow_id = decision.cow_id
            tracklet.id_source = decision.source
            tracklet.id_confidence = decision.confidence
            self.stats.by_tag += 1
            return decision

        decision = self._decide_by_biometrics(tracklet)
        tracklet.cow_id = decision.cow_id
        tracklet.id_source = decision.source
        tracklet.id_confidence = decision.confidence
        if decision.cow_id is not None:
            self.stats.by_biometric += 1
            self.stats.biometric_rescue += 1
            # Портрет, узнанный биометрией, тоже полезен: он расширяет галерею
            # новыми ракурсами. Но добавляем осторожно — только уверенные попадания,
            # иначе одна ошибка начнёт тиражировать сама себя.
            if decision.distance is not None and decision.distance < self.cfg.unknown_distance * 0.6:
                self._auto_enroll(tracklet, decision.cow_id, source="biometric", limit=8)
        else:
            self.stats.unknown += 1
        return decision

    # -- канал 1: бирка ---------------------------------------------------

    def _decide_by_tag(self, tracklet: Tracklet) -> Optional[IdentityDecision]:
        """Голосование по номерам, прочитанным на разных кадрах трека.

        OCR ошибается на отдельных кадрах — это нормально. Но ошибается он
        по-разному, а правильный номер повторяется. Большинство по треку
        отсеивает случайные ошибки почти полностью.
        """
        reads = [r for r in tracklet.tag_reads if r]
        if len(reads) < self.cfg.min_tag_reads_to_enroll:
            return None

        counter = Counter(reads)
        top_text, top_count = counter.most_common(1)[0]
        if len(counter) > 1:
            self.stats.tag_conflicts += 1

        ratio = top_count / len(reads)
        if ratio < self.cfg.vote_ratio:
            # Голоса разошлись слишком сильно — доверять нельзя, уходим в биометрию.
            return None

        cow_id = self._normalise_cow_id(top_text)
        return IdentityDecision(
            cow_id=cow_id,
            source="tag",
            confidence=ratio,
            votes=top_count,
            total_votes=len(reads),
        )

    def _normalise_cow_id(self, tag_text: str) -> str:
        """Приводит номер с бирки к виду, принятому в хозяйстве (ИСЖ/ERP).

        Полный ИНЖ (12 символов: KZ, буква области, цифра вида, 8 цифр
        порядкового номера) оставляется как есть — под ним животное записано
        в ИСЖ. Если камера прочитала только цифры, к ним добавляется приставка
        хозяйства; сопоставление с полным ИНЖ — по реестру хозяйства.
        """
        text = tag_text.strip().upper()
        if INZH.match(text):
            return text
        if text.startswith(self.tag_prefix.rstrip("-")):
            return text if "-" in text else f"{self.tag_prefix}{text[2:]}"
        return f"{self.tag_prefix}{text}"

    # -- канал 2: биометрия -----------------------------------------------

    def _decide_by_biometrics(self, tracklet: Tracklet) -> IdentityDecision:
        """Голосование по галерее: каждый кадр трека голосует своим ближайшим соседом.

        Решение по одному кадру неустойчиво — смазанный кадр или неудачный ракурс
        уводят в сторону. Голосование по десяткам кадров трека делает ответ
        заметно надёжнее и почти ничего не стоит.
        """
        if not tracklet.embeddings or self.gallery.size() == 0:
            return IdentityDecision(cow_id=None, source="unknown", confidence=0.0)

        votes: Counter = Counter()
        distances: dict[str, list[float]] = {}
        for emb in tracklet.embeddings:
            match = self.gallery.match(emb)
            if match.is_known and match.cow_id is not None:
                votes[match.cow_id] += 1
                distances.setdefault(match.cow_id, []).append(match.distance)

        total = len(tracklet.embeddings)
        if not votes:
            return IdentityDecision(
                cow_id=None, source="unknown", confidence=0.0, total_votes=total
            )

        cow_id, count = votes.most_common(1)[0]
        ratio = count / total
        mean_distance = float(np.mean(distances[cow_id]))

        if ratio < self.cfg.vote_ratio:
            # Кадры разошлись во мнениях — честнее ответить «не знаю»,
            # чем записать чужие показатели в карточку животного.
            return IdentityDecision(
                cow_id=None,
                source="unknown",
                confidence=ratio,
                votes=count,
                total_votes=total,
                distance=mean_distance,
            )

        return IdentityDecision(
            cow_id=cow_id,
            source="biometric",
            confidence=ratio,
            votes=count,
            total_votes=total,
            distance=mean_distance,
        )

    # -- самообучение -----------------------------------------------------

    def _auto_enroll(
        self,
        tracklet: Tracklet,
        cow_id: Optional[str],
        source: str = "tag",
        limit: Optional[int] = None,
    ) -> None:
        """Записывает портреты трека в галерею под полученным номером.

        Это и есть момент, в котором бирка «учит» биометрию.
        """
        if cow_id is None or not tracklet.embeddings:
            return
        embeddings = tracklet.embeddings
        if limit is not None and len(embeddings) > limit:
            # Берём равномерно по треку, чтобы захватить разные ракурсы,
            # а не десяток почти одинаковых соседних кадров.
            idx = np.linspace(0, len(embeddings) - 1, limit).astype(int)
            embeddings = [embeddings[i] for i in idx]
        self.gallery.enroll(cow_id, embeddings, source=source)
        self.stats.auto_enrolled += 1

    # -- побочный полезный сигнал -----------------------------------------

    def update_tag_health(self, cow_id: str, tag_was_read: bool) -> Optional[str]:
        """Следит, у какого животного перестала читаться бирка.

        Биркование в Казахстане обязательно, потерянная бирка — это нарушение
        и проблема для хозяйства. Система узнаёт животное биометрически и может
        сама сообщить, что бирку пора восстановить. Функция, которой нет
        ни у ошейников, ни у RFID.
        """
        if tag_was_read:
            self.tag_miss_streak[cow_id] = 0
            return None
        streak = self.tag_miss_streak.get(cow_id, 0) + 1
        self.tag_miss_streak[cow_id] = streak
        if streak >= 5:
            return (
                f"У животного {cow_id} бирка не читается {streak} дней подряд — "
                f"вероятно, потеряна или загрязнена."
            )
        return None
