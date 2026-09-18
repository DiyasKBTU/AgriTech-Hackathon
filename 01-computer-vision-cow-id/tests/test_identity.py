"""Идентификация: галерея, режим «не знаю», «бирка учит биометрию», OCR."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from cowid.config import EmbedderConfig, GalleryConfig, TagOCRConfig
from cowid.identity.embedder import CoatPatternEmbedder, build_embedder
from cowid.identity.gallery import BiometricGallery
from cowid.identity.identifier import TagTaughtIdentifier
from cowid.identity.reid_dataset import Sample, split_by_identity
from cowid.identity.tag_ocr import DigitTemplateOCR
from cowid.identity.train import evaluate_reid
from cowid.types import BBox

from .conftest import make_tracklet, unit


# --------------------------------------------------------------------------
# Галерея
# --------------------------------------------------------------------------

def test_stranger_gets_unknown_not_someone_elses_number():
    """Незнакомому животному нельзя назначать чужой номер: в чужую карточку
    польются чужие показатели, и норма здорового животного поедет."""
    gallery = BiometricGallery(GalleryConfig(unknown_distance=0.2))
    gallery.enroll("KZ-1001", [unit([1, 0, 0, 0])])
    gallery.enroll("KZ-1002", [unit([0, 1, 0, 0])])

    assert gallery.match(unit([0.98, 0.02, 0, 0])).cow_id == "KZ-1001"
    stranger = gallery.match(unit([0, 0, 1, 0]))
    assert not stranger.is_known and stranger.cow_id is None


def test_gallery_survives_save_and_load(tmp_path):
    gallery = BiometricGallery(GalleryConfig())
    gallery.enroll("KZ-1001", [unit([1, 0, 0]), unit([0.9, 0.1, 0])])
    gallery.save(tmp_path / "g.json")
    restored = BiometricGallery.load(tmp_path / "g.json", GalleryConfig())
    assert restored.known_ids() == ["KZ-1001"]
    assert restored.total_embeddings() == 2


# --------------------------------------------------------------------------
# «Бирка учит биометрию»
# --------------------------------------------------------------------------

def test_tag_read_registers_animal_without_a_human():
    gallery = BiometricGallery(GalleryConfig())
    identifier = TagTaughtIdentifier(gallery, GalleryConfig())
    decision = identifier.identify(make_tracklet(1, ["1001"] * 5, [unit([1, 0, 0])] * 10))
    assert decision.source == "tag" and decision.cow_id == "KZ-1001"
    assert gallery.known_ids() == ["KZ-1001"]
    assert gallery.total_embeddings() == 10


def test_full_kazakh_animal_number_is_kept_as_is():
    """ИНЖ с бирки должен совпасть с записью в ИСЖ символ в символ."""
    identifier = TagTaughtIdentifier(BiometricGallery(GalleryConfig()), GalleryConfig())
    decision = identifier.identify(make_tracklet(1, ["KZA400012345"] * 4, [unit([1, 0, 0])] * 4))
    assert decision.cow_id == "KZA400012345"


def test_animal_recognised_by_appearance_once_tag_taught_the_system():
    cfg = GalleryConfig(unknown_distance=0.3)
    gallery = BiometricGallery(cfg)
    identifier = TagTaughtIdentifier(gallery, cfg)
    identifier.identify(make_tracklet(1, ["1001"] * 5, [unit([1, 0, 0])] * 10))

    decision = identifier.identify(make_tracklet(2, [], [unit([0.97, 0.05, 0.02])] * 8))
    assert decision.source == "biometric" and decision.cow_id == "KZ-1001"


def test_single_ocr_misreads_are_outvoted():
    gallery = BiometricGallery(GalleryConfig())
    identifier = TagTaughtIdentifier(gallery, GalleryConfig())
    reads = ["1001"] * 7 + ["1091", "7001"]
    assert identifier.identify(make_tracklet(1, reads, [unit([1, 0, 0])] * 9)).cow_id == "KZ-1001"


def test_split_vote_gives_unknown():
    identifier = TagTaughtIdentifier(BiometricGallery(GalleryConfig()),
                                     GalleryConfig(vote_ratio=0.6))
    reads = ["1001"] * 4 + ["1002"] * 4
    assert identifier.identify(make_tracklet(1, reads, [unit([1, 0, 0])] * 8)).cow_id is None


def test_lost_tag_is_reported_after_several_days():
    identifier = TagTaughtIdentifier(BiometricGallery(GalleryConfig()), GalleryConfig())
    message = None
    for _ in range(5):
        message = identifier.update_tag_health("KZ-1001", tag_was_read=False)
    assert message and "KZ-1001" in message
    assert identifier.update_tag_health("KZ-1001", tag_was_read=True) is None


# --------------------------------------------------------------------------
# OCR бирки и дескриптор — проверка кода на простой картинке
# --------------------------------------------------------------------------

def _tag_image(number: str) -> np.ndarray:
    """Жёлтая бирка с номером на сером фоне — минимальная картинка для проверки кода."""
    img = np.full((120, 200, 3), 110, dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (150, 80), (60, 200, 250), -1)
    cv2.rectangle(img, (40, 40), (150, 80), (30, 120, 160), 1)
    cv2.putText(img, number, (48, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2,
                cv2.LINE_AA)
    return img


@pytest.mark.parametrize("number", ["1004", "2718", "5093"])
def test_digit_ocr_reads_a_clear_tag(number):
    ocr = DigitTemplateOCR(TagOCRConfig(min_confidence=0.3))
    read = ocr.read(_tag_image(number), BBox(0, 0, 199, 119))
    assert read is not None and read.text == number


def test_coat_descriptor_tells_two_patterns_apart():
    def animal(spots):
        img = np.full((160, 240, 3), 105, dtype=np.uint8)
        cv2.ellipse(img, (120, 80), (80, 40), 0, 0, 360, (225, 225, 225), -1)
        for x, y, r in spots:
            cv2.circle(img, (x, y), r, (40, 40, 40), -1)
        return img

    emb = CoatPatternEmbedder(EmbedderConfig(kind="coatpattern"))
    a = emb.embed(animal([(80, 70, 14), (150, 90, 10)]), BBox(0, 0, 239, 159))
    b = emb.embed(animal([(120, 60, 18), (170, 70, 9), (90, 95, 8)]), BBox(0, 0, 239, 159))
    assert a is not None and b is not None
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)
    assert float(np.dot(a, b)) < 0.995


def test_missing_model_gives_a_clear_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="cowid train-reid"):
        build_embedder(EmbedderConfig(kind="learned", weights=str(tmp_path / "nope.pt")))


# --------------------------------------------------------------------------
# Обучение: честное разбиение и метрики
# --------------------------------------------------------------------------

def test_split_never_puts_one_animal_in_both_train_and_test():
    """Если одно животное есть и в обучении, и в проверке, модель его просто
    запоминает, и метрика ничего не значит."""
    samples = [Sample(path=f"{i}_{k}.jpg", identity=f"cow{i}")
               for i in range(30) for k in range(6)]
    split = split_by_identity(samples, test_identity_ratio=0.3)
    assert split.summary()["identity_overlap"] == 0
    assert split.summary()["test_identities"] == 9


def test_reid_metrics_and_unknown_threshold():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(10, 32))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    def noisy(n):
        v = np.repeat(centers, n, 0) + rng.normal(0, 0.15, (10 * n, 32))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    ids = lambda n: [str(i) for i in range(10) for _ in range(n)]  # noqa: E731
    m = evaluate_reid(noisy(2), ids(2), noisy(3), ids(3))
    assert m["rank1"] > 0.9
    assert 0.0 < m["unknown_distance_at_far1"] < 1.0


def test_mask_center_blanks_only_the_middle():
    torch = pytest.importorskip("torch")
    from cowid.identity.train import MaskCenter

    t = MaskCenter(0.5)(torch.ones(3, 8, 8))
    assert float(t[:, 2:6, 2:6].abs().sum()) == 0.0
    assert float(t[:, 0, :].sum()) == 3 * 8


def test_shrink_images_keeps_folders_and_limits_size(tmp_path):
    from cowid.datasets import shrink_images

    src = tmp_path / "src" / "cow-1"
    src.mkdir(parents=True)
    cv2.imwrite(str(src / "a.jpg"), np.zeros((300, 600, 3), dtype=np.uint8))
    assert shrink_images(tmp_path / "src", tmp_path / "dst", 100, log=lambda m: None) == 1
    small = cv2.imread(str(tmp_path / "dst" / "cow-1" / "a.jpg"))
    assert max(small.shape[:2]) == 100
