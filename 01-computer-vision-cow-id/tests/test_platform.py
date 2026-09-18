"""Платформа: база, API, сквозной прогон конвейера, загрузчик датасетов."""

from __future__ import annotations

import io
import random
import zipfile
from datetime import date, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from cowid.api.app import create_app
from cowid.config import PipelineConfig, StorageConfig, ZoneConfig
from cowid.pipeline import CowIdPipeline
from cowid.store.db import Store
from cowid.types import ActivityFeatures, BBox, Detection, Event

from .conftest import unit



# --------------------------------------------------------------------------
# База
# --------------------------------------------------------------------------

def _feat(cow, day, feeder=3600.0, observed=36000.0):
    return ActivityFeatures(cow_id=cow, day=day, feeder_seconds=feeder,
                            observed_seconds=observed, distance_m=2000)


def test_features_from_two_cameras_are_summed(tmp_db):
    """Одно животное попало в две камеры — сутки должны сложиться."""
    d = date(2026, 3, 1)
    tmp_db.save_features("cam-1", [_feat("A", d, feeder=1000, observed=10000)])
    tmp_db.save_features("cam-2", [_feat("A", d, feeder=500, observed=5000)])
    rows = tmp_db.features_before(d + timedelta(days=1))
    assert len(rows) == 1
    assert rows[0].feeder_seconds == 1500 and rows[0].observed_seconds == 15000


def test_same_event_is_not_stored_twice(tmp_db):
    e = Event(cow_id="A", day=date(2026, 3, 1), severity="alert", title="t", detail="d")
    assert tmp_db.save_events([e]) == 1
    assert tmp_db.save_events([e]) == 0


def test_feedback_changes_event_status(tmp_db):
    tmp_db.save_events([Event(cow_id="A", day=date(2026, 3, 1), severity="alert",
                              title="t", detail="d")])
    event_id = tmp_db.events()[0]["event_id"]
    tmp_db.add_feedback(event_id, "false_alarm", "корова в порядке")
    assert tmp_db.events()[0]["status"] == "false_alarm"
    assert tmp_db.feedback_summary() == {"false_alarm": 1}
    with pytest.raises(KeyError):
        tmp_db.add_feedback(9999, "confirmed")


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

def test_console_and_feedback_endpoint(tmp_path):
    cfg = PipelineConfig(storage=StorageConfig(database=str(tmp_path / "db.sqlite"),
                                               evidence_dir=str(tmp_path / "ev")))
    store = Store(cfg.storage.database)
    store.save_events([Event(cow_id="KZ-1", day=date(2026, 3, 1), severity="alert",
                             title="Падение кормления", detail="подробности")])
    client = TestClient(create_app(cfg))

    page = client.get("/")
    assert page.status_code == 200 and "COW ID" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/api/events").json()[0]["cow_id"] == "KZ-1"

    event_id = client.get("/api/events").json()[0]["event_id"]
    ok = client.post(f"/api/events/{event_id}/feedback", json={"verdict": "confirmed"})
    assert ok.status_code == 200
    assert client.get("/api/events").json()[0]["status"] == "confirmed"
    assert client.post("/api/events/999/feedback", json={"verdict": "confirmed"}).status_code == 404


def test_herd_and_cow_pages_show_personal_norm(tmp_path):
    """Восемь спокойных суток и одни с резким падением кормления: стадо и
    карточка коровы должны показать её норму и отклонение от неё."""
    cfg = PipelineConfig(storage=StorageConfig(database=str(tmp_path / "db.sqlite"),
                                               evidence_dir=str(tmp_path / "ev")))
    store = Store(cfg.storage.database)
    start = date(2026, 3, 1)
    for i in range(9):
        feeder = 3600.0 + (i % 3) * 60 if i < 8 else 1200.0
        store.save_features("cam", [_feat("KZ-7", start + timedelta(days=i), feeder=feeder)])
    client = TestClient(create_app(cfg))

    herd = client.get("/api/herd").json()
    row = herd["cows"][0]
    assert row["cow_id"] == "KZ-7" and row["day"] == "2026-03-09"
    assert row["indicators"]["feeder_share"]["delta_pct"] < -50
    assert any(i["status"] == "не сделано" for i in herd["indicators"])
    # Признак словами и доска «у кого что»: падение кормления — «Ест меньше», сильно.
    assert row["signs"][0]["name"] == "Ест меньше" and row["signs"][0]["level"] == "strong"
    eats_less = next(b for b in herd["board"] if b["name"] == "Ест меньше")
    assert [h["cow_id"] for h in eats_less["strong"]] == ["KZ-7"]
    assert {i["name"] for i in herd["not_measured"]} >= {"Хромает (по походке)", "Исхудала"}
    assert herd["board"][0]["name"] == "Возможна хромота"
    # Любые сутки из истории; сутки без данных — 404.
    early = client.get("/api/herd?day=2026-03-02").json()
    assert early["day"] == "2026-03-02" and early["cows"][0]["status"] == "learning"
    assert client.get("/api/herd?day=2025-01-01").status_code == 404

    cow = client.get("/api/cows/KZ-7").json()
    assert len(cow["days"]) == 9 and cow["days"][0]["status"] == "learning"
    assert client.get("/api/cows/NOPE").status_code == 404
    assert client.get("/api/live/state").json()["running"] is False


# --------------------------------------------------------------------------
# Сквозной прогон: проверяем «трубы», а не качество
# --------------------------------------------------------------------------

class _BoxDetector:
    """Подставной детектор: находит светлый прямоугольник на тёмном кадре."""

    def detect(self, frame, frame_idx):
        mask = (frame[:, :, 0] > 200).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return []
        return [Detection(BBox(xs.min(), ys.min(), xs.max(), ys.max()), 0.9, frame_idx)]


class _ConstEmbedder:
    def embed(self, frame, bbox, corners=None):
        return unit([1, 0, 0])


class _TagReader:
    def read(self, frame, bbox):
        from cowid.identity.tag_ocr import TagRead

        return TagRead(text="1001", confidence=0.9)


def test_video_goes_all_the_way_to_the_database(tmp_path):
    """Видео -> трек -> номер -> показатели -> база. Картинка здесь — просто
    движущийся прямоугольник: тест проверяет связку модулей, а не зрение."""
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for f in range(60):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        x = 20 + f * 2
        cv2.rectangle(frame, (x, 30), (x + 60, 70), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()

    cfg = PipelineConfig(
        storage=StorageConfig(database=str(tmp_path / "db.sqlite"),
                              evidence_dir=str(tmp_path / "ev")),
        zones=[ZoneConfig(name="feeder", polygon=[(0, 0), (320, 0), (320, 120), (0, 120)])],
    )
    cfg.video.frame_stride = 2
    store = Store(cfg.storage.database)
    pipeline = CowIdPipeline(cfg, store=store, detector=_BoxDetector(),
                             embedder=_ConstEmbedder(), tag_reader=_TagReader(),
                             gallery_path=tmp_path / "gallery.json")

    result = pipeline.process_video(video, date(2026, 3, 1))

    assert result.frames_processed == 30
    assert result.identified == 1
    assert [a["cow_id"] for a in store.animals()] == ["KZ-1001"]
    history = store.animal_history("KZ-1001")
    assert history[0]["feeder_seconds"] > 0
    assert (tmp_path / "gallery.json").exists()
    assert store.runs()[0]["frames"] == 30


# --------------------------------------------------------------------------
# Загрузчик датасетов
# --------------------------------------------------------------------------

def test_download_resumes_after_network_failure(tmp_path, monkeypatch):
    """Обрыв посередине, повторный запуск — докачивается только недостающее,
    и файл совпадает с оригиналом байт в байт."""
    from cowid import datasets as d

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        z.writestr("cow.bin", random.Random(0).randbytes(3 * 1024 * 1024))
    remote = buf.getvalue()

    monkeypatch.setattr(d, "CHUNK", 256 * 1024)
    monkeypatch.setattr(d, "TARGET_DIR", tmp_path)
    monkeypatch.setattr(d, "DATASETS", {"t": d.Remote("t.zip", "http://fake/t.zip")})
    monkeypatch.setattr(d, "remote_size", lambda url: len(remote))
    monkeypatch.setattr(d.time, "sleep", lambda s: None)

    state = {"calls": 0, "fail_after": 4}

    class Resp:
        status = 206

        def __init__(self, data):
            self.data = data

        def read(self):
            return self.data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0):
        state["calls"] += 1
        if state["fail_after"] is not None and state["calls"] > state["fail_after"]:
            raise OSError("обрыв сети")
        a, b = req.headers["Range"].split("=")[1].split("-")
        return Resp(remote[int(a):int(b) + 1])

    monkeypatch.setattr(d.urllib.request, "urlopen", fake_urlopen)

    assert d.download("t", workers=2, log=lambda m: None) == 1          # обрыв
    saved = len(d.Journal(tmp_path / "t.zip.progress.json", 0).done)
    assert 0 < saved < len(remote) // d.CHUNK + 1

    state.update(calls=0, fail_after=None)
    assert d.download("t", workers=2, log=lambda m: None) == 0          # продолжение
    total = len(remote) // d.CHUNK + 1
    assert state["calls"] == total - saved
    assert (tmp_path / "t.zip").read_bytes() == remote


def test_live_frame_endpoint_and_phone_hint(tmp_path):
    cfg = PipelineConfig(storage=StorageConfig(database=str(tmp_path / "db.sqlite"),
                                               evidence_dir=str(tmp_path / "ev")))
    client = TestClient(create_app(cfg))
    # Камера браузера не включена — кадр не принимается.
    assert client.post("/api/live/frame", content=b"\xff\xd8",
                       headers={"Content-Type": "image/jpeg"}).status_code == 409
    # Без `cowid serve --phone` адреса для телефона не показываются.
    assert client.get("/api/live/phone").json()["enabled"] is False
    kinds = {s["kind"] for s in client.get("/api/live/sources").json()}
    assert {"browser", "camera", "url"} <= kinds
