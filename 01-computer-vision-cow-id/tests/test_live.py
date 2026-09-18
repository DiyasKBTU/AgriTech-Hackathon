"""Живой режим: голосование за номер, проверка картинки, чтение видео.

Проверяется код, а не качество: картинки здесь — однотонные кадры.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from collections import deque

from cowid.live import LiveEngine, LiveSource, _CowStats, _Reader, _TrackInfo, cow_signs


def test_number_is_shown_only_after_enough_agreeing_votes():
    info = _TrackInfo()
    info.votes.extend(["078", "078"])
    assert info.identity(0.5)[0] is None           # меньше трёх голосов — «?»
    info.votes.extend(["078", None])
    assert info.identity(0.5) == ("078", 3, 4)
    info.votes.extend(["101", "101", "101", None, None, None])
    assert info.identity(0.5)[0] is None           # голоса разошлись — «?»


def test_vote_window_forgets_old_votes_after_track_switch():
    """Трекер перескочил на соседку: при окне в 4 голоса номер меняется, без окна — нет."""
    windowed = _TrackInfo(votes=deque(maxlen=4))
    endless = _TrackInfo()
    for info in (windowed, endless):
        info.votes.extend(["C01"] * 6 + ["C02"] * 4)
    assert windowed.identity(0.5)[0] == "C02"
    assert endless.identity(0.5)[0] == "C01"


def _stats(minutes: int, lying: float, feeder: float) -> _CowStats:
    st = _CowStats()
    for i in range(minutes * 4):                   # кадр раз в 15 с
        t = 14 * 3600 + i * 15
        state = "лежит" if i < minutes * 4 * lying else "стоит"
        zone = "feeder" if (i % 100) < 100 * feeder and state != "лежит" else None
        st.update(t, 60, state, zone)
    return st


def test_cow_signs_speak_in_farm_words_and_use_registry():
    norm = {"lying": 0.2, "feeder": 0.3, "drinker": 0.02}
    lame_record = {"as_of": "2023-07-25",
                   "last_leg_problem": {"day": "2023-07-02", "what": "хромота", "detail": ""}}
    st = _stats(90, lying=0.6, feeder=0.0)
    signs = cow_signs(st, norm, lame_record, now=st.last_t)
    assert signs[0]["level"] == "strong" and "возможна хромота" in signs[0]["text"]
    assert "хромота 02.07" in signs[0]["text"]
    assert any("лежит больше обычного" in s["text"] for s in signs)
    # Та же корова, но наблюдали 10 минут, — выводов по долям нет.
    short = _stats(10, lying=0.6, feeder=0.0)
    assert not [s for s in cow_signs(short, norm, lame_record, short.last_t) if "лежит больше" in s["text"]]
    # Обычная корова — без признаков.
    calm = _stats(90, lying=0.2, feeder=0.3)
    assert cow_signs(calm, norm, None, calm.last_t) == []


def test_camera_health_flags_dark_and_blurry_frames():
    dark = np.full((240, 320, 3), 10, dtype=np.uint8)
    flat = np.full((240, 320, 3), 128, dtype=np.uint8)
    sharp = np.zeros((240, 320, 3), dtype=np.uint8)
    sharp[::2] = 255
    assert "темно" in LiveEngine._health(dark)["problems"]
    assert any("размыто" in p for p in LiveEngine._health(flat)["problems"])
    assert LiveEngine._health(sharp)["problems"] == []


def test_playlist_reader_starts_new_segment_for_each_file(tmp_path):
    paths = []
    for n in range(2):
        path = tmp_path / f"clip{n}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 50, (64, 48))
        for _ in range(5):
            writer.write(np.full((48, 64, 3), 100 + n * 50, dtype=np.uint8))
        writer.release()
        paths.append(str(path))

    reader = _Reader(LiveSource("t", "t", "playlist", paths, speed=10.0))
    reader.start()
    segments = set()
    deadline = time.time() + 5
    while time.time() < deadline and len(segments) < 2:
        item = reader.take()
        if item:
            segments.add(item[0])
        time.sleep(0.002)
    reader.stop_flag.set()
    reader.join(timeout=2)
    assert len(segments) >= 2


# --------------------------------------------------------------------------
# Камера браузера (телефон или ноутбук)
# --------------------------------------------------------------------------

def _jpeg(value: int = 120) -> bytes:
    frame = np.full((90, 160, 3), value, np.uint8)
    return cv2.imencode(".jpg", frame)[1].tobytes()


def test_browser_frames_are_decoded_and_time_follows_the_clock():
    from cowid.live import _PushReader

    reader = _PushReader(LiveSource("browser", "t", "browser", None))
    assert reader.error                                     # до первого кадра — «ждём кадры»
    assert reader.push(_jpeg())["frame"] == [160, 90]
    seg, idx, fps, frame = reader.take()
    assert frame.shape == (90, 160, 3) and reader.take() is None and reader.error == ""
    time.sleep(0.25)
    reader.push(_jpeg())
    later = reader.take()[1]
    assert later - idx >= 2                                 # 0,25 с при 10 кадрах/с — номер по часам
    try:
        reader.push(b"not an image")
    except ValueError:
        pass
    else:
        raise AssertionError("мусор вместо кадра должен отклоняться")


def test_browser_source_accepts_frames_only_while_it_is_on(monkeypatch):
    from cowid.config import PipelineConfig

    engine = LiveEngine()
    monkeypatch.setattr(LiveEngine, "models_for", lambda self, source: (PipelineConfig(), None))
    monkeypatch.setattr(LiveEngine, "_loop", lambda self, source, cfg, models: None)
    try:
        engine.push_frame(_jpeg())
    except RuntimeError:
        pass
    else:
        raise AssertionError("без включённой камеры браузера кадры не принимаются")
    state = engine.start(source_id="browser", view="side")
    assert state["source"]["kind"] == "browser" and state["source"]["view"] == "side"
    assert engine.push_frame(_jpeg())["ok"]
    engine.stop()
