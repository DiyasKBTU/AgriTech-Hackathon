"""Живой режим: голосование за номер, проверка картинки, чтение видео.

Проверяется код, а не качество: картинки здесь — однотонные кадры.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from collections import Counter, deque

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


def test_day_strip_merges_same_activity_and_trail_stays_in_one_camera():
    st = _CowStats()
    plan = ["лежит"] * 4 + ["стоит"] * 2
    for i, state in enumerate(plan):
        st.update(50400 + 15 * i, 60, state, None, step=15, where=(100 + i, 50, 1))
    st.update(50400 + 15 * 6, 60, "стоит", "feeder", step=15, where=(400, 60, 2))
    st.update(50400 + 15 * 20, 60, "стоит", "feeder", step=15, where=(410, 60, 2))   # пауза 3,5 мин
    assert [k for _, _, k in st.timeline] == ["лежит", "стоит", "у корма", "у корма"]
    assert st.timeline[0][:2] == [50400, 50460]       # лежит до смены занятия
    assert st.timeline[2][1] == 50400 + 15 * 6 + 15   # после паузы — новый отрезок
    assert [c for *_, c in st.trail] == [2]           # другая камера и пауза — путь заново


def test_rollcall_counts_only_shown_time_and_flags_long_absence():
    from types import SimpleNamespace

    from cowid.config import PipelineConfig
    from cowid.live import MISSING_MIN, FrameProcessor

    models = SimpleNamespace(registry={"C01": {}}, norms={}, zones=None,
                             gallery=SimpleNamespace(known_ids=lambda: ["C01", "C02", "C03"], size=lambda: 3))
    proc = FrameProcessor(PipelineConfig(), models, 15.0)
    proc.watched = (MISSING_MIN + 10) * 60
    proc.cows["C01"] = _CowStats(watched_at=proc.watched, last_t=50400)               # в кадре
    proc.cows["C02"] = _CowStats(watched_at=proc.watched - (MISSING_MIN + 5) * 60, last_t=48000)
    rc = proc.rollcall()
    by = {c["cow"]: c for c in rc["cows"]}
    assert rc["total"] == 3 and rc["missing"] == 2 and rc["in_frame"] == 1
    assert by["C01"]["away_min"] == 0 and not by["C01"]["missing"]
    assert by["C02"]["missing"] and by["C02"]["last"] == "13:20"
    assert by["C03"]["missing"] and by["C03"]["away_min"] is None                    # так и не появилась
    assert rc["cows"][0]["cow"] in ("C02", "C03")                                    # пропавшие — сверху


def test_one_number_goes_to_one_box_and_loser_takes_next_or_unknown():
    from cowid.identity.voting import assign_unique

    tallies = [Counter({"C01": 7, "C02": 1}), Counter({"C01": 4, "C03": 3}), Counter({"C01": 5})]
    totals = [8, 8, 8]
    # C01 хотят все три: достаётся самой уверенной, вторая берёт C03 (3 из 8 ≥ ⅓), третьей — «?».
    assert assign_unique(tallies, totals, min_votes=3, min_share=0.34) == ["C01", "C03", None]
    # При доле ½ запасной номер не проходит.
    assert assign_unique(tallies, totals, min_votes=3, min_share=0.5) == ["C01", None, None]
    assert assign_unique([Counter()], [2]) == [None]


def test_pose_gallery_compares_lying_with_lying_portraits():
    from cowid.config import GalleryConfig
    from cowid.identity.gallery import PoseGallery

    e = np.eye(4, dtype=np.float32)
    # C01 стоя похожа на ось 0, лёжа — на ось 1; C02 стоя — на ось 1.
    pg = PoseGallery(np.stack([e[0], e[1], e[1]]), np.array(["C01", "C01", "C02"]),
                     np.array([False, True, False]), GalleryConfig(unknown_distance=0.3))
    assert pg.match_many(np.stack([e[1], e[1], e[3]]), [True, False, False]) == ["C01", "C02", None]


def test_frame_labels_do_not_overlap_and_focus_draws():
    from cowid.render import _place, draw_tracks

    taken = []
    for _ in range(6):                                 # шесть коров в одной точке
        spot = _place((100, 100, 160, 140), 40, 16, (400, 300), taken)
        if spot:
            taken.append((spot[0], spot[1], spot[0] + 40, spot[1] + 16))
    for i, a in enumerate(taken):
        for b in taken[i + 1:]:
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
    frame = np.full((300, 400, 3), 128, np.uint8)
    rows = [{"cow": "C01", "state": "стоит", "votes": 3, "total": 3, "bbox": [10, 10, 60, 50]},
            {"cow": None, "state": "лежит", "votes": 0, "total": 2, "bbox": [200, 100, 260, 140]}]
    out = draw_tracks(frame, rows, focus={"cow": "C01", "trail": [(20, 20), (35, 30)], "note": ""})
    assert out.shape == frame.shape
    assert out[200, 350].mean() < 128                  # вне выбранной коровы кадр притушен


def test_explicit_alerts_need_her_own_limit_and_enough_watching():
    from cowid.live import explicit_alerts

    limits = {"feed_h": 1.6, "lying_h": 0.8, "own": True}          # с запасом ×1,25 — 2 ч и 1 ч
    st = _CowStats()
    for i in range(10 * 4):                          # 10 минут у корма — подход засчитан
        st.update(50400 + 15 * i, 60, "стоит", "feeder", step=15)
    t = 50400 + 15 * 40
    for i in range(int(2.5 * 3600 / 15)):            # потом 2,5 ч стоит в стороне, всё время на виду
        st.update(t + 15 * i, 60, "стоит", None, step=15)
    now = st.last_t
    texts = [a["text"] for a in explicit_alerts(st, now, limits)]
    assert texts and texts[0].startswith("не подходила к корму 2 ч 30 мин") and "самое долгое 1 ч 36 мин" in texts[0]
    # Ту же паузу камера видела урывками (меньше 60% времени) — «не ела» не утверждаем.
    st.seen_since_feeder = 0.3 * (now - st.feeder_last)
    assert explicit_alerts(st, now, limits) == []
    # Лежит подряд дольше своего предела.
    lie = _CowStats()
    for i in range(int(1.5 * 3600 / 15)):
        lie.update(60000 + 15 * i, 60, "лежит", None, step=15)
    assert any("лежит 1 ч 30 мин подряд" in a["text"] for a in explicit_alerts(lie, lie.last_t, limits))


def test_sensor_alerts_use_only_past_days_and_need_history():
    from cowid.mmcows import sensor_alerts

    def day(gap):
        w = {"feed_gap": [gap, 3600, 3600 + gap * 3600], "lying_bout": [1.0, 0, 3600]}
        return {"runs": [], "windows": {"0-14": w, "14-24": w}}

    gaps = [3.0, 3.1, 2.9, 3.0, 3.2, 4.5, 3.0]          # шестые сутки — на 40% дольше рекорда
    profiles = {"C01": {f"2023-07-{20 + i}": day(g) for i, g in enumerate(gaps)}}
    alerts = sensor_alerts(profiles)
    assert {a["day"] for a in alerts} == {"2023-07-25"}
    assert all(a["kind"] == "feed_gap" and a["record_h"] == 3.2 and a["history_days"] == 5 for a in alerts)
    # Меньше пяти прошлых суток — своего рекорда ещё нет, предупреждений нет.
    early = {"C01": {f"2023-07-{20 + i}": day(g) for i, g in enumerate([3.0, 9.0, 3.0])}}
    assert sensor_alerts(early) == []


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


def test_file_keeps_every_frame_and_jumps_over_empty_pen(tmp_path):
    """Медленная обработка не теряет кадры файла; отрезок без коров пропускается."""
    path = tmp_path / "day.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 50, (64, 48))
    for n in range(12):
        writer.write(np.full((48, 64, 3), 20 * n, dtype=np.uint8))
    writer.release()

    source = LiveSource("t", "t", "playlist", [str(path)], speed=10.0, skip=[(4, 8, "дойка")])
    reader = _Reader(source)
    reader.start()
    got, skipped = [], None
    deadline = time.time() + 5
    while time.time() < deadline and len(got) < 8:
        item = reader.take()
        if item:
            got.append(item)
            if item[1] == 8:
                skipped = list(reader.skipped)
            time.sleep(0.03)                       # обработка медленнее показа
        time.sleep(0.002)
    reader.stop_flag.set()
    reader.join(timeout=2)
    assert [idx for _, idx, _, _ in got] == [0, 1, 2, 3, 8, 9, 10, 11]
    assert abs(int(got[4][3].mean()) - 160) <= 3   # после прыжка — правда кадр 8
    assert got[4][0] != got[3][0] and skipped == ["дойка"]


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
