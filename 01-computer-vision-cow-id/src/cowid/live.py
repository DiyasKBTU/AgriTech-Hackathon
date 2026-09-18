"""Живой просмотр: камера или демо-видео → рамки, номера и индикаторы сразу.

Суточная обработка (`cowid process`) остаётся основным режимом фермы: ночью,
по записи. Живой режим нужен, чтобы при установке камеры и на показе увидеть,
что система видит прямо сейчас:

* подключена ли камера, не темно ли, не размыт ли кадр;
* сколько коров в кадре и кого из них система узнала;
* что корова делает: идёт или стоит, сколько секунд в кадре, в какой зоне.

Источники:
* камера телефона или ноутбука через браузер: страница «Камера» снимает кадры
  и отправляет их на сервер (`POST /api/live/frame`). На ноутбуке работает
  сразу, телефону браузер даёт камеру только по https — `cowid serve --phone`;
* камера компьютера с сервером по номеру (встроенная — 0, USB — 1, 2…);
* IP-камера или телефон с приложением по ссылке — RTSP (`rtsp://...`) или
  HTTP-поток (IP Webcam на Android: `http://<адрес телефона>:8080/video`);
* демо-ролики из датасетов — по кругу.

Как стоит камера: «сверху» — наш детектор и галерея Cows2021; «сбоку» —
готовая модель COCO (корова — класс 19), номера не ставятся.

Номер коровы в живом режиме — это голосование кадров текущей дорожки:
пока проголосовало меньше трёх кадров или голоса разошлись, показывается «?».
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from .config import PipelineConfig

COWS2021_VIDEOS = Path(
    "data/real/cows2021/4vnrca7qw1642qlwxjadp87h7/Sub-levels/Identification/Videos"
)

#: Ролик MmCows с 14:00 (кадр 3360) до конца суток, вырезан из полного без
#: пережатия — кадры те же. Его привозит архив для другого компьютера.
MMCOWS_FROM_14 = Path("data/real/mmcows_0725_from14.mp4")
MMCOWS_FIRST_SHOWN = 3360

#: Ролики Cows2021, где все коровы — из тех, на которых училась модель.
KNOWN_CLIPS = [
    "2020-03-08_13-36-33", "2020-03-08_14-45-1", "2020-03-09_12-19-46",
    "2020-03-09_13-22-21", "2020-03-09_13-57-29", "2020-03-11_13-17-22",
]
#: Ролики, где узнана корова, которую модель при обучении не видела
#: (она только зарегистрирована по февральским фото). Отобраны по прогону
#: `cowid check-video`, часть сверена глазами.
UNSEEN_CLIPS = [
    "2020-03-10_12-37-58", "2020-03-10_12-53-58", "2020-03-08_13-1-24",
    "2020-03-11_12-33-8", "2020-03-11_13-23-36", "2020-03-09_12-51-24",
]

#: Скорость центра, выше которой корова считается идущей, в долях ширины кадра
#: в секунду. Идущая корова пересекает кадр прохода за 3–6 секунд; дрожание
#: рамки у стоящей — сотые доли.
MOVING_SHARE_PER_S = 0.04
MIN_VOTES = 3
STREAM_WIDTH = 960


@dataclass
class LiveSource:
    id: str
    title: str
    kind: str                         # "playlist" | "camera" | "url" | "browser"
    target: Union[list[str], int, str, None]
    note: str = ""
    config: str = "configs/cows2021.yaml"
    #: Во сколько раз быстрее реального времени проигрывать файл.
    speed: float = 1.0
    #: С какого кадра начать первый проход по файлу.
    start_frame: int = 0
    #: Номер первого кадра файла в полных сутках (у вырезанного ролика — не 0).
    first_index: int = 0
    #: Сколько реальных секунд между кадрами, если видео снято с прореживанием
    #: (MmCows — кадр раз в 15 с). Тогда же показывается время суток.
    real_seconds_per_frame: Optional[float] = None
    #: Для камер: "top" — сверху (наш детектор), "side" — сбоку (COCO).
    view: str = ""

    def as_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "kind": self.kind, "note": self.note,
                "view": self.view}


def _clips(names: list[str]) -> list[str]:
    return [str(COWS2021_VIDEOS / n / "RGB.avi") for n in names
            if (COWS2021_VIDEOS / n / "RGB.avi").exists()]


def available_sources() -> list[LiveSource]:
    out: list[LiveSource] = []
    from .datasets import archive_path, is_complete

    mmcows = archive_path("mmcows_video_0725")
    # С 14:00: утренние кадры детектор видел при дообучении.
    if mmcows.exists() and is_complete("mmcows_video_0725"):
        video = {"target": [str(mmcows)], "start_frame": MMCOWS_FIRST_SHOWN}
    elif MMCOWS_FROM_14.exists():
        video = {"target": [str(MMCOWS_FROM_14)], "first_index": MMCOWS_FIRST_SHOWN}
    else:
        video = None
    if video:
        out.append(LiveSource(
            "demo_other_farm", "Демо: ферма MmCows — кто это и что это значит", "playlist",
            note="США, 4 камеры под наклоном, кадр раз в 15 с (сутки за 24 минуты), показ с 14:00. "
            "Детектор и узнавание дообучены на утренних кадрах этой фермы; 16 коров "
            "зарегистрированы, к каждой подтягивается карточка из реестра фермы.",
            config="configs/mmcows.yaml", speed=4.0, real_seconds_per_frame=15.0, **video))
    known = _clips(KNOWN_CLIPS)
    if known:
        out.append(LiveSource(
            "demo_known", "Демо: знакомые коровы (Cows2021)", "playlist", known,
            "Проход после дойки, камера сверху. Модель училась на этих коровах "
            "(по февральским фото); ролики — март, их модель не видела."))
    unseen = _clips(UNSEEN_CLIPS)
    if unseen:
        out.append(LiveSource(
            "demo_unseen", "Демо: коровы, которых модель не видела", "playlist", unseen,
            "Эти коровы не участвовали в обучении модели. Они только зарегистрированы "
            "по фото за месяц до записи — так будет с новой партией на ферме."))
    out.append(LiveSource(
        "browser", "Камера телефона или ноутбука", "browser", None,
        "Снимает браузер: на ноутбуке — встроенная камера, на телефоне — задняя. "
        "Телефон — в одной сети с компьютером (Wi-Fi или точка доступа телефона)."))
    out.append(LiveSource(
        "camera0", "Камера компьютера по номеру", "camera", 0,
        "Камера, подключённая к компьютеру с сервером: 0 — встроенная, 1 — USB."))
    out.append(LiveSource(
        "url", "IP-камера или телефон по ссылке", "url", "",
        "RTSP-камера коровника (rtsp://…) или приложение на телефоне, "
        "например IP Webcam: http://адрес-телефона:8080/video."))
    return out


# --------------------------------------------------------------------------
# Чтение кадров
# --------------------------------------------------------------------------

class _Reader(threading.Thread):
    """Читает источник в своём потоке и держит только самый свежий кадр.

    Обработка медленнее видео — старые кадры не копятся, берётся последний.
    Файлы проигрываются в реальном времени и по кругу; каждый новый файл —
    новый «отрезок», на котором трекер начинает заново.
    """

    def __init__(self, source: LiveSource):
        super().__init__(daemon=True)
        self.source = source
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()
        self.latest: Optional[tuple[int, int, float, np.ndarray]] = None
        self.segment = 0
        self.fps = 0.0
        self.error = ""
        self.clip_name = ""

    def _open(self, target) -> cv2.VideoCapture:
        if isinstance(target, int):
            cap = cv2.VideoCapture(target, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(target)
            return cap
        return cv2.VideoCapture(target)

    def _publish(self, idx: int, fps: float, frame: np.ndarray) -> None:
        with self.lock:
            self.latest = (self.segment, idx, fps, frame)

    def run(self) -> None:
        if self.source.kind == "playlist":
            self._run_playlist()
        else:
            self._run_stream()

    def _run_playlist(self) -> None:
        files = list(self.source.target)
        first = True
        while not self.stop_flag.is_set():
            for path in files:
                if self.stop_flag.is_set():
                    return
                cap = self._open(path)
                if not cap.isOpened():
                    self.error = f"не открывается {path}"
                    continue
                self.error = ""
                self.clip_name = Path(path).parent.name if Path(path).stem == "RGB" \
                    else Path(path).stem
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                self.fps = fps
                self.segment += 1
                play_fps = fps * self.source.speed
                idx = self.source.first_index
                if first and self.source.start_frame:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, self.source.start_frame)
                    idx += self.source.start_frame
                first = False
                start_idx = idx
                started = time.perf_counter()
                while not self.stop_flag.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    self._publish(idx, fps, frame)
                    idx += 1
                    # Воспроизведение в реальном времени (или с ускорением).
                    wait = started + (idx - start_idx) / play_fps - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                cap.release()

    def _run_stream(self) -> None:
        while not self.stop_flag.is_set():
            cap = self._open(self.source.target)
            if not cap.isOpened():
                self.error = "источник не открывается — проверьте номер камеры или ссылку"
                time.sleep(2.0)
                continue
            self.error = ""
            self.segment += 1
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.fps = fps
            idx = 0
            while not self.stop_flag.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.error = "поток прервался — переподключение"
                    break
                self._publish(idx, fps, frame)
                idx += 1
            cap.release()
            time.sleep(1.0)

    def take(self) -> Optional[tuple[int, int, float, np.ndarray]]:
        with self.lock:
            item, self.latest = self.latest, None
        return item


class _PushReader:
    """Кадры присылает браузер (камера телефона или ноутбука).

    Тот же интерфейс, что у `_Reader`, но без своего потока: кадр кладётся
    вызовом `push`. Номер кадра считается по часам при условной частоте
    PUSH_FPS — тогда время в кадре совпадает с реальным, даже если браузер
    присылает кадры неровно.
    """

    PUSH_FPS = 10.0
    MAX_BYTES = 5 * 1024 * 1024

    def __init__(self, source: LiveSource):
        self.source = source
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()
        self.latest: Optional[tuple[int, int, float, np.ndarray]] = None
        self.segment = 1
        self.fps = 0.0
        self.error = "ждём кадры из браузера — разрешите доступ к камере"
        self.clip_name = ""
        self._started: Optional[float] = None
        self._idx = -1
        self._times: deque = deque(maxlen=20)

    def start(self) -> None:
        pass

    def push(self, data: bytes) -> dict:
        if len(data) > self.MAX_BYTES:
            raise ValueError("кадр больше 5 МБ")
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("не JPEG/PNG")
        now = time.perf_counter()
        if self._started is None:
            self._started = now
        self._idx = max(self._idx + 1, int(round((now - self._started) * self.PUSH_FPS)))
        self._times.append(now)
        if len(self._times) > 1:
            self.fps = (len(self._times) - 1) / max(self._times[-1] - self._times[0], 1e-6)
        self.error = ""
        with self.lock:
            self.latest = (self.segment, self._idx, self.PUSH_FPS, frame)
        return {"ok": True, "frame": [frame.shape[1], frame.shape[0]]}

    def take(self) -> Optional[tuple[int, int, float, np.ndarray]]:
        with self.lock:
            item, self.latest = self.latest, None
        return item


# --------------------------------------------------------------------------
# Обработка
# --------------------------------------------------------------------------

POSES = {"standing": "стоит", "lying": "лежит"}


@dataclass
class _TrackInfo:
    votes: deque = field(default_factory=deque)
    seen_embeddings: int = 0
    first_t: float = 0.0
    last_t: float = 0.0
    centers: deque = field(default_factory=lambda: deque(maxlen=12))
    poses: deque = field(default_factory=lambda: deque(maxlen=5))
    moving_s: float = 0.0
    still_s: float = 0.0
    lying_s: float = 0.0
    zone_s: Counter = field(default_factory=Counter)

    def identity(self, vote_ratio: float) -> tuple[Optional[str], int, int]:
        tally = Counter(self.votes)
        total = len(self.votes)
        named = [(k, v) for k, v in tally.most_common() if k is not None]
        if not named:
            return None, 0, total
        cow, n = named[0]
        if n < MIN_VOTES or n / max(total, 1) < vote_ratio:
            return None, n, total
        return cow, n, total


class _Models:
    """Модели загружаются один раз на файл настроек."""

    def __init__(self, cfg: PipelineConfig):
        from .detect.detectors import build_detector
        from .identity.embedder import build_embedder
        from .identity.gallery import BiometricGallery

        self.detector = build_detector(cfg.detector)
        self.embedder = build_embedder(cfg.embedder)
        gallery_path = Path("var") / f"gallery_{cfg.camera_id}.json"
        self.gallery = (BiometricGallery.load(gallery_path, cfg.gallery)
                        if gallery_path.exists() else BiometricGallery(cfg.gallery))
        self.zones = None
        if cfg.zone_masks and Path(cfg.zone_masks).exists():
            from .activity.zones import MaskZones

            self.zones = MaskZones(cfg.zone_masks, tiles=max(1, cfg.detector.tiles))
        self.registry: dict[str, dict] = {}
        if cfg.farm.registry == "mmcows" and cfg.farm.as_of:
            from datetime import date

            from .registry import registry

            self.registry = registry(date.fromisoformat(cfg.farm.as_of))
        self.norms: dict[str, dict] = {}
        if cfg.farm.hourly_norms == "mmcows":
            from .mmcows import load_hourly_norms

            self.norms = load_hourly_norms()



ZONE_RU = {"feeder": "у корма", "drinker": "у поилки"}
#: Сколько корова должна быть на виду, чтобы сравнивать её с обычными часами.
MIN_SEEN_S = 20 * 60


@dataclass
class _CowStats:
    """Что делала узнанная корова за сеанс — по часам суток фермы."""

    seen: Counter = field(default_factory=Counter)
    lying: Counter = field(default_factory=Counter)
    feeder: Counter = field(default_factory=Counter)
    drinker: Counter = field(default_factory=Counter)
    last_t: float = -1e9
    state: str = ""
    zone: Optional[str] = None
    bout_start: Optional[float] = None
    longest_bout: float = 0.0

    def update(self, t: float, max_gap: float, state: str, zone: Optional[str]) -> None:
        gap = t - self.last_t
        dt = gap if 0 < gap <= max_gap else 0.0
        hour = int(t // 3600) % 24
        if dt:
            self.seen[hour] += dt
            self.lying[hour] += dt if state == "лежит" else 0.0
            self.feeder[hour] += dt if zone == "feeder" else 0.0
            self.drinker[hour] += dt if zone == "drinker" else 0.0
        if state == "лежит":
            if self.bout_start is None or not dt:
                self.bout_start = t
            self.longest_bout = max(self.longest_bout, t - self.bout_start)
        else:
            self.bout_start = None
        self.last_t, self.state, self.zone = t, state, zone

    def share(self, kind: str) -> Optional[float]:
        total = sum(self.seen.values())
        return sum(getattr(self, kind).values()) / total if total else None

    def expected(self, hourly: dict, kind: str) -> Optional[float]:
        """Её обычная доля за те же часы суток, что корова была на виду.

        Норма часа сглажена по соседним часам: медиана одного часа по 13 суткам
        шумит (дойка, раздача корма сдвигаются на полчаса).
        """
        values = hourly.get(kind) or []
        num = den = 0.0
        for hour, secs in self.seen.items():
            near = [values[(hour + d) % 24] for d in (-1, 0, 1)
                    if len(values) == 24 and values[(hour + d) % 24] is not None]
            if near:
                num += secs * sum(near) / len(near)
                den += secs
        return num / den if den else None


def _pct(v: Optional[float]) -> Optional[int]:
    return None if v is None else int(round(100 * v))


def cow_signs(stats: _CowStats, norm: dict, record: Optional[dict], now: float,
              own: bool = True) -> list[dict]:
    """Что значит поведение коровы — словами зоотехника.

    Лежит или стоит — само по себе ничего не говорит. Говорит отличие от её
    обычных часов и то, что о ней известно (недавняя хромота из реестра).
    Хромоту камера здесь не видит напрямую: это косвенный признак по поведению.
    `own` — норма своя (обычные часы коровы); иначе — сравнение со стадом.
    """
    than = "обычного" if own else "стада"
    signs = []
    seen = sum(stats.seen.values())
    bout = (now - stats.bout_start) if stats.bout_start is not None else 0.0
    if bout >= 120 * 60:
        signs.append({"level": "notable", "text": f"лежит без перерыва {bout / 60:.0f} мин"})
    if seen < MIN_SEEN_S:
        return signs
    lie, lie_n = stats.share("lying"), norm.get("lying")
    feed, feed_n = stats.share("feeder"), norm.get("feeder")
    water, water_n = stats.share("drinker"), norm.get("drinker")
    lies_more = False
    if lie is not None and lie_n is not None:
        diff = lie - lie_n
        if diff >= 0.15:
            lies_more = True
            signs.append({"level": "strong" if diff >= 0.25 else "notable",
                          "text": f"лежит больше {than}: {lie:.0%} времени против {lie_n:.0%}"})
        elif diff <= -0.30:
            signs.append({"level": "info",
                          "text": f"почти не ложится: {lie:.0%} времени против {lie_n:.0%}"})
    eats_less = False
    # Корова ест подходами: за полчаса «мало у корма» ничего не значит.
    if (feed is not None and feed_n is not None and seen >= 60 * 60
            and feed_n >= 0.10 and feed <= 0.5 * feed_n):
        eats_less = True
        signs.append({"level": "notable",
                      "text": f"мало у кормового стола: {feed:.0%} против {feed_n:.0%}"})
    if water is not None and water_n is not None and water >= 0.05 and water >= 2 * water_n:
        signs.append({"level": "info", "text": f"часто у поилки: {water:.0%} против {water_n:.0%}"})
    leg = (record or {}).get("last_leg_problem")
    recent_leg = bool(leg) and _days_between(leg["day"], (record or {}).get("as_of")) <= 60
    if lies_more and (eats_less or bout >= 90 * 60 or recent_leg):
        why = [f"лежит больше {than}"]
        if eats_less:
            why.append("мало у корма")
        if bout >= 90 * 60:
            why.append(f"лежит без перерыва {bout / 60:.0f} мин")
        if recent_leg:
            why.append(f"в реестре {leg['what']} {leg['day'][8:10]}.{leg['day'][5:7]}")
        signs.insert(0, {"level": "strong",
                         "text": "возможна хромота — осмотреть ноги (" + ", ".join(why) + ")"})
    return signs


def _days_between(earlier: str, later: Optional[str]) -> int:
    from datetime import date

    if not later:
        return 10 ** 6
    return (date.fromisoformat(later) - date.fromisoformat(earlier)).days


def _registry_brief(record: Optional[dict]) -> Optional[dict]:
    if not record:
        return None
    return {
        "as_of": record.get("as_of"),
        "summary": record.get("summary"), "eid": record.get("eid"), "pen": record.get("pen"),
        "milk_kg": record.get("milk_kg"), "last_leg_problem": record.get("last_leg_problem"),
        "health": record.get("health", [])[:4],
    }


def cow_row(cow: str, stats: _CowStats, hourly: Optional[dict], herd: dict,
            record: Optional[dict], now: float) -> dict:
    if hourly:
        norm = {k: stats.expected(hourly, k) for k in ("lying", "feeder", "drinker")}
        source = "её обычные часы (датчики, 13 суток)"
    else:
        norm, source = herd, "стадо в эти же часы"
    signs = cow_signs(stats, norm, record, now, own=bool(hourly))
    bout = (now - stats.bout_start) if stats.bout_start is not None else 0.0
    return {
        "cow": cow, "state": stats.state, "zone": ZONE_RU.get(stats.zone, stats.zone),
        "seen_min": round(sum(stats.seen.values()) / 60),
        "lying_pct": _pct(stats.share("lying")), "lying_norm_pct": _pct(norm.get("lying")),
        "feeder_pct": _pct(stats.share("feeder")), "feeder_norm_pct": _pct(norm.get("feeder")),
        "drinker_pct": _pct(stats.share("drinker")), "drinker_norm_pct": _pct(norm.get("drinker")),
        "bout_min": round(bout / 60), "longest_bout_min": round(stats.longest_bout / 60),
        "norm_source": source, "norm_own": bool(hourly), "signs": signs,
        "registry": _registry_brief(record),
    }


class FrameProcessor:
    """Кадр → коровы в кадре: номер по голосам дорожки, поза или движение, зона.

    Общий для живого режима и для демо-роликов (`cowid demo-video`), чтобы
    ролик показывал ровно то, что считает платформа.
    """

    def __init__(self, cfg: PipelineConfig, models: _Models,
                 real_seconds_per_frame: Optional[float] = None):
        from .activity.zones import build_zones

        self.cfg = cfg
        self.models = models
        self.real_spf = real_seconds_per_frame
        self.zones = build_zones(cfg.zones)
        self.session: dict[str, float] = {}
        self.cows: dict[str, _CowStats] = {}
        self.now = 0.0
        self.reset()

    def reset(self) -> None:
        """Новый отрезок видео: дорожки начинаются заново, журнал сеанса остаётся."""
        from .track.tracker import CowTracker

        self.tracker = CowTracker(self.cfg.tracker)
        self.tracks: dict[int, _TrackInfo] = {}

    def video_time(self, idx: int, fps: float) -> float:
        return idx * self.real_spf if self.real_spf else idx / max(fps, 1e-6)

    def process(self, frame: np.ndarray, idx: int, fps: float) -> list[dict]:
        models = self.models
        t_video = self.video_time(idx, fps)
        detections = models.detector.detect(frame, idx)
        # Галерея пуста — узнавать некого; отпечатки не считаем, это самое дорогое.
        embeddings = ([models.embedder.embed(frame, d.bbox, d.corners) for d in detections]
                      if models.gallery.size() else None)
        active = self.tracker.update(detections, idx, embeddings)
        for done in self.tracker.drain_finished():
            self.tracks.pop(done.track_id, None)

        w = frame.shape[1]
        rows = []
        for t in active:
            info = self.tracks.get(t.track_id)
            if info is None:
                info = _TrackInfo(first_t=t_video, last_t=t_video,
                                  votes=deque(maxlen=self.cfg.gallery.vote_window or None))
                self.tracks[t.track_id] = info
            for emb in t.embeddings[info.seen_embeddings:]:
                match = models.gallery.match(emb)
                info.votes.append(match.cow_id if match.is_known else None)
            info.seen_embeddings = len(t.embeddings)
            obs = t.observations[-1]
            if obs.frame_idx != idx:
                continue                       # трек сейчас держится на предсказании
            dt = max(0.0, t_video - info.last_t)
            info.last_t = t_video
            cx, cy = obs.bbox.center
            info.centers.append((t_video, cx, cy))
            speed = self._speed(info.centers) / w
            moving = speed > MOVING_SHARE_PER_S
            if obs.label in POSES:
                # Детектор различает позу — она важнее скорости: при кадре
                # раз в 15 секунд скорость ничего не говорит.
                info.poses.append(obs.label)
                pose = Counter(info.poses).most_common(1)[0][0]
                state = POSES[pose]
                if pose == "lying":
                    info.lying_s += dt
                else:
                    info.still_s += dt
            else:
                state = "идёт" if moving else "стоит"
                if moving:
                    info.moving_s += dt
                else:
                    info.still_s += dt
            zone = next((z.name for z in self.zones if z.contains((cx, cy))), None)
            if zone is None and models.zones is not None:
                zone = models.zones.zone_at((cx, cy), frame.shape)
            if zone:
                info.zone_s[zone] += dt
            cow, votes, total = info.identity(self.cfg.gallery.vote_ratio)
            if cow:
                self.session[cow] = self.session.get(cow, 0.0) + dt
            rows.append({
                "track": t.track_id, "cow": cow, "votes": votes, "total": total, "zone_key": zone,
                "seconds": round(t_video - info.first_t, 1),
                "state": state,
                "speed": round(speed * 100, 1),
                "zone": ZONE_RU.get(zone, zone), "moving_s": round(info.moving_s, 1),
                "still_s": round(info.still_s, 1), "lying_s": round(info.lying_s, 1),
                "corners": obs.corners, "bbox": obs.bbox.to_xyxy(),
                "registered": models.gallery.size() > 0,
            })
        self._update_cows(rows, t_video, fps)
        return rows

    def _update_cows(self, rows: list[dict], t_video: float, fps: float) -> None:
        """Одна корова может быть видна двум камерам — берётся дорожка с большим числом голосов."""
        step = self.real_spf or self.cfg.video.frame_stride / max(fps, 1e-6)
        best: dict[str, dict] = {}
        for r in rows:
            if r["cow"] and (r["cow"] not in best or r["votes"] > best[r["cow"]]["votes"]):
                best[r["cow"]] = r
        for cow, r in best.items():
            self.cows.setdefault(cow, _CowStats()).update(
                t_video, max(4 * step, 5.0), r["state"], r["zone_key"])
        self.now = t_video

    def summary(self, frame: np.ndarray, rows: list[dict]) -> dict:
        known = sum(1 for r in rows if r["cow"])
        out = {
            "health": camera_health(frame),
            "in_frame": len(rows),
            "known": known,
            "unknown": len(rows) - known,
            "tracks": [{k: v for k, v in r.items() if k not in ("corners", "bbox")}
                       for r in rows],
            "session": sorted(({"cow": c, "seconds": round(s, 1)}
                               for c, s in self.session.items()),
                              key=lambda x: -x["seconds"])[:30],
            "gallery": self.models.gallery.size(),
        }
        if any(i.poses for i in self.tracks.values()):
            out["lying"] = sum(1 for r in rows if r["state"] == "лежит")
            out["standing"] = len(rows) - out["lying"]
        out["cows"] = self.cow_rows()
        out["farm"] = {"registry": bool(self.models.registry), "norms": bool(self.models.norms),
                       "as_of": self.cfg.farm.as_of}
        return out

    def cow_rows(self) -> list[dict]:
        herd = {}
        for kind in ("lying", "feeder", "drinker"):
            vals = [st.share(kind) for st in self.cows.values()
                    if sum(st.seen.values()) >= MIN_SEEN_S and st.share(kind) is not None]
            herd[kind] = float(np.median(vals)) if len(vals) >= 3 else None
        rows = [cow_row(cow, st, self.models.norms.get(cow), herd,
                        self.models.registry.get(cow), self.now)
                for cow, st in self.cows.items()]
        rank = {"strong": 0, "notable": 1, "info": 2}
        rows.sort(key=lambda r: (min((rank[x["level"]] for x in r["signs"]), default=3),
                                 -r["seen_min"]))
        return rows

    @staticmethod
    def _speed(centers: deque) -> float:
        if len(centers) < 2:
            return 0.0
        t0, x0, y0 = centers[0]
        t1, x1, y1 = centers[-1]
        return float(np.hypot(x1 - x0, y1 - y0) / max(t1 - t0, 1e-3))


def camera_health(frame: np.ndarray) -> dict:
    """Яркость и резкость кадра: темно, засвечено, размыто или грязный объектив."""
    small = cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1])))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    problems = []
    if brightness < 40:
        problems.append("темно")
    if brightness > 225:
        problems.append("засвечено")
    if sharpness < 20:
        problems.append("размыто или объектив грязный")
    return {"brightness": round(brightness), "sharpness": round(sharpness),
            "problems": problems}


#: Камера сбоку или под углом: готовая модель COCO, корова — класс 19.
SIDE_DETECTOR = "models/pretrained/yolo26n.pt"


def source_config(source: LiveSource) -> PipelineConfig:
    if source.view == "side":
        cfg = PipelineConfig(camera_id="side")      # галереи нет — номера не ставятся
        cfg.detector.model = SIDE_DETECTOR
        cfg.detector.class_ids = [19]
        cfg.detector.conf = 0.3
        cfg.tracker.min_hits = 1
        return cfg
    path = Path(source.config)
    return PipelineConfig.load(path) if path.exists() else PipelineConfig()


class LiveEngine:
    """Один живой сеанс на процесс консоли."""

    #: Проверка картинки — та же функция, что и в роликах.
    _health = staticmethod(camera_health)

    def __init__(self):
        self._lock = threading.Lock()
        self._reader: Optional[_Reader] = None
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._models: dict[str, _Models] = {}
        self._jpeg: Optional[bytes] = None
        self._state: dict = {"running": False}
        self._frame_event = threading.Condition()

    def models_for(self, source: LiveSource) -> tuple[PipelineConfig, _Models]:
        cfg = source_config(source)
        key = "side" if source.view == "side" else source.config
        if key not in self._models:
            self._models[key] = _Models(cfg)
        return cfg, self._models[key]

    # -- управление --------------------------------------------------------

    def start(self, source_id: Optional[str] = None, url: Optional[str] = None,
              camera_index: Optional[int] = None, view: Optional[str] = None) -> dict:
        from dataclasses import replace

        sources = {s.id: s for s in available_sources()}
        if url:
            source = replace(sources["url"], title=f"Поток: {url}", target=url)
        elif camera_index is not None:
            source = replace(sources["camera0"], id=f"camera{camera_index}",
                             title=f"Камера №{camera_index}", target=int(camera_index))
        elif source_id in sources and source_id != "url":
            source = sources[source_id]
        else:
            raise KeyError(f"нет источника {source_id}")
        if source.kind != "playlist":
            source = replace(source, view="side" if view == "side" else "top")

        self.stop()
        cfg, models = self.models_for(source)
        self._stop.clear()
        self._reader = _PushReader(source) if source.kind == "browser" else _Reader(source)
        self._reader.start()
        self._state = {"running": True, "source": source.as_dict(), "started": time.time()}
        self._worker = threading.Thread(target=self._loop, args=(source, cfg, models),
                                        daemon=True)
        self._worker.start()
        return self._state

    def stop(self) -> None:
        self._stop.set()
        if self._reader is not None:
            self._reader.stop_flag.set()
        if self._worker is not None:
            self._worker.join(timeout=5)
        self._reader = None
        self._worker = None
        with self._lock:
            self._state = {"running": False}
            self._jpeg = None

    def state(self) -> dict:
        with self._lock:
            return dict(self._state)

    def push_frame(self, data: bytes) -> dict:
        """Кадр из браузера. Ошибка, если сейчас включён другой источник."""
        reader = self._reader
        if not isinstance(reader, _PushReader):
            raise RuntimeError("камера браузера не включена")
        return reader.push(data)

    def next_jpeg(self, timeout: float = 2.0) -> Optional[bytes]:
        with self._frame_event:
            self._frame_event.wait(timeout)
            return self._jpeg

    def latest_jpeg(self) -> Optional[bytes]:
        return self._jpeg

    # -- цикл --------------------------------------------------------------

    def _loop(self, source: LiveSource, cfg: PipelineConfig, models: _Models) -> None:
        from .render import draw_tracks

        reader = self._reader
        proc = FrameProcessor(cfg, models, source.real_seconds_per_frame)
        segment = -1
        processed = 0
        proc_times: deque = deque(maxlen=30)

        while not self._stop.is_set():
            item = reader.take() if reader else None
            if item is None:
                time.sleep(0.005)
                if reader and reader.error:
                    self._publish_error(source, reader.error)
                continue
            seg, idx, fps, frame = item
            if seg != segment:
                proc.reset()
                segment = seg
            started = time.perf_counter()
            rows = proc.process(frame, idx, fps)
            h, w = frame.shape[:2]
            scale = min(1.0, STREAM_WIDTH / w)
            annotated = draw_tracks(frame, rows, scale)
            if scale < 1:
                annotated = cv2.resize(annotated, (STREAM_WIDTH, int(h * scale)),
                                       interpolation=cv2.INTER_AREA)
            proc_times.append(time.perf_counter() - started)
            processed += 1
            t_video = proc.video_time(idx, fps)
            state = {
                "running": True,
                "source": source.as_dict(),
                "clip": reader.clip_name if reader else "",
                "clock": (time.strftime("%H:%M", time.gmtime(t_video))
                          if source.real_seconds_per_frame else None),
                "frame": [w, h],
                # У камеры браузера — сколько кадров в секунду он реально присылает.
                "fps_in": round((reader.fps if reader and reader.fps else fps), 1),
                "fps_proc": round(len(proc_times) / max(sum(proc_times), 1e-6), 1),
                "processed": processed,
                "error": reader.error if reader else "",
                **proc.summary(frame, rows),
            }
            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self._lock:
                self._state = state
            with self._frame_event:
                if ok:
                    self._jpeg = buf.tobytes()
                self._frame_event.notify_all()

    def _publish_error(self, source: LiveSource, error: str) -> None:
        with self._lock:
            self._state = {"running": True, "source": source.as_dict(), "error": error,
                           "in_frame": 0, "known": 0, "unknown": 0, "tracks": [],
                           "session": []}


ENGINE = LiveEngine()
