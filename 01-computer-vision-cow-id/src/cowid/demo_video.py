"""Демо-ролики для показа: что видит система, когда камеры под рукой нет.

Каждый ролик — кадр с рамками и номерами слева и панель справа (как страница
«Камера» в платформе). Считает тот же `FrameProcessor`, что и живой режим,
поэтому ролик показывает ровно то, что показала бы платформа.

    cowid demo-video                 все ролики и общий ролик для защиты
    cowid demo-video --only known    один ролик
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .live import FrameProcessor, LiveSource, _Models, available_sources, source_config
from .render import cow_panel, draw_tracks, fit, panel, put_texts, text_width

OUT_DIR = Path("reports/demo")
W, H = 1600, 720
VIDEO_W, PANEL_W = 1240, 360
#: Для фермы с реестром панель шире: у каждой коровы — карточка и вывод.
COW_PANEL_W = 448

Log = Callable[[str], None]

#: Что рассказывает каждый ролик. Кадры другой фермы — после 14:00: эти часы
#: дообученный детектор не видел.
PLAN = {
    "known": {
        "source": "demo_known",
        "title": "Знакомые коровы",
        "note": "Камера сверху над проходом после дойки (Cows2021, март). Модель узнавания "
                "училась на этих коровах по февральским фото. Номер ставится, когда "
                "за корову проголосовали ≥ 3 кадра дорожки и больше половины совпали.",
    },
    "unseen": {
        "source": "demo_unseen",
        "title": "Коровы, которых модель не видела",
        "note": "Эти коровы не участвовали в обучении — они только зарегистрированы по "
                "фото за месяц до записи. Так будет с новой партией на ферме. "
                "Серая рамка «?» — честное «не знаю».",
    },
    "other_farm": {
        "source": "demo_other_farm",
        "title": "Другая ферма: кто это и что это значит",
        "note": "США (MmCows), 4 камеры под наклоном, кадр раз в 15 секунд — здесь час "
                "за 10 секунд. Детектор и узнавание дообучены на утренних кадрах, "
                "показаны дневные: синяя рамка — лежит. Справа — сравнение с её "
                "обычными часами и карточка из реестра фермы.",
        "start": 3360,          # 14:00
        "frames": 600,          # до 16:30
        "fps": 6,
    },
}


def compress(raw: Path, final: Path, log: Log = print) -> Path:
    """Пережимает ролик ffmpeg (H.264, crf 27): в 5–10 раз меньше, открывается везде.

    OpenCV пишет видео без настройки битрейта — минута весит сотни мегабайт.
    Если ffmpeg недоступен, остаётся исходный файл.
    """
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raw.replace(final)
        return final
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
           "-preset", "medium", "-crf", "27", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(final)]
    subprocess.run(cmd, check=True)
    raw.unlink(missing_ok=True)
    log(f"  {final}: {final.stat().st_size / 1e6:.1f} МБ")
    return final


def _raw(path: Path) -> Path:
    return path.with_name(path.stem + ".raw.mp4")


def _writer(path: Path, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Черновик пишется в mp4v — потом его всё равно пережимает ffmpeg в H.264.
    for codec in ("mp4v", "avc1"):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (W, H))
        if w.isOpened():
            return w
    raise RuntimeError(f"не удалось открыть запись видео {path}")


def title_card(title: str, lines: list[str]) -> np.ndarray:
    img = np.full((H, W, 3), (239, 242, 243), dtype=np.uint8)
    items = [("COW ID", (80, 90), 30, (24, 27, 28), True),
             (title, (80, 170), 52, (24, 27, 28), True)]
    y = 270
    for line in lines:
        items.append((line, (80, y), 26, (86, 84, 87), False))
        y += 42
    return put_texts(img, items)


def _frames_of(source: LiveSource, spec: dict):
    """(отрезок, номер кадра, fps, кадр, подпись) по всем файлам источника."""
    files = source.target if isinstance(source.target, list) else [source.target]
    for seg, path in enumerate(files):
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        start = spec.get("start", 0)
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        limit = spec.get("frames")
        name = Path(path).parent.name if Path(path).stem == "RGB" else Path(path).stem
        idx, n = start, 0
        while limit is None or n < limit:
            ok, frame = cap.read()
            if not ok:
                break
            yield seg, idx, fps, frame, name
            idx += 1
            n += 1
        cap.release()


def render(key: str, out: Optional[Path] = None, log: Log = print,
           writer: Optional[cv2.VideoWriter] = None) -> dict:
    spec = PLAN[key]
    sources = {s.id: s for s in available_sources()}
    if spec["source"] not in sources:
        raise FileNotFoundError(f"источник {spec['source']} недоступен")
    source = sources[spec["source"]]
    cfg = source_config(source)
    models = _Models(cfg)
    proc = FrameProcessor(cfg, models, source.real_seconds_per_frame)

    own = writer is None
    fps_out = spec.get("fps", 30)
    final = out or OUT_DIR / f"{key}.mp4"
    if own:
        writer = _writer(_raw(final), fps_out)
    for _ in range(int(fps_out * 3)):
        writer.write(title_card(spec["title"], _wrap_lines(spec["note"])))

    segment = -1
    last_state: dict = {}
    stats = {"frames": 0, "rows": 0, "known": 0, "lying": 0}
    for seg, idx, fps, frame, name in _frames_of(source, spec):
        if seg != segment:
            proc.reset()
            segment = seg
        rows = proc.process(frame, idx, fps)
        state = proc.summary(frame, rows)
        h, w = frame.shape[:2]
        farm = bool(state.get("farm", {}).get("registry"))
        video_w = W - COW_PANEL_W if farm else VIDEO_W
        scale = min(video_w / w, H / h)
        left = fit(draw_tracks(frame, rows, scale), video_w, H)
        t = proc.video_time(idx, fps)
        if source.real_seconds_per_frame:
            subtitle = f"время на ферме {int(t // 3600):02d}:{int(t % 3600 // 60):02d}"
        else:
            subtitle = f"ролик {name}, {t:.1f} с"
        right = (cow_panel(COW_PANEL_W, H, spec["title"], subtitle, state) if farm
                 else panel(PANEL_W, H, spec["title"], subtitle, state, spec["note"]))
        writer.write(np.hstack([left, right]))
        last_state = state
        stats["frames"] += 1
        stats["rows"] += len(rows)
        stats["known"] += state["known"]
        stats["lying"] += state.get("lying", 0)
        if stats["frames"] % 200 == 0:
            log(f"  {key}: кадров {stats['frames']}")
    flagged = [r for r in last_state.get("cows", []) if r.get("signs")]
    if flagged:
        card = summary_card(spec["title"], flagged)
        for _ in range(int(fps_out * 6)):
            writer.write(card)
    if own:
        writer.release()
        compress(_raw(final), final, log)
    stats["flagged"] = [(r["cow"], [x["text"] for x in r["signs"]]) for r in flagged]
    stats["session_cows"] = sorted(proc.session)
    log(f"{key}: кадров {stats['frames']}, узнанных коров {len(proc.session)}")
    return stats


def summary_card(title: str, flagged: list[dict]) -> np.ndarray:
    """Итог показа: на кого обратить внимание и почему."""
    img = np.full((H, W, 3), (239, 242, 243), dtype=np.uint8)
    items = [("COW ID — итог за время показа", (80, 50), 30, (24, 27, 28), True),
             ("На кого обратить внимание зоотехнику", (80, 100), 22, (86, 84, 87), False)]
    y = 160
    colors = {"strong": (38, 34, 155), "notable": (18, 101, 138), "info": (134, 93, 43)}
    for r in flagged[:7]:
        reg = r.get("registry") or {}
        items.append((r["cow"], (80, y), 30, (24, 27, 28), True))
        items.append((reg.get("summary", ""), (190, y + 6), 18, (86, 84, 87), False))
        y += 40
        for s in r["signs"][:2]:
            items.append((s["text"], (190, y), 20, colors.get(s["level"], (24, 27, 28)), True))
            y += 28
        y += 12
        if y > H - 80:
            break
    items.append(("Система не ставит диагноз: она показывает, что изменилось у коровы "
                  "по сравнению с её обычными часами, и что о ней известно.", (80, H - 50), 18,
                  (86, 84, 87), False))
    return put_texts(img, items)


def _wrap_lines(text: str, width: int = 1400, size: int = 26) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        cand = f"{cur} {word}".strip()
        if text_width(cand, size) <= width:
            cur = cand
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def render_all(log: Log = print) -> dict[str, dict]:
    """Отдельные ролики и один общий — для защиты."""
    results = {}
    for key in PLAN:
        try:
            results[key] = render(key, log=log)
        except FileNotFoundError as exc:
            log(f"{key}: пропущен — {exc}")
    final = OUT_DIR / "cowid_demo.mp4"
    combined = _writer(_raw(final), 30)
    for _ in range(90):
        combined.write(title_card("Как работает COW ID", [
            "Камера находит коров, узнаёт каждую и считает, что она делает.",
            "Утром зоотехник получает не график, а список: кого осмотреть и почему.",
            "Дальше — три ролика на реальных данных.",
        ]))
    for key in PLAN:
        path = OUT_DIR / f"{key}.mp4"
        if not path.exists():
            continue
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        repeat = max(1, round(30 / fps))      # ролик 6 кадров/с — растягиваем до 30
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for _ in range(repeat):
                combined.write(frame)
        cap.release()
    combined.release()
    compress(_raw(final), final, log)
    log(f"Общий ролик: {final}")
    return results
