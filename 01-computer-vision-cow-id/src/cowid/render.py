"""Рисование кадра для людей: рамки коров, подписи по-русски, панель показателей.

Встроенный шрифт OpenCV не знает кириллицу, поэтому текст рисуется через
Pillow системным шрифтом. Если подходящего шрифта нет (например, в Docker),
подписи уходят латиницей — рамки и цифры остаются.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]
BOLD_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

# Цвета BGR: узнана — зелёный, «не знаю» — серый, лежит — синий.
KNOWN = (60, 150, 60)
UNKNOWN = (150, 150, 150)
LYING = (170, 110, 40)
PAPER = (239, 242, 243)
INK = (24, 27, 28)
INK2 = (86, 84, 87)
ALERT = (38, 34, 155)


@lru_cache(maxsize=16)
def font(size: int, bold: bool = False):
    from PIL import ImageFont

    for path in (BOLD_CANDIDATES if bold else []) + FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return None


def _latin(text: str) -> str:
    table = str.maketrans("абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
                          "abvgdeejziiklmnoprstufhccss'y'euaABVGDEEJZIIKLMNOPRSTUFHCCSS'Y'EUA")
    return text.translate(table)


def put_texts(img: np.ndarray, items: list[tuple[str, tuple[int, int], int, tuple, bool]]) -> np.ndarray:
    """Рисует пачку подписей за один переход в Pillow: (текст, (x, y), размер, цвет BGR, жирный)."""
    if not items:
        return img
    if font(14) is None:
        for text, (x, y), size, color, _ in items:
            cv2.putText(img, _latin(text), (x, y + size), cv2.FONT_HERSHEY_SIMPLEX,
                        size / 30, color, 1 if size < 22 else 2, cv2.LINE_AA)
        return img
    from PIL import Image, ImageDraw

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for text, (x, y), size, color, bold in items:
        draw.text((x, y), text, font=font(size, bold), fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def text_width(text: str, size: int, bold: bool = False) -> int:
    f = font(size, bold)
    if f is None:
        return int(len(text) * size * 0.6)
    return int(f.getlength(text))


def track_label(row: dict, compact: bool = False) -> str:
    # На ферме, где коровы не зарегистрированы, номер и голоса не показываем.
    if not row.get("registered", True):
        return row["state"]
    if compact:
        # Много коров в кадре: только номер. Поза видна по цвету рамки
        # (синяя — лежит), иначе подписи не помещаются между соседями.
        return row["cow"] or "?"
    who = f"№ {row['cow']}" if row.get("cow") else "?"
    return f"{who}  {row['votes']}/{row['total']}  {row['state']}"


def duration(seconds: float) -> str:
    if seconds >= 120:
        return f"{seconds / 60:.0f} мин"
    return f"{seconds:.0f} с"


#: Выбранная корова («найти корову») — янтарная рамка и путь, остальной кадр притушен.
FOCUS = (0, 176, 240)
DIM = 0.42


def _outline(r: dict) -> np.ndarray:
    if r.get("corners"):
        return np.asarray(r["corners"], dtype=np.float32).reshape(-1, 2)
    x1, y1, x2, y2 = r["bbox"]
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def _place(box: tuple[int, int, int, int], w: int, h: int, size: tuple[int, int],
           taken: list[tuple[int, int, int, int]]) -> Optional[tuple[int, int]]:
    """Левый верхний угол подписи: над рамкой, внутри сверху, под рамкой — где не
    занято подписями соседей. Нигде нет места — подписи не будет (рамка останется)."""
    x0, y0, x1, y1 = box
    W, H = size
    for x, y in ((x0, y0 - h), (x0, y0 + 2), (x0, y1), (x1 - w, y0 - h), (x0, y1 - h - 2),
                 (x1 - w, y1)):
        x = min(max(0, x), W - w)
        y = min(max(0, y), H - h)
        if all(x + w <= a or a2 <= x or y + h <= b or b2 <= y for a, b, a2, b2 in taken):
            return x, y
    return None


def draw_tracks(frame: np.ndarray, rows: list[dict], scale: float = 1.0,
                focus: Optional[dict] = None, alerted: frozenset = frozenset()) -> np.ndarray:
    """Рамки и подписи. `scale` — во сколько раз кадр уменьшен при показе,
    чтобы толщина линий и размер текста остались читаемыми.

    `focus` — выбранная корова: {"cow", "trail": [(x, y), ...], "note"}. Её рамка
    и путь — янтарные, остальной кадр притушен, чужие рамки тонкие и без подписей.
    `alerted` — коровы с явным предупреждением («не ела 4 ч»): красная рамка и «!».
    """
    out = frame.copy()
    thick = max(2, int(3 / scale))
    compact = len(rows) > 12
    size = max(14, int((20 if not compact else 15) / scale))
    pad = max(3, int(4 / scale))
    target = focus.get("cow") if focus else None
    picked = focus.get("tracks") if focus else None

    def is_chosen(r: dict) -> bool:
        if not target or r.get("cow") != target:
            return False
        return picked is None or r.get("track") in picked

    mine = [r for r in rows if is_chosen(r)]
    if focus:
        light = out.copy()
        out = (out * DIM).astype(np.uint8)
        for r in mine:
            x, y, w, h = cv2.boundingRect(_outline(r).astype(np.int32))
            m = int(h * 0.15)
            out[max(0, y - m):y + h + m, max(0, x - m):x + w + m] = \
                light[max(0, y - m):y + h + m, max(0, x - m):x + w + m]

    # Подписи — по важности: выбранная, с предупреждением, узнанные, «не знаю».
    order = sorted(rows, key=lambda r: (not is_chosen(r), r.get("cow") not in alerted, not r.get("cow")))
    taken: list[tuple[int, int, int, int]] = []
    labels = []
    for r in order:
        pts = _outline(r)
        chosen = is_chosen(r)
        warn = bool(r.get("cow")) and r["cow"] in alerted
        if warn:
            color = ALERT
        elif r.get("state") == "лежит":
            color = LYING
        else:
            color = KNOWN if r.get("cow") else UNKNOWN
        if focus and not chosen:
            cv2.polylines(out, [pts.astype(np.int32)], True, (110, 110, 110), max(1, thick // 2))
            continue
        # «Не знаю» — тонкая рамка; в толпе без подписи: «?» ничего не говорит
        # и только закрывает соседей. Узнанная — рамка толще.
        unsure = r.get("registered", True) and not r.get("cow")
        cv2.polylines(out, [pts.astype(np.int32)], True, FOCUS if chosen else color,
                      thick * 2 if chosen else (max(1, thick // 2) if unsure else thick + 1))
        if unsure and compact:
            continue
        text = track_label(r, compact=compact and not chosen)
        if chosen and r.get("state"):
            text = f"{r['cow']} · {r['state']}"
        if warn:
            text = "! " + text
        lsize = int(size * 1.25) if chosen else size
        w = text_width(text, lsize, True) + 2 * pad
        h = lsize + 2 * pad
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        spot = _place((int(x0), int(y0), int(x1), int(y1)), w, h, (out.shape[1], out.shape[0]), taken)
        if spot is None:
            continue
        x, y = spot
        taken.append((x, y, x + w, y + h))
        cv2.rectangle(out, (x, y), (x + w, y + h), FOCUS if chosen else color, -1)
        labels.append((text, (x + pad, y + pad - max(1, lsize // 8)), lsize,
                       INK if chosen else (255, 255, 255), True))

    if focus:
        trail = np.asarray(focus.get("trail") or [], dtype=np.float32).reshape(-1, 2)
        if len(trail) > 1:
            cv2.polylines(out, [trail.astype(np.int32)], False, FOCUS, max(2, thick), cv2.LINE_AA)
        for i, (px, py) in enumerate(trail):
            cv2.circle(out, (int(px), int(py)), max(3, thick + (2 if i == len(trail) - 1 else 0)),
                       FOCUS, -1, cv2.LINE_AA)
        if not mine and focus.get("note"):
            note = focus["note"]
            nsize = int(size * 1.2)
            w = text_width(note, nsize, True) + 2 * pad
            cv2.rectangle(out, (0, 0), (w, nsize + 2 * pad), FOCUS, -1)
            labels.append((note, (pad, pad - max(1, nsize // 8)), nsize, INK, True))
    return put_texts(out, labels)


def draw_tracks_small(frame: np.ndarray, rows: list[dict], width: int,
                      focus: Optional[dict] = None, alerted: frozenset = frozenset()) -> np.ndarray:
    """Уменьшает кадр до ширины `width` и рисует рамки уже на нём.

    Выглядит так же, как `draw_tracks` на полном кадре с уменьшением после,
    но на кадре 4K подписи через Pillow стоят ~45 мс, а на 960 px — единицы.
    """
    h, w = frame.shape[:2]
    k = width / w
    if k >= 1:
        return draw_tracks(frame, rows, focus=focus, alerted=frozenset(alerted))
    small = cv2.resize(frame, (width, int(round(h * k))), interpolation=cv2.INTER_AREA)
    scaled = []
    for r in rows:
        r = dict(r)
        if r.get("corners"):
            r["corners"] = [(x * k, y * k) for x, y in r["corners"]]
        if r.get("bbox"):
            r["bbox"] = [v * k for v in r["bbox"]]
        scaled.append(r)
    if focus:
        focus = {**focus, "trail": [(x * k, y * k) for x, y in focus.get("trail") or []]}
    return draw_tracks(small, scaled, focus=focus, alerted=frozenset(alerted))


def fit(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Вписывает кадр в прямоугольник с полями."""
    h, w = frame.shape[:2]
    k = min(width / w, height / h)
    small = cv2.resize(frame, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 22, dtype=np.uint8)
    y0, x0 = (height - small.shape[0]) // 2, (width - small.shape[1]) // 2
    canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    return canvas


def panel(width: int, height: int, title: str, subtitle: str, state: dict,
          note: str = "") -> np.ndarray:
    """Правая панель ролика — как страница «Камера» в платформе."""
    img = np.full((height, width, 3), PAPER, dtype=np.uint8)
    items = []
    y = 18
    items.append(("COW ID", (18, y), 22, INK, True))
    y += 36
    for line in _wrap(title, width - 36, 19, True):
        items.append((line, (18, y), 19, INK, True))
        y += 25
    for line in _wrap(subtitle, width - 36, 15, False):
        items.append((line, (18, y), 15, INK2, False))
        y += 20
    y += 10
    cv2.line(img, (18, y), (width - 18, y), (205, 208, 212), 1)
    y += 14

    big = [("коров в кадре", state.get("in_frame", 0)),
           ("узнано", state.get("known", 0)),
           ("«не знаю»", state.get("unknown", 0))]
    if "lying" in state:
        big = [("коров в кадре", state.get("in_frame", 0)),
               ("узнано", state.get("known", 0)) if state.get("gallery")
               else ("стоят", state.get("standing", 0)),
               ("лежат", state.get("lying", 0))]
    col = (width - 36) // 3
    for i, (k, v) in enumerate(big):
        items.append((str(v), (18 + i * col, y), 34, INK, True))
        items.append((k, (18 + i * col, y + 42), 14, INK2, False))
    y += 74
    cv2.line(img, (18, y), (width - 18, y), (205, 208, 212), 1)
    y += 12

    items.append(("корова", (18, y), 13, INK2, False))
    items.append(("голоса", (120, y), 13, INK2, False))
    items.append(("в кадре", (200, y), 13, INK2, False))
    items.append(("сейчас", (290, y), 13, INK2, False))
    y += 22
    for r in state.get("tracks", [])[:10]:
        who = r["cow"] if r.get("cow") else "?"
        items.append((who, (18, y), 18, INK if r.get("cow") else INK2, True))
        votes = f"{r['votes']}/{r['total']}" if r.get("registered", True) else "—"
        items.append((votes, (120, y + 2), 15, INK, False))
        items.append((duration(r["seconds"]), (200, y + 2), 15, INK, False))
        items.append((r["state"], (290, y + 2), 15, INK, False))
        y += 28
    if not state.get("tracks"):
        items.append(("коров в кадре нет", (18, y), 15, INK2, False))
        y += 28

    health = state.get("health") or {}
    yb = height - 200
    cv2.line(img, (18, yb), (width - 18, yb), (205, 208, 212), 1)
    yb += 10
    items.append((f"камера: яркость {health.get('brightness', '—')}, резкость "
                  f"{health.get('sharpness', '—')}", (18, yb), 14, INK2, False))
    problems = health.get("problems") or []
    items.append(("внимание: " + ", ".join(problems) if problems else "картинка в порядке",
                  (18, yb + 20), 14, ALERT if problems else INK2, False))
    yb += 50
    for line in _wrap(note, width - 36, 14, False)[:7]:
        items.append((line, (18, yb), 14, INK2, False))
        yb += 19
    return put_texts(img, items)


def _wrap(text: str, width: int, size: int, bold: bool) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        cand = f"{cur} {word}".strip()
        if text_width(cand, size, bold) <= width:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


WARN = (18, 101, 138)
INFO = (134, 93, 43)
LEVEL_COLOR = {"strong": ALERT, "notable": WARN, "info": INFO}


def cow_panel(width: int, height: int, title: str, subtitle: str, state: dict) -> np.ndarray:
    """Правая панель для фермы с реестром: кто это и что это значит."""
    img = np.full((height, width, 3), PAPER, dtype=np.uint8)
    items = [("COW ID", (18, 16), 20, INK, True)]
    y = 46
    for line in _wrap(title, width - 36, 19, True):
        items.append((line, (18, y), 19, INK, True))
        y += 25
    items.append((subtitle, (18, y), 15, INK2, False))
    y += 30
    big = [("в кадре", state.get("in_frame", 0)), ("узнано", state.get("known", 0)),
           ("лежат", state.get("lying", 0))]
    col = (width - 36) // 3
    for i, (k, v) in enumerate(big):
        items.append((str(v), (18 + i * col, y), 30, INK, True))
        items.append((k, (18 + i * col, y + 36), 13, INK2, False))
    y += 62
    cv2.line(img, (18, y), (width - 18, y), (205, 208, 212), 1)
    y += 10
    rows = state.get("cows", [])
    ranked = [r for r in rows if r.get("signs")] + [r for r in rows if not r.get("signs")]
    for r in ranked:
        reg = r.get("registry") or {}
        wrapped = [line for x in r.get("signs", [])[:2]
                   for line in _wrap(x["text"], width - 88, 13, True)[:2]]
        need = 20 + 17 * bool(reg.get("last_leg_problem")) + 17 + 17 * len(wrapped) + 6
        if y + need > height - 50:
            break
        head = r["cow"]
        leg = reg.get("last_leg_problem")
        brief = reg.get("summary", "").split(" · ")[:2]
        items.append((head, (18, y), 17, INK, True))
        items.append((" · ".join(brief), (70, y + 2), 13, INK2, False))
        y += 20
        if leg:
            items.append((f"в реестре: {leg['what']} {leg['day'][8:10]}.{leg['day'][5:7]}",
                          (70, y), 13, WARN, True))
            y += 17
        stat = f"лежит {r['lying_pct']}%"
        if r.get("lying_norm_pct") is not None:
            stat += f" ({'обычно' if r.get('norm_own') else 'у стада'} {r['lying_norm_pct']}%)"
        if r.get("feeder_pct") is not None:
            stat += f" · у корма {r['feeder_pct']}%"
        items.append((stat if r["seen_min"] >= 20 else f"на виду {r['seen_min']} мин — мало для вывода",
                      (70, y), 13, INK, False))
        y += 17
        for s in r.get("signs", [])[:2]:
            for line in _wrap(s["text"], width - 88, 13, True)[:2]:
                items.append((line, (70, y), 13, LEVEL_COLOR.get(s["level"], INK), True))
                y += 17
        y += 6
    items.append(("Хромота — косвенный признак по поведению, не диагноз.",
                  (18, height - 40), 13, INK2, False))
    items.append(("Карточка — реальные записи фермы на день съёмки.", (18, height - 22), 13, INK2, False))
    return put_texts(img, items)
