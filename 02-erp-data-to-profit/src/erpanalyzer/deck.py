"""Презентация для защиты (PowerPoint). `erpanalyzer deck` — снимки MVP + сборка .pptx.

Суть берётся из `concept`, экраны — снимки работающего MVP, поэтому презентация не расходится с демо.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from . import concept as K

INK = RGBColor(0x1D, 0x1D, 0x1B)
MUTED = RGBColor(0x6B, 0x6A, 0x66)
LINE = RGBColor(0xE0, 0xDF, 0xDA)
ACCENT = RGBColor(0x1F, 0x5A, 0x43)
FONT = "Calibri"

CHROME = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")
# снимок: имя, адрес, высота окна, область обрезки (слева, сверху, справа, снизу)
PAGES = [("data", "/f1", 820, (180, 0, 1260, 800)),
         ("check", "/f1/check", 1000, (180, 0, 1260, 940)),
         ("calc_credit", "/f1/check", 3200, (180, 940, 1260, 1672)),
         ("calc_subs", "/f1/check", 3200, (180, 2325, 1260, 2790)),
         ("bank", "/bank/f1", 800, (180, 0, 1260, 790))]


def shots(out: Path, base: str = "http://127.0.0.1:8010") -> None:
    out.mkdir(parents=True, exist_ok=True)
    prof = Path(tempfile.gettempdir()) / "erpanalyzer_chrome_profile"
    for name, path, h, box in PAGES:
        png = out / f"{name}.png"
        subprocess.run([str(CHROME), "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=1", f"--window-size=1440,{h}", f"--user-data-dir={prof}",
                        f"--screenshot={png}", base + path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=120)
        Image.open(png).crop(box).save(png)


# ---------------------------------------------------------------------------

def text(slide, x, y, w, h, lines, size=18, color=INK, bold=False, after=8, align=PP_ALIGN.LEFT):
    tf = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(lines if isinstance(lines, list) else [lines]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(after)
        for chunk, st in (line if isinstance(line, list) else [(line, {})]):
            r = p.add_run()
            r.text = chunk
            r.font.name = FONT
            r.font.size = Pt(st.get("size", size))
            r.font.bold = st.get("bold", bold)
            r.font.color.rgb = st.get("color", color)


def rule(slide, x, y, w):
    ln = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(0.01))
    ln.fill.solid()
    ln.fill.fore_color.rgb = LINE
    ln.line.fill.background()


def slide(prs, title=None, note=""):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    if title:
        text(s, 0.7, 0.55, 11.9, 0.8, title, size=32, bold=True)
    s.notes_slide.notes_text_frame.text = note
    return s


def picture(s, path, x, y, h):
    pic = s.shapes.add_picture(str(path), Inches(x), Inches(y), height=Inches(h))
    pic.line.color.rgb = LINE
    pic.line.width = Pt(0.75)
    return pic


# ---------------------------------------------------------------------------

def build(shots_dir: Path, out: Path) -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    # 1
    s = slide(prs, note="0:00–0:20. Мы — ERPAnalyzer. " + K.ONE_LINE)
    text(s, 0.9, 2.3, 11.5, 1.0, K.NAME, size=54, bold=True)
    text(s, 0.9, 3.45, 11.2, 1.6, K.ONE_LINE, size=24, color=MUTED)
    text(s, 0.9, 6.6, 11.5, 0.4, "Трек «ERP: от данных к прибыли» · AgriTech Hackathon 2026", size=14, color=MUTED)

    # 2
    s = slide(prs, "Проблема", "0:20–1:00. У фермера есть учёт в ERP, но дешёвых денег он не получает: банк не верит "
                               "цифрам, государство отказывает в субсидиях из-за мелочей. Причина одна — данные "
                               "никто не проверяет заранее.")
    y = 1.75
    for title, body in K.PROBLEM:
        text(s, 0.7, y, 2.6, 0.5, title, size=22, bold=True, color=ACCENT if title == "Причина" else INK)
        text(s, 3.3, y + 0.03, 9.3, 1.1, body, size=20)
        rule(s, 0.7, y + 1.25, 11.9)
        y += 1.45
    text(s, 0.7, 6.3, 11.9, 0.6, K.PROBLEM_FACT, size=15, color=MUTED)

    # 3
    s = slide(prs, "Решение", "1:00–1:40. ERPAnalyzer — модуль к ERP. Три шага: берём данные, проверяем как банк и "
                              "государство, подсказываем, что сделать.")
    text(s, 0.7, 1.6, 11.9, 1.2, K.SOLUTION, size=21)
    for i, (title, body) in enumerate(K.STEPS):
        x = 0.7 + i * 4.05
        text(s, x, 3.3, 3.7, 0.6, f"{i + 1}", size=40, bold=True, color=ACCENT)
        text(s, x, 4.15, 3.7, 0.5, title, size=22, bold=True)
        text(s, x, 4.7, 3.7, 2.2, body, size=16, color=MUTED)

    # 4
    s = slide(prs, "Какие данные приходят и что мы с ними делаем",
              "1:40–2:15. Почти всё уже есть в ERP хозяйства. Снаружи добавляем цены из статистики, правила "
              "субсидий и камеру из нашего CV-трека.")
    rows = [("Данные", "Откуда", "Что проверяем")] + [(f"{a} — {b}", src, why) for a, b, src, why in K.DATA]
    tbl = s.shapes.add_table(len(rows), 3, Inches(0.7), Inches(1.6), Inches(11.9), Inches(0.62 * len(rows))).table
    for j, wv in enumerate((5.4, 2.9, 3.6)):
        tbl.columns[j].width = Inches(wv)
    for r_i, row in enumerate(rows):
        for c_i, val in enumerate(row):
            cell = tbl.cell(r_i, c_i)
            cell.text = val
            cell.fill.background()
            run = cell.text_frame.paragraphs[0].runs[0]
            run.font.name = FONT
            run.font.size = Pt(15 if r_i else 13)
            run.font.bold = r_i == 0
            run.font.color.rgb = MUTED if r_i == 0 else INK

    # 5–7: MVP
    s = slide(prs, "MVP. Шаг 1: данные хозяйства из ERP",
              "2:15–2:35. Это наш MVP. Хозяйство открывает модуль и видит свои данные из ERP: заявку, деньги по "
              "месяцам, реестр животных, корма, а ещё цены из статистики и камеру. Нажимает «Проверить хозяйство».")
    picture(s, shots_dir / "data.png", 0.7, 1.45, 5.6)
    text(s, 8.5, 1.6, 4.1, 5.0, [
        "Модуль показывает сами данные: заявку, деньги по месяцам, реестр животных, корма.",
        "Плюс цены из статистики и камера.",
        "Одна кнопка — «Проверить хозяйство».",
    ], size=18, after=16)

    s = slide(prs, "MVP. Шаг 2: результат и что сделать",
              "2:35–3:05. Результат: сколько кредита хозяйство потянет, сколько субсидий положено, на месте ли "
              "скот. И главное — список, что сделать, с суммой. " + K.EXAMPLE)
    picture(s, shots_dir / "check.png", 0.7, 1.45, 5.6)
    text(s, 7.7, 1.6, 4.9, 5.2, [
        [("Три ответа: ", {"bold": True}), ("кредит, субсидии, залог.", {})],
        [("Список «что сделать» ", {"bold": True}), ("— простыми словами и с суммой.", {})],
        [("Пример. ", {"bold": True, "color": ACCENT}), (K.EXAMPLE, {})],
    ], size=18, after=16)

    s = slide(prs, "MVP. Как посчитан кредит",
              "3:05–3:30. Каждая цифра объяснена. Деньги за год — из финансов ERP: выручка минус расходы. "
              "Делим на все платежи — банку нужно не меньше 1,2 раза. Стоимость стада — живой вес из ERP на "
              "цену из статистики. Можно дать меньшее из двух: по деньгам и по залогу.")
    picture(s, shots_dir / "calc_credit.png", 0.7, 1.45, 5.4)
    text(s, 9.0, 1.6, 3.6, 5.2, [
        "Каждый шаг: что считаем, откуда данные, расчёт, результат.",
        "Деньги — из финансов ERP.",
        "Стадо — живой вес из ERP × цена из статистики.",
        "Можно дать — меньшее из двух.",
    ], size=17, after=14)

    s = slide(prs, "MVP. Как посчитаны субсидии",
              "3:30–3:50. Каждую голову проверяем по условиям приказа. Сколько проходят и сколько теряется — "
              "с причиной. Можно открыть проверку каждого бычка.")
    picture(s, shots_dir / "calc_subs.png", 0.7, 1.45, 3.9)
    text(s, 0.7, 5.6, 11.9, 1.4, [
        "Правило — из приказа Минсельхоза № 108. Каждую голову из реестра ERP проверяем по условиям: "
        "отец, покупатель, бирка, возраст на дату продажи, вес.",
        "Первое невыполненное условие — причина, и сумма по этой голове попадает в «можно потерять».",
    ], size=17, after=8)

    s = slide(prs, "MVP. Что видит банк",
              "3:50–4:15. Банк получает не анкету, а справку из проверенных данных: сколько можно дать, "
              "почему такая оценка, хватит ли денег на платежи в плохой год.")
    picture(s, shots_dir / "bank.png", 0.7, 1.45, 5.6)
    text(s, 8.5, 1.6, 4.1, 5.0, [
        "Вместо анкеты — справка из учёта.",
        "Почему такая оценка — простыми словами.",
        "Хватит ли денег на платежи в засуху — по настоящим скачкам цен.",
        "Надёжных можно одобрять без выезда.",
    ], size=18, after=16)

    # 8
    s = slide(prs, "Кому польза и как внедрить", "4:15–4:40. Польза для фермера, банка и государства. Внедрение — "
                                                "модуль в ERP, платит хозяйство или банк, начинаем с пилота.")
    for i, (who, what) in enumerate(K.VALUE):
        x = 0.7 + i * 4.05
        text(s, x, 1.7, 3.7, 0.5, who, size=22, bold=True)
        text(s, x, 2.25, 3.7, 1.4, what, size=18, color=MUTED)
    rule(s, 0.7, 3.75, 11.9)
    text(s, 0.7, 4.05, 11.9, 0.5, "Как внедрить", size=22, bold=True)
    text(s, 0.7, 4.65, 11.9, 2.2, ["—  " + x for x in K.ROLLOUT], size=18, after=10)

    # 9
    s = slide(prs, note="4:40–5:00. Итог одной фразой. Честно: хозяйства в демо — примеры, цены и правила — настоящие.")
    text(s, 0.9, 2.2, 11.5, 1.8, "Данные уже есть в ERP. ERPAnalyzer проверяет их заранее — и фермер получает "
                                 "кредит и субсидии без отказов.", size=34, bold=True)
    text(s, 0.9, 4.4, 11.5, 0.6, "Спасибо!", size=28, color=ACCENT, bold=True)
    text(s, 0.9, 6.4, 11.5, 0.7, K.HONEST, size=13, color=MUTED)

    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    return out
