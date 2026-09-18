"""Цены Бюро национальной статистики РК (stat.gov.kz).

Серия «Индексы цен и цены в сельском хозяйстве в Республике Казахстан»,
по файлу на месяц. Лист «5» — средние цены производителей по регионам,
в тенге за тонну. Формат файлов менялся (xls/xlsx, «5» или «5.»), поэтому
строки ищем по названию, а регионы — по строке с «Республика Казахстан».
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

# ключ → как строка начинается на листе 5 (без учёта регистра и пробелов)
PRODUCTS = {
    "cattle": "скот крупный рогатый",          # весь КРС в живом весе
    "cattle_beef": "скот крупный рогатый взрослый мясного",
    "cattle_dairy": "скот крупный рогатый взрослый молочного",
    "sheep": "овцы",
    "barley": "ячмень",
    "wheat": "пшеница",
    "hay": "сено",
}
PRODUCT_NAMES = {
    "cattle": "КРС, живой вес",
    "cattle_beef": "КРС взрослый мясного стада, живой вес",
    "cattle_dairy": "КРС взрослый молочного стада, живой вес",
    "sheep": "Овцы, живой вес",
    "barley": "Ячмень",
    "wheat": "Пшеница",
    "hay": "Сено",
}
NATIONAL = "Республика Казахстан"


def _num(v) -> float | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().replace("\xa0", "").replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None  # «x» — конфиденциально, «-» — нет данных


def _norm(s) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d\)", "", str(s))).strip().lower()


def parse_month(path: Path) -> pd.DataFrame:
    """Один файл → строки (month, region, product, tg_per_kg)."""
    month = path.stem  # файлы названы YYYY-MM при скачивании
    xl = pd.ExcelFile(path)
    sheet = next(s for s in xl.sheet_names if s.strip().rstrip(".") == "5")
    df = xl.parse(sheet, header=None, dtype=object)

    head_row = next(i for i in range(len(df))
                    if any(str(v).strip() == NATIONAL for v in df.iloc[i].tolist()))
    regions = {j: str(v).strip() for j, v in enumerate(df.iloc[head_row].tolist())
               if isinstance(v, str) and v.strip()}

    found: dict[str, int] = {}
    for i in range(head_row + 1, len(df)):
        label = _norm(df.iloc[i, 0])
        for key, prefix in PRODUCTS.items():
            if key in found or not label.startswith(prefix):
                continue
            # «скот крупный рогатый» не должен поймать строки «…взрослый…»
            if key == "cattle" and "взросл" in label:
                continue
            if key == "wheat" and label != "пшеница":
                continue
            found[key] = i

    rows = []
    for key, i in found.items():
        for j, region in regions.items():
            v = _num(df.iloc[i, j])
            if v is not None:
                rows.append((month, region, key, v / 1000.0))
    return pd.DataFrame(rows, columns=["month", "region", "product", "tg_per_kg"])


# Старая серия (2018–2021, name=26960): двуязычные файлы, строки подписаны по-казахски,
# цены растениеводства и животноводства — на разных листах. Берём только республику.
OLD_PRODUCTS = {"cattle": "ірі қара мал", "sheep": "қой", "barley": "арпа", "wheat": "бидай", "hay": "шөп"}


def parse_month_old(path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    rows = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, header=None, dtype=object)
        head = " ".join(str(v) for v in df.head(3).values.ravel() if pd.notna(v))
        if "Средние цены производителей на продукцию" not in head:
            continue
        hr = next((i for i in range(min(len(df), 12))
                   if any(str(v).strip() == "Қазақстан Республикасы" for v in df.iloc[i].tolist())), None)
        if hr is None:
            continue
        col = next(j for j, v in enumerate(df.iloc[hr].tolist()) if str(v).strip() == "Қазақстан Республикасы")
        for i in range(hr + 1, len(df)):
            label = _norm(df.iloc[i, 0])
            for key, name in OLD_PRODUCTS.items():
                if label == name:
                    v = _num(df.iloc[i, col])
                    if v is not None:
                        rows.append((path.stem, NATIONAL, key, v / 1000.0))
    out = pd.DataFrame(rows, columns=["month", "region", "product", "tg_per_kg"])
    return out.drop_duplicates(["month", "product"])


def build(raw_dir: Path, out_dir: Path, old_dir: Path | None = None) -> pd.DataFrame:
    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    files = [raw_dir / m["file"] for m in manifest]
    parts = [parse_month(f) for f in files]
    if old_dir is not None and (old_dir / "manifest.json").exists():
        # файлы .rar не распаковываем (нет распаковщика) — в ряд попадают только xls/xlsx
        for m in json.loads((old_dir / "manifest.json").read_text(encoding="utf-8")):
            f = old_dir / m["file"]
            if f.suffix in (".xls", ".xlsx"):
                parts.append(parse_month_old(f))
    df = pd.concat(parts).drop_duplicates(["month", "region", "product"])
    df = df.sort_values(["product", "region", "month"]).reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "prices_kz.csv", index=False, encoding="utf-8")
    return df


def load(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "prices_kz.csv", dtype={"month": str})


def series(prices: pd.DataFrame, product: str, region: str = NATIONAL) -> pd.Series:
    s = prices[(prices["product"] == product) & (prices["region"] == region)]
    return s.set_index("month")["tg_per_kg"].sort_index()


def latest(prices: pd.DataFrame, product: str, region: str = NATIONAL) -> tuple[str, float]:
    """Последняя цена региона; если в регионе нет — по республике."""
    s = series(prices, product, region)
    if s.empty and region != NATIONAL:
        s = series(prices, product, NATIONAL)
    return s.index[-1], float(s.iloc[-1])


def _month_no(m: str) -> int:
    y, mm = m.split("-")
    return int(y) * 12 + int(mm) - 1
