"""Реестр фермы: что известно о корове, кроме того, что видит камера.

Камера отвечает «это C07, лежит 3 часа». Зоотехнику этого мало: важно, что
C07 — четвёртая лактация, 205 дней после отёла, стельная, и три недели назад
хромала. Тогда «лежит дольше обычного» превращается в «осмотреть ноги».

На ферме такой реестр — это учётная программа или ИСЖ (номер бирки, отёлы,
осеменения, болезни, обработки копыт). Для показа берутся **реальные записи**
фермы MmCows (`sub_data/health_records`, удой с днями лактации) на дату
съёмки. Ничего не придумывается: чего нет в записях, того нет и в карточке.

Карточка строится «на дату»: события после неё не показываются — система не
знает будущего.
"""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

ROOT = Path("data/real/mmcows_sensor/sensor_data")
HEALTH = ROOT / "sub_data" / "health_records"
MILK = ROOT / "main_data" / "milk"

#: Как назвать событие зоотехнику. Ключ — (событие, уточнение) или событие.
EVENT_RU = {
    ("Medical diagnosis", "Lame"): "хромота",
    ("Medical diagnosis", "Clinical Mastitis"): "мастит",
    ("Medical diagnosis", "Ketosis"): "кетоз",
    ("Medical diagnosis", "Metritis"): "метрит",
    ("Medical diagnosis", "Milk Fever"): "родильный парез",
    ("Medical diagnosis", "Pneumonia"): "пневмония",
    ("Medical diagnosis", "Scours"): "диарея",
    ("Medical diagnosis", "DA"): "смещение сычуга",
    ("Medical diagnosis", "Retained Placenta"): "задержание последа",
    ("Medical diagnosis", "Other"): "диагноз (другое)",
    ("Hoof Maintenance", "Trim"): "расчистка копыт",
    ("Hoof Maintenance", "Block"): "блок на копыто",
    ("Hoof Maintenance", "Lame Trim"): "расчистка копыт при хромоте",
    "Injury": "травма",
    "Heat": "охота",
    "Bred": "осеменение",
    "Preg Check": "проверка стельности",
    "Freshen": "отёл",
    "Dry Off": "запуск",
    "Abortion": "аборт",
    "Vaccination": "вакцинация",
    "Treatment": "лечение",
    "Withhold Milk": "молоко не сдавать",
    "Watch List": "под наблюдением",
    "Sold Cull": "выбраковка",
}
#: Что в карточке называется «здоровьем» (остальное — учёт и воспроизводство).
HEALTH_EVENTS = {"хромота", "мастит", "кетоз", "метрит", "родильный парез", "пневмония",
                 "диарея", "смещение сычуга", "задержание последа", "диагноз (другое)",
                 "расчистка копыт при хромоте", "блок на копыто", "травма", "лечение",
                 "молоко не сдавать", "под наблюдением"}
LEG_EVENTS = {"хромота", "расчистка копыт при хромоте", "блок на копыто"}


@dataclass
class Event:
    day: date
    what: str
    detail: str = ""

    def as_dict(self) -> dict:
        return {"day": self.day.isoformat(), "what": self.what, "detail": self.detail}


@dataclass
class CowRecord:
    cow_id: str
    as_of: date
    eid: Optional[str] = None
    pen: Optional[str] = None
    lactation: Optional[int] = None
    calved: Optional[date] = None
    days_in_milk: Optional[int] = None
    pregnancy: Optional[str] = None
    milk_kg: Optional[float] = None
    health: list[Event] = field(default_factory=list)
    reproduction: list[Event] = field(default_factory=list)

    @property
    def last_leg_problem(self) -> Optional[Event]:
        legs = [e for e in self.health if e.what in LEG_EVENTS]
        return legs[0] if legs else None

    def summary(self) -> str:
        """Одна строка для таблиц: «лактация 4 · 205 дн. после отёла · стельная»."""
        parts = []
        if self.lactation:
            parts.append(f"лактация {self.lactation}")
        if self.days_in_milk is not None:
            parts.append(f"{self.days_in_milk} дн. после отёла")
        if self.pregnancy:
            parts.append(self.pregnancy)
        return " · ".join(parts)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["as_of"] = self.as_of.isoformat()
        d["calved"] = self.calved.isoformat() if self.calved else None
        d["health"] = [e.as_dict() for e in self.health]
        d["reproduction"] = [e.as_dict() for e in self.reproduction]
        leg = self.last_leg_problem
        d["last_leg_problem"] = leg.as_dict() if leg else None
        d["summary"] = self.summary()
        return d


def _parse_day(text: str) -> Optional[date]:
    try:
        return datetime.strptime(text.strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _name(event: str, specific: str) -> Optional[str]:
    return EVENT_RU.get((event, specific)) or EVENT_RU.get(event)


@lru_cache(maxsize=64)
def _rows(cow_id: str) -> tuple:
    path = HEALTH / f"{cow_id}.csv"
    if not path.exists():
        return ()
    out = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in list(csv.reader(f))[1:]:
            if len(row) < 3:
                continue
            day = _parse_day(row[0])
            if day:
                out.append((day, row[1].strip(), row[2].strip(),
                            " ".join(row[3].split()) if len(row) > 3 else ""))
    out.sort(key=lambda r: r[0], reverse=True)
    return tuple(out)


def _pregnancy(rows, as_of: date) -> Optional[str]:
    """Последняя проверка стельности и осеменения после неё."""
    for day, event, _, desc in rows:
        if event == "Preg Check":
            if desc.lower().startswith("preg"):
                m = re.search(r"(\d+)\s*DAYS", desc)
                since = (as_of - day).days
                return (f"стельная, {int(m.group(1)) + since} дн." if m
                        else "стельная") + f" (проверка {day:%d.%m})"
            if desc.lower().startswith("open"):
                return f"не стельная (проверка {day:%d.%m})"
        if event == "Bred":
            return f"осеменена {day:%d.%m}"
    return None


def _milk(cow_id: str, as_of: date) -> tuple[Optional[float], Optional[int]]:
    path = MILK / f"{cow_id}.csv"
    if not path.exists():
        return None, None
    rows = []
    with path.open(encoding="utf-8-sig") as f:
        for row in list(csv.reader(f))[1:]:
            try:
                day = datetime.fromtimestamp(float(row[0]), timezone.utc).date()
                rows.append((day, float(row[1]), int(float(row[2]))))
            except (ValueError, IndexError):
                continue
    past = [r for r in rows if r[0] <= as_of]
    if not past:
        return None, None
    week = [r[1] for r in past if r[0] > as_of - timedelta(days=7)]
    milk = round(sum(week) / len(week), 1) if week else None
    last = max(past, key=lambda r: r[0])
    return milk, last[2] + (as_of - last[0]).days


def cow_record(cow_id: str, as_of: date, health_days: int = 120) -> Optional[CowRecord]:
    """Карточка коровы на дату `as_of` из записей фермы (или None, если записей нет)."""
    rows = [r for r in _rows(cow_id) if r[0] <= as_of]
    if not rows and not (MILK / f"{cow_id}.csv").exists():
        return None
    rec = CowRecord(cow_id=cow_id, as_of=as_of)
    for day, event, specific, desc in rows:
        if rec.eid is None and event == "Change EID":
            m = re.search(r"New EID:\s*(\d+)", desc)
            rec.eid = m.group(1) if m else None
        if rec.pen is None and event == "Move":
            rec.pen = desc.replace("Pen:", "").strip() or None
    calvings = [r for r in rows if r[1] == "Freshen"]
    prev = next((r for r in rows if r[1] == "Prev lactations"), None)
    rec.lactation = len(calvings) or None
    if prev and re.search(r"\d+", prev[3]):
        rec.lactation = int(re.search(r"\d+", prev[3]).group()) + len(
            [c for c in calvings if c[0] > prev[0]])
    rec.calved = calvings[0][0] if calvings else None
    rec.milk_kg, rec.days_in_milk = _milk(cow_id, as_of)
    if rec.days_in_milk is None and rec.calved:
        rec.days_in_milk = (as_of - rec.calved).days
    rec.pregnancy = _pregnancy(rows, as_of)
    since = as_of - timedelta(days=health_days)
    seen = set()
    for day, event, specific, desc in rows:
        name = _name(event, specific)
        if not name or day < since:
            continue
        detail = _detail(name, desc)
        if (day, name, detail) in seen:          # ветеринар часто пишет одно событие дважды
            continue
        seen.add((day, name, detail))
        target = rec.health if name in HEALTH_EVENTS or name == "расчистка копыт" else rec.reproduction
        target.append(Event(day, name, detail))
    return rec


def _detail(name: str, desc: str) -> str:
    """Пояснение словами — только то, что удалось прочитать; сырые коды не показываем."""
    if name == "хромота":
        return _lame_detail(desc)
    if name == "проверка стельности":
        low = desc.lower()
        if low.startswith("preg"):
            m = re.search(r"(\d+)\s*DAYS", desc)
            return f"стельная, {m.group(1)} дн." if m else "стельная"
        if low.startswith("open"):
            return "не стельная"
    if name == "блок на копыто" and "RECHECK" in desc.upper():
        return "повторный осмотр"
    return ""


#: Сокращения ветеринара в записи о хромоте, которые удалось прочитать.
LAME_CODES = {"BLK": "блок", "WRAP": "повязка", "SOLE": "подошва", "FRCTR": "трещина",
              "WART": "пальцевый дерматит", "LATD": "латеральный палец"}


def _lame_detail(desc: str) -> str:
    text = desc.upper()
    found = [ru for code, ru in LAME_CODES.items() if code in text]
    return ", ".join(found)


def registry(as_of: date, cows: Optional[list[str]] = None) -> dict[str, dict]:
    ids = cows or sorted(p.stem for p in HEALTH.glob("C*.csv"))
    out = {}
    for cow in ids:
        rec = cow_record(cow, as_of)
        if rec:
            out[cow] = rec.as_dict()
    return out
