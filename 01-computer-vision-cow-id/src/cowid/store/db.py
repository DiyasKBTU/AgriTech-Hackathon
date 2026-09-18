"""Хранилище: единственный источник правды о животных, сутках и событиях.

SQLite выбран сознательно. На ферме нет администратора баз данных, а часто нет
и стабильного интернета. SQLite — это один файл без отдельного сервера: его
можно скопировать, переслать, открыть любым инструментом. Когда хозяйство
подключит несколько площадок и центральный сервер, схема переносится
в PostgreSQL без изменений в логике — запросы здесь стандартные.

Персональная норма животного в базе не хранится. Она каждый раз строится
заново по суточным показателям — это дёшево (сотни чисел на животное) и
исключает расхождение между «нормой в памяти» и фактами в базе.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator, Optional

from ..types import ActivityFeatures, Event

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id    TEXT NOT NULL,
    video        TEXT NOT NULL,
    day          TEXT NOT NULL,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    frames       INTEGER DEFAULT 0,
    tracklets    INTEGER DEFAULT 0,
    identified   INTEGER DEFAULT 0,
    stats        TEXT
);

CREATE TABLE IF NOT EXISTS animals (
    cow_id       TEXT PRIMARY KEY,
    first_seen   TEXT NOT NULL,
    last_seen    TEXT NOT NULL,
    id_source    TEXT
);

-- Суточные показатели. Одно животное может попадать в несколько камер,
-- поэтому ключ включает камеру; сводка по животному — сумма по камерам.
CREATE TABLE IF NOT EXISTS daily_features (
    cow_id            TEXT NOT NULL,
    day               TEXT NOT NULL,
    camera_id         TEXT NOT NULL,
    feeder_seconds    REAL DEFAULT 0,
    drinker_seconds   REAL DEFAULT 0,
    drinker_visits    INTEGER DEFAULT 0,
    resting_seconds   REAL DEFAULT 0,
    standing_seconds  REAL DEFAULT 0,
    distance_m        REAL DEFAULT 0,
    observed_seconds  REAL DEFAULT 0,
    tracks_count      INTEGER DEFAULT 0,
    milk_kg           REAL,
    PRIMARY KEY (cow_id, day, camera_id)
);

CREATE TABLE IF NOT EXISTS events (
    event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    cow_id       TEXT NOT NULL,
    day          TEXT NOT NULL,
    severity     TEXT NOT NULL,
    title        TEXT NOT NULL,
    detail       TEXT NOT NULL,
    deviations   TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    -- new | confirmed | false_alarm
    status       TEXT NOT NULL DEFAULT 'new',
    UNIQUE (cow_id, day, title)
);

-- Ответ зоотехника на событие. Это не просто отметка в интерфейсе:
-- подтверждённые события и ложные тревоги — готовая разметка, по которой
-- потом настраиваются пороги и дообучается модель.
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id     INTEGER NOT NULL REFERENCES events(event_id),
    verdict      TEXT NOT NULL CHECK (verdict IN ('confirmed', 'false_alarm')),
    comment      TEXT,
    created_at   TEXT NOT NULL
);

-- Кадры-доказательства: пути к снимкам и траектории по животному за сутки.
CREATE TABLE IF NOT EXISTS evidence (
    cow_id       TEXT NOT NULL,
    day          TEXT NOT NULL,
    payload      TEXT NOT NULL,
    PRIMARY KEY (cow_id, day)
);

CREATE INDEX IF NOT EXISTS idx_events_day ON events(day);
CREATE INDEX IF NOT EXISTS idx_features_cow ON daily_features(cow_id, day);
"""

FEATURE_COLUMNS = (
    "feeder_seconds", "drinker_seconds", "drinker_visits", "resting_seconds",
    "standing_seconds", "distance_m", "observed_seconds", "tracks_count", "milk_kg",
)

#: Как сводить сутки из нескольких источников. Время и путь с разных камер
#: складываются; удой приходит из одного места, его складывать нельзя.
AGGREGATE = {c: "SUM" for c in FEATURE_COLUMNS}
AGGREGATE["milk_kg"] = "MAX"


def _select_features() -> str:
    return ", ".join(f"{AGGREGATE[c]}({c}) AS {c}" for c in FEATURE_COLUMNS)


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            # База, созданная до появления удоя, получает новый столбец.
            columns = {r["name"] for r in conn.execute("PRAGMA table_info(daily_features)")}
            if "milk_kg" not in columns:
                conn.execute("ALTER TABLE daily_features ADD COLUMN milk_kg REAL")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- прогоны ----------------------------------------------------------

    def start_run(self, camera_id: str, video: str, day: date) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO runs (camera_id, video, day, started_at) VALUES (?, ?, ?, ?)",
                (camera_id, video, day.isoformat(), _now()),
            )
            return int(cur.lastrowid)

    def finish_run(self, run_id: int, frames: int, tracklets: int,
                   identified: int, stats: dict) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE runs SET finished_at=?, frames=?, tracklets=?, identified=?, stats=? "
                "WHERE run_id=?",
                (_now(), frames, tracklets, identified,
                 json.dumps(stats, ensure_ascii=False), run_id),
            )

    def runs(self, limit: int = 50) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY run_id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row(r, json_fields=("stats",)) for r in rows]

    # -- животные и сутки -------------------------------------------------

    def save_features(self, camera_id: str, features: list[ActivityFeatures],
                      sources: Optional[dict[str, str]] = None) -> None:
        sources = sources or {}
        with self.connect() as conn:
            for f in features:
                day = f.day.isoformat()
                conn.execute(
                    f"INSERT OR REPLACE INTO daily_features "
                    f"(cow_id, day, camera_id, {', '.join(FEATURE_COLUMNS)}) "
                    f"VALUES (?, ?, ?, {', '.join('?' for _ in FEATURE_COLUMNS)})",
                    (f.cow_id, day, camera_id, *[getattr(f, c) for c in FEATURE_COLUMNS]),
                )
                conn.execute(
                    "INSERT INTO animals (cow_id, first_seen, last_seen, id_source) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(cow_id) DO UPDATE SET "
                    "  first_seen = MIN(first_seen, excluded.first_seen), "
                    "  last_seen = MAX(last_seen, excluded.last_seen), "
                    "  id_source = COALESCE(excluded.id_source, id_source)",
                    (f.cow_id, day, day, sources.get(f.cow_id)),
                )

    def features_before(self, day: date) -> list[ActivityFeatures]:
        """Все суточные показатели до указанной даты, сведённые по камерам.

        Из них перед оценкой новых суток заново строится персональная норма.
        """
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT cow_id, day, {_select_features()} "
                f"FROM daily_features WHERE day < ? GROUP BY cow_id, day ORDER BY day",
                (day.isoformat(),),
            ).fetchall()
        return [_features(r) for r in rows]

    def animals(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM animals ORDER BY cow_id").fetchall()
        return [_row(r) for r in rows]

    def animal_history(self, cow_id: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT day, {_select_features()} "
                f"FROM daily_features WHERE cow_id = ? GROUP BY day ORDER BY day",
                (cow_id,),
            ).fetchall()
        return [_row(r) for r in rows]

    def days(self) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT day FROM daily_features ORDER BY day"
            ).fetchall()
        return [r["day"] for r in rows]

    # -- события и ответы зоотехника -------------------------------------

    def save_events(self, events: list[Event]) -> int:
        saved = 0
        with self.connect() as conn:
            for e in events:
                d = e.as_dict()
                cur = conn.execute(
                    "INSERT OR IGNORE INTO events "
                    "(cow_id, day, severity, title, detail, deviations, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (d["cow_id"], d["day"], d["severity"], d["title"], d["detail"],
                     json.dumps(d["deviations"], ensure_ascii=False), d["created_at"]),
                )
                saved += cur.rowcount
        return saved

    def events(self, status: Optional[str] = None, limit: int = 200) -> list[dict]:
        query = "SELECT * FROM events"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += (" ORDER BY day DESC, CASE severity WHEN 'alert' THEN 0 "
                  "WHEN 'warning' THEN 1 ELSE 2 END, event_id DESC LIMIT ?")
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row(r, json_fields=("deviations",)) for r in rows]

    def add_feedback(self, event_id: int, verdict: str, comment: str = "") -> None:
        if verdict not in ("confirmed", "false_alarm"):
            raise ValueError("verdict должен быть 'confirmed' или 'false_alarm'")
        with self.connect() as conn:
            found = conn.execute(
                "SELECT 1 FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
            if not found:
                raise KeyError(f"событие {event_id} не найдено")
            conn.execute(
                "INSERT INTO feedback (event_id, verdict, comment, created_at) "
                "VALUES (?, ?, ?, ?)",
                (event_id, verdict, comment, _now()),
            )
            conn.execute("UPDATE events SET status = ? WHERE event_id = ?",
                         (verdict, event_id))

    def feedback_summary(self) -> dict[str, int]:
        """Сколько событий подтверждено и сколько оказалось ложными.

        Это и есть настоящая точность системы на конкретной ферме — та цифра,
        которую нельзя получить ни на одном датасете.
        """
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM events GROUP BY status"
            ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    # -- доказательства ---------------------------------------------------

    def save_evidence(self, cow_id: str, day: date, payload: dict) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO evidence (cow_id, day, payload) VALUES (?, ?, ?)",
                (cow_id, day.isoformat(), json.dumps(payload, ensure_ascii=False)),
            )

    def evidence(self, cow_id: str, day: str) -> Optional[dict]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT payload FROM evidence WHERE cow_id = ? AND day = ?", (cow_id, day)
            ).fetchone()
        return json.loads(row["payload"]) if row else None


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _row(row: sqlite3.Row, json_fields: tuple[str, ...] = ()) -> dict:
    d = dict(row)
    for key in json_fields:
        if d.get(key):
            d[key] = json.loads(d[key])
    return d


def _features(row: sqlite3.Row) -> ActivityFeatures:
    return ActivityFeatures(
        cow_id=row["cow_id"],
        day=date.fromisoformat(row["day"]),
        **{c: row[c] for c in FEATURE_COLUMNS},
    )
