"""«Живой залог»: подтверждение поголовья системой наблюдения CV-трека.

Банк не принимает скот в залог, потому что не может убедиться, что он есть.
Здесь ERP-список голов в помещении под наблюдением сверяется с тем, кого
система наблюдения CV-трека (COW ID) видела за последние сутки.

Честно: в демо система работает на открытом наборе MmCows (10 коров, присутствие
по датчикам). Узнавание именно по камере проверено отдельно (метрики берём из
того же API). Сверка с ИСЖ не подключена.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

CV_URL = "http://127.0.0.1:8000"


def _get(path: str, timeout: float) -> dict:
    with urllib.request.urlopen(CV_URL + path, timeout=timeout) as r:
        return json.load(r)


def load_presence(snapshot: Path, timeout: float = 1.0) -> dict:
    """Живой ответ CV-трека; если сервер не запущен — сохранённый снимок."""
    try:
        herd = _get("/api/herd", timeout)
        quality = _get("/api/quality", timeout)
        return {"live": True, "source": f"CV-трек COW ID, {CV_URL}", "taken": herd.get("day"),
                "herd": herd, "reid": quality.get("reid") or {}, "video": quality.get("video") or {}}
    except Exception:
        snap = json.loads(snapshot.read_text(encoding="utf-8"))
        return {"live": False, "source": snap["source"], "taken": snap["taken"][:10],
                "herd": snap["herd"], "reid": snap.get("reid") or {}, "video": snap.get("video") or {}}


def verify(card: dict, presence: dict) -> dict:
    """Сверка ERP-списка голов под наблюдением с тем, кого видела система."""
    mon = card.get("monitored")
    if not mon:
        return {"connected": False}
    cows = {c["cow_id"]: c for c in presence["herd"].get("cows", [])}
    rows = []
    for tag, cv_id in mon["tags"].items():
        c = cows.get(cv_id)
        seen = bool(c and (c.get("observed_h") or 0) > 0)
        rows.append({"tag": tag, "cv_id": cv_id, "seen": seen,
                     "hours": (c or {}).get("observed_h"), "status": (c or {}).get("status_label", "—"),
                     "signs": [s["name"] for s in (c or {}).get("signs", [])]})
    seen = sum(r["seen"] for r in rows)
    reid, video = presence.get("reid", {}), presence.get("video", {})
    return {
        "connected": True, "place": mon["place"], "day": presence["herd"].get("day"),
        "expected": len(rows), "seen": seen, "share": seen / len(rows) if rows else 0.0,
        "rows": rows, "live": presence["live"], "source": presence["source"], "taken": presence["taken"],
        "herd_total": card.get("herd_total"),
        "reid_rank1": reid.get("rank1"), "reid_new": reid.get("n_test_identities"),
        "video_identified": video.get("identified"), "video_tracks": video.get("tracks"),
    }
