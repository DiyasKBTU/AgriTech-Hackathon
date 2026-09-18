"""Платформа COW ID: веб-интерфейс и API.

Интерфейс — одна страница без внешних библиотек (`static/`): на ферме может
не быть интернета. Всё, что видит пользователь, доступно и через API — чтобы
события можно было передать в учётную систему (ERP), в мессенджер или
в мобильное приложение без переделки ядра.

Страницы: «Сегодня» (кого осмотреть), «Стадо» (признаки по всем коровам),
«Корова» (её норма и дни), «Камера» (живой просмотр), «Качество» (чем и как
проверены модели).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import PipelineConfig
from ..herd import cow_detail, herd_summary
from ..store.db import Store

STATIC = Path(__file__).parent / "static"


class FeedbackIn(BaseModel):
    verdict: Literal["confirmed", "false_alarm"]
    comment: str = ""


class LiveStartIn(BaseModel):
    source: Optional[str] = None
    url: Optional[str] = None
    camera: Optional[int] = None
    #: Как стоит камера: "top" — сверху (наш детектор), "side" — сбоку (COCO).
    view: Optional[str] = None


def lan_addresses() -> list[str]:
    """Адреса компьютера в локальной сети — их набирают на телефоне."""
    import socket

    found = []
    try:                                # основной адрес: куда ушёл бы пакет наружу
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("10.255.255.255", 1))
            found.append(sock.getsockname()[0])
    except OSError:
        pass
    try:
        found += socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        pass
    out = []
    for ip in found:
        if not ip.startswith(("127.", "169.254.", "0.")) and ip not in out:
            out.append(ip)
    return out


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


#: Где искать openssl, если его нет в PATH (ставится вместе с Git for Windows).
OPENSSL_PLACES = (r"C:\Program Files\Git\mingw64\bin\openssl.exe",
                  r"C:\Program Files\Git\usr\bin\openssl.exe")


def phone_certificate(folder: Path, addresses: list[str]) -> Optional[tuple[Path, Path]]:
    """Самоподписанный сертификат для https: без него браузер телефона не даёт камеру.

    Создаётся один раз через openssl; браузер предупредит, что сертификат
    не проверен, — это нормально для адреса в своей сети.
    """
    import os
    import shutil
    import subprocess

    cert, key = folder / "cert.pem", folder / "key.pem"
    if cert.exists() and key.exists():
        return cert, key
    openssl = shutil.which("openssl") or next(
        (p for p in OPENSSL_PLACES if Path(p).exists()), None)
    if openssl is None:
        return None
    folder.mkdir(parents=True, exist_ok=True)
    names = ",".join([f"IP:{ip}" for ip in ["127.0.0.1", *addresses]] + ["DNS:localhost"])
    done = subprocess.run(
        [openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "825",
         "-keyout", str(key), "-out", str(cert), "-subj", "/CN=COW ID",
         "-addext", f"subjectAltName={names}"],
        capture_output=True, text=True, env={**os.environ, "MSYS_NO_PATHCONV": "1"})
    if done.returncode != 0 or not cert.exists():
        return None
    return cert, key


def create_app(cfg: PipelineConfig) -> FastAPI:
    store = Store(cfg.storage.database)
    app = FastAPI(title="COW ID", docs_url="/api/docs")

    evidence_dir = Path(cfg.storage.evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/evidence", StaticFiles(directory=str(evidence_dir)), name="evidence")
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> HTMLResponse:
        # Метка версии в ссылках: после обновления браузер не возьмёт старый скрипт.
        version = int(max((STATIC / n).stat().st_mtime for n in ("app.js", "app.css")))
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        html = html.replace("/static/app.js", f"/static/app.js?v={version}")
        html = html.replace("/static/app.css", f"/static/app.css?v={version}")
        return HTMLResponse(html, headers={"Cache-Control": "no-cache"})

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "database": str(store.path)}

    # -- стадо и события ---------------------------------------------------

    @app.get("/api/summary")
    def summary() -> dict:
        runs = store.runs(limit=1)
        return {
            "camera_id": cfg.camera_id,
            "database": str(store.path),
            "animals": len(store.animals()),
            "days": store.days(),
            "events_by_status": store.feedback_summary(),
            "last_run": runs[0] if runs else None,
        }

    @app.get("/api/events")
    def events(status: Optional[str] = None, day: Optional[str] = None,
               limit: int = 500) -> list[dict]:
        rows = store.events(status=status, limit=limit)
        if day:
            rows = [r for r in rows if r["day"] == day]
        return rows

    @app.post("/api/events/{event_id}/feedback")
    def feedback(event_id: int, body: FeedbackIn) -> JSONResponse:
        try:
            store.add_feedback(event_id, body.verdict, body.comment)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse({"event_id": event_id, "status": body.verdict})

    @app.get("/api/herd")
    def herd(day: Optional[date] = None) -> dict:
        try:
            return herd_summary(store, cfg.baseline, day)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/cows/{cow_id}")
    def cow(cow_id: str) -> dict:
        detail = cow_detail(store, cfg.baseline, cow_id)
        if not detail["days"] and not detail["events"]:
            raise HTTPException(status_code=404, detail=f"Нет данных по корове {cow_id}")
        return detail

    @app.get("/api/registry/{cow_id}")
    def registry_card(cow_id: str, day: Optional[date] = None) -> dict:
        """Карточка коровы из реестра фермы на дату (для живого показа — на день съёмки)."""
        from ..live import ENGINE
        from ..registry import cow_record

        live_day = (ENGINE.state().get("farm") or {}).get("as_of")
        as_of = day or (date.fromisoformat(live_day) if live_day else None)
        if cfg.farm.registry != "mmcows" and not live_day:
            raise HTTPException(status_code=404, detail="Реестр фермы не подключён")
        if as_of is None:
            days = store.days()
            as_of = date.fromisoformat(days[-1]) if days else date.today()
        record = cow_record(cow_id, as_of)
        if record is None:
            raise HTTPException(status_code=404, detail=f"В реестре нет коровы {cow_id}")
        return record.as_dict()

    @app.get("/api/runs")
    def runs(limit: int = 50) -> list[dict]:
        return store.runs(limit=limit)

    # -- качество моделей --------------------------------------------------

    @app.get("/api/quality")
    def quality() -> dict:
        return {
            "detector": _read_json(Path("models/detector/evaluation.json")),
            "reid": _read_json(Path("models/reid/evaluation.json")),
            "reid_controls": _read_json(Path("models/reid/controls.json")),
            "reid_voting": _read_json(Path("models/reid/voting_effect.json")),
            "face_controls": _read_json(Path("models/reid_face/controls.json")),
            "video": _read_json(Path("reports/video_check/summary.json")),
            "video_unknown": _read_json(Path("reports/video_check/unknown_analysis.json")),
            "sensor_check": _read_json(Path("reports/mmcows/detector_on_sensors.json")),
            "lying_check": _read_json(Path("reports/mmcows/lying_check.json")),
            "farm_chain": _read_json(Path("reports/mmcows/chain.json")),
        }

    # -- живой просмотр ----------------------------------------------------

    @app.get("/api/live/sources")
    def live_sources() -> list[dict]:
        from ..live import available_sources

        return [s.as_dict() for s in available_sources()]

    @app.post("/api/live/start")
    def live_start(body: LiveStartIn) -> dict:
        from ..live import ENGINE

        try:
            return ENGINE.start(source_id=body.source, url=body.url, camera_index=body.camera,
                                view=body.view)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/live/stop")
    def live_stop() -> dict:
        from ..live import ENGINE

        ENGINE.stop()
        return {"running": False}

    @app.get("/api/live/state")
    def live_state() -> dict:
        from ..live import ENGINE

        return ENGINE.state()

    @app.post("/api/live/frame", include_in_schema=False)
    async def live_frame(request: Request) -> dict:
        """Кадр с камеры браузера (телефон или ноутбук), тело — JPEG."""
        from ..live import ENGINE

        data = await request.body()
        try:
            return ENGINE.push_frame(data)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/live/phone")
    def live_phone() -> dict:
        """Как открыть платформу с телефона: адреса и порт https (`cowid serve --phone`)."""
        phone = getattr(app.state, "phone", None) or {}
        return {"enabled": bool(phone), "https_port": phone.get("https_port"),
                "http_port": phone.get("http_port"), "addresses": lan_addresses() if phone else []}

    @app.get("/api/live/stream.mjpg", include_in_schema=False)
    def live_stream() -> StreamingResponse:
        from ..live import ENGINE

        def frames():
            idle = 0
            while idle < 30:
                jpeg = ENGINE.next_jpeg(timeout=1.0)
                if jpeg is None:
                    idle += 1
                    continue
                idle = 0
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")

        return StreamingResponse(frames(),
                                 media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/live/latest.jpg", include_in_schema=False)
    def live_latest() -> Response:
        """Последний кадр одним снимком — запасной путь, если поток mjpg не доходит
        до страницы (антивирус или прокси держат поток, пока он не закончится)."""
        from ..live import ENGINE

        jpeg = ENGINE.latest_jpeg()
        if jpeg is None:
            return Response(status_code=204)
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    return app
