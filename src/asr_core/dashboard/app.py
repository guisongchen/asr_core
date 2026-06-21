import os
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..client import ASRCoreClient
from ..config import MODEL_CHOICES, SOCKET_PATH
from .systemd import SystemdManager

app = FastAPI(title="ASRCore Dashboard")

static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

systemd = SystemdManager(user=True)


def _asr_status() -> dict:
    if not os.path.exists(SOCKET_PATH):
        return {"state": "unloaded", "model_name": None, "gpu_memory_mb": None}
    try:
        with ASRCoreClient(auto_start=False) as client:
            return client.status()
    except Exception as e:
        return {"state": "error", "model_name": None, "error_message": str(e)}


def _asr_stats() -> dict:
    if not os.path.exists(SOCKET_PATH):
        return {}
    try:
        with ASRCoreClient(auto_start=False) as client:
            return client.stats()
    except Exception:
        return {}


def _services() -> list[dict]:
    return [
        {"name": s.name, "active": s.active, "status": s.status, "uptime": s.uptime}
        for s in systemd.all_statuses()
    ]


def _full_status() -> dict:
    return {"asr": _asr_status(), "stats": _asr_stats(), "services": _services()}


# ── pages ────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"asr": _asr_status(), "stats": _asr_stats(), "services": _services(), "models": MODEL_CHOICES},
    )


# ── api ───────────────────────────────────────────────────────────

@app.get("/api/status")
def api_status():
    return _full_status()


@app.post("/api/load", response_class=HTMLResponse)
def api_load(request: Request, model_name: str = Form(...)):
    try:
        with ASRCoreClient(auto_start=False) as client:
            client.load_model(model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return partial_status(request)


@app.post("/api/unload", response_class=HTMLResponse)
def api_unload(request: Request):
    try:
        with ASRCoreClient(auto_start=False) as client:
            client.unload_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return partial_status(request)


@app.post("/api/services/{name}/start", response_class=HTMLResponse)
def api_service_start(request: Request, name: str):
    systemd.start(name)
    return partial_status(request)


@app.post("/api/services/{name}/stop", response_class=HTMLResponse)
def api_service_stop(request: Request, name: str):
    systemd.stop(name)
    return partial_status(request)


@app.post("/api/services/{name}/restart", response_class=HTMLResponse)
def api_service_restart(request: Request, name: str):
    systemd.restart(name)
    return partial_status(request)


@app.get("/api/services/{name}/logs")
def api_service_logs(name: str, lines: int = 50):
    return {"name": name, "logs": systemd.logs(name, lines)}


@app.post("/api/refresh")
def api_refresh():
    return Response(
        status_code=200,
        headers={"HX-Trigger": '{"refresh-status": "", "refresh-logs": ""}'},
    )


# ── partials ──────────────────────────────────────────────────────

@app.get("/partials/status", response_class=HTMLResponse)
def partial_status(request: Request):
    return templates.TemplateResponse(
        request,
        "partials/status.html",
        {"asr": _asr_status(), "stats": _asr_stats(), "services": _services(), "models": MODEL_CHOICES},
    )


@app.get("/partials/logs/{name}", response_class=HTMLResponse)
def partial_logs(request: Request, name: str, lines: int = 50):
    return _render_logs(request, name, lines)


@app.get("/partials/logs", response_class=HTMLResponse)
def partial_logs_query(request: Request, name: str, lines: int = 50):
    return _render_logs(request, name, lines)


def _render_logs(request: Request, name: str, lines: int):
    return templates.TemplateResponse(
        request,
        "partials/logs.html",
        {"name": name, "logs": systemd.logs(name, lines)},
    )
