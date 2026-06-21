import os
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..client import ASRCoreClient
from ..config import MODEL_CHOICES, SOCKET_PATH

app = FastAPI(title="ASRCore Dashboard")

static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))


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


def _full_status() -> dict:
    return {"asr": _asr_status(), "stats": _asr_stats()}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    status = _full_status()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "asr": status["asr"],
            "stats": status["stats"],
            "models": MODEL_CHOICES,
        },
    )


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


@app.get("/partials/status", response_class=HTMLResponse)
def partial_status(request: Request):
    status = _full_status()
    return templates.TemplateResponse(
        request,
        "partials/status.html",
        {
            "asr": status["asr"],
            "stats": status["stats"],
            "models": MODEL_CHOICES,
        },
    )
