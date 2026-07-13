import time
from contextlib import asynccontextmanager
from pathlib import Path

import psutil
import torch
from fastapi import FastAPI, HTTPException

from .audio import AudioPreprocessor
from .config import MODEL_CHOICES, MODEL_SIZE_DEFAULT, SOCKET_PATH
from .models import LoadRequest, ModelStatus, TranscribeRequest, TranscribeResponse
from .transcriber import AudioTranscriber

_transcriber: AudioTranscriber | None = None
_stats = {"requests_total": 0, "requests_failed": 0}


def _gpu_memory_mb() -> float | None:
    try:
        return torch.cuda.memory_allocated() / 1024 / 1024
    except Exception:
        return None


def _gpu_system_mb() -> float | None:
    try:
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1024 / 1024
    except Exception:
        return None


def _get_transcriber() -> AudioTranscriber:
    if _transcriber is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _transcriber


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if _transcriber is not None:
        _transcriber.unload()


app = FastAPI(title="ASRCore", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status", response_model=ModelStatus)
def status():
    if _transcriber is None:
        return ModelStatus(state="unloaded", gpu_memory_mb=_gpu_memory_mb(), **_stats)
    return ModelStatus(
        state=_transcriber.state,
        model_name=_transcriber.model_name,
        error_message=str(_transcriber._error) if _transcriber._error else None,
        gpu_memory_mb=_gpu_memory_mb(),
        **_stats,
    )


@app.post("/load")
def load(req: LoadRequest):
    global _transcriber
    if _transcriber is not None and _transcriber.model_name == req.model_name:
        if _transcriber.state == "error":
            _transcriber.unload()
        else:
            return {"state": _transcriber.state, "model_name": req.model_name}

    if _transcriber is not None:
        _transcriber.unload()

    if req.model_name not in MODEL_CHOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{req.model_name}'. Choices: {MODEL_CHOICES}",
        )

    _transcriber = AudioTranscriber(model_name=req.model_name)
    _transcriber.load()
    return {"state": _transcriber.state, "model_name": req.model_name}


@app.post("/unload")
def unload():
    global _transcriber
    if _transcriber is not None:
        _transcriber.unload()
        _transcriber = None
    return {"state": "unloaded"}


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(req: TranscribeRequest):
    global _transcriber
    _stats["requests_total"] += 1

    processed_path = None
    try:
        requested = req.model_name
        if requested:
            if _transcriber is None or _transcriber.model_name != requested:
                load(LoadRequest(model_name=requested))
        elif _transcriber is None:
            load(LoadRequest(model_name=MODEL_SIZE_DEFAULT))

        transcriber = _get_transcriber()
        processed_path = AudioPreprocessor.preprocess(req.audio_path)
        result = transcriber.transcribe(processed_path, language=req.language)
        return TranscribeResponse(**result)
    except HTTPException:
        _stats["requests_failed"] += 1
        raise
    except Exception as e:
        _stats["requests_failed"] += 1
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if processed_path and processed_path != req.audio_path:
            Path(processed_path).unlink(missing_ok=True)


@app.get("/stats")
def stats():
    return {
        **_stats,
        "gpu_memory_mb": _gpu_system_mb(),
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
    }
