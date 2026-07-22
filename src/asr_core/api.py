import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import psutil
import torch
from fastapi import FastAPI, HTTPException

from .audio import AudioPreprocessor
from .config import MODEL_CHOICES, MODEL_SIZE_DEFAULT
from .models import LoadRequest, ModelStatus, TranscribeRequest, TranscribeResponse
from .transcriber import AudioTranscriber

logger = logging.getLogger(__name__)

_transcriber: AudioTranscriber | None = None
_transcriber_lock = threading.Lock()
_stats_lock = threading.Lock()
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
    """Get the current transcriber or raise 503. Must be called under _transcriber_lock."""
    if _transcriber is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _transcriber


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    with _transcriber_lock:
        if _transcriber is not None:
            _transcriber.unload()


app = FastAPI(title="ASRCore", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status", response_model=ModelStatus)
def status():
    with _transcriber_lock:
        if _transcriber is None:
            with _stats_lock:
                return ModelStatus(state="unloaded", gpu_memory_mb=_gpu_memory_mb(), **_stats)
        with _stats_lock:
            return ModelStatus(
                state=_transcriber.state,
                model_name=_transcriber.model_name,
                error_message=str(_transcriber.error) if _transcriber.error else None,
                gpu_memory_mb=_gpu_memory_mb(),
                **_stats,
            )


@app.post("/load")
def load(req: LoadRequest):
    global _transcriber

    with _transcriber_lock:
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

    # Wait for the model to be ready before returning
    try:
        _transcriber.wait_for_ready()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load failed: {e}")

    return {"state": _transcriber.state, "model_name": req.model_name}


@app.post("/unload")
def unload():
    global _transcriber
    with _transcriber_lock:
        if _transcriber is not None:
            _transcriber.unload()
            _transcriber = None
    return {"state": "unloaded"}


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(req: TranscribeRequest):
    global _transcriber

    with _stats_lock:
        _stats["requests_total"] += 1

    processed_path = None
    try:
        with _transcriber_lock:
            requested = req.model_name
            if requested:
                if _transcriber is None or _transcriber.model_name != requested:
                    # Release lock before calling load() which acquires it
                    pass
                else:
                    requested = None  # already loaded
            elif _transcriber is None:
                requested = MODEL_SIZE_DEFAULT

        # Load outside the lock if needed (load() acquires the lock internally)
        if requested:
            load(LoadRequest(model_name=requested))

        with _transcriber_lock:
            transcriber = _get_transcriber()

        processed_path = AudioPreprocessor.preprocess(req.audio_path)
        result = transcriber.transcribe(processed_path, language=req.language)
        return TranscribeResponse(**result)
    except HTTPException:
        with _stats_lock:
            _stats["requests_failed"] += 1
        raise
    except Exception as e:
        with _stats_lock:
            _stats["requests_failed"] += 1
        logger.exception("Transcription failed for '%s'", req.audio_path)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if processed_path and processed_path != req.audio_path:
            Path(processed_path).unlink(missing_ok=True)


@app.get("/stats")
def stats():
    with _stats_lock:
        return {
            **_stats,
            "gpu_memory_mb": _gpu_system_mb(),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
        }
