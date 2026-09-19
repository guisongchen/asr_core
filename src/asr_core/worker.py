"""Model worker subprocess — the only place torch / CUDA is touched.

The supervisor (FastAPI daemon) never imports torch for CUDA work; it spawns
this worker via ``multiprocessing.get_context("spawn")`` and talks to it over a
pipe. Unloading the model = terminating this process, so the CUDA driver
reclaims ALL of its VRAM deterministically — no PyTorch allocator residue.

Protocol (multiprocessing.Pipe, plain dicts):
    supervisor → worker:  {"cmd": "transcribe", "audio_path": str, "language": str | None}
                          {"cmd": "shutdown"}
    worker → supervisor:  {"event": "ready"}                       (model loaded)
                          {"event": "load_error", "error": str}
                          {"event": "result", "text": str, "detected_language": str,
                           "duration_seconds": float}
                          {"event": "transcribe_error", "error": str}

``model_name=None`` puts the worker in idle test mode: it sends "ready"
immediately without importing torch, so the protocol is testable without a GPU.
"""

import gc
import logging
from pathlib import Path

from .config import ALLOWED_LANGUAGES

logger = logging.getLogger(__name__)


def _load_model(model_name: str, model_dir: str):
    import torch
    from qwen_asr import Qwen3ASRModel

    local_path = Path(model_dir) / model_name
    if not local_path.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {local_path}. "
            f"Place the model at {local_path} or update model_dir in asr_core.toml."
        )
    logger.info("Worker loading model '%s' from %s", model_name, local_path)
    return Qwen3ASRModel.from_pretrained(
        str(local_path),
        dtype=torch.bfloat16,
        device_map="cuda",
        max_inference_batch_size=1,
        max_new_tokens=1024,
        local_files_only=True,
    )


def _transcribe(model, audio_path: str, language: str | None) -> dict:
    import time

    start = time.monotonic()
    results = model.transcribe(audio=str(audio_path), language=language)
    duration = time.monotonic() - start

    detected = results[0].language.lower() if results[0].language else ""
    text = results[0].text.strip()

    # If model hallucinated a wrong language, force English as fallback
    if text and detected and detected not in ALLOWED_LANGUAGES:
        logger.warning("Unexpected language '%s', re-transcribing as English", detected)
        results = model.transcribe(audio=str(audio_path), language="English")
        text = results[0].text.strip()
        detected = results[0].language.lower() if results[0].language else ""

    # Release intermediate GPU tensors cached by PyTorch's CUDA allocator
    del results
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "text": text,
        "detected_language": detected,
        "duration_seconds": duration,
    }


def run_worker(conn, model_name: str | None, model_dir: str):
    """Worker entry point — runs in the child process."""
    model = None
    try:
        if model_name is not None:
            try:
                model = _load_model(model_name, model_dir)
            except Exception as e:
                logger.error("Worker failed to load model '%s': %s", model_name, e)
                conn.send({"event": "load_error", "error": str(e)})
                return
            logger.info("Worker loaded model '%s'", model_name)

        conn.send({"event": "ready"})

        while True:
            msg = conn.recv()
            cmd = msg.get("cmd")
            if cmd == "shutdown":
                break
            if cmd == "transcribe":
                if model is None:
                    conn.send({"event": "transcribe_error", "error": "no model loaded"})
                    continue
                try:
                    conn.send(
                        {
                            "event": "result",
                            **_transcribe(model, msg["audio_path"], msg.get("language")),
                        }
                    )
                except Exception as e:
                    logger.exception("Worker transcription failed")
                    conn.send({"event": "transcribe_error", "error": str(e)})
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        # Process exit reclaims VRAM no matter what; this just lets a clean
        # shutdown release the model a bit earlier.
        model = None
        gc.collect()
