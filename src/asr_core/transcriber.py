import gc
import logging
import threading
import time

import torch
from qwen_asr import Qwen3ASRModel

from .config import (
    ALLOWED_LANGUAGES,
    MODEL_DIR,
    MODEL_READY_TIMEOUT,
    MODEL_SIZE_DEFAULT,
)

logger = logging.getLogger(__name__)


class AudioTranscriber:
    """Qwen3-ASR model wrapper with async loading and lifecycle management."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or MODEL_SIZE_DEFAULT
        self._model = None
        self._ready = threading.Event()
        self._error: Exception | None = None
        self._lock = threading.Lock()
        self._load_thread: threading.Thread | None = None
        self._cancelled = False

    @property
    def error(self) -> Exception | None:
        """Public accessor for the last load error."""
        return self._error

    def load(self):
        """Start loading the model in a background thread."""
        with self._lock:
            if self._load_thread is not None and self._load_thread.is_alive():
                return
            self._cancelled = False
            self._ready.clear()
            self._error = None
            self._load_thread = threading.Thread(target=self._load, daemon=True)
            self._load_thread.start()

    def _load(self):
        try:
            local_path = MODEL_DIR / self.model_name
            if not local_path.is_dir():
                raise FileNotFoundError(
                    f"Model directory not found: {local_path}. "
                    f"Place the model at {local_path} or update model_dir in asr_core.toml."
                )

            logger.info("Loading model '%s' from %s", self.model_name, local_path)
            self._model = Qwen3ASRModel.from_pretrained(
                str(local_path),
                dtype=torch.bfloat16,
                device_map="cuda",
                max_inference_batch_size=1,
                max_new_tokens=1024,
                local_files_only=True,
            )
            logger.info("Model '%s' loaded successfully", self.model_name)
        except Exception as e:
            if not self._cancelled:
                logger.error("Failed to load model '%s': %s", self.model_name, e)
                self._error = e
        finally:
            self._ready.set()

    def unload(self):
        """Unload the model and free GPU memory."""
        with self._lock:
            self._cancelled = True

            # Wait for an in-progress load to finish before tearing down
            if self._load_thread is not None and self._load_thread.is_alive():
                self._load_thread.join(timeout=30)

            if self._model is not None:
                del self._model
                self._model = None
            self._ready.clear()
            self._error = None
            self._load_thread = None

        gc.collect()

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info("Model '%s' unloaded, GPU memory released", self.model_name)

    def wait_for_ready(self, timeout: float = MODEL_READY_TIMEOUT):
        if not self._ready.wait(timeout=timeout):
            raise RuntimeError(f"Model loading timed out after {timeout}s")
        if self._error:
            raise self._error
        return self._model

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set() and self._error is None and self._model is not None

    @property
    def state(self) -> str:
        if self._error:
            return "error"
        if self.is_ready:
            return "loaded"
        if self._load_thread is not None and self._load_thread.is_alive():
            return "loading"
        return "unloaded"

    def transcribe(self, audio_path: str, language: str = None) -> dict:
        model = self.wait_for_ready()

        start = time.monotonic()
        results = model.transcribe(audio=str(audio_path), language=language)
        duration = time.monotonic() - start

        detected = results[0].language.lower() if results[0].language else ""
        text = results[0].text.strip()

        # If model hallucinated a wrong language, force English as fallback
        if text and detected and detected not in ALLOWED_LANGUAGES:
            logger.warning(
                "Unexpected language '%s', re-transcribing as English", detected
            )
            results = model.transcribe(audio=str(audio_path), language="English")
            text = results[0].text.strip()
            detected = results[0].language.lower() if results[0].language else ""

        # Release intermediate GPU tensors cached by PyTorch's CUDA allocator
        # to prevent gradual memory growth across repeated calls.
        del results
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        return {
            "text": text,
            "detected_language": detected,
            "duration_seconds": duration,
        }
