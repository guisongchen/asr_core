"""Supervisor-side handle for the ASR model worker subprocess.

Keeps the public interface of the old in-process AudioTranscriber
(load / unload / wait_for_ready / is_ready / state / transcribe) so api.py
needs only minimal changes — but the model itself lives in a spawned child
process (see worker.py). Unload = terminate the child, which guarantees the
CUDA driver reclaims all of its VRAM.
"""

import logging
import multiprocessing as mp
import shutil
import subprocess
import threading

from .config import MODEL_DIR, MODEL_READY_TIMEOUT, MODEL_SIZE_DEFAULT
from .worker import run_worker

logger = logging.getLogger(__name__)

# Long audio files can take minutes; transcribe waits far longer than load.
TRANSCRIBE_TIMEOUT = 600


def _pid_gpu_memory_mb(pid: int) -> float | None:
    """VRAM used by a specific PID, via nvidia-smi (no CUDA context needed)."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and int(parts[0]) == pid:
                return float(parts[1])
        return 0.0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError, ValueError):
        return None


class AudioTranscriber:
    """Qwen3-ASR worker-process wrapper with async loading and lifecycle management."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or MODEL_SIZE_DEFAULT
        self._proc: mp.Process | None = None
        self._conn = None
        self._ready = threading.Event()
        self._error: Exception | None = None
        self._lock = threading.Lock()  # serializes pipe access
        self._cancelled = False
        self._stopped_cleanly = False

    @property
    def error(self) -> Exception | None:
        """Public accessor for the last load error."""
        return self._error

    def load(self):
        """Spawn the worker process; the model loads inside it."""
        with self._lock:
            if self._proc is not None and self._proc.is_alive():
                return
            self._cancelled = False
            self._stopped_cleanly = False
            self._ready.clear()
            self._error = None
            ctx = mp.get_context("spawn")
            parent_conn, child_conn = ctx.Pipe()
            self._conn = parent_conn
            self._proc = ctx.Process(
                target=run_worker,
                args=(child_conn, self.model_name, str(MODEL_DIR)),
                daemon=True,
                name=f"asr-worker-{self.model_name}",
            )
            self._proc.start()
            child_conn.close()
            logger.info("Worker spawned for model '%s' (pid %s)", self.model_name, self._proc.pid)

    def _await_first_message(self, timeout: float):
        """Consume the worker's ready/load_error handshake. Caller holds _lock."""
        if self._ready.is_set() or self._error is not None:
            return
        if self._conn is None or not self._conn.poll(timeout):
            raise RuntimeError(f"Model loading timed out after {timeout}s")
        msg = self._conn.recv()
        if msg.get("event") == "ready":
            self._ready.set()
        else:
            self._error = RuntimeError(msg.get("error", "unknown load error"))

    def unload(self):
        """Terminate the worker; process exit frees all of its VRAM."""
        with self._lock:
            self._cancelled = True
            proc, conn = self._proc, self._conn
            self._proc = None
            self._conn = None
            self._ready.clear()
            self._error = None
            self._stopped_cleanly = True

        if proc is None:
            return
        try:
            if conn is not None and proc.is_alive():
                conn.send({"cmd": "shutdown"})
        except (OSError, BrokenPipeError):
            pass
        proc.join(timeout=15)
        if proc.is_alive():
            # Worker may be blocked in model load or inference and never saw
            # the shutdown request — escalate.
            logger.warning("Worker pid %s did not exit, terminating", proc.pid)
            proc.terminate()
            proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        logger.info("Model '%s' unloaded, worker reaped, VRAM released", self.model_name)

    def wait_for_ready(self, timeout: float = MODEL_READY_TIMEOUT):
        with self._lock:
            self._await_first_message(timeout)
            if self._error:
                raise self._error
            return self

    @property
    def is_ready(self) -> bool:
        return (
            self._ready.is_set()
            and self._error is None
            and self._proc is not None
            and self._proc.is_alive()
        )

    @property
    def state(self) -> str:
        if self._error:
            return "error"
        if self._proc is not None and not self._proc.is_alive() and not self._stopped_cleanly:
            return "error"  # worker died unexpectedly (OOM kill, crash)
        if self.is_ready:
            return "loaded"
        if self._proc is not None and self._proc.is_alive():
            return "loading"
        return "unloaded"

    def gpu_memory_mb(self) -> float | None:
        proc = self._proc
        if proc is None or not proc.is_alive() or proc.pid is None:
            return None
        return _pid_gpu_memory_mb(proc.pid)

    def transcribe(self, audio_path: str, language: str = None) -> dict:
        with self._lock:
            self._await_first_message(MODEL_READY_TIMEOUT)
            if self._error:
                raise self._error
            if self._proc is None or not self._proc.is_alive():
                raise RuntimeError("Worker process is not running")
            self._conn.send(
                {"cmd": "transcribe", "audio_path": str(audio_path), "language": language}
            )
            if not self._conn.poll(TRANSCRIBE_TIMEOUT):
                raise RuntimeError(f"Transcription timed out after {TRANSCRIBE_TIMEOUT}s")
            msg = self._conn.recv()

        if msg.get("event") == "result":
            return {
                "text": msg["text"],
                "detected_language": msg.get("detected_language", ""),
                "duration_seconds": msg.get("duration_seconds"),
            }
        raise RuntimeError(msg.get("error", "unknown transcribe error"))
