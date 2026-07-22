import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from .config import MODEL_SIZE_DEFAULT, SOCKET_PATH

logger = logging.getLogger(__name__)


def _find_python() -> str:
    """Find the best Python interpreter for spawning the daemon."""
    # 1. Project venv (editable install layout)
    project_root = Path(__file__).parent.parent.parent
    venv_python = project_root / ".venv" / "bin" / "python3"
    if venv_python.exists():
        return str(venv_python)
    # 2. Current interpreter (works if asr_core is importable)
    return sys.executable


class ASRCoreClient:
    """Client for the ASRCore HTTP-over-Unix-socket service."""

    def __init__(self, socket_path: str = SOCKET_PATH, auto_start: bool = True):
        self.socket_path = socket_path
        self.auto_start = auto_start
        self._transport = httpx.HTTPTransport(uds=socket_path)
        self._client = httpx.Client(
            transport=self._transport, timeout=300.0, proxy=None
        )

    def _ensure_running(self):
        if os.path.exists(self.socket_path):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                    s.settimeout(0.5)
                    s.connect(self.socket_path)
                    return
            except OSError:
                pass
        if not self.auto_start:
            raise ConnectionError(
                f"ASRCore socket not found at {self.socket_path}"
            )
        self._spawn_daemon()

    def _spawn_daemon(self):
        python = _find_python()
        logger.info("Auto-starting ASRCore daemon with %s", python)
        subprocess.Popen(
            [python, "-m", "asr_core", "--detach"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if os.path.exists(self.socket_path):
                try:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                        s.settimeout(0.5)
                        s.connect(self.socket_path)
                        logger.info("ASRCore daemon started successfully")
                        return
                except OSError:
                    pass
            time.sleep(0.2)
        raise ConnectionError("ASRCore daemon did not start within 10s")

    def _request(self, method: str, path: str, **kwargs):
        self._ensure_running()
        url = f"http://asr_core{path}"
        response = self._client.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()

    def health(self) -> dict:
        return self._request("GET", "/health")

    def status(self) -> dict:
        return self._request("GET", "/status")

    def load_model(self, model_name: str = MODEL_SIZE_DEFAULT) -> dict:
        return self._request("POST", "/load", json={"model_name": model_name})

    def unload_model(self) -> dict:
        return self._request("POST", "/unload")

    def transcribe(
        self, audio_path: str, model_name: str = None, language: str = None
    ) -> dict:
        payload = {"audio_path": str(audio_path)}
        if model_name:
            payload["model_name"] = model_name
        if language:
            payload["language"] = language
        return self._request("POST", "/transcribe", json=payload)

    def stats(self) -> dict:
        return self._request("GET", "/stats")

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
