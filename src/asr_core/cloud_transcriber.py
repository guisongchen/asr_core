"""Cloud ASR backend: Alibaba DashScope qwen3-asr-flash (OpenAI-compatible).

Mirrors the AudioTranscriber public interface (load / unload /
wait_for_ready / is_ready / state / transcribe) so api.py can swap backends
transparently. No GPU, worker process, or model files are involved: "load"
only validates configuration, and transcribe() sends the audio as a base64
data URI to the DashScope chat/completions endpoint. Usage is billed per
token on the DashScope account.

Limits (per DashScope docs): base64-encoded audio must stay under 10 MB,
which at 16 kHz mono WAV (~32 KB/s raw, ~43 KB/s base64) allows roughly
4 minutes per request — far above typical voice-input snippets.
"""

import base64
import logging
import mimetypes
import os
import threading
from pathlib import Path

import httpx

from .config import CLOUD_MODEL_DEFAULT, DASHSCOPE_BASE_URL

logger = logging.getLogger(__name__)

# Generous ceiling: short voice snippets return in seconds, but a slow
# network plus a multi-minute clip should not spuriously time out.
TRANSCRIBE_TIMEOUT = 180

# One retry on transport-level failures (connection reset, flaky DNS).
MAX_TRANSPORT_RETRIES = 1

# Base64-encoded request ceiling from the DashScope docs.
MAX_BASE64_BYTES = 10 * 1024 * 1024

# asr_core uses language names; DashScope asr_options.language wants codes.
_LANGUAGE_CODE_MAP = {
    "chinese": "zh",
    "english": "en",
    "cantonese": "yue",
    "japanese": "ja",
    "korean": "ko",
}

_AUDIO_MIME_BY_SUFFIX = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".aac": "audio/aac",
}


def _audio_data_uri(audio_path: str) -> str:
    """Encode a local audio file as a data URI for input_audio.data."""
    path = Path(audio_path)
    mime = _AUDIO_MIME_BY_SUFFIX.get(path.suffix.lower())
    if mime is None:
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    raw = path.read_bytes()
    encoded = base64.b64encode(raw)
    if len(encoded) > MAX_BASE64_BYTES:
        raise RuntimeError(
            f"Audio too large for cloud ASR: base64 size {len(encoded)} bytes "
            f"exceeds the 10 MB limit. Split or shorten the recording."
        )
    return f"data:{mime};base64,{encoded.decode('ascii')}"


class CloudTranscriber:
    """DashScope qwen3-asr-flash transcriber with the AudioTranscriber interface."""

    def __init__(self, model_name: str = None, api_key: str = None, base_url: str = None):
        self.model_name = model_name or CLOUD_MODEL_DEFAULT
        self._api_key = api_key if api_key is not None else os.environ.get(
            "DASHSCOPE_API_KEY", ""
        )
        self._base_url = (base_url or DASHSCOPE_BASE_URL).rstrip("/")
        self._loaded = False
        self._error: Exception | None = None
        self._lock = threading.Lock()

    @property
    def error(self) -> Exception | None:
        return self._error

    def load(self):
        """Validate configuration. No network call — nothing to warm up."""
        with self._lock:
            if self._loaded:
                return
            self._error = None
            if not self._api_key:
                self._error = RuntimeError(
                    "Cloud ASR backend selected but DASHSCOPE_API_KEY is not set. "
                    "Export it or switch ASR_BACKEND back to 'local'."
                )
                raise self._error
            self._loaded = True
            logger.info(
                "Cloud ASR backend ready (model=%s, endpoint=%s)",
                self.model_name,
                self._base_url,
            )

    def unload(self):
        with self._lock:
            self._loaded = False
            self._error = None
        logger.info("Cloud ASR backend unloaded (model=%s)", self.model_name)

    def wait_for_ready(self, timeout: float = None):
        if self._error:
            raise self._error
        if not self._loaded:
            raise RuntimeError("Cloud transcriber not loaded; call load() first")
        return self

    @property
    def is_ready(self) -> bool:
        return self._loaded and self._error is None

    @property
    def state(self) -> str:
        if self._error:
            return "error"
        return "loaded" if self._loaded else "unloaded"

    def gpu_memory_mb(self) -> float | None:
        return None  # cloud backend uses no local VRAM

    def transcribe(self, audio_path: str, language: str = None) -> dict:
        if not self.is_ready:
            raise RuntimeError("Cloud transcriber is not ready (not loaded or in error state)")

        asr_options = {"enable_itn": True}
        if language:
            code = _LANGUAGE_CODE_MAP.get(language.lower())
            if code:
                asr_options["language"] = code
            else:
                logger.warning("Unsupported language '%s' for cloud ASR, auto-detecting", language)

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": _audio_data_uri(audio_path)},
                        }
                    ],
                }
            ],
            "stream": False,
            "asr_options": asr_options,
        }

        response = self._post_with_retry(payload)
        text = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        usage = response.get("usage") or {}
        if usage:
            logger.info(
                "Cloud ASR usage: total_tokens=%s (prompt=%s, completion=%s)",
                usage.get("total_tokens"),
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
            )
        return {
            "text": text,
            "detected_language": "",
            "duration_seconds": None,
        }

    def _post_with_retry(self, payload: dict) -> dict:
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        attempts = 1 + MAX_TRANSPORT_RETRIES
        for attempt in range(1, attempts + 1):
            try:
                with httpx.Client(timeout=TRANSCRIBE_TIMEOUT, proxy=None) as client:
                    resp = client.post(url, headers=headers, json=payload)
            except httpx.TransportError as e:
                if attempt < attempts:
                    logger.warning("Cloud ASR transport error (attempt %d/%d): %s", attempt, attempts, e)
                    continue
                raise RuntimeError(f"Cloud ASR request failed: {e}") from e
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as e:
                    raise RuntimeError(f"Cloud ASR returned invalid JSON: {e}") from e
            detail = resp.text[:500]
            # 429 / 5xx are worth one retry; 4xx (auth, bad request) are not.
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < attempts:
                logger.warning(
                    "Cloud ASR HTTP %d (attempt %d/%d): %s",
                    resp.status_code, attempt, attempts, detail,
                )
                continue
            raise RuntimeError(f"Cloud ASR HTTP {resp.status_code}: {detail}")
        raise RuntimeError("Cloud ASR request failed after retries")  # unreachable
