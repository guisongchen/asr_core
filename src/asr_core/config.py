import os
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

# Server
SOCKET_PATH = os.environ.get("ASR_CORE_SOCKET", "/tmp/asr_core.sock")
HOST = "127.0.0.1"
PORT = 8123

# Dashboard
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8124

# Audio
SAMPLE_RATE = 16000

# Request timeout for model readiness
MODEL_READY_TIMEOUT = 120

# Allowed output languages for fallback logic
ALLOWED_LANGUAGES = {"english", "chinese"}

# Model defaults
_MODEL_DEFAULTS = {
    "model_dir": "/home/ccc/models/asr",
    "default_model": "qwen3-asr-0.6b",
    "allowed_models": ["qwen3-asr-0.6b", "Qwen3-ASR-1.7B"],
}


def _find_config_path() -> Path:
    """Locate asr_core.toml via env var or project root heuristic."""
    env_path = os.environ.get("ASR_CORE_CONFIG")
    if env_path:
        return Path(env_path)
    # Fallback: assume src layout (src/asr_core/config.py -> project root)
    return Path(__file__).parent.parent.parent / "asr_core.toml"


def _load_config() -> dict:
    config_path = _find_config_path()
    if not config_path.is_file():
        return {}
    try:
        with config_path.open("rb") as f:
            data = tomllib.load(f)
    except Exception:
        return {}
    return data.get("models", {})


_config = _load_config()

MODEL_DIR = Path(_config.get("model_dir", _MODEL_DEFAULTS["model_dir"]))
MODEL_SIZE_DEFAULT = _config.get("default_model", _MODEL_DEFAULTS["default_model"])
MODEL_CHOICES = _config.get("allowed_models", _MODEL_DEFAULTS["allowed_models"])
