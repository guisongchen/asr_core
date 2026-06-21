from pathlib import Path

# Server
SOCKET_PATH = "/tmp/asr_core.sock"
HOST = "127.0.0.1"
PORT = 8123

# Model defaults
MODEL_SIZE_DEFAULT = "qwen3-asr-0.6b"
MODEL_CHOICES = ["qwen3-asr-0.6b", "Qwen3-ASR-1.7B"]

# Path resolution: prefer local project models, then HF cache fallback
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_LOCAL_PATH = str(PROJECT_ROOT / "models" / MODEL_SIZE_DEFAULT)

# Dashboard
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8124

# Audio
SAMPLE_RATE = 16000

# Request timeout for model readiness
MODEL_READY_TIMEOUT = 120

# Allowed output languages for fallback logic
ALLOWED_LANGUAGES = {"english", "chinese"}
