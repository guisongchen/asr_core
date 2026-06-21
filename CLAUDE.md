# CLAUDE.md

This file provides guidance to Claude Code when working in the `asr_core` repository.

## Project Overview

`asr_core` is a shared local ASR (Automatic Speech Recognition) service that owns the Qwen3-ASR model lifecycle. It exposes an HTTP API over a Unix domain socket and provides a Python client library used by `voice_to_text` and `video_to_md`.

## Architecture

```
asr_core/
├── src/asr_core/
│   ├── server.py      # Uvicorn + FastAPI daemon entry point
│   ├── api.py         # FastAPI routes: /health, /status, /load, /unload, /transcribe, /stats
│   ├── models.py      # Pydantic request/response models
│   ├── client.py      # ASRCoreClient — Python client over Unix socket
│   ├── transcriber.py # AudioTranscriber — wraps qwen_asr.Qwen3ASRModel
│   ├── audio.py       # AudioPreprocessor — resample, normalize, noise reduction
│   └── config.py      # Paths, defaults, constants
├── services/
│   └── asr-core.service
└── models/            # Local model directories or symlinks (gitignored)
```

## Environment

```bash
cd /home/ccc/projects/asr_core
python3 -m venv .venv
uv pip install -e .
```

The project expects Python 3.10+ and a CUDA-capable GPU for model inference.

## Running the service

### Manually

```bash
.venv/bin/python3 -m asr_core
```

### Via systemd

```bash
systemctl --user enable --now $PWD/services/asr-core.service
```

Socket path: `/tmp/asr_core.sock`

## API

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/health` | GET | — | Liveness |
| `/status` | GET | — | Model state |
| `/load` | POST | `{"model_name":"qwen3-asr-0.6b"}` | Load model |
| `/unload` | POST | — | Unload model, free GPU |
| `/transcribe` | POST | `{"audio_path":"/path/to.wav"}` | Transcribe audio |
| `/stats` | GET | — | Request stats + GPU memory |

## Model resolution

When loading a model, `AudioTranscriber` resolves the path in this order:

1. `asr_core/models/<model_name>` if it exists
2. Fallback HF identifier `Qwen/<model_name>` (only if not in offline mode)

Create symlinks to share models with other projects:

```bash
ln -s /home/ccc/projects/voice_to_text/models/qwen3-asr-0.6b models/qwen3-asr-0.6b
ln -s /home/ccc/projects/video_to_md/models/Qwen3-ASR-1.7B models/Qwen3-ASR-1.7B
```

## Client usage

```python
from asr_core.client import ASRCoreClient

with ASRCoreClient() as client:
    client.load_model("qwen3-asr-0.6b")
    result = client.transcribe("/tmp/audio.wav")
    print(result["text"])
```

`ASRCoreClient` auto-starts the daemon if `/tmp/asr_core.sock` is missing (configurable).

## Important notes

- `models/` is gitignored. Do not commit model weights.
- The service uses `local_files_only=True` when loading models, so models must be present locally.
- Only one model is loaded at a time; loading a different model unloads the previous one.
- If the daemon hits `CUDA out of memory`, stop other GPU processes or unload the current model first.
