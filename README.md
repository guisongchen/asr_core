# ASRCore

Shared ASR (Automatic Speech Recognition) service for local Qwen3-ASR models.

## What it does

ASRCore owns the ASR model lifecycle and exposes an HTTP API over a Unix domain socket. Other applications (like `voice_to_text` and `video_to_md`) send audio files to ASRCore and receive transcriptions.

## Quick start

```bash
cd /home/ccc/projects/asr_core
uv pip install -e .

# Start the daemon
python -m asr_core

# In another terminal
curl --noproxy '*' --unix-socket /tmp/asr_core.sock http://asr_core/status
curl --noproxy '*' --unix-socket /tmp/asr_core.sock -X POST -H "Content-Type: application/json" \
  -d '{"model_name":"qwen3-asr-0.6b"}' http://asr_core/load
```

## systemd

```bash
systemctl --user enable --now $PWD/services/asr-core.service
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check |
| `/status` | GET | Model state: `unloaded`, `loading`, `loaded`, `error` |
| `/load` | POST | Load a model by name |
| `/unload` | POST | Unload the current model |
| `/transcribe` | POST | Transcribe an audio file |
| `/stats` | GET | Request stats and GPU memory |

## Architecture

The model runs in a spawned **worker subprocess** (`worker.py`); the FastAPI supervisor never
touches CUDA. Unload = terminate the worker, so the driver reclaims all VRAM — no PyTorch
allocator residue, safe across repeated load/unload cycles.

Monitoring (model status, GPU memory, request stats) lives in the unified
[system_services_manager](https://github.com/guisongchen/system_services_manager) console;
the standalone dashboard was removed.

## Models

Models are loaded from the directory configured in `asr_core.toml`:

```toml
[models]
model_dir = "/home/ccc/models/asr"
```

Place model directories there:

```bash
mv /path/to/qwen3-asr-0.6b /home/ccc/models/asr/qwen3-asr-0.6b
mv /path/to/Qwen3-ASR-1.7B /home/ccc/models/asr/Qwen3-ASR-1.7B
```

If `asr_core.toml` is missing, the default model directory is `/home/ccc/models/asr`.

## Cloud ASR backend (no GPU)

For machines without a GPU, ASRCore can forward transcription to Alibaba
DashScope `qwen3-asr-flash` (billed per token) instead of running a local model.

```toml
# asr_core.toml
[cloud]
backend = "cloud"   # "local" (default) | "cloud"
```

or via environment variable (wins over the file):

```bash
export ASR_BACKEND=cloud
export DASHSCOPE_API_KEY=sk-...   # required; never commit it
```

Notes:

- Audio is sent as a base64 data URI to the OpenAI-compatible endpoint
  (`DASHSCOPE_BASE_URL`, default `https://dashscope.aliyuncs.com/compatible-mode/v1`).
- Base64-encoded audio is limited to 10 MB (~4 min of 16 kHz mono WAV).
- Optional `asr_options.language` mapping: `chinese`→`zh`, `english`→`en`, etc.;
  unknown values fall back to auto-detection. ITN is enabled by default.
- In cloud mode, `/load` only validates the API key (no warm-up) and
  `/unload` is a no-op state transition; the Unix-socket API is unchanged,
  so existing clients (voice_to_text) work without modification.
