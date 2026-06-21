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
