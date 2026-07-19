# ADR-0001: GPU Memory Cleanup After Each Transcription

## Status

Accepted — implemented 2026-07-19.

## Context

Users reported gradual GPU memory growth when using the 0.6B Qwen3-ASR model
over time. Each call to `/transcribe` caused `nvidia-smi` (and the `/stats`
endpoint) to show increasing memory usage, even though transcription results
were returned successfully and no errors were logged.

## Root Cause Analysis

### Primary cause: PyTorch CUDA caching allocator

PyTorch's default memory allocator (`torch.cuda.caching_allocator`) does **not**
return freed memory to the OS. When tensors are freed by Python, the underlying
CUDA memory is held in an internal cache for potential reuse. This is by design
— it avoids expensive `cudaMalloc` / `cudaFree` calls on every allocation.

The problem: our code only called `torch.cuda.empty_cache()` during
`AudioTranscriber.unload()` (model switch or server shutdown). Between
individual `/transcribe` requests, nothing flushed the PyTorch cache, so
`nvidia-smi` and `torch.cuda.mem_get_info()` (used by the `/stats` endpoint)
reported ever-increasing usage.

### Contributing factors

| Factor | Code Location | Impact |
|---|---|---|
| KV cache in `model.generate()` | `qwen_asr/inference/qwen3_asr.py:510` | Each call allocates fresh KV cache tensors (up to `max_new_tokens=1024`). Freed after return but kept in PyTorch cache. |
| Processor input tensors | `qwen_asr/inference/qwen3_asr.py:507-508` | `input_ids`, `input_features`, `attention_mask` on GPU. Freed by Python GC on an unpredictable schedule. |
| Re-transcription on language mismatch | `transcriber.py:108-114` | When detected language is unexpected, a second `model.transcribe()` runs for the same audio, doubling temporary allocations before GC. |
| No GC between calls | `api.py:100-125` | Python GC may delay collection across multiple rapid requests. |

### Why the model weights themselves don't grow

Model weights (parameters) are allocated once at load time via
`from_pretrained()` and persist for the lifetime of the `AudioTranscriber`.
They are not affected by the caching allocator — the growth comes entirely
from per-inference temporary tensors being cached and never released back
to the GPU driver.

## Decision

Add explicit GPU memory cleanup at the end of every `AudioTranscriber.transcribe()`
call:

```python
del results
gc.collect()
try:
    torch.cuda.empty_cache()
except Exception:
    pass
```

### What each line does

- **`del results`** — removes the reference to the transcription result list,
  allowing the `ASRTranscription` objects and any held tensors to be collected.
- **`gc.collect()`** — forces Python's garbage collector to run immediately,
  releasing any unreferenced CPU-side objects that may hold GPU tensor views.
- **`torch.cuda.empty_cache()`** — releases all unoccupied cached memory from
  PyTorch's CUDA caching allocator back to the GPU driver. This does **not**
  unload the model weights.

### What this does NOT affect

- Model weights remain loaded and untouched.
- Inference performance impact is negligible — the overhead of a few
  milliseconds of cache cleanup is dwarfed by the multi-second transcription
  time.
- The next inference still benefits from PyTorch allocator speed: freeing the
  cache does not disable the allocator, it only releases idle cached blocks.

## Alternatives Considered

### A. Periodic cleanup (every N requests)

Count requests and only clean up every 5–10 calls. Rejected because:
- Adds complexity (counter state) without meaningful benefit.
- The cleanup overhead (~1–5 ms) is negligible compared to inference (2–10 s).

### B. Cleanup in the API layer (`api.py`)

Call `empty_cache()` in the `/transcribe` handler's `finally` block instead of
in `AudioTranscriber`. Rejected because:
- The cache management concern belongs to the model wrapper, not the HTTP layer.
- `AudioTranscriber.unload()` already owns cache cleanup; `transcribe()` should
  do the same for consistency.

### C. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

Use PyTorch's expandable segments feature to reduce fragmentation. Partially
mitigates growth but does not eliminate it — the cache still accumulates freed
blocks across calls. Not mutually exclusive with this fix; could be applied
additionally.

## Consequences

- **Positive**: GPU memory usage stabilizes after the first few calls. The
  `/stats` endpoint and `nvidia-smi` report consistent, predictable memory
  levels regardless of call count or uptime.
- **Positive**: Long-running daemons can process arbitrarily many transcription
  requests without running out of GPU memory.
- **Neutral**: Each transcription incurs ~1–5 ms of additional cleanup overhead.
- **Neutral**: The first call after cleanup re-allocates cached blocks, which is
  the normal PyTorch behavior and no slower than the first-ever call.

## References

- [PyTorch CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- `src/asr_core/transcriber.py` — `AudioTranscriber.transcribe()` and `unload()`
- `src/asr_core/api.py` — `/transcribe` and `/stats` endpoints
- `.venv/.../qwen_asr/inference/qwen3_asr.py` — `_infer_asr_transformers()`
- Commit `95182bd` — prior fix adding `del` + `gc.collect()` + `empty_cache()` to `unload()`
