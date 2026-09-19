"""Worker-process protocol tests — no GPU required (model_name=None = idle mode)."""

import multiprocessing as mp

from asr_core.worker import run_worker


def _spawn(model_name=None):
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=run_worker, args=(child, model_name, "/tmp"), daemon=True)
    proc.start()
    child.close()
    return proc, parent


def test_worker_ready_and_clean_shutdown():
    proc, conn = _spawn()
    try:
        assert conn.poll(30), "worker did not send ready"
        assert conn.recv() == {"event": "ready"}
        conn.send({"cmd": "shutdown"})
        proc.join(timeout=15)
        assert not proc.is_alive()
        assert proc.exitcode == 0
    finally:
        if proc.is_alive():
            proc.kill()
        conn.close()


def test_worker_transcribe_without_model_errors():
    proc, conn = _spawn()
    try:
        assert conn.poll(30)
        conn.recv()  # ready
        conn.send({"cmd": "transcribe", "audio_path": "/x.wav", "language": None})
        assert conn.poll(30)
        msg = conn.recv()
        assert msg["event"] == "transcribe_error"
        conn.send({"cmd": "shutdown"})
        proc.join(timeout=15)
    finally:
        if proc.is_alive():
            proc.kill()
        conn.close()
