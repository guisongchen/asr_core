import argparse
import os
import signal
import socket
import sys
import threading
import time

import uvicorn

from .api import app
from .config import HOST, PORT, SOCKET_PATH


def _wait_for_socket(path: str, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(path):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                    s.connect(path)
                    return True
            except OSError:
                pass
        time.sleep(0.1)
    return False


def main():
    parser = argparse.ArgumentParser(description="ASRCore HTTP daemon")
    parser.add_argument(
        "--tcp",
        action="store_true",
        help=f"Listen on TCP {HOST}:{PORT} instead of Unix socket",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Detach from terminal after socket is ready",
    )
    args = parser.parse_args()

    # Clean up stale socket only if nothing is listening
    if not args.tcp and os.path.exists(SOCKET_PATH):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                s.connect(SOCKET_PATH)
            print(f"ASRCore is already running on {SOCKET_PATH}", file=sys.stderr)
            sys.exit(1)
        except (OSError, ConnectionRefusedError):
            os.unlink(SOCKET_PATH)

    config = uvicorn.Config(
        app,
        host=HOST if args.tcp else None,
        port=PORT if args.tcp else None,
        uds=None if args.tcp else SOCKET_PATH,
        log_level="info",
    )
    server = uvicorn.Server(config)

    # Handle signals cleanly and remove socket on exit
    def handle_signal(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        server.should_exit = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        if args.detach:
            ready = threading.Event()

            def wait_and_exit():
                ok = _wait_for_socket(SOCKET_PATH, timeout=10.0) if not args.tcp else True
                if ok:
                    print("ASRCore daemon ready")
                else:
                    print("ASRCore daemon failed to start within timeout")
                    os._exit(1)

            threading.Thread(target=wait_and_exit, daemon=True).start()
            server.run()
        else:
            server.run()
    finally:
        if not args.tcp:
            try:
                os.unlink(SOCKET_PATH)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
