import argparse
import logging
import os
import signal
import socket
import sys
import threading
import time

import uvicorn

from .api import app
from .config import HOST, PORT, SOCKET_PATH

logger = logging.getLogger(__name__)


def _setup_logging():
    """Configure logging for the daemon."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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
        help="Run in background-ready mode (prints readiness, used by auto-start)",
    )
    args = parser.parse_args()

    _setup_logging()

    # Clean up stale socket only if nothing is listening
    if not args.tcp and os.path.exists(SOCKET_PATH):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                s.connect(SOCKET_PATH)
            logger.error("ASRCore is already running on %s", SOCKET_PATH)
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
        logger.info("Received signal %d, shutting down...", signum)
        server.should_exit = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        if args.detach:
            def wait_and_report():
                ok = _wait_for_socket(SOCKET_PATH, timeout=10.0) if not args.tcp else True
                if ok:
                    logger.info("ASRCore daemon ready")
                else:
                    logger.error("ASRCore daemon failed to start within timeout")
                    server.should_exit = True

            threading.Thread(target=wait_and_report, daemon=True).start()
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
