import argparse
import sys

import uvicorn

from ..config import DASHBOARD_HOST, DASHBOARD_PORT
from .app import app


def main():
    parser = argparse.ArgumentParser(description="ASRCore Web Dashboard")
    parser.add_argument(
        "--host", default=DASHBOARD_HOST, help=f"Bind host (default: {DASHBOARD_HOST})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DASHBOARD_PORT,
        help=f"Bind port (default: {DASHBOARD_PORT})",
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    sys.exit(main())
