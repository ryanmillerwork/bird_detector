#!/usr/bin/env python3
"""
Serve the `detections/` directory over HTTP (for Home Assistant dashboards).

This exposes:
  - /latest.jpg
  - /detection_<ts>.jpg (historical, limited by KEEP_LAST_ANNOTATED)

Config via environment variables (can live in .env):
  - DETECTIONS_HTTP_HOST (default: 0.0.0.0)
  - DETECTIONS_HTTP_PORT (default: 8765)
  - DETECTIONS_DIR (default: ./detections)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from bd.env import load_dotenv
from bd.http_server import start_detections_http_server


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    load_dotenv(base_dir / ".env", override=False)

    host = (os.environ.get("DETECTIONS_HTTP_HOST") or "0.0.0.0").strip() or "0.0.0.0"
    port = int((os.environ.get("DETECTIONS_HTTP_PORT") or "8765").strip() or "8765")
    detections_dir = Path(os.environ.get("DETECTIONS_DIR", str(base_dir / "detections"))).resolve()
    detections_dir.mkdir(exist_ok=True)

    print(f"Serving {detections_dir} on http://{host}:{port}/ (Ctrl+C to stop)")
    server = start_detections_http_server(host=host, port=port, directory=detections_dir)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()


