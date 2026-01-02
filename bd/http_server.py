from __future__ import annotations

from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any


class _NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        # Avoid HA/browser caching "latest.jpg" aggressively.
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()


@dataclass
class DetectionsHTTPServer:
    host: str
    port: int
    directory: Path
    _httpd: ThreadingHTTPServer
    _thread: Thread

    def stop(self) -> None:
        try:
            self._httpd.shutdown()
        except Exception:
            pass
        try:
            self._httpd.server_close()
        except Exception:
            pass


def start_detections_http_server(*, host: str, port: int, directory: Path) -> DetectionsHTTPServer:
    directory = Path(directory).resolve()
    directory.mkdir(exist_ok=True)

    def handler(*args: Any, **kwargs: Any) -> _NoCacheHandler:
        return _NoCacheHandler(*args, directory=str(directory), **kwargs)

    httpd = ThreadingHTTPServer((host, int(port)), handler)
    t = Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return DetectionsHTTPServer(host=host, port=int(port), directory=directory, _httpd=httpd, _thread=t)


