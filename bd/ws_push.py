from __future__ import annotations

import asyncio
import contextlib
import os
import queue
import threading
from dataclasses import dataclass
from typing import Any


def _ws_send_timeout_s() -> float:
    """Override with DETECTIONS_WS_SEND_TIMEOUT_S (e.g. 20 on slow WiFi + ~0.7MB JPEGs)."""
    try:
        return max(2.0, float((os.environ.get("DETECTIONS_WS_SEND_TIMEOUT_S") or "20").strip() or "20"))
    except ValueError:
        return 20.0


def _ws_jpeg_scale() -> float:
    """
    1.0 = full size (read bytes as-is for WS).
    0.5 = half width and height (~1/4 pixels) for a smaller WS payload on slow links.
    Set with DETECTIONS_WS_JPEG_SCALE (e.g. 0.5 for diagnosis).
    """
    try:
        raw = (os.environ.get("DETECTIONS_WS_JPEG_SCALE") or "1").strip() or "1"
        s = float(raw)
    except ValueError:
        return 1.0
    if s <= 0 or s > 1.0:
        return 1.0
    return s


def _ws_jpeg_quality() -> int:
    try:
        q = int((os.environ.get("DETECTIONS_WS_JPEG_QUALITY") or "85").strip() or "85")
    except ValueError:
        return 85
    return min(100, max(40, q))


def scale_jpeg_for_ws(jpeg_bytes: bytes) -> bytes:
    """
    Optionally downscale+re-encode a JPEG for WebSocket push only (HTTP `latest.jpg` unchanged).
    """
    scale = _ws_jpeg_scale()
    if not jpeg_bytes or scale >= 0.9999:
        return jpeg_bytes
    try:
        import cv2
        import numpy as np
    except ImportError:
        return jpeg_bytes
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return jpeg_bytes
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if new_w == w and new_h == h:
        return jpeg_bytes
    out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    q = _ws_jpeg_quality()
    ok, buf = cv2.imencode(".jpg", out, (int(cv2.IMWRITE_JPEG_QUALITY), q))
    if not ok or buf is None:
        return jpeg_bytes
    return bytes(buf)


@dataclass
class DetectionsWSServer:
    host: str
    port: int
    _queue: "queue.Queue[bytes | None]"
    _thread: threading.Thread
    _in_flight: threading.Event
    _loop: asyncio.AbstractEventLoop | None = None

    def stop(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._queue.put_nowait(None)
        except Exception:
            try:
                self._queue.put(None, timeout=0.1)
            except Exception:
                pass
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(lambda: None)  # wake
            except Exception:
                pass
        self._thread.join(timeout=8.0)
        if self._thread.is_alive():
            # Best-effort; the daemon will die with the process.
            pass

    def notify_jpeg(self, data: bytes) -> None:
        """
        Enqueue a JPEG for broadcast when idle; never cancel an in-progress send.

        If a send is in flight to clients, the new image is dropped (not queued).
        When not sending, at most one not-yet-picked payload is held (replaced on repeat notify).
        """
        if not data:
            return
        if self._in_flight.is_set():
            return
        try:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._queue.put_nowait(data)
            print(f"[ws] jpeg queued {len(data)} bytes for broadcast", flush=True)
        except Exception:
            pass

    @property
    def ws_url(self) -> str:
        h = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"ws://{h}:{self.port}"


def _run_server_loop(
    host: str,
    port: int,
    q: "queue.Queue[bytes | None]",
    in_flight: threading.Event,
    loop_ref: list[Any],
) -> None:
    import websockets

    clients: set[Any] = set()
    clients_lock = asyncio.Lock()

    async def register(websocket: Any) -> None:
        async with clients_lock:
            clients.add(websocket)
        try:
            async for _ in websocket:
                pass
        except Exception:
            pass
        finally:
            async with clients_lock:
                clients.discard(websocket)

    send_timeout = _ws_send_timeout_s()
    jscale = _ws_jpeg_scale()
    try:
        print(
            f"[ws] per-client send timeout {send_timeout:.0f}s (DETECTIONS_WS_SEND_TIMEOUT_S)",
            flush=True,
        )
        if jscale < 0.9999:
            print(
                f"[ws] WebSocket push: scaled JPEG {jscale} (DETECTIONS_WS_JPEG_SCALE), quality {_ws_jpeg_quality()} (DETECTIONS_WS_JPEG_QUALITY)",
                flush=True,
            )
    except OSError:
        pass

    async def broadcast_once(data: bytes) -> None:
        cset: list[Any] = []
        async with clients_lock:
            cset = list(clients)
        if not cset:
            return

        async def send_to(ws: Any) -> bool:
            try:
                await asyncio.wait_for(ws.send(data), timeout=send_timeout)
                return True
            except Exception:
                with contextlib.suppress(Exception):
                    await ws.close(1011, "stale or slow")
                return False

        outcomes = await asyncio.gather(
            *(send_to(ws) for ws in cset),
            return_exceptions=True,
        )
        n_ok = sum(1 for o in outcomes if o is True)  # type: ignore[union-attr, assignment]
        n_err = len(outcomes) - n_ok
        print(
            f"[ws] jpeg sent {len(data)} bytes -> {n_ok}/{len(cset)} client(s) ok"
            + (f" ({n_err} failed/dropped)" if n_err else "")
            + f" (send timeout {send_timeout:.0f}s)",
            flush=True,
        )
        to_remove: list[Any] = [ws for ws, o in zip(cset, outcomes, strict=True) if o is not True]
        if to_remove:
            async with clients_lock:
                for w in to_remove:
                    clients.discard(w)

    def get_next() -> bytes | None:
        data = q.get()
        if data is None:
            return None
        in_flight.set()
        return data

    async def fanout() -> None:
        loop = asyncio.get_running_loop()
        while True:
            data = await loop.run_in_executor(None, get_next)
            if data is None:
                break
            try:
                await broadcast_once(data)
            finally:
                in_flight.clear()

    async def runner() -> None:
        async with websockets.serve(register, host, port) as _server:  # noqa: SIM117
            await fanout()

    loop = asyncio.new_event_loop()
    loop_ref[0] = loop
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(runner())
    finally:
        try:
            loop.stop()
        except Exception:
            pass
        try:
            loop.close()
        except Exception:
            pass


def start_detections_ws_server(*, host: str, port: int) -> DetectionsWSServer | None:
    """
    Broadcast each JPEG to all WebSocket clients on a dedicated port.
    `notify_jpeg` must be called from the birdwatch thread only after `latest.jpg` is written.
    """
    try:
        import websockets  # noqa: F401
    except ImportError:
        return None

    q: queue.Queue[bytes | None] = queue.Queue(maxsize=1)
    in_flight = threading.Event()
    loop_ref: list[Any] = [None]
    t = threading.Thread(
        target=_run_server_loop,
        args=(host, int(port), q, in_flight, loop_ref),
        name="bd-ws-push",
        daemon=True,
    )
    t.start()
    for _ in range(100):
        if loop_ref[0] is not None:
            break
        import time

        time.sleep(0.01)
    return DetectionsWSServer(
        host=host,
        port=int(port),
        _queue=q,
        _thread=t,
        _in_flight=in_flight,
        _loop=loop_ref[0] if loop_ref[0] else None,
    )
