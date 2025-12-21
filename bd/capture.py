from __future__ import annotations

import math
import time
from typing import Tuple
from urllib.request import Request, urlopen

import cv2
import numpy as np


def next_aligned_time(now_s: float, interval_s: float, phase_s: float = 0.0) -> float:
    """
    Return the next wall-clock aligned time T such that:
      T = phase_s + k*interval_s, and T > now_s
    For interval_s=2.0 and phase_s=0.0, this aligns to ...:00.000, ...:02.000, ...:04.000, etc.
    """
    if interval_s <= 0:
        return now_s
    k = math.floor((now_s - phase_s) / interval_s) + 1
    return phase_s + k * interval_s


def grab_frame_jpeg_url(jpeg_url: str, timeout_s: float) -> Tuple[np.ndarray | None, str]:
    """Fetch a single JPEG over HTTP and decode into an OpenCV BGR frame."""
    req = Request(
        jpeg_url,
        headers={
            "User-Agent": "bird_detector/birdwatch.py",
            "Accept": "image/jpeg,image/*;q=0.9,*/*;q=0.1",
        },
        method="GET",
    )
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            data = resp.read()
    except Exception as e:
        return None, str(e)

    if not data:
        return None, "empty response"

    img_array = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        return None, f"decode failed ({len(data)} bytes)"
    return frame, ""


def capture_loop_jpeg(
    *,
    jpeg_url: str,
    timeout_s: float,
    interval_s: float,
    stop_event,
    frame_queue,
    is_corrupted_frame,
) -> None:
    """
    Producer loop: wall-clock aligned JPEG fetch. Drops old frames by using a size-1 queue.

    frame_queue holds tuples: (scheduled_at_s, frame, grab_ms)
    """
    consecutive_failures = 0
    last_error_log = 0.0
    # Startup logging is handled by birdwatch.py; keep producer quiet unless errors occur.

    while not stop_event.is_set():
        scheduled_at = next_aligned_time(time.time(), interval_s, 0.0)
        sleep_s = scheduled_at - time.time()
        if sleep_s > 0:
            time.sleep(sleep_s)

        snap_start = time.time()
        frame, err = grab_frame_jpeg_url(jpeg_url, timeout_s=timeout_s)
        if frame is None:
            consecutive_failures += 1
            now = time.time()
            if consecutive_failures == 1 or now - last_error_log > 5:
                prefix = f"[producer] JPEG fetch failed {consecutive_failures}x in a row"
                print(f"{prefix}: {err}" if err else prefix)
                last_error_log = now
            backoff = min(0.2 * consecutive_failures, 5.0)
            if backoff > 0:
                time.sleep(backoff)
            continue

        consecutive_failures = 0
        grab_ms = (time.time() - snap_start) * 1000.0

        if is_corrupted_frame(frame):
            print("[producer] Corrupted frame detected, skipping...")
            continue

        try:
            if frame_queue.full():
                frame_queue.get_nowait()
            frame_queue.put_nowait((scheduled_at, frame, grab_ms))
        except Exception:
            # Don't let producer die because of queue edge cases.
            pass



