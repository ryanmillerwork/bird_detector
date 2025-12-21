#!/usr/bin/env python3
"""Capture a single frame from either RTSP (go2rtc) or an HTTP JPEG endpoint (e.g. Frigate latest.jpg)."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from urllib.request import Request, urlopen

import cv2
import numpy as np


DEFAULT_RTSP_URL = "rtsp://192.168.0.50:8554/bird_cam"
# python capture_frame.py /tmp/test.jpg   --jpeg-url "http://192.168.0.50:5000/api/bird/latest.jpg"


def _default_output_path() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"frame_{timestamp}.jpg"

def capture_frame_rtsp(rtsp_url: str) -> np.ndarray | None:
    """Grab a single frame from an RTSP stream via OpenCV/FFmpeg."""
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream: {rtsp_url}")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("Error: Could not read frame from RTSP stream")
        return None
    return frame


def capture_frame_jpeg_url(jpeg_url: str, timeout_s: float = 3.0) -> np.ndarray | None:
    """Fetch a JPEG over HTTP and decode it into an OpenCV BGR frame."""
    req = Request(
        jpeg_url,
        headers={
            # Some servers behave better with an explicit UA
            "User-Agent": "bird_detector/capture_frame.py",
            "Accept": "image/jpeg,image/*;q=0.9,*/*;q=0.1",
        },
        method="GET",
    )
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            data = resp.read()
    except Exception as e:
        print(f"Error: Failed to fetch JPEG from {jpeg_url}: {e}")
        return None

    if not data:
        print(f"Error: Empty response from {jpeg_url}")
        return None

    img_array = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        print(f"Error: Failed to decode JPEG bytes from {jpeg_url} (got {len(data)} bytes)")
        return None
    return frame


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output JPG path (default: frame_YYYYmmdd_HHMMSS.jpg).",
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--rtsp",
        default=None,
        help=f"RTSP URL (default: {DEFAULT_RTSP_URL}).",
    )
    src.add_argument(
        "--jpeg-url",
        default=None,
        help="HTTP URL that returns a single JPEG (e.g. Frigate /api/<camera>/latest.jpg).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("JPEG_TIMEOUT_S", "3.0")),
        help="HTTP timeout for --jpeg-url in seconds (default: 3.0 or env JPEG_TIMEOUT_S).",
    )

    args = parser.parse_args(argv)

    output_path = args.output or _default_output_path()

    if args.jpeg_url:
        frame = capture_frame_jpeg_url(args.jpeg_url, timeout_s=args.timeout)
    else:
        frame = capture_frame_rtsp(args.rtsp or DEFAULT_RTSP_URL)

    if frame is None:
        return 2

    ok = cv2.imwrite(output_path, frame)
    if not ok:
        print(f"Error: Failed to write output file: {output_path}")
        return 3

    print(f"Saved frame to {output_path} ({frame.shape[1]}x{frame.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


