#!/usr/bin/env python3
"""Capture a single frame from an RTSP stream (go2rtc or direct camera)."""

import cv2
import sys
from datetime import datetime
from pathlib import Path


def load_env():
    """Load credentials from .env file."""
    env = {}
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().strip().split("\n"):
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                env[key.strip()] = val.strip()
    return env


_env = load_env()

RTSP_URL_OVERRIDE = _env.get("RTSP_URL") or None
RTSP_PATH = _env.get("RTSP_PATH", "rtsp://192.168.0.50:8554/bird_cam")


def build_rtsp_url():
    if RTSP_URL_OVERRIDE:
        return RTSP_URL_OVERRIDE
    if RTSP_PATH.startswith("rtsp://"):
        return RTSP_PATH
    required = ("CAMERA_USER", "CAMERA_PASS", "CAMERA_IP", "CAMERA_PORT")
    missing = [k for k in required if not _env.get(k)]
    if missing:
        raise ValueError(
            "RTSP_PATH is not a full rtsp:// URL, but camera credentials are missing.\n"
            f"Missing: {', '.join(missing)}\n"
            "Fix: set RTSP_URL to a full URL (recommended with go2rtc) or set CAMERA_USER/CAMERA_PASS/CAMERA_IP/CAMERA_PORT."
        )
    return f"rtsp://{_env['CAMERA_USER']}:{_env['CAMERA_PASS']}@{_env['CAMERA_IP']}:{_env['CAMERA_PORT']}/{RTSP_PATH}"

def capture_frame(output_path=None):
    """Grab a single frame from the RTSP stream."""
    
    # Open the stream
    try:
        rtsp_url = build_rtsp_url()
    except ValueError as e:
        print(str(e))
        return None

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print("Error: Could not open RTSP stream")
        return None
    
    # Grab a frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read frame")
        return None
    
    # Generate filename if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"frame_{timestamp}.jpg"
    
    # Save the frame
    cv2.imwrite(output_path, frame)
    print(f"Saved frame to {output_path} ({frame.shape[1]}x{frame.shape[0]})")
    
    return frame

if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else None
    capture_frame(output)


