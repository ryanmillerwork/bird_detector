#!/usr/bin/env python3
"""Capture a single frame from Reolink camera."""

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
RTSP_URL = f"rtsp://{_env['CAMERA_USER']}:{_env['CAMERA_PASS']}@{_env['CAMERA_IP']}:{_env['CAMERA_PORT']}/h264Preview_01_main"

def capture_frame(output_path=None):
    """Grab a single frame from the RTSP stream."""
    
    # Open the stream
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    
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


