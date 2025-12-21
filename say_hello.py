#!/usr/bin/env python3
"""
Minimal TTS test for Raspberry Pi audio output (e.g., Bluetooth speaker).

Usage:
  python3 say_hello.py
  python3 say_hello.py --text "hello"
  python3 say_hello.py --list-engines

It tries common Pi-friendly engines in this order:
  1) espeak-ng
  2) espeak
  3) piper (offline neural TTS; requires model file)
  4) pico2wave + aplay
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def which(cmd: str) -> str | None:
    return shutil.which(cmd)


def run(cmd: list[str]) -> int:
    # Avoid hanging forever if audio backends misbehave.
    # Most TTS calls should finish quickly.
    try:
        p = subprocess.run(cmd, check=False, timeout=20)
        return int(p.returncode)
    except subprocess.TimeoutExpired:
        print(f"Timed out running: {' '.join(cmd)}")
        return 124


def speak_espeak(text: str, prefer_ng: bool) -> int:
    exe = "espeak-ng" if prefer_ng else "espeak"
    if not which(exe):
        return 127
    # Keep it simple: default voice, moderate speed. Quote-free argv.
    return run([exe, "-s", "160", text])


def speak_pico2wave(text: str) -> int:
    if not which("pico2wave") or not which("aplay"):
        return 127
    # pico2wave writes a wav file; aplay plays it (ALSA).
    with tempfile.NamedTemporaryFile(prefix="tts_", suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        rc = run(["pico2wave", "-w", wav_path, text])
        if rc != 0:
            return rc
        return run(["aplay", "-q", wav_path])
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


def find_piper_exe() -> str | None:
    """
    Prefer an activated environment's `piper` on PATH, but also support the
    repo-local uv/venv at ./birds/bin/piper so `python3 say_hello.py` can work.
    """
    exe = which("piper")
    if exe:
        return exe
    local = Path(__file__).parent / "birds" / "bin" / "piper"
    if local.exists():
        return str(local)
    return None


def default_piper_model_path() -> Path:
    return Path(__file__).parent / "tts_models" / "piper" / "en_US-libritts_r-medium.onnx"


def speak_piper(text: str, model_path: str | None, *, verbose: bool) -> int:
    piper_exe = find_piper_exe()
    if not piper_exe or not which("aplay"):
        return 127

    model = Path(model_path) if model_path else default_piper_model_path()
    model_json = Path(str(model) + ".json")
    if not model.exists() or not model_json.exists():
        if verbose:
            print("Piper model not found.")
            print(f"Expected:\n  {model}\n  {model_json}")
        return 127

    with tempfile.NamedTemporaryFile(prefix="piper_", suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        try:
            p = subprocess.run(
                [piper_exe, "--model", str(model), "--output_file", wav_path],
                input=(text + "\n").encode("utf-8"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=40,
                check=False,
            )
        except subprocess.TimeoutExpired:
            if verbose:
                print("Timed out running piper")
            return 124

        if p.returncode != 0:
            if verbose:
                err = p.stderr.decode(errors="ignore").strip()
                if err:
                    print(err)
            return int(p.returncode)

        return run(["aplay", "-q", wav_path])
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


def detect_engines() -> list[str]:
    engines: list[str] = []
    if which("espeak-ng"):
        engines.append("espeak-ng")
    if which("espeak"):
        engines.append("espeak")
    if find_piper_exe() and which("aplay"):
        engines.append("piper")
    if which("pico2wave") and which("aplay"):
        engines.append("pico2wave+aplay")
    return engines


def main() -> int:
    parser = argparse.ArgumentParser(description="Speak a short phrase to test audio output.")
    parser.add_argument("--text", default="hello", help="Text to speak (default: hello)")
    parser.add_argument(
        "--engine",
        default="auto",
        choices=["auto", "piper", "espeak-ng", "espeak", "pico2wave"],
        help="Which engine to use (default: auto)",
    )
    parser.add_argument(
        "--piper-model",
        default=str(default_piper_model_path()),
        help="Path to Piper .onnx model (default: repo-local en_US libritts_r medium)",
    )
    parser.add_argument("--list-engines", action="store_true", help="Print available engines and exit")
    args = parser.parse_args()

    available = detect_engines()
    if args.list_engines:
        if available:
            print("Available engines:", ", ".join(available))
        else:
            print("Available engines: (none detected)")
        return 0

    text = str(args.text).strip()
    if not text:
        print("Error: --text cannot be empty")
        return 2

    if args.engine == "espeak-ng":
        rc = speak_espeak(text, prefer_ng=True)
    elif args.engine == "espeak":
        rc = speak_espeak(text, prefer_ng=False)
    elif args.engine == "piper":
        rc = speak_piper(text, model_path=args.piper_model, verbose=True)
    elif args.engine == "pico2wave":
        rc = speak_pico2wave(text)
    else:
        # auto
        rc = speak_piper(text, model_path=args.piper_model, verbose=False)
        if rc == 127:
            rc = speak_espeak(text, prefer_ng=True)
        if rc == 127:
            rc = speak_espeak(text, prefer_ng=False)
        if rc == 127:
            rc = speak_pico2wave(text)

    if rc == 127:
        print(
            "No supported TTS engine found.\n"
            "Install one of:\n"
            "  - piper-tts (provides `piper`) + alsa-utils (aplay)\n"
            "  - espeak-ng   (recommended)\n"
            "  - espeak\n"
            "  - libttspico-utils (pico2wave) + alsa-utils (aplay)\n"
            "\nThen re-run:\n"
            "  python3 say_hello.py --list-engines"
        )
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())


