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
import json
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


def _piper_sample_rate(model: Path) -> int | None:
    model_json = Path(str(model) + ".json")
    if not model_json.exists():
        return None
    try:
        data = json.loads(model_json.read_text(encoding="utf-8"))
        sr = int(data.get("audio", {}).get("sample_rate", 0))
        return sr if sr > 0 else None
    except Exception:
        return None


class _PiperRawStream:
    """
    Keep Piper loaded by running it once with --output-raw and streaming audio
    into a long-lived `aplay` process.
    """

    def __init__(self, *, model: Path) -> None:
        self.model = model
        self._piper: subprocess.Popen[bytes] | None = None
        self._aplay: subprocess.Popen[bytes] | None = None

    def start(self) -> int:
        piper_exe = find_piper_exe()
        if not piper_exe or not which("aplay"):
            return 127

        model_json = Path(str(self.model) + ".json")
        if not self.model.exists() or not model_json.exists():
            return 127

        sample_rate = _piper_sample_rate(self.model) or 22050

        # Piper reads text from stdin, writes raw PCM to stdout. We pipe that into aplay.
        self._piper = subprocess.Popen(
            [
                piper_exe,
                "--model",
                str(self.model),
                "--output_raw",
                "--sentence_silence",
                "0.15",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        assert self._piper.stdout is not None
        self._aplay = subprocess.Popen(
            [
                "aplay",
                "-q",
                "-t",
                "raw",
                "-f",
                "S16_LE",
                "-c",
                "1",
                "-r",
                str(sample_rate),
            ],
            stdin=self._piper.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Let aplay own the pipe; if we keep this fd open, EOF handling gets weird.
        self._piper.stdout.close()
        return 0

    def speak(self, text: str) -> int:
        if not self._piper or not self._piper.stdin:
            return 127
        if self._piper.poll() is not None:
            return int(self._piper.returncode or 1)

        phrase = str(text).strip()
        if not phrase:
            return 0
        try:
            self._piper.stdin.write((phrase + "\n").encode("utf-8"))
            self._piper.stdin.flush()
            return 0
        except BrokenPipeError:
            return 1

    def stop(self) -> None:
        for p in (self._piper, self._aplay):
            if p is None:
                continue
            try:
                if p.poll() is None:
                    p.terminate()
            except Exception:
                pass
        for p in (self._piper, self._aplay):
            if p is None:
                continue
            try:
                p.wait(timeout=2)
            except Exception:
                try:
                    p.kill()
                except Exception:
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


def say_hello(
    text: str | None,
    *,
    engine: str,
    piper_model: str,
    prompt: str = "Type what you want me to say, then press Enter: ",
) -> int:
    """
    Interactive mode: keep prompting/speaking until Ctrl+C.

    If text is provided, it is spoken first, then the prompt continues.
    Piper is kept loaded by running a long-lived `piper --output-raw` process.
    """
    chosen = str(engine).strip()
    if chosen == "auto":
        # Prefer Piper if possible; otherwise fall back to espeak-ng/espeak/pico2wave.
        if find_piper_exe() and which("aplay"):
            chosen = "piper"
        elif which("espeak-ng"):
            chosen = "espeak-ng"
        elif which("espeak"):
            chosen = "espeak"
        else:
            chosen = "pico2wave"

    piper_stream: _PiperRawStream | None = None
    if chosen == "piper":
        model = Path(piper_model) if piper_model else default_piper_model_path()
        piper_stream = _PiperRawStream(model=model)
        rc = piper_stream.start()
        if rc != 0:
            # Keep behavior consistent with the rest of the script.
            return int(rc)

    def speak_once(phrase: str) -> int:
        if chosen == "piper":
            assert piper_stream is not None
            return piper_stream.speak(phrase)
        if chosen == "espeak-ng":
            return speak_espeak(phrase, prefer_ng=True)
        if chosen == "espeak":
            return speak_espeak(phrase, prefer_ng=False)
        if chosen == "pico2wave":
            return speak_pico2wave(phrase)
        return 2

    try:
        # Optional initial phrase.
        if text is not None:
            first = str(text).strip()
            if first:
                speak_once(first)

        # REPL loop until Ctrl+C / EOF.
        while True:
            try:
                line = input(prompt)
            except EOFError:
                break
            phrase = str(line)
            if not phrase.strip():
                continue
            speak_once(phrase)
    except KeyboardInterrupt:
        pass
    finally:
        if piper_stream is not None:
            piper_stream.stop()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Speak a short phrase to test audio output.")
    parser.add_argument(
        "--text",
        default=None,
        help="Optional initial text to speak before entering interactive mode.",
    )
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

    rc = say_hello(
        args.text,
        engine=str(args.engine),
        piper_model=str(args.piper_model),
    )

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


