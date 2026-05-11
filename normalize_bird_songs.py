#!/usr/bin/env python3
from __future__ import annotations

"""
Loudness-normalize bird song clips for birdwatch / bd.tts playback.

Requires ffmpeg and ffprobe on PATH (same toolchain as ffplay).

Reads masters from --input-dir (default: <bird_songs>/originals). Each clip is
downmixed to mono when needed (stereo: equal-power L+R), EBU loudnorm is applied
on that mono signal, then duplicated to dual-mono stereo (L=R). Outputs are
192 kbps MP3 (<stem>.mp3) by default.

  python3 normalize_bird_songs.py --dry-run
  python3 normalize_bird_songs.py
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


AUDIO_EXTS = frozenset({".mp3", ".wav", ".ogg", ".flac", ".m4a"})
JUNK_NAME_RE = re.compile(r"^\._")

# ffmpeg stderr may contain non-UTF-8 metadata (e.g. Latin-1 ©); avoid decode errors.
_SUBPROCESS_TEXT_KW = {"encoding": "utf-8", "errors": "replace"}


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    default_out = Path(os.environ.get("BIRD_SONGS_DIR", str(base / "bird_songs"))).expanduser()

    p = argparse.ArgumentParser(
        description=(
            "Normalize bird_songs clips (mono loudnorm, dual-mono stereo MP3) via ffmpeg loudnorm."
        )
    )
    p.add_argument(
        "--dir",
        type=Path,
        default=default_out,
        help=f"Output directory for normalized files (default: env BIRD_SONGS_DIR or {base / 'bird_songs'}).",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory of source clips (default: <output-dir>/originals).",
    )
    p.add_argument(
        "--i",
        dest="target_i",
        type=float,
        default=-16.0,
        help="Target integrated loudness in LUFS (default: -16).",
    )
    p.add_argument(
        "--tp",
        dest="target_tp",
        type=float,
        default=-1.5,
        help="Target true peak in dBTP (default: -1.5).",
    )
    p.add_argument(
        "--lra",
        dest="target_lra",
        type=float,
        default=11.0,
        help="Target loudness range (default: 11).",
    )
    p.add_argument(
        "--bitrate",
        type=int,
        default=192,
        metavar="KBPS",
        help="Output MP3 bitrate in kbps (default: 192).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print actions only.")
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return p.parse_args()


def which_or_die(name: str) -> str:
    path = shutil.which(name)
    if not path:
        print(f"error: `{name}` not found on PATH (install ffmpeg).", file=sys.stderr)
        sys.exit(1)
    return path


def is_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_EXTS and not JUNK_NAME_RE.match(path.name)


def encode_args_mp3_kbps(bitrate: int) -> list[str]:
    if bitrate < 32 or bitrate > 320:
        raise ValueError(f"bitrate out of range (32–320): {bitrate}")
    return ["-c:a", "libmp3lame", "-b:a", f"{bitrate}k"]


def audio_channels(ffprobe: str, path: Path) -> int:
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
        **_SUBPROCESS_TEXT_KW,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {path} (exit {proc.returncode}):\n{proc.stderr}"
        )
    line = (proc.stdout or "").strip().splitlines()
    if not line:
        raise RuntimeError(f"ffprobe returned no channel info for {path}")
    raw = line[0].strip().split(",")[0].strip()
    try:
        return int(raw)
    except ValueError as e:
        raise RuntimeError(f"ffprobe invalid channels for {path}: {line[0]!r}") from e


def mono_downmix_prefix(channels: int) -> str:
    """Return pan filter to feed loudnorm, or empty when already mono."""
    if channels == 1:
        return ""
    if channels == 2:
        return "pan=mono|c0=0.5*c0+0.5*c1"
    raise ValueError(
        f"unsupported audio channel count ({channels}); only mono (1) and stereo (2) are supported"
    )


def join_af(*parts: str) -> str:
    return ",".join(p for p in parts if p)


def extract_loudnorm_json(stderr_text: str) -> dict:
    key = '"input_i"'
    pos = stderr_text.find(key)
    if pos == -1:
        raise ValueError("loudnorm JSON not found in ffmpeg stderr (missing input_i).")
    start = stderr_text.rfind("{", 0, pos)
    if start == -1:
        raise ValueError("loudnorm JSON start brace not found.")
    depth = 0
    end = -1
    for i in range(start, len(stderr_text)):
        c = stderr_text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        raise ValueError("loudnorm JSON end brace not found.")
    return json.loads(stderr_text[start:end])


def loudnorm_measure(ffmpeg: str, inp: Path, af: str) -> dict:
    proc = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(inp),
            "-af",
            af,
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
        **_SUBPROCESS_TEXT_KW,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"loudnorm measure failed for {inp} (exit {proc.returncode}):\n{proc.stderr}"
        )
    return extract_loudnorm_json(proc.stderr)


def loudnorm_apply(
    ffmpeg: str,
    inp: Path,
    outp: Path,
    af: str,
    extra_encode: list[str],
    *,
    force: bool,
) -> None:
    cmd = [ffmpeg, "-hide_banner", "-nostats"]
    if force:
        cmd.append("-y")
    cmd.extend(
        [
            "-i",
            str(inp),
            "-af",
            af,
            *extra_encode,
            str(outp),
        ]
    )
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        **_SUBPROCESS_TEXT_KW,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"loudnorm encode failed for {inp} -> {outp} (exit {proc.returncode}):\n"
            f"{proc.stderr}"
        )


def main() -> None:
    args = parse_args()
    out_dir = args.dir.expanduser().resolve()
    input_dir = (
        args.input_dir.expanduser().resolve()
        if args.input_dir is not None
        else (out_dir / "originals").resolve()
    )

    if not out_dir.is_dir():
        print(f"error: output directory does not exist: {out_dir}", file=sys.stderr)
        sys.exit(1)
    if not input_dir.is_dir():
        print(f"error: input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    ffmpeg = which_or_die("ffmpeg")
    ffprobe = which_or_die("ffprobe")

    inputs = sorted(p for p in input_dir.iterdir() if is_audio_file(p))
    if not inputs:
        print(f"[normalize] no audio files under {input_dir}")
        return

    print(
        f"[normalize] input={input_dir} output={out_dir} "
        f"I={args.target_i} LUFS TP={args.target_tp} dBTP LRA={args.target_lra} "
        f"mp3={args.bitrate}k stereo_dual_mono ({len(inputs)} file(s))"
    )

    for inp in inputs:
        outp = out_dir / f"{inp.stem}.mp3"
        try:
            ch = audio_channels(ffprobe, inp)
            prefix = mono_downmix_prefix(ch)
        except (RuntimeError, ValueError) as e:
            print(f"error: {inp.name}: {e}", file=sys.stderr)
            sys.exit(1)

        ch_note = f"{ch}ch"
        if ch == 2:
            ch_note += "->mono"
        elif ch == 1:
            ch_note += " mono"

        if outp.exists() and not args.force:
            print(f"[normalize] skip (exists, use --force): {inp.stem}.mp3 <- {inp.name}")
            continue
        print(f"[normalize] {inp.name} -> {outp.name} ({ch_note})")
        if args.dry_run:
            extra = encode_args_mp3_kbps(args.bitrate)
            measure_af = join_af(
                prefix,
                f"loudnorm=I={args.target_i}:TP={args.target_tp}:LRA={args.target_lra}:print_format=json",
            )
            encode_af = join_af(
                prefix,
                f"loudnorm=I={args.target_i}:TP={args.target_tp}:LRA={args.target_lra}:measured_*",
                "pan=stereo|c0=c0|c1=c0",
            )
            print(
                f"  (dry-run) measure -af {measure_af!r}; "
                f"encode -af {encode_af!r}; {' '.join(extra)}"
            )
            continue

        measure_af = join_af(
            prefix,
            f"loudnorm=I={args.target_i}:TP={args.target_tp}:LRA={args.target_lra}:print_format=json",
        )
        try:
            measured = loudnorm_measure(ffmpeg, inp, measure_af)
            mi = float(measured["input_i"])
            mtp = float(measured["input_tp"])
            mlra = float(measured["input_lra"])
            mth = float(measured["input_thresh"])
            off = float(measured["target_offset"])
        except (RuntimeError, ValueError, KeyError) as e:
            print(f"error: {inp.name}: {e}", file=sys.stderr)
            sys.exit(1)

        loudnorm_enc = (
            f"loudnorm=I={args.target_i}:TP={args.target_tp}:LRA={args.target_lra}"
            f":measured_I={mi}:measured_TP={mtp}:measured_LRA={mlra}"
            f":measured_thresh={mth}:offset={off}:linear=true"
        )
        encode_af = join_af(prefix, loudnorm_enc, "pan=stereo|c0=c0|c1=c0")

        try:
            extra = encode_args_mp3_kbps(args.bitrate)
            loudnorm_apply(ffmpeg, inp, outp, encode_af, extra, force=args.force)
            print(f"         measured_I={measured.get('input_i')} -> wrote {outp}")
        except (RuntimeError, ValueError) as e:
            print(f"error: {inp.name}: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
