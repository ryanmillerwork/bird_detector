from __future__ import annotations

import os
import queue
import shutil
import subprocess
import tempfile
import threading
import wave
from pathlib import Path


class TTSSpeaker:
    """
    Lightweight, non-blocking Piper-only TTS wrapper (optional bird songs).

    Env vars (typically wired by the caller):
      - TTS_ENABLED: 0/1 (default 1)
      - TTS_PIPER_MODEL: path to .onnx model (expects matching .onnx.json)
      - TTS_MIN_CONF: minimum species confidence to speak (default 0.0)
      - TTS_COOLDOWN_S: minimum seconds between repeating the same phrase (default 15)
      - TTS_MAX_QUEUE: max queued phrases (default 10)
      - TTS_PREROLL_MS: leading silence (ms) added to audio before speech (default 650)
      - BIRD_SONGS_ENABLED: 0/1 (default 1 when TTS is enabled)
      - BIRD_SONGS_DIR: directory of audio files named like "<class>.(mp3|wav)" (default: ./bird_songs)
      - BIRD_SONGS_MAX_S: max seconds of bird song to play after speech (default 10)
    """

    def __init__(
        self,
        *,
        enabled: bool,
        base_dir: Path,
        piper_model: Path,
        min_conf: float,
        cooldown_s: float,
        max_queue: int,
        preroll_ms: int,
        bird_songs_enabled: bool,
        bird_songs_dir: Path,
        bird_songs_max_s: float,
    ) -> None:
        self.enabled = bool(enabled)
        self.base_dir = base_dir
        self.piper_model = piper_model
        self.min_conf = float(min_conf)
        self.cooldown_s = float(cooldown_s)
        self.preroll_ms = max(0, int(preroll_ms))
        self.bird_songs_enabled = bool(bird_songs_enabled) and bool(enabled)
        self.bird_songs_dir = bird_songs_dir
        self.bird_songs_max_s = max(0.0, float(bird_songs_max_s))
        self._stop = threading.Event()
        self._q: queue.Queue[tuple[str, str, bool, bool]] = queue.Queue(maxsize=max(1, int(max_queue)))
        self._thread = threading.Thread(target=self._worker, daemon=True)

        if self.enabled:
            piper_exe = self._find_piper_exe()
            has_aplay = bool(self._which("aplay"))
            model_json = Path(str(self.piper_model) + ".json")
            if not piper_exe or not has_aplay:
                print("[tts] Piper unavailable (missing `piper` and/or `aplay`); disabling TTS.")
                self.enabled = False
                self.bird_songs_enabled = False
            elif not self.piper_model.exists() or not model_json.exists():
                print(f"[tts] Piper model missing; disabling TTS: {self.piper_model} (+ .json)")
                self.enabled = False
                self.bird_songs_enabled = False
            else:
                print(
                    f"[tts] enabled engine=piper, "
                    f"min_conf={self.min_conf:.2f}, cooldown_s={self.cooldown_s:.0f}, preroll_ms={self.preroll_ms}, "
                    f"bird_songs={'on' if self.bird_songs_enabled else 'off'}"
                )
                self._thread.start()

    @staticmethod
    def _which(cmd: str) -> str | None:
        return shutil.which(cmd)

    def _find_piper_exe(self) -> str | None:
        exe = self._which("piper")
        if exe:
            return exe
        local = self.base_dir / "birds" / "bin" / "piper"
        if local.exists():
            return str(local)
        return None

    def _speak_piper(self, text: str) -> int:
        piper_exe = self._find_piper_exe()
        if not piper_exe or not self._which("aplay"):
            return 127

        model_json = Path(str(self.piper_model) + ".json")
        if not self.piper_model.exists() or not model_json.exists():
            return 127

        with tempfile.NamedTemporaryFile(prefix="piper_", suffix=".wav", delete=False) as f:
            wav_path = f.name
        padded_path = None
        try:
            try:
                p = subprocess.run(
                    [piper_exe, "--model", str(self.piper_model), "--output_file", wav_path],
                    input=(text + "\n").encode("utf-8"),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=40,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return 124

            if p.returncode != 0:
                err = p.stderr.decode(errors="ignore").strip()
                if err:
                    print(f"[tts] piper error: {err}")
                return int(p.returncode)

            play_path = wav_path
            if self.preroll_ms > 0:
                padded_path = self._add_wav_preroll(wav_path, self.preroll_ms)
                if padded_path:
                    play_path = padded_path

            p2 = subprocess.run(["aplay", "-q", play_path], check=False, timeout=20)
            return int(p2.returncode)
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass
            if padded_path:
                try:
                    os.unlink(padded_path)
                except OSError:
                    pass

    @staticmethod
    def _add_wav_preroll(wav_path: str, preroll_ms: int) -> str | None:
        """Prepend silence to a PCM WAV file so Bluetooth speakers don't clip the first syllable."""
        try:
            with wave.open(wav_path, "rb") as r:
                params = r.getparams()
                audio = r.readframes(r.getnframes())
        except Exception as e:
            print(f"[tts] could not read wav for preroll: {e}")
            return None

        framerate = int(params.framerate)
        channels = int(params.nchannels)
        sampwidth = int(params.sampwidth)
        silent_frames = int((framerate * int(preroll_ms)) / 1000)
        if silent_frames <= 0:
            return None
        silence = b"\x00" * (silent_frames * channels * sampwidth)

        with tempfile.NamedTemporaryFile(prefix="piper_pad_", suffix=".wav", delete=False) as f:
            out_path = f.name
        try:
            with wave.open(out_path, "wb") as w:
                w.setparams(params)
                w.writeframes(silence)
                w.writeframes(audio)
            return out_path
        except Exception as e:
            print(f"[tts] could not write padded wav: {e}")
            try:
                os.unlink(out_path)
            except OSError:
                pass
            return None

    def enqueue(self, raw_label: str, spoken_text: str, *, speak: bool, play_song: bool) -> None:
        if not self.enabled:
            return
        raw = str(raw_label).strip()
        spoken = str(spoken_text).strip()
        if not raw or not spoken:
            return
        if not speak and not play_song:
            return
        try:
            self._q.put_nowait((raw, spoken, bool(speak), bool(play_song)))
        except queue.Full:
            pass

    def _find_bird_song(self, raw_label: str) -> Path | None:
        if not self.bird_songs_enabled:
            return None
        base = raw_label.strip()
        if not base:
            return None
        d = self.bird_songs_dir
        if not d.exists() or not d.is_dir():
            return None

        for ext in (".mp3", ".wav", ".ogg", ".flac", ".m4a"):
            p = d / f"{base}{ext}"
            if p.exists() and p.is_file():
                return p
        return None

    def _play_audio(self, path: Path) -> int:
        """Play up to bird_songs_max_s seconds of an audio file."""
        max_s = float(self.bird_songs_max_s)
        if max_s <= 0:
            return 0

        ffplay = self._which("ffplay")
        if ffplay:
            try:
                p = subprocess.run(
                    [
                        ffplay,
                        "-nodisp",
                        "-autoexit",
                        "-loglevel",
                        "error",
                        "-t",
                        f"{max_s:.3f}",
                        str(path),
                    ],
                    check=False,
                    timeout=max_s + 5.0,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return int(p.returncode)
            except subprocess.TimeoutExpired:
                return 124

        if path.suffix.lower() == ".wav" and self._which("aplay"):
            try:
                p = subprocess.run(
                    ["aplay", "-q", str(path)],
                    check=False,
                    timeout=max_s + 5.0,
                )
                return int(p.returncode)
            except subprocess.TimeoutExpired:
                return 124

        return 127

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                raw, spoken, should_speak, should_song = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if should_speak:
                    self._speak_piper(spoken)
                if should_song:
                    song = self._find_bird_song(raw)
                    if song is not None:
                        self._play_audio(song)
            except Exception as e:
                print(f"[tts] error: {e}")

    def stop(self) -> None:
        self._stop.set()


