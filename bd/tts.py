from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any


class TTSSpeaker:
    """
    Piper TTS + optional bird songs. **No queue:** each `enqueue` replaces the
    previous request; any in-flight audio (Piper, aplay, ffplay) is stopped so
    the latest announcement can start promptly.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        base_dir: Path,
        piper_model: Path,
        min_conf: float,
        cooldown_s: float,
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
        self._lock = threading.Lock()
        # Monotonic token: new enqueue increments; playing code aborts if token changes.
        self._play_seq = 0
        self._args: tuple[str, str, bool, bool] | None = None
        self._wake = threading.Event()
        self._active_popen: subprocess.Popen[Any] | None = None
        self._thread = threading.Thread(target=self._worker, daemon=True, name="tts")

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
                    f"[tts] enabled engine=piper (latest-wins, no queue), "
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

    def _set_child(self, p: subprocess.Popen[Any] | None) -> None:
        self._active_popen = p

    def _terminate_current_child(self) -> None:
        p = self._active_popen
        if p is None:
            return
        if p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=0.35)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        self._active_popen = None

    def _still_current(self, token: int) -> bool:
        with self._lock:
            return self._play_seq == token

    def _wait_popen(
        self,
        p: subprocess.Popen[Any],
        token: int,
        *,
        kill_on_supersede: bool = True,
    ) -> bool:
        """
        Return True if process exited with this token still the active play.
        Return False if superseded (newer enqueue) or process failed.
        """
        while p.poll() is None:
            if not self._still_current(token) and kill_on_supersede:
                try:
                    p.terminate()
                except Exception:
                    pass
                return False
            time.sleep(0.04)
        if not self._still_current(token) and kill_on_supersede:
            return False
        return p.returncode == 0

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
            print(f"[tts] could not write preroll wav: {e}")
            try:
                os.unlink(out_path)
            except OSError:
                pass
            return None

    def _speak_piper(self, text: str, token: int) -> bool:
        """Synthesize and play. Returns True if this token is still current at end and playback succeeded."""
        piper_exe = self._find_piper_exe()
        if not piper_exe or not self._which("aplay"):
            return False
        model_json = Path(str(self.piper_model) + ".json")
        if not self.piper_model.exists() or not model_json.exists():
            return False
        with tempfile.NamedTemporaryFile(prefix="piper_", suffix=".wav", delete=False) as f:
            wav_path = f.name
        padded_path: str | None = None
        p2: subprocess.Popen[Any] | None = None
        try:
            p1 = subprocess.Popen(
                [piper_exe, "--model", str(self.piper_model), "--output_file", wav_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self._set_child(p1)
            if p1.stdin is not None:
                try:
                    p1.stdin.write((text + "\n").encode("utf-8"))
                finally:
                    try:
                        p1.stdin.close()
                    except BrokenPipeError:
                        pass
            if not self._wait_popen(p1, token):
                return False
            if p1.returncode != 0:
                err = (p1.stderr.read() or b"").decode(errors="ignore").strip() if p1.stderr else ""
                if err:
                    print(f"[tts] piper error: {err}")
                return False

            play_path = wav_path
            if self.preroll_ms > 0:
                padded_path = self._add_wav_preroll(wav_path, self.preroll_ms)
                if padded_path:
                    play_path = padded_path

            p2 = subprocess.Popen(
                ["aplay", "-q", play_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._set_child(p2)
            if not self._wait_popen(p2, token):
                return False
            if p2.returncode != 0:
                return False
            return self._still_current(token)
        except Exception as e:
            print(f"[tts] _speak_piper: {e}")
            return False
        finally:
            self._set_child(None)
            try:
                os.unlink(wav_path)
            except OSError:
                pass
            if padded_path:
                try:
                    os.unlink(padded_path)
                except OSError:
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

    def _play_audio(self, path: Path, token: int) -> bool:
        max_s = float(self.bird_songs_max_s)
        if max_s <= 0:
            return self._still_current(token)
        ffplay = self._which("ffplay")
        try:
            if ffplay:
                p = subprocess.Popen(
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
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._set_child(p)
                if not self._wait_popen(p, token):
                    return False
                return self._still_current(token) and (p.returncode or 0) == 0
            if path.suffix.lower() == ".wav" and self._which("aplay"):
                p = subprocess.Popen(
                    ["aplay", "-q", str(path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._set_child(p)
                if not self._wait_popen(p, token):
                    return False
                return self._still_current(token) and (p.returncode or 0) == 0
            return self._still_current(token)
        finally:
            self._set_child(None)

    def _play_request(self, raw: str, spoken: str, should_speak: bool, should_song: bool, token: int) -> None:
        if should_speak:
            if not self._speak_piper(spoken, token):
                return
        if not self._still_current(token):
            return
        if should_song:
            song = self._find_bird_song(raw)
            if song is not None and self._still_current(token):
                self._play_audio(song, token)

    def _worker(self) -> None:
        while not self._stop.is_set():
            if not self._wake.wait(timeout=0.25):
                continue
            self._wake.clear()
            with self._lock:
                args = self._args
                tok = self._play_seq
            if args is None:
                continue
            raw, spoken, should_speak, should_song = args
            try:
                self._play_request(str(raw), str(spoken), bool(should_speak), bool(should_song), tok)
            except Exception as e:
                print(f"[tts] error: {e}")

    def enqueue(self, raw_label: str, spoken_text: str, *, speak: bool, play_song: bool) -> None:
        if not self.enabled:
            return
        raw = str(raw_label).strip()
        spoken = str(spoken_text).strip()
        if not raw or not spoken:
            return
        if not speak and not play_song:
            return
        with self._lock:
            self._play_seq += 1
            self._args = (raw, spoken, bool(speak), bool(play_song))
        self._terminate_current_child()
        self._wake.set()
        parts = []
        if speak:
            parts.append("speak")
        if play_song:
            parts.append("song")
        what = "+".join(parts) if parts else "none"
        print(
            f'[tts] audio event: label={raw!r} text={spoken!r} ({what}, replaces in-flight)',
            flush=True,
        )

    def stop(self) -> None:
        self._stop.set()
        self._terminate_current_child()
        self._wake.set()
