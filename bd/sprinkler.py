from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import SprinklerConfig


class SprinklerController:
    """
    Optional HTTP sprinkler trigger used by birdwatch runtime.

    Fires a POST request when a target species is detected, with a global cooldown
    to avoid repeated activations while the animal remains in frame.
    """

    def __init__(self, cfg: SprinklerConfig) -> None:
        self.cfg = cfg
        self._last_triggered_at = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()

        if not cfg.enabled:
            return

        host = cfg.host.rstrip("/")
        duration = int(cfg.duration_s)
        self._url = f"{host}/number/open_for_seconds/set?value={duration}"
        species = ",".join(sorted(cfg.species))
        print(
            f"[sprinkler] enabled host={host} duration={duration}s "
            f"cooldown={cfg.cooldown_s:.0f}s species={species}"
        )

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def maybe_trigger(self, species: str, confidence: float) -> bool:
        if not self.enabled:
            return False

        sp = str(species).strip()
        if sp not in self.cfg.species:
            return False

        conf = float(confidence)
        if conf < self.cfg.min_conf:
            return False

        now = time.time()
        with self._lock:
            if now - self._last_triggered_at < self.cfg.cooldown_s:
                return False
            self._last_triggered_at = now

        threading.Thread(
            target=self._fire,
            args=(sp, conf),
            daemon=True,
            name="sprinkler",
        ).start()
        return True

    def _fire(self, species: str, confidence: float) -> None:
        if self._stop.is_set():
            return
        try:
            import requests
        except Exception as e:
            print(f"[sprinkler] requests not available; skipping trigger ({e})")
            return

        try:
            response = requests.post(self._url, data="", timeout=3.0)
            if response.ok:
                print(f"[sprinkler] triggered {species} (conf={confidence:.2f})")
            else:
                print(
                    f"[sprinkler] trigger failed {species} (conf={confidence:.2f}): "
                    f"HTTP {response.status_code}"
                )
        except Exception as e:
            print(f"[sprinkler] trigger failed {species} (conf={confidence:.2f}): {e}")

    def stop(self) -> None:
        self._stop.set()
