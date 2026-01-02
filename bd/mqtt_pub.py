from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any

from .config import MQTTConfig


class MQTTPublisher:
    """
    Tiny MQTT publisher wrapper used by birdwatch runtime.

    - Publishes per-frame detection events (non-retained)
    - Publishes "latest state" (retained)
    - Keeps paho-mqtt as an optional dependency; if missing or connection fails,
      publishing becomes a no-op (and birdwatch continues).
    """

    def __init__(self, cfg: MQTTConfig):
        self.cfg = cfg
        self._client = None
        self._connected = False

        if not cfg.enabled:
            return

        try:
            import paho.mqtt.client as mqtt  # type: ignore
        except Exception as e:
            print(f"[mqtt] enabled but paho-mqtt is not installed; disabling mqtt ({e})")
            return

        # paho-mqtt v2+ defaults to deprecated callback API v1; opt into v2 when available.
        try:
            client = mqtt.Client(
                client_id=cfg.client_id,
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                protocol=mqtt.MQTTv311,
            )
        except Exception:
            client = mqtt.Client(client_id=cfg.client_id, protocol=mqtt.MQTTv311)

        if cfg.username:
            client.username_pw_set(cfg.username, cfg.password)

        def on_connect(_client: Any, _userdata: Any, _flags: Any, rc: Any, _properties: Any = None) -> None:
            ok = False
            try:
                ok = int(rc) == 0
            except Exception:
                ok = str(rc) == "Success"

            self._connected = ok
            if ok:
                print(f"[mqtt] connected: {cfg.host}:{cfg.port} topic_event={cfg.topic_event!r}")
            else:
                print(f"[mqtt] connect failed rc={rc}; disabling mqtt publishes")

        def on_disconnect(_client: Any, _userdata: Any, rc: Any, _properties: Any = None) -> None:
            self._connected = False
            if rc:
                print(f"[mqtt] disconnected rc={rc}")

        client.on_connect = on_connect
        client.on_disconnect = on_disconnect

        self._client = client

        try:
            # connect_async + loop_start makes connection setup consistent across paho versions.
            client.loop_start()
            client.connect_async(cfg.host, int(cfg.port), keepalive=30)
            # Give it a brief moment to connect; we don't want to block startup for long.
            t0 = time.time()
            while time.time() - t0 < 2.0 and not self._connected:
                time.sleep(0.05)
        except Exception as e:
            print(f"[mqtt] connection failed; disabling mqtt publishes ({e})")
            try:
                client.loop_stop()
            except Exception:
                pass
            self._client = None
            self._connected = False

    def _publish_json(self, topic: str, payload: dict[str, Any], *, retain: bool) -> None:
        if not self.cfg.enabled or self._client is None or not self._connected:
            return
        try:
            msg = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
            self._client.publish(topic, msg, qos=int(self.cfg.qos), retain=bool(retain))
        except Exception as e:
            print(f"[mqtt] publish failed: {e}")

    def publish_event(self, payload: dict[str, Any]) -> None:
        self._publish_json(self.cfg.topic_event, payload, retain=False)

    def publish_state(self, payload: dict[str, Any]) -> None:
        self._publish_json(self.cfg.topic_state, payload, retain=bool(self.cfg.retain_state))

    def stop(self) -> None:
        if self._client is None:
            return
        try:
            self._client.loop_stop()
        except Exception:
            pass
        try:
            self._client.disconnect()
        except Exception:
            pass

    def debug_summary(self) -> dict[str, Any]:
        d = asdict(self.cfg)
        d["connected"] = bool(self._connected)
        if d.get("password"):
            d["password"] = "***"
        return d


