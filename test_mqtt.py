#!/usr/bin/env python3
"""
Minimal MQTT test utility for Home Assistant integration.

Publishes a single JSON message to the topic (default: "bird_detector"), or
subscribes and prints messages.

Examples:
  # Publish one event
  MQTT_HOST=192.168.0.84 MQTT_USER=... MQTT_PASS=... python3 test_mqtt.py publish

  # Subscribe (to verify HA/broker traffic)
  MQTT_HOST=192.168.0.84 python3 test_mqtt.py sub
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is None:
        return default
    v = v.strip()
    return v if v else default


def _get_cfg() -> dict[str, Any]:
    topic = _env("MQTT_TOPIC", None) or _env("MQTT_TOPIC_EVENT", "bird_detector/event")
    return {
        "host": _env("MQTT_HOST", "192.168.0.84"),
        "port": int(_env("MQTT_PORT", "1883") or "1883"),
        "user": _env("MQTT_USER", None),
        "password": _env("MQTT_PASS", None),
        "topic": topic,
        "client_id": _env("MQTT_CLIENT_ID", "bird_detector_test"),
        "qos": int(_env("MQTT_QOS", "0") or "0"),
        "retain": (_env("MQTT_RETAIN", "0") or "0") in ("1", "true", "True", "yes", "YES"),
    }


def _sample_payload() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "event_id": f"test-{int(time.time() * 1000)}",
        "ts": now.isoformat(),
        "epoch_s": now.timestamp(),
        "source": "bird_detector/test_mqtt.py",
        "yolo_label": "bird",
        "yolo_conf": 0.90,
        "species": "cardinal",
        "species_conf": 0.87,
        "annotated_image": "file://detections/detection_123456_260101.jpg",
    }


def publish_once() -> int:
    try:
        import paho.mqtt.client as mqtt  # type: ignore
    except Exception as e:
        print("Missing dependency: paho-mqtt")
        print("Install with: pip install paho-mqtt")
        print(f"Import error: {e}")
        return 2

    cfg = _get_cfg()
    payload = _sample_payload()

    # paho-mqtt v2+ defaults to deprecated callback API v1; opt into v2 to avoid warnings.
    try:
        client = mqtt.Client(
            client_id=cfg["client_id"],
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            protocol=mqtt.MQTTv311,
        )
    except Exception:
        client = mqtt.Client(client_id=cfg["client_id"], protocol=mqtt.MQTTv311)
    if cfg["user"]:
        client.username_pw_set(cfg["user"], cfg["password"])

    connected = threading.Event()
    connect_error: dict[str, Any] = {"rc": None}
    connect_ok: dict[str, bool] = {"ok": False}

    def on_connect(_client: Any, _userdata: Any, _flags: Any, rc: Any, _properties: Any = None) -> None:
        connect_error["rc"] = rc
        try:
            ok = int(rc) == 0
        except Exception:
            ok = str(rc) == "Success"
        connect_ok["ok"] = ok
        connected.set()

    client.on_connect = on_connect

    # More robust than connect()+loop_start ordering across paho versions.
    client.loop_start()
    try:
        client.connect_async(cfg["host"], cfg["port"], keepalive=30)
        if not connected.wait(timeout=5.0):
            print(f"MQTT connect timeout (rc={connect_error['rc']}) host={cfg['host']} port={cfg['port']}")
            return 2
        if not connect_ok["ok"]:
            print(f"MQTT connect failed (rc={connect_error['rc']}) host={cfg['host']} port={cfg['port']}")
            return 2

        msg = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        info = client.publish(cfg["topic"], msg, qos=cfg["qos"], retain=cfg["retain"])
        info.wait_for_publish(timeout=5)
        print(f"published topic={cfg['topic']!r} qos={cfg['qos']} retain={cfg['retain']}: {msg}")
        return 0
    finally:
        client.loop_stop()
        try:
            client.disconnect()
        except Exception:
            pass


def subscribe_forever() -> int:
    try:
        import paho.mqtt.client as mqtt  # type: ignore
    except Exception as e:
        print("Missing dependency: paho-mqtt")
        print("Install with: pip install paho-mqtt")
        print(f"Import error: {e}")
        return 2

    cfg = _get_cfg()

    def on_connect(client: Any, _userdata: Any, _flags: Any, rc: Any, _properties: Any = None) -> None:
        # rc is an int in older callback API versions, and a ReasonCode in v2.
        ok = False
        try:
            ok = int(rc) == 0
        except Exception:
            ok = str(rc) == "Success"

        if ok:
            print(f"connected; subscribing to {cfg['topic']!r} ...")
            client.subscribe(cfg["topic"], qos=cfg["qos"])
        else:
            print(f"connect failed rc={rc}")

    def on_message(_client: Any, _userdata: Any, msg: Any) -> None:
        try:
            s = msg.payload.decode("utf-8", errors="replace")
        except Exception:
            s = repr(msg.payload)
        print(f"[{msg.topic}] {s}")

    # paho-mqtt v2+ defaults to deprecated callback API v1; opt into v2 to avoid warnings.
    try:
        client = mqtt.Client(client_id=cfg["client_id"], callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    except Exception:
        client = mqtt.Client(client_id=cfg["client_id"])
    client.on_connect = on_connect
    client.on_message = on_message
    if cfg["user"]:
        client.username_pw_set(cfg["user"], cfg["password"])

    client.connect(cfg["host"], cfg["port"], keepalive=30)
    client.loop_forever()
    return 0


def main() -> int:
    # Load repo-local .env if present (real environment variables still win).
    # This mirrors `birdwatch.py` so you can configure MQTT params in one place.
    try:
        from bd.env import load_dotenv  # type: ignore

        base_dir = Path(__file__).resolve().parent
        load_dotenv(base_dir / ".env", override=False)
    except Exception:
        # If bd/ isn't importable for some reason, fall back to plain env vars.
        pass

    cmd = (sys.argv[1] if len(sys.argv) > 1 else "publish").strip().lower()
    if cmd in ("publish", "pub", "p"):
        return publish_once()
    if cmd in ("subscribe", "sub", "s"):
        return subscribe_forever()
    print("Usage:")
    print("  python3 test_mqtt.py publish")
    print("  python3 test_mqtt.py sub")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


