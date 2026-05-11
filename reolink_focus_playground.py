#!/usr/bin/env python3
"""Reolink zoom/focus playground using reolink-aio (>=0.19).

Use this repo's uv-managed venv (not ``/birds`` on the filesystem)::

    cd ~/bird_detector
    source birds/bin/activate
    export REOLINK_HOST=192.168.x.x
    export REOLINK_USER=admin
    export REOLINK_PASSWORD=secret
    # optional: REOLINK_CHANNEL=0
    #
    # LAN cameras often expose the CGI API on HTTP port 80 only. Leave REOLINK_PORT and
    # REOLINK_USE_HTTPS unset so reolink-aio can try HTTPS:443 then HTTP:80. If you set
    # *both* REOLINK_PORT=443 and REOLINK_USE_HTTPS=1, login is pinned to HTTPS only and
    # may fail with "Cannot connect ... 443". Then use either:
    #   birds/bin/python reolink_focus_playground.py --no-https --port 80
    # or unset those two variables.
    birds/bin/python reolink_focus_playground.py
    birds/bin/python reolink_focus_playground.py --set-focus 128
    #
    # If HTTP to :80 and HTTPS to :443 both fail with "Connect call failed", the Pi cannot
    # open a TCP route to the camera (wrong IP, VLAN, guest WiFi / AP isolation, or HTTP
    # disabled). Try Baichuan-only (default TCP 9000), which some WiFi models still answer:
    #   birds/bin/python reolink_focus_playground.py --bc-only
    #   birds/bin/python reolink_focus_playground.py --bc-only --bc-port 9000

Install dependency into that venv::

    uv pip install --python birds/bin/python 'reolink-aio>=0.19'
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any

from reolink_aio.api import Host
from reolink_aio.exceptions import ReolinkError


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str) -> int | None:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return None
    return int(v)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print Reolink zoom/focus range and optionally set focus.")
    p.add_argument("--host", default=os.environ.get("REOLINK_HOST"), help="Camera IP/hostname (or REOLINK_HOST)")
    p.add_argument("--user", "--username", dest="username", default=os.environ.get("REOLINK_USER"), help="REOLINK_USER")
    p.add_argument("--password", default=os.environ.get("REOLINK_PASSWORD"), help="REOLINK_PASSWORD")
    p.add_argument(
        "--port",
        type=int,
        default=_env_int("REOLINK_PORT"),
        help="HTTP(S) port (or REOLINK_PORT)",
    )
    p.add_argument("--https", action="store_true", help="Force HTTPS (overrides REOLINK_USE_HTTPS)")
    p.add_argument(
        "--no-https",
        action="store_true",
        help="Force HTTP (typical for LAN; use with --port 80 if login to :443 fails)",
    )
    p.add_argument(
        "--channel",
        type=int,
        default=_env_int("REOLINK_CHANNEL") or 0,
        help="Channel index (REOLINK_CHANNEL, default 0)",
    )
    p.add_argument(
        "--set-focus",
        type=int,
        default=None,
        metavar="N",
        help="Set absolute focus position to N (clamped to device range)",
    )
    p.add_argument(
        "--bc-only",
        action="store_true",
        default=_env_bool("REOLINK_BC_ONLY"),
        help="Use Baichuan protocol only (no HTTP CGI); default port 9000 (REOLINK_BC_ONLY=1)",
    )
    p.add_argument(
        "--bc-port",
        type=int,
        default=_env_int("REOLINK_BC_PORT") or 9000,
        metavar="PORT",
        help="Baichuan TCP port (default 9000, or REOLINK_BC_PORT)",
    )
    args = p.parse_args()
    if args.https and args.no_https:
        print("Use only one of --https and --no-https.", file=sys.stderr)
        raise SystemExit(2)
    return args


def _use_https_from_args(args: argparse.Namespace) -> bool | None:
    if args.https:
        return True
    if args.no_https:
        return False
    if os.environ.get("REOLINK_USE_HTTPS") is None:
        return None
    return _env_bool("REOLINK_USE_HTTPS")


def _require_credentials(args: argparse.Namespace) -> None:
    missing = [n for n, v in (("host", args.host), ("username", args.username), ("password", args.password)) if not v]
    if missing:
        print(
            "Missing: " + ", ".join(missing) + ". "
            "Set REOLINK_HOST, REOLINK_USER, REOLINK_PASSWORD or pass --host / --user / --password.",
            file=sys.stderr,
        )
        raise SystemExit(2)


async def _run(args: argparse.Namespace) -> None:
    _require_credentials(args)
    assert args.host and args.username and args.password

    host = Host(
        args.host,
        args.username,
        args.password,
        port=args.port,
        use_https=_use_https_from_args(args),
        bc_port=args.bc_port,
        bc_only=args.bc_only,
    )
    try:
        if args.bc_only:
            print(f"mode=baichuan_only bc_port={args.bc_port} (HTTP port {args.port!r} ignored for API)", flush=True)
        await host.get_host_data()
        await host.get_states()

        ch = args.channel
        print(f"model={host.model!r} is_nvr={host.is_nvr} num_channels={host.num_channels} channel={ch}")

        for feat in ("zoom", "focus"):
            ok = host.supported(ch, feat)
            print(f"supported({ch}, {feat!r})={ok}")

        zr: dict[str, Any] | None = None
        try:
            if host.supported(ch, "zoom") or host.supported(ch, "focus"):
                zr = host.zoom_range(ch)
                print(f"zoom_range({ch})={zr!r}")
        except ReolinkError as err:
            print(f"zoom_range failed: {err}", file=sys.stderr)

        if host.supported(ch, "zoom"):
            print(f"get_zoom({ch})={host.get_zoom(ch)}")
        if host.supported(ch, "focus"):
            print(f"get_focus({ch})={host.get_focus(ch)}")

        if args.set_focus is not None:
            if not host.supported(ch, "focus"):
                print("Focus not supported on this channel; skipping --set-focus.", file=sys.stderr)
                return
            if zr is None:
                zr = host.zoom_range(ch)
            fmin = int(zr["focus"]["min"])
            fmax = int(zr["focus"]["max"])
            raw = int(args.set_focus)
            val = max(fmin, min(fmax, raw))
            if val != raw:
                print(f"Clamped focus {raw} -> {val} (device range {fmin}..{fmax})")
            await host.set_focus(ch, val)
            await host.get_states()
            print(f"get_focus({ch}) after set_focus={host.get_focus(ch)}")
    finally:
        await host.logout()


def _login_failure_hint(err: ReolinkError) -> str | None:
    s = str(err)
    msg = s.lower()
    if "connect call failed" in msg or "cannot connect to host" in msg:
        lines = [
            "Hint: TCP connection failed (not an SSL problem — aiohttp often still says 'ssl:default').",
            "  • From this machine: ping the camera IP; try: nc -zv <host> 80  and  nc -zv <host> 9000",
            "  • Confirm the IP (Reolink app / router DHCP). Wrong subnet, VLAN, or Wi‑Fi "
            "client isolation blocks all ports.",
            "  • If HTTP is disabled on the cam but Baichuan is up, try:",
            "      python reolink_focus_playground.py --bc-only",
        ]
        if ":443" in s:
            lines.insert(
                1,
                "  • Port 443: many LAN cameras only serve the CGI API on HTTP :80 — try "
                "`--no-https --port 80` or unset REOLINK_PORT and REOLINK_USE_HTTPS.",
            )
        elif ":80" in s:
            lines.insert(1, "  • Port 80 refused: the Pi cannot reach the camera on HTTP; fix routing/IP/firewall first.")
        return "\n".join(lines)
    if ":443" in s or ("443" in msg and ("connect" in msg or "ssl" in msg or "tls" in msg)):
        return (
            "Hint: login used HTTPS port 443. Many Reolinks only serve the HTTP API on port 80 on the LAN.\n"
            "  python reolink_focus_playground.py --no-https --port 80\n"
            "Or unset REOLINK_PORT and REOLINK_USE_HTTPS so the library can auto-try 443 then 80."
        )
    if ("ssl" in msg or "certificate" in msg) and ":80" not in s:
        return "Hint: try --no-https --port 80 for plain HTTP to the camera on your LAN."
    return None


def main() -> None:
    args = _parse_args()
    try:
        asyncio.run(_run(args))
    except ReolinkError as err:
        print(f"ReolinkError: {err}", file=sys.stderr)
        hint = _login_failure_hint(err)
        if hint:
            print(hint, file=sys.stderr)
        raise SystemExit(1) from err


if __name__ == "__main__":
    main()
