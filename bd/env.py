from __future__ import annotations

import os
from pathlib import Path


def env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None:
        return float(default)
    try:
        return float(val.strip())
    except ValueError:
        return float(default)


def env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return int(default)
    try:
        return int(val.strip())
    except ValueError:
        return int(default)


def load_dotenv(path: str | Path, *, override: bool = False) -> dict[str, str]:
    """
    Minimal .env loader (no external dependency).

    Supports lines like:
      KEY=value
      export KEY=value
    Ignores blank lines and comments starting with '#'.

    By default does NOT override already-set environment variables.
    Returns the dict of parsed key/value pairs.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        return {}

    out: dict[str, str] = {}
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        # Strip simple surrounding quotes.
        if len(val) >= 2 and ((val[0] == val[-1]) and val[0] in {"'", '"'}):
            val = val[1:-1]
        out[key] = val

        if override or key not in os.environ:
            os.environ[key] = val

    return out


