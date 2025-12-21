from __future__ import annotations

"""
Postgres helper.

This module is intentionally small and dependency-light. It reads connection
settings from environment variables and returns a ready Postgres connection.

Supported env vars:
- DATABASE_URL=postgresql://user:pass@host:5432/dbname
  OR the standard libpq variables:
  - PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD, PGSSLMODE, PGCONNECT_TIMEOUT

Driver support:
- psycopg (v3) preferred
- psycopg2 supported as a fallback
"""

import os
from dataclasses import dataclass
from typing import Any


class DBConfigError(RuntimeError):
    pass


class DBDriverMissingError(RuntimeError):
    pass


@dataclass(frozen=True)
class PostgresConfig:
    dsn: str | None
    host: str | None
    port: int | None
    dbname: str | None
    user: str | None
    password: str | None
    sslmode: str | None
    connect_timeout_s: int | None
    application_name: str

    @classmethod
    def from_env(cls) -> "PostgresConfig":
        dsn = os.environ.get("DATABASE_URL") or None

        host = os.environ.get("PGHOST") or None
        port_raw = os.environ.get("PGPORT") or None
        dbname = os.environ.get("PGDATABASE") or None
        user = os.environ.get("PGUSER") or None
        password = os.environ.get("PGPASSWORD") or None
        sslmode = os.environ.get("PGSSLMODE") or None
        timeout_raw = os.environ.get("PGCONNECT_TIMEOUT") or None

        port = None
        if port_raw:
            try:
                port = int(port_raw.strip())
            except ValueError:
                port = None

        connect_timeout_s = None
        if timeout_raw:
            try:
                connect_timeout_s = int(timeout_raw.strip())
            except ValueError:
                connect_timeout_s = None

        application_name = os.environ.get("PGAPPLICATION_NAME", "bird_detector")

        return cls(
            dsn=dsn,
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            sslmode=sslmode,
            connect_timeout_s=connect_timeout_s,
            application_name=application_name,
        )

    def conninfo(self) -> str:
        """
        Build a libpq-style conninfo string.

        Note: if DATABASE_URL is present, prefer that directly.
        """
        if self.dsn:
            return self.dsn

        missing = [k for k, v in {"PGHOST": self.host, "PGDATABASE": self.dbname, "PGUSER": self.user}.items() if not v]
        if missing:
            raise DBConfigError(
                "Missing required Postgres env vars. Provide DATABASE_URL or set: "
                + ", ".join(missing)
            )

        parts: list[str] = []
        parts.append(f"host={self.host}")
        if self.port is not None:
            parts.append(f"port={self.port}")
        parts.append(f"dbname={self.dbname}")
        parts.append(f"user={self.user}")
        if self.password is not None:
            parts.append(f"password={self.password}")
        if self.sslmode is not None:
            parts.append(f"sslmode={self.sslmode}")
        if self.connect_timeout_s is not None:
            parts.append(f"connect_timeout={self.connect_timeout_s}")
        if self.application_name:
            parts.append(f"application_name={self.application_name}")
        return " ".join(parts)


def _import_driver() -> tuple[str, Any]:
    """
    Returns (driver_name, driver_module).
    """
    try:
        import psycopg  # type: ignore

        return "psycopg", psycopg
    except Exception:
        pass

    try:
        import psycopg2  # type: ignore

        return "psycopg2", psycopg2
    except Exception:
        pass

    raise DBDriverMissingError(
        "No Postgres driver installed. Install one of:\n"
        "- psycopg (v3): `uv pip install psycopg[binary]`\n"
        "- psycopg2 (v2): `uv pip install psycopg2-binary`"
    )


def connect(*, autocommit: bool = True):
    """
    Create and return a Postgres connection.
    """
    cfg = PostgresConfig.from_env()
    conninfo = cfg.conninfo()
    driver_name, driver = _import_driver()

    if driver_name == "psycopg":
        conn = driver.connect(conninfo)  # type: ignore[attr-defined]
        conn.autocommit = autocommit
        return conn

    # psycopg2 fallback
    conn = driver.connect(conninfo)  # type: ignore[attr-defined]
    conn.autocommit = autocommit
    return conn


def test_connection() -> None:
    """
    Quick smoke test: connects and runs SELECT 1.
    """
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute("select 1;")
        row = cur.fetchone()
        print(f"DB OK, SELECT 1 -> {row}")
    finally:
        conn.close()


if __name__ == "__main__":
    test_connection()





