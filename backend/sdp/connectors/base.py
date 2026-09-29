"""P3. Database connectors: load generated tables straight into PostgreSQL, MySQL, MongoDB (and SQLite, used for tests and demos).

Every load is transactional where the database allows it, and `dry_run=True` (the default) performs the full load inside a transaction,
verifies row counts, then ROLLS BACK, so nothing is left behind. Databases that cannot roll back DDL (MySQL) or multi-document writes
(MongoDB without a replica set) fall back to plan-only validation or compensating cleanup, and the result says which one happened.

Safety: connection strings come from the caller, so the host must be on an allowlist (`SDP_CONNECTOR_ALLOW_HOSTS`, default localhost only),
SQLite files must live under the data directory, passwords are never echoed, and table names are always quoted.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pandas as pd


class ConnectorError(RuntimeError):
    pass


SCHEMES = {"postgresql": "postgres", "postgres": "postgres", "mysql": "mysql", "mariadb": "mysql", "mongodb": "mongo", "sqlite": "sqlite"}


@dataclass
class LoadResult:
    target: str
    dialect: str
    mode: str
    dry_run: bool
    ok: bool = False
    committed: bool = False
    rolled_back: bool = False
    strategy: str = "transaction"           # transaction | plan_only | compensating
    tables: dict[str, int] = field(default_factory=dict)      # rows written (or that would be)
    verified: dict[str, int] = field(default_factory=dict)    # rows counted in the database before rollback/commit
    statements: list[str] = field(default_factory=list)       # first few statements, for review
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def mask(url: str) -> str:
    p = urlparse(url)
    if not p.netloc:
        return url
    host = p.hostname or ""
    port = f":{p.port}" if p.port else ""
    user = f"{p.username}:***@" if p.password else (f"{p.username}@" if p.username else "")
    return f"{p.scheme}://{user}{host}{port}{p.path}"


def allowed_hosts() -> set[str]:
    extra = {h.strip().lower() for h in os.environ.get("SDP_CONNECTOR_ALLOW_HOSTS", "").split(",") if h.strip()}
    return {"localhost", "127.0.0.1", "::1"} | extra


def data_dir() -> Path:
    return Path(os.environ.get("SDP_DATA_DIR") or Path(__file__).resolve().parents[2] / "data").resolve()


@dataclass
class Target:
    url: str
    dialect: str
    host: str | None
    database: str | None
    path: Path | None = None


def parse_target(url: str) -> Target:
    p = urlparse(url)
    dialect = SCHEMES.get(p.scheme.split("+")[0])
    if not dialect:
        raise ConnectorError(f"unsupported connection scheme {p.scheme!r}; use postgresql://, mysql://, mongodb:// or sqlite:///")
    if dialect == "sqlite":
        raw = unquote(p.path)[1:]       # sqlite:///rel.db is relative, sqlite:////abs/path.db is absolute (SQLAlchemy convention)
        if raw in ("", ":memory:"):
            return Target(url, dialect, None, ":memory:", None)
        path = Path(raw)
        if not path.is_absolute():
            path = data_dir() / path
        path = path.resolve()
        if os.environ.get("SDP_CONNECTOR_ALLOW_SQLITE_ANY") != "1" and data_dir() not in path.parents:
            raise ConnectorError(f"SQLite files must be inside the data directory ({data_dir()})")
        return Target(url, dialect, None, path.name, path)
    host = (p.hostname or "").lower()
    if host not in allowed_hosts():
        raise ConnectorError(f"host {host or '(none)'!r} is not allowed; add it to SDP_CONNECTOR_ALLOW_HOSTS (default: localhost only)")
    db = unquote(p.path.lstrip("/")) or None
    return Target(url, dialect, host, db)


def py_value(v: Any, kind: str) -> Any:
    """numpy/pandas scalars -> plain Python objects the DB drivers accept; NaN/NaT -> None."""
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, float) and v != v:
        return None
    if hasattr(v, "item") and not isinstance(v, (pd.Timestamp,)):
        v = v.item()
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime() if kind == "datetime" else v.date()
    if kind == "bool":
        return bool(v)
    if kind == "int":
        return int(v)
    if kind == "float":
        f = float(v)
        return None if f != f or f in (float("inf"), float("-inf")) else f
    return v
