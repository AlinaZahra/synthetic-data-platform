"""P3. Load generated tables into PostgreSQL / MySQL / MongoDB (dry-run by default, transactional, allowlisted hosts)."""

from __future__ import annotations

from typing import Any

from sdp.connectors.base import ConnectorError, LoadResult, mask, parse_target
from sdp.connectors.mongo import MongoConnector
from sdp.connectors.sql import SqlConnector


def get_connector(url: str, **kw: Any) -> SqlConnector | MongoConnector:
    t = parse_target(url)
    return MongoConnector(url, **kw) if t.dialect == "mongo" else SqlConnector(url, **kw)


__all__ = ["ConnectorError", "LoadResult", "MongoConnector", "SqlConnector", "get_connector", "mask", "parse_target"]
