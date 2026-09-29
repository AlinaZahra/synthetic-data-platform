"""UTF-8 exports. `bom=True` prefixes the UTF-8 byte-order mark, which Excel needs to open Urdu/Arabic/Chinese CSVs correctly."""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import pandas as pd

BOM = b"\xef\xbb\xbf"


def _with_bom(data: bytes, bom: bool) -> bytes:
    return BOM + data if bom else data


def json_bytes(obj: Any, bom: bool = False, indent: int | None = 2) -> bytes:
    """ensure_ascii=False keeps Arabic/Urdu/CJK readable in the file instead of \\uXXXX escapes."""
    return _with_bom(json.dumps(obj, ensure_ascii=False, indent=indent, default=str).encode("utf-8"), bom)


def csv_bytes(df: pd.DataFrame, bom: bool = False) -> bytes:
    return _with_bom(df.to_csv(index=False).encode("utf-8"), bom)


def zip_bytes(tables: dict[str, pd.DataFrame], bom: bool = False, extra: dict[str, bytes] | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in tables.items():
            z.writestr(f"{name}.csv", csv_bytes(df, bom))
        for name, payload in (extra or {}).items():
            z.writestr(name, payload)
    return buf.getvalue()
