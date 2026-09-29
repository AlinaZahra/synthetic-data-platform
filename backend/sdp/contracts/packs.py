"""Starter packs: data + contract in one file each (`packs/<name>.json`). A pack names a generator and lists expectations.
Add a pack = add a JSON file (generator must be one of GENERATORS)."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from sdp.contracts.expectations import to_great_expectations, validate_suite

PACK_DIR = Path(__file__).parent / "packs"


class PackError(ValueError):
    pass


def _banking(rows: int, seed: int, locale: str) -> dict[str, pd.DataFrame]:
    from sdp.nl import DatasetConfig, generate_dataset
    from sdp.nl.parser import domains
    dom = domains()["bank_customers"]
    cfg = DatasetConfig(domain="bank_customers", locale=locale, rows=rows, seed=seed, flag={"name": "is_fraud", "rate": 0.02}, history_months=6,
                        schema_columns=dom["columns"], rules=dom["rules"])
    return generate_dataset(cfg).tables


def _ecommerce(rows: int, seed: int, locale: str) -> dict[str, pd.DataFrame]:
    from sdp.datasets import make_shop_full
    return make_shop_full(rows, seed=seed)


def _healthcare(rows: int, seed: int, locale: str) -> dict[str, pd.DataFrame]:
    from sdp.contracts.healthcare import make_healthcare
    return make_healthcare(rows, seed=seed, locale=locale)


GENERATORS: dict[str, Callable[[int, int, str], dict[str, pd.DataFrame]]] = {"banking": _banking, "ecommerce": _ecommerce, "healthcare": _healthcare}


@lru_cache(maxsize=1)
def _load() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    dirs = [PACK_DIR] + ([Path(os.environ["SDP_PACK_DIR"])] if os.environ.get("SDP_PACK_DIR") else [])
    for d in dirs:
        for f in sorted(d.glob("*.json")):
            p = json.loads(f.read_text(encoding="utf-8"))
            if p.get("generator") not in GENERATORS:
                raise PackError(f"{f.name}: unknown generator {p.get('generator')!r}; available: {sorted(GENERATORS)}")
            out[p["name"]] = p
    return out


def list_packs() -> list[dict[str, Any]]:
    return [{"name": p["name"], "title": p["title"], "description": p["description"], "generator": p["generator"], "default_locale": p.get("default_locale", "en-US"),
             "default_rows": p.get("default_rows", 300), "tables": p["tables"], "expectations": len(p["expectations"])} for p in _load().values()]


def get_pack(name: str) -> dict[str, Any]:
    if name not in _load():
        raise PackError(f"unknown pack {name!r}; available: {sorted(_load())}")
    return _load()[name]


def generate_pack(name: str, rows: int | None = None, seed: int = 0, locale: str | None = None) -> dict[str, pd.DataFrame]:
    p = get_pack(name)
    rows = rows or p.get("default_rows", 300)
    if not 10 <= rows <= 200_000:
        raise PackError("rows must be between 10 and 200,000")
    return GENERATORS[p["generator"]](rows, seed, locale or p.get("default_locale", "en-US"))


def validate_pack(name: str, tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    p = get_pack(name)
    return validate_suite(tables, p["expectations"], suite_name=p["name"])


def export_suite(name: str) -> dict[str, dict[str, Any]]:
    return to_great_expectations(get_pack(name)["expectations"], name)
