"""Shared helpers: deterministic seeding and dtype classification."""

from __future__ import annotations

import numpy as np
import pandas as pd


def resolve_seed(seed: int | None) -> int:
    """Return `seed`, or a fresh random 32-bit seed when None (so it can be logged)."""
    if seed is not None:
        return int(seed)
    return int(np.random.SeedSequence().generate_state(1)[0])


def spawn_rngs(seed: int, n: int) -> list[np.random.Generator]:
    """`n` statistically independent generators derived from one seed."""
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n)]


def is_identifier_like(s: pd.Series, min_unique: int = 50, ratio: float = 0.5) -> bool:
    """Text column whose values are (nearly) all distinct - names, codes, ids. Resampling such values would copy real records."""
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_bool_dtype(s):
        return False
    v = s.dropna()
    return v.nunique() >= min_unique and v.nunique() / max(len(v), 1) > ratio


def dtype_family(s: pd.Series) -> str:
    """Coarse dtype family: bool | int | float | datetime | str."""
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_integer_dtype(s):
        return "int"
    if pd.api.types.is_float_dtype(s):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    return "str"
