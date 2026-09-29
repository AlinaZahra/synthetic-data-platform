"""Controlled, logged injection of nulls and outliers after sampling."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from sdp.tabular.config import GenConfig, OutlierMethod

logger = logging.getLogger("sdp.tabular")

_MAX_LOGGED_ROWS = 50


@dataclass
class InjectionRecord:
    column: str
    action: str  # "null" | "outlier"
    requested_rate: float
    count: int
    method: str | None = None
    row_indices: list[int] = field(default_factory=list)  # truncated to first 50
    detail: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _outlier_values(
    clean: np.ndarray, method: OutlierMethod, k: int, rng: np.random.Generator
) -> np.ndarray:
    q1, q3 = np.percentile(clean, [25, 75])
    iqr = q3 - q1
    std = float(np.std(clean)) or 1.0
    sign = rng.choice([-1.0, 1.0], size=k)
    if method == "iqr":
        spread = iqr if iqr > 0 else std
        dist = rng.uniform(1.5, 3.0, size=k) * spread
        return np.where(sign > 0, q3 + dist, q1 - dist)
    if method == "zscore":
        return float(np.mean(clean)) + sign * rng.uniform(4.0, 6.0, size=k) * std
    raise AssertionError(method)  # "scale" handled by caller (needs the original values)


def inject(df: pd.DataFrame, cfg: GenConfig, rng: np.random.Generator) -> tuple[pd.DataFrame, list[InjectionRecord]]:
    """Return a modified copy of `df` plus a log. Deterministic given `rng`.

    Columns are processed in DataFrame order; within a column nulls are drawn first, and
    outliers are drawn only from rows that are still non-null.
    """
    unknown = (set(cfg.null_rate) | set(cfg.outlier_rate)) - set(df.columns)
    if unknown:
        raise ValueError(f"unknown columns in config: {sorted(unknown)}")

    out = df.copy()
    log: list[InjectionRecord] = []
    n = len(out)

    for col in df.columns:
        n_null = round(cfg.null_rate.get(col, 0.0) * n)
        n_out = round(cfg.outlier_rate.get(col, 0.0) * n)
        if not (n_null or n_out):
            continue

        null_rows = np.empty(0, dtype=int)
        if n_null:
            null_rows = np.sort(rng.choice(n, size=n_null, replace=False))

        if n_out:
            s = out[col]
            if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                raise ValueError(f"outliers require a numeric column; {col!r} is {s.dtype}")
            eligible = np.setdiff1d(np.arange(n), null_rows)
            n_out = min(n_out, len(eligible))
            rows = np.sort(rng.choice(eligible, size=n_out, replace=False))
            clean = s.to_numpy(dtype=float)
            if cfg.outlier_method == "scale":
                factor = rng.uniform(5.0, 10.0, size=n_out)
                base = clean[rows]
                new = np.where(base != 0, base * factor, (np.std(clean) or 1.0) * factor)
            else:
                new = _outlier_values(clean, cfg.outlier_method, n_out, rng)
            if pd.api.types.is_integer_dtype(s):
                new = np.rint(new)
            out[col] = _assign(s, rows, new)
            rec = InjectionRecord(
                column=col, action="outlier", requested_rate=cfg.outlier_rate[col], count=int(n_out),
                method=cfg.outlier_method, row_indices=rows[:_MAX_LOGGED_ROWS].tolist(),
                detail={"min_injected": float(new.min()), "max_injected": float(new.max())},
            )
            log.append(rec)
            logger.info("injected %d %s outliers into %r", rec.count, rec.method, col)

        if n_null:
            out[col] = _set_null(out[col], null_rows)
            rec = InjectionRecord(
                column=col, action="null", requested_rate=cfg.null_rate[col], count=int(n_null),
                row_indices=null_rows[:_MAX_LOGGED_ROWS].tolist(),
            )
            log.append(rec)
            logger.info("injected %d nulls into %r", rec.count, col)
    return out, log


def _assign(s: pd.Series, rows: np.ndarray, values: np.ndarray) -> pd.Series:
    arr = s.to_numpy(dtype=float).copy()
    arr[rows] = values
    if pd.api.types.is_integer_dtype(s):
        return pd.Series(np.rint(arr).astype(s.dtype), index=s.index)
    return pd.Series(arr, index=s.index).astype(s.dtype)


def _set_null(s: pd.Series, rows: np.ndarray) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        s = s.astype("boolean")
    elif pd.api.types.is_integer_dtype(s) and not pd.api.types.is_extension_array_dtype(s):
        s = s.astype("Int64")
    s = s.copy()
    s.iloc[rows] = pd.NA if pd.api.types.is_extension_array_dtype(s) else np.nan
    return s
