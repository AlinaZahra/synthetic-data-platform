"""T6. Scenario packs: rare patterns and boundary values injected at a controlled rate.

Each pack takes a rate (fraction of rows), tags every affected row in a hidden `_edge_case` column
("pack:case;pack:case", empty for untouched rows) and contributes to an edge-case coverage report.

Coverage = (pack, case) pairs actually exercised / pairs that are *applicable* to this dataset's columns.
Rows are assigned to cases round-robin over a shuffled selection, so every applicable case appears once
as soon as the pack touches at least as many rows as it has cases. Deterministic for a given seed.

Packs: boundary_values, duplicates, typos, negative_balances, leap_year_dates, timezone_shifts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

TAG = "_edge_case"
BALANCE_NAME = re.compile(r"balance|amount|total|price|income|spend|salary|credit|debit|value|cost|fee", re.I)
KEYBOARD = {c: n for row in ("qwertyuiop", "asdfghjkl", "zxcvbnm") for i, c in enumerate(row)
            for n in [row[max(0, i - 1)] + row[min(len(row) - 1, i + 1)]]}


class EdgeCaseError(ValueError):
    pass


def visible(df: pd.DataFrame) -> pd.DataFrame:
    """The dataset without hidden bookkeeping columns (those starting with an underscore)."""
    return df[[c for c in df.columns if not str(c).startswith("_")]]


def _is_num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _is_str(s: pd.Series) -> bool:
    return not (_is_num(s) or pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_bool_dtype(s)
                or isinstance(s.dtype, pd.CategoricalDtype)) and s.dropna().map(lambda v: isinstance(v, str)).all()


def _cols(df: pd.DataFrame, protect: set[str], pred: Callable[[pd.Series], bool]) -> list[str]:
    return [c for c in df.columns if c not in protect and not str(c).startswith("_") and pred(df[c])]


# ------------------------------------------------------------------ packs
@dataclass
class Pack:
    name: str
    cases: Callable[[pd.DataFrame, set[str]], dict[str, list[str]]]                     # case -> applicable columns
    apply: Callable[[pd.DataFrame, str, str, int, np.random.Generator], str | None]     # (df, case, col, row, rng) -> detail


def _boundary_cases(df: pd.DataFrame, protect: set[str]) -> dict[str, list[str]]:
    ints = _cols(df, protect, lambda s: pd.api.types.is_integer_dtype(s))
    floats = _cols(df, protect, lambda s: pd.api.types.is_float_dtype(s))
    strs = _cols(df, protect, _is_str)
    dts = _cols(df, protect, pd.api.types.is_datetime64_any_dtype)
    out = {"zero": ints + floats, "max_value": ints + floats, "min_value": ints + floats, "negative_one": ints,
           "empty_string": strs, "very_long_string": strs, "single_space": strs, "epoch_date": dts, "max_date": dts}
    return {k: v for k, v in out.items() if v}


def _boundary_apply(df, case, col, i, rng):
    s = df[col]
    if case == "empty_string":
        df.at[i, col] = ""
    elif case == "single_space":
        df.at[i, col] = " "
    elif case == "very_long_string":
        df.at[i, col] = "x" * 1024
    elif case == "epoch_date":
        df.at[i, col] = pd.Timestamp("1970-01-01")
    elif case == "max_date":
        df.at[i, col] = pd.Timestamp("2262-04-11")
    elif pd.api.types.is_integer_dtype(s):
        info = np.iinfo(s.dtype.numpy_dtype if hasattr(s.dtype, "numpy_dtype") else s.dtype)
        df.at[i, col] = {"zero": 0, "max_value": info.max, "min_value": info.min, "negative_one": -1 if info.min < 0 else 0}[case]
    else:
        df.at[i, col] = {"zero": 0.0, "max_value": np.finfo(np.float64).max, "min_value": -np.finfo(np.float64).max}[case]
    return case


def _typos_cases(df, protect):
    strs = [c for c in _cols(df, protect, _is_str) if df[c].dropna().astype(str).str.len().ge(3).any()]
    return {k: strs for k in ("swap_adjacent", "drop_char", "double_char", "keyboard_neighbor", "case_flip", "trailing_space")} if strs else {}


def _typos_apply(df, case, col, i, rng):
    v = df.at[i, col]
    if not isinstance(v, str) or len(v) < 3:
        return None
    k = int(rng.integers(1, len(v) - 1))
    if case == "swap_adjacent":
        v2 = v[:k - 1] + v[k] + v[k - 1] + v[k + 1:]
    elif case == "drop_char":
        v2 = v[:k] + v[k + 1:]
    elif case == "double_char":
        v2 = v[:k] + v[k] + v[k:]
    elif case == "keyboard_neighbor":
        c = v[k]
        nb = KEYBOARD.get(c.lower())
        v2 = v[:k] + (nb[int(rng.integers(0, 2))] if nb else c) + v[k + 1:]
    elif case == "case_flip":
        v2 = v[:k] + v[k].swapcase() + v[k + 1:]
    else:
        v2 = v + " "
    if v2 == v:
        v2 = v + " "  # guarantee a visible change (e.g. swapping identical letters)
    df.at[i, col] = v2
    return case


def _neg_cases(df, protect):
    named = _cols(df, protect, lambda s: _is_num(s) and not pd.api.types.is_unsigned_integer_dtype(s))
    pref = [c for c in named if BALANCE_NAME.search(c)] or [c for c in named if pd.api.types.is_float_dtype(df[c])]
    return {k: pref for k in ("negated", "overdrawn_large", "just_below_zero")} if pref else {}


def _neg_apply(df, case, col, i, rng):
    v = df.at[i, col]
    if pd.isna(v):
        return None
    isint = pd.api.types.is_integer_dtype(df[col])
    big = float(abs(v))
    lim = float(np.iinfo(np.int64).max) if isint else float(np.finfo(np.float64).max)   # never overflow to inf when packs stack
    new = {"negated": -big, "overdrawn_large": -min(big * 10 + 1, lim), "just_below_zero": -1 if isint else -0.01}[case]
    df.at[i, col] = int(max(new, -lim)) if isint else float(new)
    return case


def _date_cols(df, protect):
    return _cols(df, protect, pd.api.types.is_datetime64_any_dtype)


def _leap_cases(df, protect):
    d = _date_cols(df, protect)
    return {k: d for k in ("feb29_leap_year", "feb28_before_non_leap_march", "mar1_after_non_leap_feb", "century_non_leap_1900", "year_end_rollover")} if d else {}


def _leap_apply(df, case, col, i, rng):
    v = df.at[i, col]
    if pd.isna(v):
        return None
    tod = pd.Timedelta(hours=v.hour, minutes=v.minute, seconds=v.second)
    base = {"feb29_leap_year": [f"{y}-02-29" for y in (2000, 2020, 2024)], "feb28_before_non_leap_march": ["2023-02-28", "2021-02-28"],
            "mar1_after_non_leap_feb": ["2023-03-01", "2021-03-01"], "century_non_leap_1900": ["1900-02-28", "1900-03-01"],
            "year_end_rollover": ["2023-12-31", "2024-12-31"]}[case]
    df.at[i, col] = pd.Timestamp(base[int(rng.integers(0, len(base)))]) + tod
    return case


def _tz_cases(df, protect):
    d = _date_cols(df, protect)
    return {k: d for k in ("shift_plus_hours", "shift_minus_hours", "dst_spring_forward_gap", "dst_fall_back_overlap", "midnight_boundary")} if d else {}


def _tz_apply(df, case, col, i, rng):
    v = df.at[i, col]
    if pd.isna(v):
        return None
    if case == "shift_plus_hours":
        df.at[i, col] = v + pd.Timedelta(hours=int(rng.integers(1, 15)))
    elif case == "shift_minus_hours":
        df.at[i, col] = v - pd.Timedelta(hours=int(rng.integers(1, 13)))
    elif case == "dst_spring_forward_gap":
        df.at[i, col] = pd.Timestamp("2024-03-10 02:30:00")   # a wall-clock time that does not exist in US Eastern
    elif case == "dst_fall_back_overlap":
        df.at[i, col] = pd.Timestamp("2024-11-03 01:30:00")   # a wall-clock time that occurs twice
    else:
        df.at[i, col] = v.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return case


PACKS: dict[str, Pack] = {
    "boundary_values": Pack("boundary_values", _boundary_cases, _boundary_apply),
    "typos": Pack("typos", _typos_cases, _typos_apply),
    "negative_balances": Pack("negative_balances", _neg_cases, _neg_apply),
    "leap_year_dates": Pack("leap_year_dates", _leap_cases, _leap_apply),
    "timezone_shifts": Pack("timezone_shifts", _tz_cases, _tz_apply),
}
from sdp.locale.langedge import LANG_PACKS as _LANG_PACKS  # noqa: E402  M8 language packs share the same tag/coverage machinery

PACKS.update({name: Pack(name, fns[0], fns[1]) for name, fns in _LANG_PACKS.items()})
DUPLICATE_CASES = ("exact_duplicate", "near_duplicate_whitespace", "near_duplicate_case")
ALL_PACKS = [*PACKS, "duplicates"]


@dataclass
class EdgeResult:
    data: pd.DataFrame
    report: dict[str, Any] = field(default_factory=dict)


def _append_tag(df: pd.DataFrame, i: int, tag: str) -> None:
    cur = df.at[i, TAG]
    df.at[i, TAG] = tag if not cur else f"{cur};{tag}"


def inject_edge_cases(df: pd.DataFrame, packs: dict[str, float], seed: int = 0, protect: list[str] | None = None) -> EdgeResult:
    """Return (copy of df with `_edge_case` column, report). `packs` maps pack name -> rate in [0, 1]."""
    unknown = set(packs) - set(ALL_PACKS)
    if unknown:
        raise EdgeCaseError(f"unknown edge-case packs {sorted(unknown)}; available: {ALL_PACKS}")
    for name, r in packs.items():
        if not 0.0 <= r <= 1.0:
            raise EdgeCaseError(f"rate for {name!r} must be within [0, 1]")
    protect_set = set(protect or [])
    rng = np.random.default_rng(seed)
    out = df.copy().reset_index(drop=True)
    out[TAG] = pd.Series([""] * len(out), dtype="object")
    n0 = len(out)
    report: dict[str, Any] = {"packs": {}, "rows": n0}
    possible_total = exercised_total = 0

    for name in [p for p in ALL_PACKS if p in packs]:
        rate = packs[name]
        k = min(n0, int(round(rate * n0)))
        rows = rng.permutation(n0)[:k]
        if name == "duplicates":
            possible = list(DUPLICATE_CASES) if _cols(out, protect_set, _is_str) or True else []
            exercised: dict[str, int] = {}
            added = []
            strs = _cols(out, protect_set, _is_str)
            for j, i in enumerate(rows):
                case = possible[j % len(possible)]
                if case != "exact_duplicate" and not strs:
                    case = "exact_duplicate"
                row = out.iloc[[int(i)]].copy()
                if case != "exact_duplicate":
                    col = strs[int(rng.integers(0, len(strs)))]
                    v = row.iloc[0][col]
                    if isinstance(v, str):
                        row[col] = (v + " ") if case == "near_duplicate_whitespace" else (v.swapcase() if v.swapcase() != v else v + " ")
                tag = f"duplicates:{case}"
                row[TAG] = tag
                _append_tag(out, int(i), tag)
                added.append(row)
                exercised[case] = exercised.get(case, 0) + 1
            if added:
                out = pd.concat([out, *added], ignore_index=True)
            info = {"rate": rate, "rows_affected": len(rows), "rows_added": len(added), "cases_possible": possible, "cases_exercised": sorted(exercised),
                    "case_counts": exercised, "columns": []}
        else:
            pack = PACKS[name]
            cases = pack.cases(out.iloc[:n0], protect_set)
            case_names = sorted(cases)
            exercised = {}
            touched: set[str] = set()
            for j, i in enumerate(rows):
                if not case_names:
                    break
                case = case_names[j % len(case_names)]
                col = cases[case][int(rng.integers(0, len(cases[case])))]
                if pack.apply(out, case, col, int(i), rng):
                    _append_tag(out, int(i), f"{name}:{case}")
                    exercised[case] = exercised.get(case, 0) + 1
                    touched.add(col)
            info = {"rate": rate, "rows_affected": int(sum(exercised.values())), "cases_possible": case_names, "cases_exercised": sorted(exercised),
                    "case_counts": exercised, "columns": sorted(touched)}
        info["coverage_pct"] = 100.0 * len(info["cases_exercised"]) / len(info["cases_possible"]) if info["cases_possible"] else None
        info["applicable"] = bool(info["cases_possible"])
        report["packs"][name] = info
        possible_total += len(info["cases_possible"])
        exercised_total += len(info["cases_exercised"])

    out[TAG] = out[TAG].replace("", None)
    report["coverage_pct"] = 100.0 * exercised_total / possible_total if possible_total else None
    report["cases_possible"], report["cases_exercised"] = possible_total, exercised_total
    report["rows_tagged"] = int(out[TAG].notna().sum())
    report["not_applicable"] = [n for n in packs if not report["packs"][n]["applicable"]]
    return EdgeResult(out, report)
