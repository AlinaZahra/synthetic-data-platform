"""Business-rule constraints and edge-case coverage.

Rule dicts:
  {"type": "range", "column": "age", "min": 18, "max": 100}
  {"type": "not_null" | "unique", "column": "id"}
  {"type": "in_set", "column": "plan", "values": ["basic", "pro"]}
  {"type": "regex", "column": "code", "pattern": "^[A-Z]{3}$"}
  {"type": "expr", "expr": "end_date >= start_date"}      (pandas eval; must be True for every row)
Null cells do not violate range/in_set/regex (use not_null for that).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _violations(df: pd.DataFrame, rule: dict[str, Any]) -> pd.Series:
    t = rule["type"]
    if t == "expr":
        ok = df.eval(rule["expr"], engine="python")
        return ~ok.fillna(False).astype(bool)
    s = df[rule["column"]]
    if t == "range":
        bad = pd.Series(False, index=df.index)
        if rule.get("min") is not None:
            bad |= s < rule["min"]
        if rule.get("max") is not None:
            bad |= s > rule["max"]
        return bad.fillna(False)
    if t == "not_null":
        return s.isna()
    if t == "unique":
        return s.duplicated(keep="first") & s.notna()
    if t == "in_set":
        return (~s.isin(rule["values"])) & s.notna()
    if t == "regex":
        return (~s.astype(str).str.match(rule["pattern"])) & s.notna()
    raise ValueError(f"unknown rule type {t!r}")


def check_constraints(df: pd.DataFrame, rules: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(df)
    any_bad = pd.Series(False, index=df.index)
    results = []
    for rule in rules:
        bad = _violations(df, rule)
        any_bad |= bad
        results.append({"rule": rule, "violations": int(bad.sum()), "examples": np.flatnonzero(bad.to_numpy())[:5].tolist()})
    return {"n_rules": len(rules), "n_rows": n, "rows_violating": int(any_bad.sum()),
            "pass_pct": 100.0 * (1 - any_bad.mean()) if n else 100.0,
            "violations": [r for r in results if r["violations"]], "rules": results}


RARE_CATEGORY = 0.05


def edge_case_coverage(real: pd.DataFrame, synth: pd.DataFrame) -> dict[str, Any]:
    """Share of the real data's edge cases that the synthetic data also exhibits.

    numeric: lower tail (synth min <= real p1), upper tail (synth max >= real p99), zero present, negatives present,
             nulls present; categorical: every category with frequency <= 5% appears, nulls present.
    """
    total, covered, missing = 0, 0, []

    def case(col: str, name: str, real_has: bool, synth_has: bool) -> None:
        nonlocal total, covered
        if not real_has:
            return
        total += 1
        if synth_has:
            covered += 1
        else:
            missing.append({"column": col, "case": name})

    for c in real.columns:
        r, s = real[c], synth[c]
        case(c, "nulls", bool(r.isna().any()), bool(s.isna().any()))
        if pd.api.types.is_numeric_dtype(r) and not pd.api.types.is_bool_dtype(r):
            rv, sv = r.dropna().astype(float), s.dropna().astype(float)
            if rv.empty or sv.empty:
                continue
            case(c, "lower tail", True, sv.min() <= rv.quantile(0.01))
            case(c, "upper tail", True, sv.max() >= rv.quantile(0.99))
            case(c, "zero value", bool((rv == 0).any()), bool((sv == 0).any()))
            case(c, "negative values", bool((rv < 0).any()), bool((sv < 0).any()))
        elif not pd.api.types.is_datetime64_any_dtype(r):
            freq = r.dropna().astype(str).value_counts(normalize=True)
            present = set(s.dropna().astype(str).unique())
            for cat, f in freq.items():
                if f <= RARE_CATEGORY and r.nunique() <= 50:
                    case(c, f"rare category {cat!r}", True, cat in present)
    return {"score": 100.0 * covered / total if total else 100.0, "covered": covered, "total": total, "missing": missing[:50]}
