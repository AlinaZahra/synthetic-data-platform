"""Great-Expectations-style data contracts, implemented natively (no dependency).

A suite is a list of expectations `{expectation_type, table, kwargs, meta?}` using GE's names and kwargs (including `mostly`). Running a
suite yields a report shaped like GE's validation result: per-expectation success, element/unexpected counts, unexpected percent, a
partial list of offending values, and overall statistics. `to_great_expectations` exports one real GE suite per table so the same
contract can be run in a GE deployment.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable

import numpy as np
import pandas as pd


class ExpectationError(ValueError):
    pass


PARTIAL = 20
TYPE_CHECKS: dict[str, Callable[[pd.Series], bool]] = {
    "int": lambda s: pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s),
    "float": lambda s: pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s),
    "number": lambda s: pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s),
    "string": lambda s: bool(s.dropna().map(lambda v: isinstance(v, str)).all()) and not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s),
    "datetime": lambda s: pd.api.types.is_datetime64_any_dtype(s),
    "bool": lambda s: pd.api.types.is_bool_dtype(s),
}


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        raise ExpectationError(f"column {name!r} does not exist")
    return df[name]


def _result(mask_bad: pd.Series, values: pd.Series, total: int, mostly: float | None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    bad = int(mask_bad.sum())
    pct = 100.0 * bad / total if total else 0.0
    ok = (100.0 - pct) / 100.0 >= (mostly if mostly is not None else 1.0) - 1e-12 if total else True
    partial = [v.item() if hasattr(v, "item") else (v.isoformat() if hasattr(v, "isoformat") else v) for v in values[mask_bad].head(PARTIAL).tolist()]
    return {"success": bool(ok), "result": {"element_count": total, "unexpected_count": bad, "unexpected_percent": round(pct, 4), "partial_unexpected_list": partial, **(extra or {})}}


def _pos(v: Any) -> Any:
    return v.item() if hasattr(v, "item") else v


# ---------------------------------------------------------------- expectations
def e_row_count_between(df, tables, kw):
    n = len(df)
    lo, hi = kw.get("min_value"), kw.get("max_value")
    return {"success": (lo is None or n >= lo) and (hi is None or n <= hi), "result": {"observed_value": n}}


def e_column_exist(df, tables, kw):
    return {"success": kw["column"] in df.columns, "result": {}}


def e_columns_match_set(df, tables, kw):
    want, have = set(kw["column_set"]), set(map(str, df.columns))
    ok = have == want if kw.get("exact_match", True) else want <= have
    return {"success": ok, "result": {"observed_value": sorted(have), "details": {"missing": sorted(want - have), "unexpected": sorted(have - want) if kw.get("exact_match", True) else []}}}


def e_not_null(df, tables, kw):
    s = _col(df, kw["column"])
    return _result(s.isna(), s, len(s), kw.get("mostly"))


def e_unique(df, tables, kw):
    s = _col(df, kw["column"])
    nn = s.dropna()
    return _result(nn.duplicated(keep=False), nn, len(nn), kw.get("mostly"))


def e_between(df, tables, kw):
    s = _col(df, kw["column"])
    nn = s.dropna()
    lo, hi = kw.get("min_value"), kw.get("max_value")
    if pd.api.types.is_datetime64_any_dtype(nn):
        lo = pd.Timestamp(lo) if lo is not None else None
        hi = pd.Timestamp(hi) if hi is not None else None
    elif not pd.api.types.is_numeric_dtype(nn):
        raise ExpectationError(f"column {kw['column']!r} is not numeric or date")
    bad = pd.Series(False, index=nn.index)
    if lo is not None:
        bad |= (nn <= lo) if kw.get("strict_min") else (nn < lo)
    if hi is not None:
        bad |= (nn >= hi) if kw.get("strict_max") else (nn > hi)
    return _result(bad, nn, len(nn), kw.get("mostly"))


def e_in_set(df, tables, kw):
    s = _col(df, kw["column"]).dropna()
    allowed = set(kw["value_set"])
    return _result(~s.map(lambda v: _pos(v) in allowed), s, len(s), kw.get("mostly"))


def e_regex(df, tables, kw):
    s = _col(df, kw["column"]).dropna()
    try:
        pat = re.compile(kw["regex"])
    except re.error as e:
        raise ExpectationError(f"bad regex: {e}") from e
    return _result(~s.astype(str).map(lambda v: bool(pat.fullmatch(v))), s, len(s), kw.get("mostly"))


def e_type(df, tables, kw):
    s = _col(df, kw["column"])
    t = kw["type_"]
    if t not in TYPE_CHECKS:
        raise ExpectationError(f"unknown type {t!r}; available: {sorted(TYPE_CHECKS)}")
    return {"success": bool(TYPE_CHECKS[t](s)), "result": {"observed_value": str(s.dtype)}}


def e_pair_greater(df, tables, kw):
    a, b = _col(df, kw["column_A"]), _col(df, kw["column_B"])
    both = a.notna() & b.notna()
    bad = (a < b) if kw.get("or_equal", False) else (a <= b)
    return _result(bad[both], pd.Series([f"{x} vs {y}" for x, y in zip(a[both], b[both])], index=a[both].index), int(both.sum()), kw.get("mostly"))


def e_in_other_table(df, tables, kw):
    s = _col(df, kw["column"]).dropna()
    ref = kw["ref_table"]
    if ref not in tables:
        raise ExpectationError(f"table {ref!r} is not in the dataset")
    allowed = set(_col(tables[ref], kw["ref_column"]).dropna().tolist())
    return _result(~s.isin(allowed), s, len(s), kw.get("mostly"))


def e_mean_between(df, tables, kw):
    s = _col(df, kw["column"]).dropna()
    m = float(s.mean()) if len(s) else float("nan")
    lo, hi = kw.get("min_value"), kw.get("max_value")
    return {"success": bool(len(s) and (lo is None or m >= lo) and (hi is None or m <= hi)), "result": {"observed_value": m}}


def e_proportion_true(df, tables, kw):
    s = _col(df, kw["column"]).dropna()
    p = float((s == 1).mean()) if len(s) else float("nan")
    lo, hi = kw.get("min_value"), kw.get("max_value")
    return {"success": bool(len(s) and (lo is None or p >= lo) and (hi is None or p <= hi)), "result": {"observed_value": p}}


def e_child_sum(df, tables, kw):
    """Custom: parent[column] equals the sum of child[value] grouped by the key (within `tolerance`), e.g. orders.total vs order_items."""
    child = tables[kw["child_table"]]
    grouped = child.groupby(kw["child_key"])[kw["child_value"]].sum()
    if "child_multiplier" in kw:
        grouped = (child[kw["child_value"]] * child[kw["child_multiplier"]]).groupby(child[kw["child_key"]]).sum()
    parent = df.set_index(kw["key"])[kw["column"]]
    exp = grouped.reindex(parent.index).fillna(0.0)
    bad = (parent - exp).abs() > kw.get("tolerance", 0.01)
    return _result(bad, parent, len(parent), kw.get("mostly"))


REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "expect_table_row_count_to_be_between": e_row_count_between,
    "expect_column_to_exist": e_column_exist,
    "expect_table_columns_to_match_set": e_columns_match_set,
    "expect_column_values_to_not_be_null": e_not_null,
    "expect_column_values_to_be_unique": e_unique,
    "expect_column_values_to_be_between": e_between,
    "expect_column_values_to_be_in_set": e_in_set,
    "expect_column_values_to_match_regex": e_regex,
    "expect_column_values_to_be_of_type": e_type,
    "expect_column_pair_values_a_to_be_greater_than_b": e_pair_greater,
    "expect_column_values_to_be_in_other_table": e_in_other_table,     # custom (referential integrity)
    "expect_column_mean_to_be_between": e_mean_between,
    "expect_column_proportion_of_values_equal_one_to_be_between": e_proportion_true,   # custom (flag rates)
    "expect_column_values_to_equal_sum_of_child_values": e_child_sum,  # custom (cross-table reconciliation)
}
CUSTOM = {"expect_column_values_to_be_in_other_table", "expect_column_proportion_of_values_equal_one_to_be_between", "expect_column_values_to_equal_sum_of_child_values"}


def validate_suite(tables: dict[str, pd.DataFrame], suite: list[dict[str, Any]], suite_name: str = "contract") -> dict[str, Any]:
    """Run every expectation. A broken expectation (unknown table/column/type) is reported as an error result, never raised."""
    results = []
    t0 = time.perf_counter()
    for exp in suite:
        et, table, kw = exp.get("expectation_type"), exp.get("table"), dict(exp.get("kwargs", {}))
        entry: dict[str, Any] = {"expectation_config": {"expectation_type": et, "table": table, "kwargs": kw, "meta": exp.get("meta", {})}}
        try:
            if et not in REGISTRY:
                raise ExpectationError(f"unknown expectation {et!r}")
            if table not in tables:
                raise ExpectationError(f"table {table!r} not found in the dataset")
            entry.update(REGISTRY[et](tables[table], tables, kw))
            entry["exception_info"] = None
        except ExpectationError as e:
            entry.update({"success": False, "result": {}, "exception_info": {"raised_exception": True, "exception_message": str(e)}})
        results.append(entry)
    ok = sum(1 for r in results if r["success"])
    by_table: dict[str, dict[str, int]] = {}
    for r in results:
        t = r["expectation_config"]["table"] or "?"
        d = by_table.setdefault(t, {"evaluated": 0, "successful": 0})
        d["evaluated"] += 1
        d["successful"] += int(r["success"])
    return {"success": ok == len(results), "suite_name": suite_name,
            "statistics": {"evaluated_expectations": len(results), "successful_expectations": ok, "unsuccessful_expectations": len(results) - ok,
                           "success_percent": round(100.0 * ok / len(results), 2) if results else 100.0, "by_table": by_table},
            "results": results, "meta": {"engine": "sdp-contracts", "run_seconds": round(time.perf_counter() - t0, 4), "tables": {k: len(v) for k, v in tables.items()}}}


def to_great_expectations(suite: list[dict[str, Any]], suite_name: str = "contract") -> dict[str, dict[str, Any]]:
    """One GE expectation-suite JSON per table (custom expectations are kept with a `meta.custom` marker; GE ignores them until registered)."""
    out: dict[str, dict[str, Any]] = {}
    for e in suite:
        t = e["table"]
        s = out.setdefault(t, {"expectation_suite_name": f"{suite_name}.{t}", "expectations": [], "meta": {"great_expectations_version": "0.18", "generated_by": "sdp"}})
        s["expectations"].append({"expectation_type": e["expectation_type"], "kwargs": e.get("kwargs", {}), "meta": {**e.get("meta", {}), **({"custom": True} if e["expectation_type"] in CUSTOM else {})}})
    return out


def report_html(report: dict[str, Any]) -> str:
    from html import escape
    rows = []
    for r in report["results"]:
        c = r["expectation_config"]
        res = r.get("result", {})
        detail = (r.get("exception_info") or {}).get("exception_message") or (f"{res.get('unexpected_count', 0)} of {res.get('element_count', '')} unexpected" if "unexpected_count" in res else str(res.get("observed_value", "")))
        rows.append(f"<tr class='{'ok' if r['success'] else 'bad'}'><td>{'PASS' if r['success'] else 'FAIL'}</td><td>{escape(str(c['table']))}</td><td>{escape(c['expectation_type'])}</td>"
                    f"<td>{escape(str(c['kwargs']))}</td><td>{escape(str(detail))}</td></tr>")
    st = report["statistics"]
    return (f"<h1>{escape(report['suite_name'])}: {'PASSED' if report['success'] else 'FAILED'}</h1><p>{st['successful_expectations']}/{st['evaluated_expectations']} expectations met ({st['success_percent']}%)</p>"
            "<table><tr><th>result</th><th>table</th><th>expectation</th><th>kwargs</th><th>detail</th></tr>" + "".join(rows) + "</table>")
