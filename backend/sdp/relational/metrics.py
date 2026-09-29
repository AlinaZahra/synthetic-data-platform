"""R5. Relation-preservation metrics: how well do the *relationships* survive, not just each table.

Score 0-100 = weighted mean of (weights renormalised over components that apply):
  cardinality   0.25  children-per-parent histogram match for every FK (1 - total-variation distance)
  join_size     0.15  rows reachable through FK chains (child -> parent -> grandparent), per ancestor row, real vs synthetic
  cross_table   0.25  strongest cross-table associations in the real data (e.g. customer segment vs order value):
                      effect size (eta^2) and per-group profile for categorical->numeric, Spearman for numeric->numeric
  fk_coverage   0.15  share of parents with >=1 child, and share of non-null FK values that resolve to a parent
  queries       0.20  the same aggregate JOIN queries (SQLite, auto-generated or supplied) run on both databases; additive
                      columns are compared as shares of the total, level columns (avg/min/max/ratio) by relative error
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from sdp.relational.cardinality import compare_distributions
from sdp.relational.inference import children_per_parent
from sdp.relational.schema import RelationshipGraph
from sdp.rules.engine import Evaluator

WEIGHTS = {"cardinality": 0.25, "join_size": 0.15, "cross_table": 0.25, "fk_coverage": 0.15, "queries": 0.20}
MIN_ANCESTOR_ROWS = 60  # effect sizes against tiny dimension tables are sampling noise
LEVEL_PREFIXES = ("avg", "mean", "min", "max", "ratio", "pct", "rate", "median")


def _num_cols(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
            and df[c].nunique() > 12]


def _cat_cols(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    return [c for c in df.columns if c not in exclude and 2 <= df[c].nunique() <= 12 and not pd.api.types.is_datetime64_any_dtype(df[c])]


def _keys(graph: RelationshipGraph, t: str) -> set[str]:
    tb = graph.table(t)
    return set(tb.primary_key) | {c for f in graph.fks_of(t) for c in f.child_columns}


def _ancestors(graph: RelationshipGraph, ev: Evaluator, table: str, depth: int = 2) -> list[str]:
    seen, order, frontier = {table}, [], [table]
    for _ in range(depth):
        nxt = []
        for t in frontier:
            for fk in graph.fks_of(t):
                if not fk.is_self_reference and fk.parent_table not in seen:
                    seen.add(fk.parent_table)
                    order.append(fk.parent_table)
                    nxt.append(fk.parent_table)
        frontier = nxt
    return order


def _attr(ev: Evaluator, tables: dict[str, pd.DataFrame], base: str, anc: str, col: str) -> pd.Series:
    if anc == base:
        return tables[base][col].reset_index(drop=True)
    pos = ev.positions(base, anc)
    return tables[anc][col].reset_index(drop=True).take(np.maximum(pos, 0)).reset_index(drop=True).where(pos >= 0)


def _rel_err(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-9)


# ---------------------------------------------------------------- components
def cardinality_component(real, synth, graph) -> dict[str, Any]:
    per, scores = {}, []
    for fk in graph.foreign_keys:
        r = children_per_parent(real[fk.child_table], fk, real[fk.parent_table])
        s = children_per_parent(synth[fk.child_table], fk, synth[fk.parent_table])
        m = compare_distributions(r, s)
        per[fk.key] = {k: m[k] for k in ("match_score", "mean_real", "mean_synth", "ks_statistic") if k in m}
        scores.append(m["match_score"])
    return {"score": 100 * float(np.mean(scores)) if scores else None,
            "summary": f"children-per-parent histograms match {100 * np.mean(scores):.0f}% on average" if scores else "no foreign keys",
            "details": per}


def join_size_component(real, synth, graph) -> dict[str, Any]:
    er, es = Evaluator(real, graph), Evaluator(synth, graph)
    per, scores = {}, []
    for t in (x.name for x in graph.tables):
        for anc in _ancestors(graph, er, t, depth=3):
            r = float((er.positions(t, anc) >= 0).sum()) / max(1, len(real[anc]))
            s = float((es.positions(t, anc) >= 0).sum()) / max(1, len(synth[anc]))
            sc = 100 * (1 - _rel_err(r, s))
            per[f"{anc} <- {t}"] = {"real_rows_per_parent": r, "synthetic_rows_per_parent": s, "score": sc}
            scores.append(sc)
    return {"score": float(np.mean(scores)) if scores else None,
            "summary": f"{len(scores)} join paths compared (rows per parent row)", "details": per}


def cross_table_component(real, synth, graph, max_pairs: int = 8) -> dict[str, Any]:
    er, es = Evaluator(real, graph), Evaluator(synth, graph)
    cands: list[tuple[float, dict[str, Any]]] = []
    for t in (x.name for x in graph.tables):
        keys = _keys(graph, t)
        for anc in _ancestors(graph, er, t):
            for col in (_cat_cols(real[anc], _keys(graph, anc)) if len(real[anc]) >= MIN_ANCESTOR_ROWS else []):
                cr = _attr(er, real, t, anc, col)
                for v in _num_cols(real[t], keys):
                    y = real[t][v].reset_index(drop=True).astype(float)
                    eta = _eta2(cr, y)
                    if eta is not None and eta >= 0.02:
                        cands.append((eta, {"kind": "cat->num", "child": t, "child_col": v, "anc": anc, "anc_col": col}))
            for col in _num_cols(real[anc], _keys(graph, anc)):
                cr = _attr(er, real, t, anc, col).astype(float)
                for v in _num_cols(real[t], keys):
                    rho = cr.corr(real[t][v].reset_index(drop=True).astype(float), method="spearman")
                    if pd.notna(rho) and abs(rho) >= 0.2:
                        cands.append((abs(rho), {"kind": "num->num", "child": t, "child_col": v, "anc": anc, "anc_col": col}))
    cands.sort(key=lambda x: -x[0])
    per, scores = [], []
    for strength, p in cands[:max_pairs]:
        t, v, anc, col = p["child"], p["child_col"], p["anc"], p["anc_col"]
        if p["kind"] == "cat->num":
            yr, ys = real[t][v].reset_index(drop=True).astype(float), synth[t][v].reset_index(drop=True).astype(float)
            cr, cs = _attr(er, real, t, anc, col), _attr(es, synth, t, anc, col)
            eta_r, eta_s = _eta2(cr, yr), _eta2(cs, ys)
            gr, gs = yr.groupby(cr).mean(), ys.groupby(cs).mean()
            common = gr.index.intersection(gs.index)
            sd = yr.std() or 1.0
            prof_err = float(np.mean(np.abs(gr.reindex(gr.index).fillna(0) - gs.reindex(gr.index).fillna(gr.mean())) / sd)) if len(common) else 1.0
            sc = 0.6 * 100 * max(0.0, 1 - prof_err / 0.5) + 0.4 * 100 * max(0.0, 1 - abs(eta_r - (eta_s or 0.0)) / 0.15)
            per.append({**p, "eta2_real": eta_r, "eta2_synthetic": eta_s, "profile_error_sd": prof_err, "score": sc,
                        "group_means_real": {str(k): float(x) for k, x in gr.items()}, "group_means_synthetic": {str(k): float(x) for k, x in gs.items()}})
        else:
            rr = _attr(er, real, t, anc, col).astype(float).corr(real[t][v].reset_index(drop=True).astype(float), method="spearman")
            rs = _attr(es, synth, t, anc, col).astype(float).corr(synth[t][v].reset_index(drop=True).astype(float), method="spearman")
            sc = 100 * max(0.0, 1 - abs(rr - (0 if pd.isna(rs) else rs)) / 0.3)
            per.append({**p, "spearman_real": float(rr), "spearman_synthetic": None if pd.isna(rs) else float(rs), "score": sc})
        scores.append(sc)
    return {"score": float(np.mean(scores)) if scores else None,
            "summary": (f"{len(scores)} strongest real cross-table relationships compared" if scores else "no strong cross-table relationships found"),
            "details": per}


def _eta2(cat: pd.Series, y: pd.Series) -> float | None:
    d = pd.DataFrame({"c": cat.to_numpy(), "y": y.to_numpy()}).dropna()
    if len(d) < 10 or d["y"].var() == 0:
        return None
    g = d.groupby("c")["y"]
    return float((g.size() * (g.mean() - d["y"].mean()) ** 2).sum() / (d["y"].var(ddof=0) * len(d)))


def fk_coverage_component(real, synth, graph) -> dict[str, Any]:
    per, scores = {}, []
    for fk in graph.foreign_keys:
        cov = {}
        for name, T in (("real", real), ("synthetic", synth)):
            counts = children_per_parent(T[fk.child_table], fk, T[fk.parent_table])
            ev = Evaluator(T, graph)
            child = T[fk.child_table][fk.child_columns]
            nonnull = ~child.isna().any(axis=1).to_numpy()
            pos = _hop_positions(ev, fk, T)
            valid = float((pos[nonnull] >= 0).mean()) if nonnull.any() else 1.0
            cov[name] = {"parent_coverage": float((counts > 0).mean()) if len(counts) else 0.0, "resolves": valid}
        sc = 100 * (1 - abs(cov["real"]["parent_coverage"] - cov["synthetic"]["parent_coverage"])) * cov["synthetic"]["resolves"]
        per[fk.key] = {**cov, "score": sc}
        scores.append(sc)
    return {"score": float(np.mean(scores)) if scores else None,
            "summary": "parents with children and resolvable foreign keys, real vs synthetic", "details": per}


def _hop_positions(ev: Evaluator, fk, T) -> np.ndarray:
    return Evaluator._hop(T[fk.child_table], fk, T[fk.parent_table])


# --------------------------------------------------------- SQL equivalence
def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _load(tables: dict[str, pd.DataFrame]) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    for name, df in tables.items():
        d = df.copy()
        for c in d.columns:
            if pd.api.types.is_datetime64_any_dtype(d[c]):
                d[c] = d[c].dt.strftime("%Y-%m-%d").where(d[c].notna(), None)
            elif str(d[c].dtype) in ("boolean", "bool"):
                d[c] = d[c].astype("Int64")
        d.to_sql(name, con, index=False)
    return con


def auto_queries(real: dict[str, pd.DataFrame], graph: RelationshipGraph, max_queries: int = 6) -> list[str]:
    """Aggregate JOIN queries for the strongest cat->num relationships, one and two hops."""
    er = Evaluator(real, graph)
    found: list[tuple[float, str]] = []
    for t in (x.name for x in graph.tables):
        keys = _keys(graph, t)
        for fk in graph.fks_of(t):
            if fk.is_self_reference:
                continue
            for hop2 in [None] + [g for g in graph.fks_of(fk.parent_table) if not g.is_self_reference]:
                anc = fk.parent_table if hop2 is None else hop2.parent_table
                for col in (_cat_cols(real[anc], _keys(graph, anc)) if len(real[anc]) >= MIN_ANCESTOR_ROWS else []):
                    for v in _num_cols(real[t], keys):
                        eta = _eta2(_attr(er, real, t, anc, col), real[t][v].reset_index(drop=True).astype(float))
                        if eta is None:
                            continue
                        on1 = " AND ".join(f"c.{_q(a)} = p.{_q(b)}" for a, b in zip(fk.child_columns, fk.parent_columns))
                        if hop2 is None:
                            sql = (f"SELECT p.{_q(col)} AS grp, COUNT(*) AS n_rows, SUM(c.{_q(v)}) AS sum_{v}, AVG(c.{_q(v)}) AS avg_{v} "
                                   f"FROM {_q(t)} c JOIN {_q(anc)} p ON {on1} GROUP BY p.{_q(col)} ORDER BY grp")
                        else:
                            on2 = " AND ".join(f"p.{_q(a)} = g.{_q(b)}" for a, b in zip(hop2.child_columns, hop2.parent_columns))
                            sql = (f"SELECT g.{_q(col)} AS grp, COUNT(*) AS n_rows, SUM(c.{_q(v)}) AS sum_{v}, AVG(c.{_q(v)}) AS avg_{v} "
                                   f"FROM {_q(t)} c JOIN {_q(fk.parent_table)} p ON {on1} JOIN {_q(anc)} g ON {on2} GROUP BY g.{_q(col)} ORDER BY grp")
                        found.append((eta, sql))
    found.sort(key=lambda x: -x[0])
    out: list[str] = []
    for _, sql in found:
        if sql not in out:
            out.append(sql)
        if len(out) >= max_queries:
            break
    return out


def _compare_result(r: pd.DataFrame, s: pd.DataFrame) -> dict[str, Any]:
    key = r.columns[0]
    r, s = r.set_index(key), s.set_index(key)
    groups = r.index.union(s.index)
    per_col, scores = {}, []
    for col in r.columns:
        a = pd.to_numeric(r[col], errors="coerce").reindex(groups).fillna(0.0)
        b = pd.to_numeric(s[col], errors="coerce").reindex(groups).fillna(0.0) if col in s.columns else pd.Series(0.0, index=groups)
        if col.lower().startswith(LEVEL_PREFIXES):
            present = a.index.isin(r.index) & a.index.isin(s.index)
            err = np.minimum(1.0, np.abs(a - b) / np.maximum(np.abs(a), 1e-9))[present]
            sc = 100 * (1 - float(err.mean())) if len(err) else 0.0
            kind = "level"
        else:
            pa, pb = a / max(a.sum(), 1e-9), b / max(b.sum(), 1e-9)
            sc = 100 * (1 - 0.5 * float(np.abs(pa - pb).sum()))
            kind = "share"
        per_col[col] = {"kind": kind, "score": sc}
        scores.append(sc)
    return {"score": float(np.mean(scores)) if scores else 0.0, "columns": per_col, "groups": [str(g) for g in groups]}


def queries_component(real, synth, graph, queries: list[str] | None = None) -> dict[str, Any]:
    sqls = queries or auto_queries(real, graph)
    if not sqls:
        return {"score": None, "summary": "no aggregate queries could be generated", "details": []}
    cr, cs = _load(real), _load(synth)
    out, scores = [], []
    try:
        for sql in sqls:
            if not re.match(r"(?is)^\s*select\b", sql) or re.search(r"(?i)\b(insert|update|delete|drop|alter|attach|pragma)\b", sql):
                out.append({"sql": sql, "error": "only SELECT statements are allowed", "score": 0.0})
                scores.append(0.0)
                continue
            try:
                r, s = pd.read_sql_query(sql, cr), pd.read_sql_query(sql, cs)
            except Exception as e:  # noqa: BLE001 - report SQL errors per query rather than failing the whole score
                out.append({"sql": sql, "error": str(e), "score": 0.0})
                scores.append(0.0)
                continue
            cmp = _compare_result(r, s)
            out.append({"sql": sql, **cmp, "real_rows": r.head(12).to_dict("records"), "synthetic_rows": s.head(12).to_dict("records")})
            scores.append(cmp["score"])
    finally:
        cr.close()
        cs.close()
    return {"score": float(np.mean(scores)), "summary": f"{len(sqls)} JOIN aggregate queries run on both databases", "details": out}


# ------------------------------------------------------------------- total
def relation_metrics(real: dict[str, pd.DataFrame], synth: dict[str, pd.DataFrame], graph: RelationshipGraph,
                     queries: list[str] | None = None) -> dict[str, Any]:
    comps = {
        "cardinality": cardinality_component(real, synth, graph),
        "join_size": join_size_component(real, synth, graph),
        "cross_table": cross_table_component(real, synth, graph),
        "fk_coverage": fk_coverage_component(real, synth, graph),
        "queries": queries_component(real, synth, graph, queries),
    }
    live = {k: v for k, v in comps.items() if v["score"] is not None}
    total_w = sum(WEIGHTS[k] for k in live)
    for k, v in live.items():
        v["weight"] = WEIGHTS[k] / total_w
    score = sum(v["score"] * v["weight"] for v in live.values()) if live else 0.0
    for k, v in comps.items():
        v.setdefault("weight", 0.0)
    return {"score": float(score), "components": comps}
