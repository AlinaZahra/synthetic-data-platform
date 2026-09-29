"""Infer PKs, FKs and cardinalities from DataFrames (uniqueness + value-overlap + name evidence)."""

from __future__ import annotations

import itertools
import re

import numpy as np
import pandas as pd

from sdp.common import dtype_family
from sdp.relational.schema import Column, ForeignKey, RelationshipGraph, Table, _singular

MIN_OVERLAP = 0.98          # fraction of child values that must exist in the parent key
SELF_REF_HINTS = ("parent", "manager", "supervisor", "boss", "reports_to", "referrer", "ancestor", "mentor")


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _idish(name: str) -> bool:
    n = _norm(name)
    return n == "id" or n.endswith("_id") or n.endswith("id") or n.endswith("_key") or n.endswith("_code")


_SEQ_HINTS = ("_no", "_num", "_number", "_seq", "_idx", "_index", "_pos", "_position", "line", "seq")


def _keyish(name: str) -> bool:
    n = _norm(name)
    return _idish(name) or any(n.endswith(h) or n == h.strip("_") for h in _SEQ_HINTS)


# ------------------------------------------------------------------ primary keys
def infer_primary_key(table: str, df: pd.DataFrame) -> list[str]:
    n = len(df)
    if n == 0:
        return []
    t, sing = _norm(table), _singular(_norm(table))
    scored: list[tuple[float, int, str]] = []
    for pos, c in enumerate(df.columns):
        fam = dtype_family(df[c])
        if fam not in ("int", "str") or df[c].isna().any() or df[c].nunique() != n:
            continue
        cn = _norm(c)
        score = 0.5 if pos == 0 else 0.0
        score += 3 if cn in ("id", f"{sing}_id", f"{t}_id", f"{sing}id") else 0
        score += 1 if _idish(c) else 0
        score += 1 if fam == "int" else 0
        scored.append((score, -pos, c))
    idish = [x for x in scored if _idish(x[2])]
    if idish:
        return [max(idish)[2]]
    # composite: a pair of id-like columns that is unique together
    cands = [c for c in df.columns if dtype_family(df[c]) in ("int", "str") and _keyish(c) and not df[c].isna().any()]
    for a, b in itertools.combinations(cands, 2):
        if not df.duplicated([a, b]).any():
            return [a, b]
    if scored:  # last resort: any unique int/str column (e.g. a natural key like `sku`)
        return [max(scored)[2]]
    return []


# ------------------------------------------------------------------ foreign keys
def _name_match(col: str, parent: str, pk_col: str) -> bool:
    c, p, sing = _norm(col), _norm(pk_col), _singular(_norm(parent))
    if c == p and p != "id":
        return True
    if c in (f"{sing}_id", f"{sing}id", f"{_norm(parent)}_id", f"{sing}_key", f"{sing}_code"):
        return True
    return sing in c and _idish(c)


def _key_family(s: pd.Series) -> str:
    """dtype family for key matching: nullable int columns arrive as float, so whole floats count as int."""
    fam = dtype_family(s)
    if fam == "float":
        v = s.dropna()
        if len(v) and (v % 1 == 0).all():
            return "int"
    return fam


def _overlap(child: pd.Series, parent_vals: pd.Series) -> float:
    v = child.dropna()
    if v.empty:
        return 0.0
    return float(v.isin(parent_vals).mean())


def _tuple_overlap(child: pd.DataFrame, parent: pd.DataFrame) -> float:
    c = child.dropna()
    if c.empty:
        return 0.0
    idx = pd.MultiIndex.from_frame(parent)
    return float(pd.MultiIndex.from_frame(c).isin(idx).mean())


def infer_graph(tables: dict[str, pd.DataFrame], min_overlap: float = MIN_OVERLAP) -> RelationshipGraph:
    pks = {name: infer_primary_key(name, df) for name, df in tables.items()}
    schema_tables = [
        Table(name=name, primary_key=pks[name], row_count=len(df),
              columns=[Column(name=c, dtype=dtype_family(df[c]), nullable=bool(df[c].isna().any()) and c not in pks[name])
                       for c in df.columns])
        for name, df in tables.items()
    ]
    fks: list[ForeignKey] = []
    for child, cdf in tables.items():
        chosen: dict[str, tuple[float, ForeignKey]] = {}  # child column -> best candidate
        for parent, pdf in tables.items():
            ppk = pks[parent]
            if len(ppk) == 1:
                cand = _single_col_candidates(child, cdf, pks[child], parent, pdf, ppk[0], min_overlap)
            elif len(ppk) > 1 and parent != child:
                cand = _composite_candidates(child, cdf, parent, pdf, ppk, min_overlap)
            else:
                cand = []
            for score, fk in cand:
                key = ",".join(fk.child_columns)
                if key not in chosen or score > chosen[key][0]:
                    chosen[key] = (score, fk)
        fks += [fk for _, fk in chosen.values()]
    graph = RelationshipGraph(tables=schema_tables, foreign_keys=fks)
    annotate_cardinalities(graph, tables)
    return graph


def _single_col_candidates(child: str, cdf: pd.DataFrame, child_pk: list[str], parent: str,
                           pdf: pd.DataFrame, pk: str, min_overlap: float) -> list[tuple[float, ForeignKey]]:
    out = []
    pvals = pdf[pk]
    pfam = _key_family(pvals)
    for col in cdf.columns:
        if _key_family(cdf[col]) != pfam or pfam not in ("int", "str"):
            continue
        if child_pk == [col]:
            continue  # a table's own single-column PK is not treated as an FK
        if child == parent:
            if col == pk or not any(h in _norm(col) for h in SELF_REF_HINTS + (_singular(_norm(child)),)):
                continue
        ov = _overlap(cdf[col], pvals)
        if ov < min_overlap:
            continue
        named = _name_match(col, parent, pk) or child == parent  # self-ref already passed the hint filter
        if not named and (pfam == "int" or cdf[col].nunique() < 5):
            continue  # small ints overlap everything; demand name evidence
        score = ov + (1.0 if named else 0.0)
        fk = ForeignKey(child_table=child, child_columns=[col], parent_table=parent, parent_columns=[pk],
                        nullable=bool(cdf[col].isna().any()), confidence=round(ov * (1.0 if named else 0.8), 3))
        out.append((score, fk))
    return out


def _composite_candidates(child: str, cdf: pd.DataFrame, parent: str, pdf: pd.DataFrame,
                          ppk: list[str], min_overlap: float) -> list[tuple[float, ForeignKey]]:
    if not all(c in cdf.columns for c in ppk):
        return []
    if _tuple_overlap(cdf[ppk], pdf[ppk]) < min_overlap:
        return []
    ov = _tuple_overlap(cdf[ppk], pdf[ppk])
    fk = ForeignKey(child_table=child, child_columns=list(ppk), parent_table=parent, parent_columns=list(ppk),
                    nullable=bool(cdf[ppk].isna().any().any()), confidence=round(ov, 3))
    return [(ov + 1.5, fk)]


# ------------------------------------------------------------------ cardinality
def children_per_parent(child_df: pd.DataFrame, fk: ForeignKey, parent_df: pd.DataFrame) -> np.ndarray:
    """Child-row count for every parent row (zeros included), aligned with parent_df order."""
    if len(fk.child_columns) == 1:
        counts = child_df[fk.child_columns[0]].value_counts()
        return counts.reindex(parent_df[fk.parent_columns[0]]).fillna(0).to_numpy(dtype=int)
    grp = child_df.dropna(subset=fk.child_columns).groupby(fk.child_columns).size()
    keys = pd.MultiIndex.from_frame(parent_df[fk.parent_columns])
    return grp.reindex(keys).fillna(0).to_numpy(dtype=int)


def annotate_cardinalities(graph: RelationshipGraph, tables: dict[str, pd.DataFrame]) -> None:
    """Fill fk.cardinality and fk.stats from data (mutates graph)."""
    for fk in graph.foreign_keys:
        cdf, pdf = tables[fk.child_table], tables[fk.parent_table]
        counts = children_per_parent(cdf, fk, pdf)
        vals = cdf[fk.child_columns].dropna()
        orphan = 1.0 - (_tuple_overlap(cdf[fk.child_columns], pdf[fk.parent_columns]) if len(vals) else 1.0)
        fk.cardinality = "1:1" if counts.max(initial=0) <= 1 else "1:N"
        fk.stats = {
            "mean_children": float(counts.mean()) if len(counts) else 0.0,
            "max_children": float(counts.max(initial=0)),
            "zero_child_fraction": float((counts == 0).mean()) if len(counts) else 0.0,
            "null_rate": float(cdf[fk.child_columns].isna().any(axis=1).mean()),
            "orphan_rate": float(orphan),
        }
