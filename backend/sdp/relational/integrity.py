"""Referential-integrity checker: orphan FKs, duplicate/null PKs, null-FK violations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from sdp.relational.schema import RelationshipGraph


@dataclass
class Violation:
    kind: str  # orphan_fk | duplicate_pk | null_pk | null_fk | partial_null_fk
    table: str
    columns: list[str]
    count: int
    examples: list = field(default_factory=list)
    fk: str | None = None


@dataclass
class IntegrityReport:
    violations: list[Violation]
    rows_checked: int

    @property
    def total(self) -> int:
        return sum(v.count for v in self.violations)

    @property
    def ok(self) -> bool:
        return self.total == 0

    def by_kind(self) -> dict[str, int]:
        out = {k: 0 for k in ("orphan_fk", "duplicate_pk", "null_pk", "null_fk", "partial_null_fk")}
        for v in self.violations:
            out[v.kind] += v.count
        return out

    def to_dict(self) -> dict:
        return {"ok": self.ok, "total_violations": self.total, "by_kind": self.by_kind(),
                "rows_checked": self.rows_checked, "violations": [asdict(v) for v in self.violations]}


def _examples(df: pd.DataFrame, mask, cols: list[str], k: int = 5) -> list:
    sub = df.loc[mask, cols].head(k)
    return sub.to_numpy().tolist() if len(cols) > 1 else sub[cols[0]].tolist()


def check_integrity(graph: RelationshipGraph, tables: dict[str, pd.DataFrame]) -> IntegrityReport:
    out: list[Violation] = []
    for t in graph.tables:
        df = tables[t.name]
        pk = t.primary_key
        if pk:
            null_pk = df[pk].isna().any(axis=1)
            if null_pk.any():
                out.append(Violation("null_pk", t.name, pk, int(null_pk.sum()), _examples(df, null_pk, pk)))
            dup = df.duplicated(pk, keep="first") & ~null_pk
            if dup.any():
                out.append(Violation("duplicate_pk", t.name, pk, int(dup.sum()), _examples(df, dup, pk)))

    for fk in graph.foreign_keys:
        child, parent = tables[fk.child_table], tables[fk.parent_table]
        cols = fk.child_columns
        nulls = child[cols].isna()
        any_null, all_null = nulls.any(axis=1), nulls.all(axis=1)
        if not fk.nullable and any_null.any():
            out.append(Violation("null_fk", fk.child_table, cols, int(any_null.sum()), _examples(child, any_null, cols), fk.key))
        elif fk.nullable and (any_null & ~all_null).any():
            m = any_null & ~all_null
            out.append(Violation("partial_null_fk", fk.child_table, cols, int(m.sum()), _examples(child, m, cols), fk.key))
        present = child[~any_null]
        if present.empty:
            continue
        if len(cols) == 1:
            ok = present[cols[0]].isin(parent[fk.parent_columns[0]].dropna())
        else:
            ok = pd.MultiIndex.from_frame(present[cols]).isin(pd.MultiIndex.from_frame(parent[fk.parent_columns].dropna()))
            ok = pd.Series(ok, index=present.index)
        if not ok.all():
            bad = present.index[~ok.to_numpy()]
            mask = child.index.isin(bad)
            out.append(Violation("orphan_fk", fk.child_table, cols, int(len(bad)), _examples(child, mask, cols), fk.key))
    return IntegrityReport(out, rows_checked=sum(len(d) for d in tables.values()))
