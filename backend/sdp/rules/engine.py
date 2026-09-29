"""Compile rules against a relationship graph and evaluate them vectorised (three-valued logic, no eval()).

Cross-table rules are evaluated on their *owner* table: the child-most table mentioned. Columns of ancestor tables
are gathered along foreign keys; aggregates (SUM/COUNT/... of a child table) are grouped by that child's FK to the owner.
NULL semantics follow SQL: a comparison involving NULL is *unknown* and does not count as a violation.
"""

from __future__ import annotations

import operator
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sdp.relational.schema import ForeignKey, RelationshipGraph
from sdp.rules import dsl
from sdp.rules.dsl import AGG_FUNCS, Between, Bin, Func, In, IsNull, Neg, Not, Num, Ref, RuleSyntaxError, Str

EQ_TOL = 0.005  # float equality tolerance (half a cent)
CMP = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "=": operator.eq, "!=": operator.ne}
FLIP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "=": "=", "!=": "!="}


@dataclass(frozen=True)
class Agg:
    fn: str
    table: str
    expr: Any | None


@dataclass
class Compiled:
    text: str                 # what the user typed
    dsl: str                  # what it compiled to (shown for confirmation)
    ast: Any
    owner: str
    tables: set[str]
    kind: str                 # derive | bound | conditional | unique | check
    target: tuple[str, str] | None = None
    name: str = ""
    translated: bool = False
    reads: set[tuple[str, str]] = field(default_factory=set)

    def describe(self) -> dict[str, Any]:
        return {"input": self.text, "dsl": self.dsl, "translated_from_plain_language": self.translated,
                "owner": self.owner, "kind": self.kind, "tables": sorted(self.tables)}


# ----------------------------------------------------------------- compile
def single_table_graph(name: str, df: pd.DataFrame) -> RelationshipGraph:
    from sdp.common import dtype_family
    from sdp.relational.schema import Column, Table
    return RelationshipGraph(tables=[Table(name=name, columns=[Column(name=c, dtype=dtype_family(df[c])) for c in df.columns])])


def _all_columns(graph: RelationshipGraph, default: str | None) -> list[str]:
    out = []
    for t in graph.tables:
        for c in t.columns:
            out.append(c.name if t.name == default else f"{t.name}.{c.name}")
    return out


def compile_rule(text: str, graph: RelationshipGraph, default_table: str | None = None, name: str = "") -> Compiled:
    if default_table is None and len(graph.tables) == 1:
        default_table = graph.tables[0].name
    dsl_text, translated = dsl.to_dsl(text, _all_columns(graph, default_table))
    ast = dsl.parse(dsl_text)
    tnames = {t.name for t in graph.tables}
    cols = {t.name: {c.name for c in t.columns} for t in graph.tables}

    def resolve_ref(r: Ref, agg_table: str | None = None) -> Ref:
        if r.table is not None:
            if r.table not in tnames:
                raise RuleSyntaxError(f"unknown table {r.table!r}")
            if r.column not in cols[r.table]:
                raise RuleSyntaxError(f"unknown column {r.table}.{r.column}")
            return r
        ctx = agg_table or default_table
        if ctx and r.column in cols.get(ctx, ()):
            return Ref(ctx, r.column)
        hits = [t for t in tnames if r.column in cols[t]]
        if len(hits) == 1:
            return Ref(hits[0], r.column)
        raise RuleSyntaxError(f"column {r.column!r} is " + ("ambiguous; qualify it as table.column" if hits else "unknown"))

    def walk(n: Any, agg_table: str | None = None) -> Any:
        if isinstance(n, Ref):
            return resolve_ref(n, agg_table)
        if isinstance(n, (Num, Str)):
            return n
        if isinstance(n, Bin):
            return Bin(n.op, walk(n.left, agg_table), walk(n.right, agg_table))
        if isinstance(n, Not):
            return Not(walk(n.x, agg_table))
        if isinstance(n, Neg):
            return Neg(walk(n.x, agg_table))
        if isinstance(n, IsNull):
            return IsNull(walk(n.x, agg_table), n.negated)
        if isinstance(n, In):
            return In(walk(n.x, agg_table), tuple(walk(v, agg_table) for v in n.values), n.negated)
        if isinstance(n, Between):
            return Between(walk(n.x, agg_table), walk(n.lo, agg_table), walk(n.hi, agg_table))
        if isinstance(n, Func) and n.name in AGG_FUNCS:
            arg = n.args[0] if n.args else None
            if n.name == "COUNT" and isinstance(arg, Ref) and arg.table is None and arg.column in tnames:
                return Agg("COUNT", arg.column, None)
            if arg is None:
                raise RuleSyntaxError(f"{n.name}() needs an argument, e.g. {n.name}(order_items.price)")
            qualified = [r.table for r in dsl.refs_in(arg) if r.table]
            child = qualified[0] if qualified else None
            if child is None:
                raise RuleSyntaxError(f"qualify columns inside {n.name}(...) as table.column")
            return Agg(n.name, child, walk(arg, child))
        if isinstance(n, Func):
            return Func(n.name, tuple(walk(a, agg_table) for a in n.args))
        raise RuleSyntaxError(f"unsupported expression {n!r}")

    resolved = walk(ast)

    plain_refs: list[Ref] = []
    aggs: list[Agg] = []

    def collect(n: Any) -> None:
        if isinstance(n, Agg):
            aggs.append(n)
        elif isinstance(n, Ref):
            plain_refs.append(n)
        elif isinstance(n, Bin):
            collect(n.left)
            collect(n.right)
        elif isinstance(n, (Not, Neg, IsNull)):
            collect(n.x)
        elif isinstance(n, In):
            collect(n.x)
            [collect(v) for v in n.values]
        elif isinstance(n, Between):
            collect(n.x)
            collect(n.lo)
            collect(n.hi)
        elif isinstance(n, Func):
            [collect(a) for a in n.args]
    collect(resolved)

    ev = Evaluator({}, graph)
    ref_tables = {r.table for r in plain_refs if r.table}
    if ref_tables:
        candidates = [t for t in ref_tables if all(o == t or ev.fk_path(t, o) is not None for o in ref_tables)]
        if not candidates:
            raise RuleSyntaxError(f"tables {sorted(ref_tables)} are not connected by foreign keys in a single chain")
        owner = sorted(candidates)[0]
    elif aggs:
        parents = {fk.parent_table for a in aggs for fk in graph.fks_of(a.table)}
        if not parents:
            raise RuleSyntaxError("aggregated table has no foreign key to an owner table")
        owner = sorted(parents)[0]
    else:
        raise RuleSyntaxError("rule references no columns")
    for a in aggs:
        if ev.fk_path(a.table, owner) is None or a.table == owner:
            raise RuleSyntaxError(f"{a.fn}({a.table}...) needs {a.table} to reference {owner} through a foreign key")

    kind, target = _classify(resolved, owner, graph)
    reads = reads_of(resolved)
    return Compiled(text=text, dsl=dsl_text, ast=resolved, owner=owner, tables=ref_tables | {a.table for a in aggs} | {owner},
                    kind=kind, target=target, name=name or dsl_text, translated=translated, reads=reads)


def reads_of(n: Any) -> set[tuple[str, str]]:
    """(table, column) pairs an expression reads; COUNT(table) reads (table, '*')."""
    if isinstance(n, Ref):
        return {(n.table or "", n.column)}
    if isinstance(n, Agg):
        return reads_of(n.expr) if n.expr is not None else {(n.table, "*")}
    if isinstance(n, Bin):
        return reads_of(n.left) | reads_of(n.right)
    if isinstance(n, (Not, Neg, IsNull)):
        return reads_of(n.x)
    if isinstance(n, In):
        return reads_of(n.x).union(*[reads_of(v) for v in n.values])
    if isinstance(n, Between):
        return reads_of(n.x) | reads_of(n.lo) | reads_of(n.hi)
    if isinstance(n, Func):
        return set().union(*[reads_of(a) for a in n.args]) if n.args else set()
    return set()


def _bare(n: Any) -> bool:
    return isinstance(n, Ref)


def _classify(ast: Any, owner: str, graph: RelationshipGraph) -> tuple[str, tuple[str, str] | None]:
    keys = {(t.name, c) for t in graph.tables for c in t.primary_key} | {
        (f.child_table, c) for f in graph.foreign_keys for c in f.child_columns}
    if isinstance(ast, Bin) and ast.op == "implies":
        return "conditional", None
    if isinstance(ast, Func) and ast.name == "UNIQUE":
        return "unique", None
    if isinstance(ast, Bin) and ast.op in CMP:
        l, r = ast.left, ast.right
        if ast.op == "=":
            for lhs, rhs in ((l, r), (r, l)):
                if _bare(lhs) and lhs.table == owner and (lhs.table, lhs.column) not in keys                         and (lhs.table, lhs.column) not in reads_of(rhs):
                    return ("bound" if isinstance(rhs, (Num, Str)) else "derive"), (lhs.table, lhs.column)
        for side in (l, r):
            if _bare(side) and side.table == owner:
                return "bound", (side.table, side.column)
        for side in (l, r):
            if _bare(side):
                return "bound", (side.table, side.column)
        return "check", None
    if isinstance(ast, (IsNull, In, Between)) and _bare(ast.x):
        return "bound", (ast.x.table, ast.x.column)
    return "check", None


# ---------------------------------------------------------------- evaluate
@dataclass
class Check:
    mask: np.ndarray          # True where the owner row violates the rule
    n: int
    unknown: int = 0

    @property
    def violations(self) -> int:
        return int(self.mask.sum())


class Evaluator:
    def __init__(self, tables: dict[str, pd.DataFrame], graph: RelationshipGraph) -> None:
        self.tables, self.graph = tables, graph
        self._pos: dict[tuple[str, str], np.ndarray] = {}

    # ---- foreign-key paths
    def fk_path(self, src: str, dst: str) -> list[ForeignKey] | None:
        if src == dst:
            return []
        q, seen = deque([(src, [])]), {src}
        while q:
            t, path = q.popleft()
            for fk in self.graph.fks_of(t):
                if fk.is_self_reference or fk.parent_table in seen:
                    continue
                if fk.parent_table == dst:
                    return path + [fk]
                seen.add(fk.parent_table)
                q.append((fk.parent_table, path + [fk]))
        return None

    def positions(self, src: str, dst: str) -> np.ndarray:
        """Row position in `dst` for every row of `src` (-1 where there is none)."""
        if src == dst:
            return np.arange(len(self.tables[src]))
        if (src, dst) in self._pos:
            return self._pos[(src, dst)]
        path = self.fk_path(src, dst)
        if path is None:
            raise RuleSyntaxError(f"no foreign-key path from {src} to {dst}")
        pos = np.arange(len(self.tables[src]))
        cur = src
        for fk in path:
            hop = self._hop(self.tables[cur], fk, self.tables[fk.parent_table])
            pos = np.where(pos >= 0, hop[np.maximum(pos, 0)], -1)
            cur = fk.parent_table
        self._pos[(src, dst)] = pos
        return pos

    @staticmethod
    def _hop(child: pd.DataFrame, fk: ForeignKey, parent: pd.DataFrame) -> np.ndarray:
        if len(fk.child_columns) == 1:
            pidx = pd.Index(parent[fk.parent_columns[0]].to_numpy())
            cidx = pd.Index(child[fk.child_columns[0]].to_numpy())
        else:
            pidx = pd.MultiIndex.from_frame(parent[fk.parent_columns])
            cidx = pd.MultiIndex.from_frame(child[fk.child_columns])
        lookup = pd.Series(np.arange(len(parent)), index=pidx)
        lookup = lookup[~lookup.index.duplicated(keep="first")]
        got = lookup.reindex(cidx).to_numpy()
        return np.where(np.isnan(got.astype(float)), -1, got.astype(float)).astype(int)

    # ---- expression evaluation
    def value(self, node: Any, owner: str) -> Any:
        n = len(self.tables[owner])
        if isinstance(node, Num):
            return node.value
        if isinstance(node, Str):
            return node.value
        if isinstance(node, Ref):
            col = self.tables[node.table][node.column].reset_index(drop=True)
            if node.table == owner:
                return col
            pos = self.positions(owner, node.table)
            return col.take(np.maximum(pos, 0)).reset_index(drop=True).where(pos >= 0)
        if isinstance(node, Neg):
            return -_num(self.value(node.x, owner), n)
        if isinstance(node, Not):
            return ~_bool(self.value(node.x, owner), n)
        if isinstance(node, IsNull):
            s = _series(self.value(node.x, owner), n)
            return (~s.isna() if node.negated else s.isna()).astype("boolean")
        if isinstance(node, In):
            s = _series(self.value(node.x, owner), n)
            vals = [self.value(v, owner) for v in node.values]
            res = s.isin(vals).astype("boolean")
            res = ~res if node.negated else res
            return res.mask(s.isna(), pd.NA)
        if isinstance(node, Between):
            x, lo, hi = (self.value(v, owner) for v in (node.x, node.lo, node.hi))
            return _cmp(">=", x, lo, n) & _cmp("<=", x, hi, n)
        if isinstance(node, Agg):
            return self._agg(node, owner)
        if isinstance(node, Func):
            return self._func(node, owner, n)
        if isinstance(node, Bin):
            if node.op in ("and", "or", "implies"):
                a, b = _bool(self.value(node.left, owner), n), _bool(self.value(node.right, owner), n)
                return (a & b) if node.op == "and" else (a | b) if node.op == "or" else ((~a) | b)
            a, b = self.value(node.left, owner), self.value(node.right, owner)
            if node.op in CMP:
                return _cmp(node.op, a, b, n)
            x, y = _num(a, n), _num(b, n)
            with np.errstate(divide="ignore", invalid="ignore"):
                res = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}[node.op](x, y)
            return res.replace([np.inf, -np.inf], np.nan) if isinstance(res, pd.Series) else res
        raise RuleSyntaxError(f"cannot evaluate {node!r}")

    def _func(self, f: Func, owner: str, n: int) -> Any:
        args = [self.value(a, owner) for a in f.args]
        if f.name == "ABS":
            return _num(args[0], n).abs()
        if f.name == "ROUND":
            return _num(args[0], n).round(int(args[1]) if len(args) > 1 else 0)
        if f.name in ("LOWER", "UPPER"):
            s = _series(args[0], n).astype("object")
            return s.map(lambda v: v if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)) else (v.lower() if f.name == "LOWER" else v.upper()))
        if f.name == "LEN":
            return _series(args[0], n).astype("object").map(lambda v: np.nan if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)) else float(len(str(v))))
        if f.name == "DATEDIFF":
            a, b = pd.to_datetime(_series(args[0], n)), pd.to_datetime(_series(args[1], n))
            return (a - b).dt.total_seconds() / 86400.0
        if f.name == "UNIQUE":
            s = _series(args[0], n)
            return (~(s.duplicated(keep="first") & s.notna())).astype("boolean")
        raise RuleSyntaxError(f"cannot evaluate {f.name}()")

    def _agg(self, a: Agg, owner: str) -> pd.Series:
        fks = [fk for fk in self.graph.fks_of(a.table) if fk.parent_table == owner]
        if not fks:
            raise RuleSyntaxError(f"{a.table} has no foreign key to {owner}")
        fk = fks[0]
        child = self.tables[a.table]
        cn = len(child)
        vals = pd.Series(np.ones(cn)) if a.expr is None else _series(self.value(a.expr, a.table), cn)
        keys = child[list(fk.child_columns)].reset_index(drop=True)
        frame = pd.concat([keys, vals.rename("__v")], axis=1)
        frame = frame.dropna(subset=list(fk.child_columns))
        g = frame.groupby(list(fk.child_columns), dropna=True)["__v"]
        if a.fn == "COUNT":
            grouped = g.count() if a.expr is not None else g.size()
        else:
            grouped = getattr(g, {"SUM": "sum", "AVG": "mean", "MIN": "min", "MAX": "max"}[a.fn])()
        pk = self.tables[owner][list(fk.parent_columns)].reset_index(drop=True)
        if len(fk.child_columns) == 1:
            out = grouped.reindex(pd.Index(pk.iloc[:, 0].to_numpy()))
        else:
            out = grouped.reindex(pd.MultiIndex.from_frame(pk))
        res = pd.Series(out.to_numpy(dtype=float))
        return res.fillna(0.0) if a.fn in ("SUM", "COUNT") else res

    # ---- checks
    def check(self, rule: Compiled) -> Check:
        n = len(self.tables[rule.owner])
        res = self.value(rule.ast, rule.owner)
        if isinstance(res, (bool, np.bool_)):
            res = pd.Series([bool(res)] * n, dtype="boolean")
        res = _bool(res, n)
        return Check(mask=res.eq(False).fillna(False).to_numpy(dtype=bool), n=n, unknown=int(res.isna().sum()))


# ------------------------------------------------------------ value helpers
def _series(x: Any, n: int) -> pd.Series:
    return x.reset_index(drop=True) if isinstance(x, pd.Series) else pd.Series([x] * n)


def _bool(x: Any, n: int) -> pd.Series:
    s = _series(x, n)
    return s.astype("boolean") if s.dtype != "boolean" else s


def _num(x: Any, n: int) -> pd.Series:
    s = _series(x, n)
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_numeric(s, errors="coerce").astype(float)


def _cmp(op: str, a: Any, b: Any, n: int) -> pd.Series:
    a, b = _series(a, n), _series(b, n)
    a_dt, b_dt = pd.api.types.is_datetime64_any_dtype(a), pd.api.types.is_datetime64_any_dtype(b)
    if a_dt and not b_dt:
        b = pd.to_datetime(b, errors="coerce")
    if b_dt and not a_dt:
        a = pd.to_datetime(a, errors="coerce")
    na = (a.isna() | b.isna()).to_numpy()
    numeric = pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b) and \
        not pd.api.types.is_bool_dtype(a) and not pd.api.types.is_bool_dtype(b)
    if numeric:
        x, y = a.to_numpy(dtype=float, na_value=np.nan), b.to_numpy(dtype=float, na_value=np.nan)
        if op in ("=", "!="):
            eq = np.abs(x - y) <= EQ_TOL
            res = eq if op == "=" else ~eq
        else:
            with np.errstate(invalid="ignore"):
                res = CMP[op](x, y)
    elif a_dt or b_dt:
        res = CMP[op](a.to_numpy(), b.to_numpy())
    else:
        x = a.astype(object).where(~a.isna(), "").to_numpy()
        y = b.astype(object).where(~b.isna(), "").to_numpy()
        x, y = np.array([str(v) for v in x], dtype=object), np.array([str(v) for v in y], dtype=object)
        res = CMP[op](x, y)
    out = pd.Series(np.asarray(res, dtype=bool), dtype="boolean")
    return out.mask(na, pd.NA)
