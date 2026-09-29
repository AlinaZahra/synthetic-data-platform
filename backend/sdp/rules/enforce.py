"""Enforcement: repair generated tables so rules hold, then verify. Works across tables (R4) and on one frame (A4).

Repair strategies, by rule kind
  derive       `T.x = expr` (e.g. orders.total = SUM(order_items.qty*order_items.price)): assign the computed value
  bound        comparisons on a bare column: dates get ref +/- 0-3 days, numbers are reflected across the bound (keeps the
               density shape better than clipping), NOT NULL is filled from observed values, IN/BETWEEN are snapped
  conditional  `A => B`: first satisfy B if it is a bare atom (possibly on a parent table), otherwise re-draw the
               categorical column in A from observed values until every rule reading that column holds
  unique       duplicates get fresh values
Key columns (PK/FK) are never modified; a rule that can only be satisfied by editing one is reported as unrepairable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sdp.relational.schema import RelationshipGraph
from sdp.rules.dsl import Between, Bin, Func, In, IsNull, Num, Ref, Str
from sdp.rules.engine import CMP, FLIP, Compiled, Evaluator, compile_rule, reads_of, single_table_graph


@dataclass
class RuleResult:
    name: str
    dsl: str
    owner: str
    kind: str
    checked: int
    violations_before: int
    violations_after: int = 0
    strategy: str = ""
    unrepairable: str | None = None

    @property
    def pass_rate_before(self) -> float:
        return 100.0 * (1 - self.violations_before / self.checked) if self.checked else 100.0

    @property
    def pass_rate_after(self) -> float:
        return 100.0 * (1 - self.violations_after / self.checked) if self.checked else 100.0

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dsl": self.dsl, "owner": self.owner, "kind": self.kind, "checked": self.checked,
                "violations_before": self.violations_before, "violations_after": self.violations_after,
                "pass_rate_before": self.pass_rate_before, "pass_rate_after": self.pass_rate_after,
                "strategy": self.strategy, "unrepairable": self.unrepairable}


@dataclass
class EnforceReport:
    rules: list[RuleResult] = field(default_factory=list)

    def _rate(self, which: str, kinds: set[str] | None = None) -> float:
        rs = [r for r in self.rules if kinds is None or r.kind in kinds]
        checked = sum(r.checked for r in rs)
        bad = sum(r.violations_before if which == "before" else r.violations_after for r in rs)
        return 100.0 * (1 - bad / checked) if checked else 100.0

    @property
    def total_violations_after(self) -> int:
        return sum(r.violations_after for r in self.rules)

    @property
    def total_violations_before(self) -> int:
        return sum(r.violations_before for r in self.rules)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rules": [r.to_dict() for r in self.rules],
            "violations_before": self.total_violations_before, "violations_after": self.total_violations_after,
            "pass_rate_before": self._rate("before"), "reconciliation_pass_rate": self._rate("after"),
            "derived_pass_rate_before": self._rate("before", {"derive"}), "derived_pass_rate": self._rate("after", {"derive"}),
        }


# ---------------------------------------------------------------- helpers
def compile_rules(texts: list[str], graph: RelationshipGraph, default_table: str | None = None) -> list[Compiled]:
    return [compile_rule(t, graph, default_table, name=f"R{i + 1}") for i, t in enumerate(texts)]


def evaluate_rules(tables: dict[str, pd.DataFrame], graph: RelationshipGraph, rules: list[Compiled]) -> list[dict[str, Any]]:
    ev = Evaluator(tables, graph)
    out = []
    for r in rules:
        c = ev.check(r)
        out.append({"name": r.name, "dsl": r.dsl, "owner": r.owner, "kind": r.kind, "checked": c.n, "violations": c.violations})
    return out


def _decimals(s: pd.Series) -> int:
    if not pd.api.types.is_float_dtype(s):
        return 0
    v = s.dropna().head(200).to_numpy(dtype=float)
    for d in range(0, 5):
        if np.allclose(np.round(v, d), v, atol=1e-9):
            return d
    return 4


def _atoms(n: Any) -> list[Any]:
    if isinstance(n, Bin) and n.op == "and":
        return _atoms(n.left) + _atoms(n.right)
    return [n]


class Enforcer:
    def __init__(self, tables: dict[str, pd.DataFrame], graph: RelationshipGraph, rules: list[Compiled], seed: int = 0) -> None:
        self.tables, self.graph, self.rules = tables, graph, rules
        self.rng = np.random.default_rng(seed)
        self.ev = Evaluator(tables, graph)
        self.protected = {(t.name, c) for t in graph.tables for c in t.primary_key} | {
            (f.child_table, c) for f in graph.foreign_keys for c in f.child_columns}
        self.notes: dict[str, str] = {}

    # ------------------------------------------------------------ driver
    def run(self, max_passes: int = 4) -> EnforceReport:
        before = {r.name: self.ev.check(r) for r in self.rules}
        results = {r.name: RuleResult(r.name, r.dsl, r.owner, r.kind, before[r.name].n, before[r.name].violations)
                   for r in self.rules}
        for _ in range(max_passes):
            changed = False
            for rule in self._ordered():
                chk = self.ev.check(rule)
                if chk.violations == 0:
                    continue
                strategy, ok = self._repair(rule, chk.mask)
                if strategy:
                    results[rule.name].strategy = strategy
                changed |= ok
            if not changed or all(self.ev.check(r).violations == 0 for r in self.rules):
                break
        for rule in self._ordered():  # snap derived values exactly (tolerance-level differences are not left behind)
            if rule.kind == "derive":
                self._derive(rule, np.ones(len(self.tables[rule.owner]), dtype=bool))
        for rule in self.rules:
            res = results[rule.name]
            res.violations_after = self.ev.check(rule).violations
            if res.violations_after and rule.name in self.notes:
                res.unrepairable = self.notes[rule.name]
            elif res.violations_after and not res.strategy:
                res.unrepairable = "no repair strategy for this rule shape"
        return EnforceReport([results[r.name] for r in self.rules])

    def _ordered(self) -> list[Compiled]:
        derive = [r for r in self.rules if r.kind == "derive"]
        ordered: list[Compiled] = []
        while derive:
            pick = next((r for r in derive if not any(o is not r and o.target in r.reads for o in derive)), derive[0])
            ordered.append(pick)
            derive.remove(pick)
        rest = [r for k in ("bound", "unique", "conditional", "check") for r in self.rules if r.kind == k]
        return ordered + rest

    def _vals(self, node: Any, owner: str) -> pd.Series:
        """Expression value as a Series with one entry per owner row (scalars broadcast)."""
        v = self.ev.value(node, owner)
        return v.reset_index(drop=True) if isinstance(v, pd.Series) else pd.Series([v] * len(self.tables[owner]))

    # -------------------------------------------------------- assignment
    def _set(self, table: str, col: str, pos: np.ndarray, values: Any) -> bool:
        if (table, col) in self.protected:
            return False
        pos = np.asarray(pos, dtype=int)
        if len(pos) == 0:
            return True
        s = self.tables[table][col].copy()
        vals = pd.Series(values).reset_index(drop=True)
        if pd.api.types.is_integer_dtype(s):
            vals = vals.round()
            if vals.isna().any() or pd.api.types.is_extension_array_dtype(s):
                s = s.astype("Int64")
                s.iloc[pos] = vals.astype("Int64")
            else:
                s.iloc[pos] = vals.astype("int64")
        elif pd.api.types.is_float_dtype(s):
            s.iloc[pos] = vals.astype(float).to_numpy()
        elif pd.api.types.is_datetime64_any_dtype(s):
            s.iloc[pos] = pd.DatetimeIndex(pd.to_datetime(vals)).astype(s.dtype)
        else:
            s.iloc[pos] = vals.astype(object).to_numpy()
        self.tables[table][col] = s
        return True

    def _rows_of(self, mask: np.ndarray, owner: str, table: str) -> np.ndarray:
        idx = np.flatnonzero(mask)
        if table == owner:
            return idx
        pos = self.ev.positions(owner, table)[idx]
        return pos[pos >= 0]

    # ------------------------------------------------------------ repair
    def _repair(self, rule: Compiled, mask: np.ndarray) -> tuple[str, bool]:
        ast, owner = rule.ast, rule.owner
        if rule.kind == "derive":
            return self._derive(rule, mask)
        if rule.kind == "bound":
            ok = self._atom(ast, owner, mask, rule)
            return ("bound repair" if ok else ""), ok
        if rule.kind == "unique":
            return self._unique(rule, mask)
        if rule.kind == "conditional":
            return self._conditional(rule, mask)
        self.notes[rule.name] = "composite check rules can only be verified, not repaired"
        return "", False

    def _derive(self, rule: Compiled, mask: np.ndarray) -> tuple[str, bool]:
        t, c = rule.target  # type: ignore[misc]
        ast: Bin = rule.ast
        expr = ast.right if (isinstance(ast.left, Ref) and (ast.left.table, ast.left.column) == (t, c)) else ast.left
        val = self._vals(expr, rule.owner)
        rows = np.flatnonzero(mask & val.notna().to_numpy())
        vals = val.iloc[rows]
        if not isinstance(expr, Ref):  # a bare column is copied exactly; computed values are rounded to money precision
            vals = vals.round(min(_decimals(self.tables[t][c]), 2))
        ok = self._set(t, c, rows, vals.to_numpy())
        if not ok:
            self.notes[rule.name] = f"{t}.{c} is a key column"
        return "derived value assigned", ok

    def _unique(self, rule: Compiled, mask: np.ndarray) -> tuple[str, bool]:
        ref = rule.ast.args[0]
        s = self.tables[ref.table][ref.column]
        rows = np.flatnonzero(mask)
        if pd.api.types.is_integer_dtype(s):
            new = int(s.max()) + 1 + np.arange(len(rows))
        elif pd.api.types.is_float_dtype(s):
            self.notes[rule.name] = "float columns are not made unique automatically"
            return "", False
        else:
            new = [f"{s.iloc[i]}-{k + 1}" for k, i in enumerate(rows)]
        ok = self._set(ref.table, ref.column, rows, new)
        if not ok:
            self.notes[rule.name] = f"{ref.table}.{ref.column} is a key column"
        return "duplicates renumbered", ok

    def _atom(self, node: Any, owner: str, mask: np.ndarray, rule: Compiled) -> bool:
        """Repair rows (owner positions where mask) so `node` holds. Returns False if it cannot be repaired."""
        ok = True
        if isinstance(node, Bin) and node.op == "and":
            return self._atom(node.left, owner, mask, rule) & self._atom(node.right, owner, mask, rule)
        if isinstance(node, Bin) and node.op in CMP:
            l, r, op = node.left, node.right, node.op
            if not isinstance(l, Ref) or (l.table != owner and isinstance(r, Ref) and r.table == owner):
                if isinstance(r, Ref):
                    l, r, op = r, l, FLIP[op]
                else:
                    self.notes[rule.name] = "neither side of the comparison is a plain column"
                    return False
            t, c = l.table, l.column
            if (t, c) in self.protected:
                self.notes[rule.name] = f"{t}.{c} is a key column"
                return False
            rows_o = np.flatnonzero(mask)
            rv = self._vals(r, owner).to_numpy()
            pos = rows_o if t == owner else self.ev.positions(owner, t)[rows_o]
            keep = pos >= 0
            rows_o, pos = rows_o[keep], pos[keep]
            if len(pos) == 0:
                return True
            col = self.tables[t][c]
            bound = rv[rows_o]
            cur = col.iloc[pos].to_numpy()
            if pd.api.types.is_datetime64_any_dtype(col):
                b = pd.to_datetime(pd.Series(bound)).reset_index(drop=True)
                strict = op in ("<", ">")
                jit = self.rng.integers(1 if strict else 0, 4, size=len(b))
                delta = pd.to_timedelta(jit, unit="D")
                if op in (">", ">="):
                    new = b + delta
                elif op in ("<", "<="):
                    new = b - delta
                elif op == "=":
                    new = b
                else:
                    self.notes[rule.name] = "!= on dates is not repairable"
                    return False
                good = ~pd.isna(new)
                return self._set(t, c, pos[good.to_numpy()], new[good].to_numpy())
            if pd.api.types.is_numeric_dtype(col):
                x = pd.Series(cur).astype(float).to_numpy()
                bnd = pd.Series(bound).astype(float).to_numpy()
                eps = 10.0 ** -max(_decimals(col), 0) if op in ("<", ">") else 0.0
                if pd.api.types.is_integer_dtype(col):
                    eps = 1.0 if op in ("<", ">") else 0.0
                if op in ("<=", "<"):
                    new = np.minimum(2 * bnd - x, bnd - eps)
                elif op in (">=", ">"):
                    new = np.maximum(2 * bnd - x, bnd + eps)
                elif op == "=":
                    new = bnd
                else:
                    self.notes[rule.name] = "!= on numbers is not repairable"
                    return False
                good = ~np.isnan(new)
                return self._set(t, c, pos[good], np.round(new[good], _decimals(col)))
            if op == "=":
                return self._set(t, c, pos, bound)
            self.notes[rule.name] = "ordering comparisons on text are not repairable"
            return False
        if isinstance(node, IsNull) and isinstance(node.x, Ref) and node.negated:
            t, c = node.x.table, node.x.column
            pos = self._rows_of(mask, owner, t)
            pool = self.tables[t][c].dropna()
            if pool.empty:
                self.notes[rule.name] = "no observed values to fill from"
                return False
            ok = self._set(t, c, pos, pool.sample(len(pos), replace=True, random_state=int(self.rng.integers(2**31))).to_numpy())
            return ok
        if isinstance(node, In) and isinstance(node.x, Ref) and not node.negated:
            t, c = node.x.table, node.x.column
            allowed = [v.value for v in node.values if isinstance(v, (Num, Str))]
            pos = self._rows_of(mask, owner, t)
            return self._set(t, c, pos, self.rng.choice(np.array(allowed, dtype=object), size=len(pos)))
        if isinstance(node, Between) and isinstance(node.x, Ref):
            t, c = node.x.table, node.x.column
            pos = self._rows_of(mask, owner, t)
            lo = self._vals(node.lo, owner).iloc[np.flatnonzero(mask)].astype(float).to_numpy()
            hi = self._vals(node.hi, owner).iloc[np.flatnonzero(mask)].astype(float).to_numpy()
            cur = self.tables[t][c].iloc[pos].astype(float).to_numpy()
            return self._set(t, c, pos, np.clip(cur, lo[: len(pos)], hi[: len(pos)]))
        self.notes[rule.name] = "unsupported atom shape"
        return False

    # ----------------------------------------------------------- implies
    def _conditional(self, rule: Compiled, mask: np.ndarray) -> tuple[str, bool]:
        ast: Bin = rule.ast
        consequent, antecedent = ast.right, ast.left
        changed = False
        strategy = ""
        atoms = _atoms(consequent)
        if all(self._is_atom(a) for a in atoms):
            ok = all([self._atom(a, rule.owner, mask, rule) for a in atoms])
            changed |= ok
            strategy = "consequent set"
            if self.ev.check(rule).violations == 0:
                return strategy, True
        # otherwise re-draw the categorical column named in the antecedent
        for a in _atoms(antecedent):
            ref = a.left if isinstance(a, Bin) and isinstance(a.left, Ref) else a.x if isinstance(a, In) and isinstance(a.x, Ref) else None
            if ref is None or (ref.table, ref.column) in self.protected:
                continue
            still = self.ev.check(rule).mask
            rows = self._rows_of(still, rule.owner, ref.table)
            readers = [r for r in self.rules if (ref.table, ref.column) in r.reads]
            fixed = self._candidate_search(ref.table, ref.column, rows, readers)
            if fixed:
                strategy = "categorical re-draw" if not strategy else strategy + " + categorical re-draw"
                changed = True
            break
        if not strategy:
            self.notes[rule.name] = "could not find a repairable column in the rule"
        return strategy, changed

    @staticmethod
    def _is_atom(n: Any) -> bool:
        return (isinstance(n, Bin) and n.op in CMP and (isinstance(n.left, Ref) or isinstance(n.right, Ref))) or \
            (isinstance(n, (IsNull, In, Between)) and isinstance(n.x, Ref))

    def _failing_rows(self, table: str, rules: list[Compiled]) -> np.ndarray:
        fails = [self._rows_of(self.ev.check(r).mask, r.owner, table) for r in rules]
        return np.unique(np.concatenate(fails)) if fails else np.array([], dtype=int)

    def _candidate_search(self, table: str, col: str, positions: np.ndarray, rules: list[Compiled]) -> int:
        remaining = np.unique(positions)
        if len(remaining) == 0:
            return 0
        current = self.tables[table][col].copy()
        cands = list(current.dropna().value_counts().index)
        fixed = 0
        for v in cands:
            if len(remaining) == 0:
                break
            trial = current.copy()
            trial.iloc[remaining] = v
            self.tables[table][col] = trial
            bad = self._failing_rows(table, rules)
            ok = remaining[~np.isin(remaining, bad)]
            if len(ok):
                current.iloc[ok] = v
                remaining = np.setdiff1d(remaining, ok)
                fixed += len(ok)
        self.tables[table][col] = current
        return fixed


def enforce(tables: dict[str, pd.DataFrame], graph: RelationshipGraph, rules: list[Compiled], seed: int = 0) -> EnforceReport:
    """Repair `tables` in place and return before/after violation counts per rule."""
    return Enforcer(tables, graph, rules, seed).run()
