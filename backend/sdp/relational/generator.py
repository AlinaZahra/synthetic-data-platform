"""Relational generator: parents first (topological order), children sample valid parent keys.

Per child table:
  1. one *driver* FK decides row count: each parent row gets k children, k ~ CardinalityConfig
     (learned from real data by default, or Poisson/Zipf/fixed);
  2. other FKs (e.g. the product side of a junction) sample parent rows with learned popularity
     weights; if the composite PK spans both FKs, partners are drawn without replacement so pairs
     stay unique (this is how N:N junction tables are generated);
  3. non-key PK parts are generated (sequences), self-references point only at earlier rows
     (a forest, so no cycles), nullable FKs get their learned null rate;
  4. remaining attribute columns come from a per-table TabularGenerator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sdp.relational.cardinality import CardinalityConfig, compare_distributions, fit_to_total, sample_counts
from sdp.relational.inference import children_per_parent
from sdp.relational.integrity import IntegrityReport, check_integrity
from sdp.relational.schema import ForeignKey, RelationshipGraph, Table
from sdp.tabular.generator import TabularGenerator

POPULARITY_FLOOR = 0.1


@dataclass
class RelationalResult:
    tables: dict[str, pd.DataFrame]
    integrity: IntegrityReport
    cardinality: dict[str, dict]
    notes: list[str] = field(default_factory=list)
    seed: int = 0
    rules: dict | None = None  # rule enforcement report (before/after, reconciliation pass rate)

    def scorecard(self) -> dict:
        scores = [c["match_score"] for c in self.cardinality.values()]
        return {
            "row_counts": {k: len(v) for k, v in self.tables.items()},
            "integrity": self.integrity.to_dict(),
            "cardinality": self.cardinality,
            "mean_cardinality_match": float(np.mean(scores)) if scores else None,
            "notes": self.notes,
            "seed": self.seed,
            "rules": self.rules,
        }


class RelationalGenerator:
    def __init__(self, graph: RelationshipGraph, cardinality: dict[str, CardinalityConfig] | None = None,
                 driver_fk: dict[str, str] | None = None, method: str = "gaussian_copula",
                 condition_on_parents: bool = True, max_context_columns: int = 10) -> None:
        self.graph = graph
        self.cardinality = cardinality or {}
        self.driver_fk = driver_fk or {}
        self.method = method
        self._tab: dict[str, TabularGenerator] = {}
        self._learned: dict[str, np.ndarray] = {}
        self._null_rate: dict[str, float] = {}
        self._real_rows: dict[str, int] = {}
        self.condition_on_parents = condition_on_parents and method == "gaussian_copula"
        self.max_context_columns = max_context_columns
        self._ctx: dict[str, list[tuple[str, str]]] = {}
        self._ctx_enc: dict[str, dict[str, dict]] = {}
        self._strat: dict[str, tuple[str, str, dict[Any, np.ndarray]]] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, tables: dict[str, pd.DataFrame], seed: int = 0) -> "RelationalGenerator":
        errs = self.graph.validate_graph()
        if errs:
            raise ValueError("invalid relationship graph: " + "; ".join(errs))
        from sdp.rules.engine import Evaluator
        ev = Evaluator(tables, self.graph)
        for t in self.graph.tables:
            df = tables[t.name]
            self._real_rows[t.name] = len(df)
            keys = set(t.primary_key) | {c for f in self.graph.fks_of(t.name) for c in f.child_columns}
            feats = [c.name for c in t.columns if c.name not in keys]
            if not feats:
                continue
            ctx = self._context_columns(t.name, tables) if self.condition_on_parents else []
            frame = df[feats].reset_index(drop=True)
            if ctx:
                extra = {}
                for parent, col in ctx:
                    pos = ev.positions(t.name, parent)
                    extra[f"{parent}.{col}"] = tables[parent][col].reset_index(drop=True).take(np.maximum(pos, 0)).reset_index(drop=True).where(pos >= 0)
                ctx_df = pd.DataFrame(extra)
                self._ctx_enc[t.name] = self._encode_context(ctx_df, frame)
                frame = pd.concat([frame, ctx_df], axis=1)
                self._ctx[t.name] = ctx
            self._tab[t.name] = TabularGenerator(self.method).fit(frame, seed=seed)
        for fk in self.graph.foreign_keys:
            self._learned[fk.key] = children_per_parent(tables[fk.child_table], fk, tables[fk.parent_table])
            self._null_rate[fk.key] = float(tables[fk.child_table][fk.child_columns].isna().any(axis=1).mean())
            if self.condition_on_parents and not fk.is_self_reference:
                strat = self._stratify(fk, tables, self._learned[fk.key])
                if strat:
                    self._strat[fk.key] = strat
        return self

    @staticmethod
    def _encode_context(ctx_df: pd.DataFrame, feats: pd.DataFrame) -> dict[str, dict]:
        """Turn categorical parent attributes into numbers ordered by how they shift the child's numeric features.

        A Gaussian copula can only express monotone dependence, and a category has no natural order. Encoding each
        category by the mean rank of the child's numeric columns makes 'segment -> order size' monotone, so the copula
        captures it. Mutates ctx_df in place; returns the mappings so generation can apply the same encoding."""
        num = [c for c in feats.columns if (pd.api.types.is_numeric_dtype(feats[c]) and not pd.api.types.is_bool_dtype(feats[c]))
               or pd.api.types.is_datetime64_any_dtype(feats[c])]
        enc: dict[str, dict] = {}
        if not num:
            return enc
        y = pd.concat([feats[c].rank(pct=True) for c in num], axis=1).mean(axis=1)
        for name in ctx_df.columns:
            s = ctx_df[name]
            numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
            if pd.api.types.is_datetime64_any_dtype(s) or (numeric and s.nunique() > 12):
                continue
            mapping = y.groupby(s).mean().to_dict()
            ctx_df[name] = s.map(mapping).astype(float)
            enc[name] = mapping
        return enc

    def _ancestors(self, table: str, depth: int = 2) -> list[str]:
        """Tables reachable by following foreign keys upward (nearest first), excluding self-references."""
        seen, order, frontier = {table}, [], [table]
        for _ in range(depth):
            nxt = []
            for t in frontier:
                for fk in self.graph.fks_of(t):
                    if not fk.is_self_reference and fk.parent_table not in seen:
                        seen.add(fk.parent_table)
                        order.append(fk.parent_table)
                        nxt.append(fk.parent_table)
            frontier = nxt
        return order

    def _attr_columns(self, table: str, tables: dict[str, pd.DataFrame]) -> list[tuple[str, str]]:
        """Informative non-key columns of `table` (numeric, datetime, or <= 12 distinct values)."""
        p = self.graph.table(table)
        keys = set(p.primary_key) | {c for f in self.graph.fks_of(p.name) for c in f.child_columns}
        out = []
        for c in p.columns:
            s = tables[p.name][c.name]
            if c.name in keys or s.nunique() < 2:
                continue
            numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
            if numeric or pd.api.types.is_datetime64_any_dtype(s) or s.nunique() <= 12:
                out.append((table, c.name))
        return out

    def _context_columns(self, table: str, tables: dict[str, pd.DataFrame]) -> list[tuple[str, str]]:
        """Ancestor attributes a child row should depend on (parents first, then grandparents)."""
        out: list[tuple[str, str]] = []
        for anc in self._ancestors(table):
            out += self._attr_columns(anc, tables)
        return out[: self.max_context_columns]

    def _series_from(self, tables: dict[str, pd.DataFrame], base: str, anc: str, col: str) -> pd.Series:
        from sdp.rules.engine import Evaluator
        if anc == base:
            return tables[base][col].reset_index(drop=True)
        pos = Evaluator(tables, self.graph).positions(base, anc)
        return tables[anc][col].reset_index(drop=True).take(np.maximum(pos, 0)).reset_index(drop=True).where(pos >= 0)

    def _stratify(self, fk: ForeignKey, tables: dict[str, pd.DataFrame], counts: np.ndarray) -> tuple[str, str, dict[Any, np.ndarray]] | None:
        """If an attribute of the parent (or a grandparent) explains how many children a parent has - e.g. customer
        segment -> orders - keep the children-per-parent distribution per value. Picked by eta-squared (>= 0.03),
        requiring at least 5 parents per group."""
        total_var = counts.var()
        if total_var <= 0:
            return None
        best: tuple[float, str, str] | None = None
        for anc in [fk.parent_table] + self._ancestors(fk.parent_table):
            for _, col in self._attr_columns(anc, tables):
                v = self._series_from(tables, fk.parent_table, anc, col)
                if not 2 <= v.nunique() <= 12:
                    continue
                g = pd.Series(counts).groupby(v)
                if (g.size() < 5).any():
                    continue
                eta2 = float((g.size() * (g.mean() - counts.mean()) ** 2).sum() / (total_var * len(counts)))
                if eta2 >= 0.03 and (best is None or eta2 > best[0]):
                    best = (eta2, anc, col)
        if best is None:
            return None
        _, anc, col = best
        v = self._series_from(tables, fk.parent_table, anc, col)
        return anc, col, {val: counts[(v == val).to_numpy()] for val in v.dropna().unique()}

    def _counts_for(self, fk: ForeignKey, cfg: CardinalityConfig, out: dict[str, pd.DataFrame], rng: np.random.Generator) -> np.ndarray:
        parents = out[fk.parent_table]
        strat = self._strat.get(fk.key)
        if cfg.kind != "learned" or strat is None:
            return sample_counts(cfg, self._learned.get(fk.key), len(parents), rng)
        anc, col, groups = strat
        pooled = self._learned[fk.key]
        vals = self._series_from(out, fk.parent_table, anc, col)
        counts = np.empty(len(parents), dtype=int)
        for v in vals.dropna().unique():
            idx = np.flatnonzero((vals == v).to_numpy())
            counts[idx] = rng.choice(groups.get(v, pooled), size=len(idx), replace=True)
        rest = np.flatnonzero(vals.isna().to_numpy())
        if len(rest):
            counts[rest] = rng.choice(pooled, size=len(rest), replace=True)
        return np.clip(counts, cfg.min_children, cfg.max_children)

    # ------------------------------------------------------------- generate
    def generate(self, rows: dict[str, int] | None = None, scale: float = 1.0, seed: int = 0,
                 rules: list[str] | None = None, enforce: bool = True) -> RelationalResult:
        """`rows` sets exact sizes for named tables (roots: row count; children: nudges child totals).

        `rules` (DSL or plain language, may span tables) are compiled against the graph; with enforce=True the generated
        tables are repaired so they hold, and the result carries before/after violation counts (naive vs enforced)."""
        if not self._real_rows:
            raise RuntimeError("call fit() before generate()")
        rows = rows or {}
        out: dict[str, pd.DataFrame] = {}
        notes: list[str] = []
        for i, name in enumerate(self.graph.topological_order()):
            rng = np.random.default_rng(np.random.SeedSequence([seed, i]))
            out[name] = self._gen_table(self.graph.table(name), out, rows, scale, rng, notes)

        rules_report = None
        if rules:
            from sdp.rules.enforce import EnforceReport, Enforcer, RuleResult, compile_rules, evaluate_rules
            compiled = compile_rules(rules, self.graph)
            if enforce:
                rules_report = Enforcer(out, self.graph, compiled, seed).run().to_dict()
            else:
                ev = evaluate_rules(out, self.graph, compiled)
                rr = [RuleResult(e["name"], e["dsl"], e["owner"], e["kind"], e["checked"], e["violations"], e["violations"]) for e in ev]
                rules_report = EnforceReport(rr).to_dict()

        card: dict[str, dict] = {}
        for fk in self.graph.foreign_keys:
            real = self._learned.get(fk.key)
            synth = children_per_parent(out[fk.child_table], fk, out[fk.parent_table])
            rep = compare_distributions(real, synth) if real is not None else {"match_score": 0.0}
            cfg = self.cardinality.get(fk.key, CardinalityConfig())
            rep["target"] = cfg.kind if not fk.is_self_reference else "learned-popularity"
            card[fk.key] = rep
        return RelationalResult(out, check_integrity(self.graph, out), card, notes, seed, rules_report)

    # ---------------------------------------------------------------- table
    def _pick_driver(self, table: str, fks: list[ForeignKey], out: dict[str, pd.DataFrame]) -> ForeignKey:
        wanted = self.driver_fk.get(table)
        if wanted:
            return next(f for f in fks if f.key == wanted)
        return max(fks, key=lambda f: len(out[f.parent_table]))  # ties -> first declared

    def _gen_table(self, t: Table, out: dict[str, pd.DataFrame], rows: dict[str, int], scale: float,
                   rng: np.random.Generator, notes: list[str]) -> pd.DataFrame:
        fks = [f for f in self.graph.fks_of(t.name) if not f.is_self_reference]
        selfs = [f for f in self.graph.fks_of(t.name) if f.is_self_reference]
        pk = t.primary_key
        df: pd.DataFrame
        driver: ForeignKey | None = None
        others: list[ForeignKey] = []

        if not fks:
            n = rows.get(t.name) or max(1, round(self._real_rows.get(t.name, 1000) * scale))
            df = pd.DataFrame(index=range(n))
        else:
            driver = self._pick_driver(t.name, fks, out)
            others = [f for f in fks if f is not driver]
            parents = out[driver.parent_table]
            cfg = self.cardinality.get(driver.key, CardinalityConfig())
            counts = self._counts_for(driver, cfg, out, rng)
            unique_others = [f for f in others if set(pk) >= set(driver.child_columns) | set(f.child_columns)]
            for f in unique_others:  # cannot have more distinct partners than partner rows
                counts = np.minimum(counts, len(out[f.parent_table]))
            if rows.get(t.name):
                counts = fit_to_total(counts, rows[t.name], rng)
                if unique_others:
                    counts = np.minimum(counts, min(len(out[f.parent_table]) for f in unique_others))
            parent_idx = np.repeat(np.arange(len(parents)), counts)
            n = len(parent_idx)
            df = pd.DataFrame(index=range(n))
            for cc, pc in zip(driver.child_columns, driver.parent_columns):
                df[cc] = parents[pc].to_numpy()[parent_idx]
            for f in others:
                self._assign_other_fk(df, f, out[f.parent_table], counts if f in unique_others else None, rng)

        # primary-key columns that are not FK columns
        for c in pk:
            if c in df.columns:
                continue
            if len(pk) > 1 and driver is not None:
                df[c] = df.groupby(driver.child_columns).cumcount().to_numpy() + 1  # line numbers within parent
            else:
                df[c] = self._sequence(t, c, n)

        for f in selfs:
            self._assign_self_fk(df, f, rng)

        tab = self._tab.get(t.name)
        if tab is not None:
            tseed = int(rng.integers(0, 2**31 - 1))
            ctx = self._ctx.get(t.name)
            if ctx and n:
                from sdp.rules.engine import Evaluator
                ev = Evaluator({**out, t.name: df}, self.graph)
                extra = {}
                for parent, col in ctx:
                    pos = ev.positions(t.name, parent)
                    extra[f"{parent}.{col}"] = out[parent][col].reset_index(drop=True).take(np.maximum(pos, 0)).reset_index(drop=True).where(pos >= 0)
                ctx_df = pd.DataFrame(extra)
                for name, mapping in self._ctx_enc.get(t.name, {}).items():
                    ctx_df[name] = ctx_df[name].map(mapping).astype(float)
                attrs = tab.sample_conditional(n, ctx_df, seed=tseed)
            else:
                attrs = tab.sample(n, seed=tseed)
            for c in attrs.columns:
                df[c] = attrs[c].to_numpy() if not isinstance(attrs[c].dtype, pd.CategoricalDtype) else attrs[c]

        df = df[[c.name for c in t.columns]]
        if pk and df.duplicated(pk).any():
            dropped = int(df.duplicated(pk).sum())
            df = df[~df.duplicated(pk)].reset_index(drop=True)
            notes.append(f"{t.name}: dropped {dropped} rows that duplicated the composite primary key {pk}")
        return self._cast_keys(t, df)

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _sequence(t: Table, col: str, n: int) -> np.ndarray:
        if t.column(col).dtype == "str":
            prefix = t.name[:3].upper()
            return np.array([f"{prefix}-{i:06d}" for i in range(1, n + 1)], dtype=object)
        return np.arange(1, n + 1)

    def _assign_other_fk(self, df: pd.DataFrame, fk: ForeignKey, parents: pd.DataFrame,
                         unique_counts: np.ndarray | None, rng: np.random.Generator) -> None:
        m = len(parents)
        learned = self._learned.get(fk.key)
        pop = (rng.choice(learned, size=m) if learned is not None and len(learned) else np.ones(m)) + POPULARITY_FLOOR
        p = pop / pop.sum()
        if unique_counts is not None:  # junction: distinct partners per driver parent
            idx = np.empty(int(unique_counts.sum()), dtype=int)
            pos = 0
            for k in unique_counts:
                if k:
                    idx[pos:pos + k] = rng.choice(m, size=k, replace=False, p=p)
                    pos += k
        else:
            idx = rng.choice(m, size=len(df), p=p)
        for cc, pc in zip(fk.child_columns, fk.parent_columns):
            df[cc] = parents[pc].to_numpy()[idx]
        rate = self._null_rate.get(fk.key, 0.0) if fk.nullable else 0.0
        if rate > 0:
            mask = rng.random(len(df)) < rate
            for cc in fk.child_columns:
                col = df[cc].astype("Int64") if pd.api.types.is_integer_dtype(df[cc]) else df[cc].astype(object)
                df[cc] = col.mask(mask)

    def _assign_self_fk(self, df: pd.DataFrame, fk: ForeignKey, rng: np.random.Generator) -> None:
        """Each row points at a strictly earlier row (or is a root: NULL, or itself if not nullable)."""
        n = len(df)
        learned = self._learned.get(fk.key)
        w = (rng.choice(learned, size=n) if learned is not None and len(learned) else np.ones(n)) + POPULARITY_FLOOR
        cs = np.cumsum(w)
        idx = np.zeros(n, dtype=int)
        if n > 1:
            idx[1:] = np.searchsorted(cs, rng.random(n - 1) * cs[:-1], side="right")
        rate = self._null_rate.get(fk.key, 0.0)
        root = rng.random(n) < rate
        root[0] = True
        idx = np.where(root, np.arange(n), idx)
        for cc, pc in zip(fk.child_columns, fk.parent_columns):
            vals = df[pc].to_numpy()[idx]
            if fk.nullable:
                col = pd.Series(vals).astype("Int64" if pd.api.types.is_integer_dtype(df[pc]) else object)
                df[cc] = col.mask(root)
            else:
                df[cc] = vals

    def _cast_keys(self, t: Table, df: pd.DataFrame) -> pd.DataFrame:
        keycols = set(t.primary_key) | {c for f in self.graph.fks_of(t.name) for c in f.child_columns}
        for c in keycols:
            dt = t.column(c).dtype
            if dt == "int":
                df[c] = df[c].astype("Int64" if df[c].isna().any() else "int64")
        return df.reset_index(drop=True)
