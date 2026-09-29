"""A4 for a single table: compile rules, enforce during sampling (repair / reject / hybrid) and compare with naive sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from sdp.rules.enforce import EnforceReport, Enforcer, compile_rules
from sdp.rules.engine import Compiled, Evaluator, single_table_graph
from sdp.tabular.generator import TabularGenerator

TABLE = "data"
Mode = Literal["repair", "reject", "hybrid"]


def compile_frame_rules(columns: pd.DataFrame, texts: list[str]) -> tuple[Any, list[Compiled]]:
    graph = single_table_graph(TABLE, columns)
    return graph, compile_rules(texts, graph, TABLE)


def violations_per_rule(df: pd.DataFrame, graph: Any, rules: list[Compiled]) -> dict[str, int]:
    ev = Evaluator({TABLE: df}, graph)
    return {r.name: ev.check(r).violations for r in rules}


def violating_rows(df: pd.DataFrame, graph: Any, rules: list[Compiled]) -> np.ndarray:
    ev = Evaluator({TABLE: df}, graph)
    bad = np.zeros(len(df), dtype=bool)
    for r in rules:
        bad |= ev.check(r).mask
    return bad


@dataclass
class RuleSampleResult:
    data: pd.DataFrame
    report: EnforceReport
    rows_dropped: int
    rounds: int
    naive_violations: dict[str, int]
    compiled: list[dict[str, Any]] = field(default_factory=list)  # what each rule compiled to, for confirmation in the UI


def sample_with_rules(gen: TabularGenerator, n: int, seed: int, rules_text: list[str], mode: Mode = "hybrid",
                      max_rounds: int = 12) -> RuleSampleResult:
    """Sample n rows that satisfy every rule.

    repair  fix violating rows in place (no rows dropped; a rule that cannot be repaired may still show violations)
    reject  rejection sampling: draw batches, keep only rows with zero violations (exact but can be slow for rare events)
    hybrid  repair first, then reject whatever is still invalid and top up (default)
    """
    base = gen.sample(n, seed)
    graph, rules = compile_frame_rules(base, rules_text)
    naive = violations_per_rule(base, graph, rules)
    rng_seed = seed
    dropped = rounds = 0
    if mode == "repair":
        tables = {TABLE: base.copy()}
        report = Enforcer(tables, graph, rules, seed).run()
        return RuleSampleResult(tables[TABLE], report, 0, 1, naive, [r.describe() for r in rules])

    kept: list[pd.DataFrame] = []
    have = 0
    batch = base
    while have < n and rounds < max_rounds:
        rounds += 1
        if mode == "hybrid":
            t = {TABLE: batch.copy()}
            Enforcer(t, graph, rules, rng_seed).run()
            batch = t[TABLE]
        ok = ~violating_rows(batch, graph, rules)
        dropped += int((~ok).sum())
        kept.append(batch[ok])
        have += int(ok.sum())
        if have >= n:
            break
        rate = max(have / max(1, (rounds * n)), 0.02)
        rng_seed += 1
        batch = gen.sample(int(min(500_000, np.ceil((n - have) / rate * 1.3)) + 10), rng_seed + 10_000)
    data = pd.concat(kept, ignore_index=True).head(n) if kept else base.head(0)
    tables = {TABLE: data.copy()}
    # final verification (no repair): what the caller actually gets
    ev = Evaluator(tables, graph)
    from sdp.rules.enforce import RuleResult
    results = []
    for r in rules:
        chk = ev.check(r)
        results.append(RuleResult(r.name, r.dsl, r.owner, r.kind, len(data), naive[r.name], chk.violations,
                                  strategy="rejection sampling" if mode == "reject" else "repair + rejection"))
    return RuleSampleResult(data, EnforceReport(results), dropped, rounds, naive, [r.describe() for r in rules])


def compare_naive_vs_enforced(gen: TabularGenerator, n: int, seed: int, rules_text: list[str], real: pd.DataFrame | None = None,
                              mode: Mode = "hybrid") -> dict[str, Any]:
    """Naive sampling ignores the rules; ours enforces them. Reports violations and, given real data, the fidelity cost."""
    naive = gen.sample(n, seed)
    graph, rules = compile_frame_rules(naive, rules_text)
    nv = violations_per_rule(naive, graph, rules)
    res = sample_with_rules(gen, n, seed, rules_text, mode)
    ev = violations_per_rule(res.data, graph, rules)

    def mean_ks(df: pd.DataFrame) -> float | None:
        if real is None:
            return None
        cols = [c for c in real.columns if pd.api.types.is_numeric_dtype(real[c]) and not pd.api.types.is_bool_dtype(real[c])]
        return float(np.mean([stats.ks_2samp(real[c].dropna().astype(float), df[c].dropna().astype(float)).statistic for c in cols])) if cols else None

    naive_bad = int(violating_rows(naive, graph, rules).sum())
    ours_bad = int(violating_rows(res.data, graph, rules).sum())
    return {
        "rows": n, "mode": mode,
        "rules": [{"name": r.name, "dsl": r.dsl, "naive_violations": nv[r.name], "enforced_violations": ev[r.name]} for r in rules],
        "naive": {"rows_violating": naive_bad, "pass_rate": 100.0 * (1 - naive_bad / n), "mean_ks_vs_real": mean_ks(naive)},
        "enforced": {"rows_violating": ours_bad, "pass_rate": 100.0 * (1 - ours_bad / max(1, len(res.data))),
                     "mean_ks_vs_real": mean_ks(res.data), "rows_dropped": res.rows_dropped, "rows_returned": len(res.data)},
    }
