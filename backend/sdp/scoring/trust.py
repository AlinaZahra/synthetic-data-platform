"""Q4. One Trust Score: Fidelity (0.40) + Privacy (0.30) + Validity & coverage (0.30).

Validity & coverage is the equal-weight mean of whichever of these were supplied:
locale validity, business constraints, referential integrity, edge-case coverage (always present).
Gates cap the verdict regardless of the number: exact copies of real rows, or any integrity violation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from sdp.scoring.constraints import check_constraints, edge_case_coverage
from sdp.scoring.fidelity import fidelity_score
from sdp.scoring.locale_validity import locale_validity
from sdp.scoring.privacy import privacy_score

WEIGHTS = {"fidelity": 0.40, "privacy": 0.30, "validity": 0.30}
LABELS = {"fidelity": "Fidelity", "privacy": "Privacy", "validity": "Validity & coverage"}
BANDS = [(85, "Strong"), (70, "Good, with caveats"), (50, "Needs work"), (0, "Not recommended")]


def _band(score: float) -> str:
    return next(label for floor, label in BANDS if score >= floor)


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


def build_trust_report(real: pd.DataFrame, synthetic: pd.DataFrame, *, real_holdout: pd.DataFrame | None = None,
                       tstr: dict | None = None, relational: dict | None = None, locale_spec: dict | None = None,
                       rules: list[dict] | None = None, integrity: dict | None = None,
                       title: str = "Synthetic dataset", seed: int = 0, dsl_rules: list[str] | None = None) -> dict[str, Any]:
    fid = fidelity_score(real, synthetic, tstr=tstr, relational=relational)
    prv = privacy_score(real, synthetic, real_holdout=real_holdout, seed=seed)

    vcomps: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    if locale_spec:
        loc = locale_validity(synthetic, **locale_spec)
        details["locale"] = loc
        vcomps.append({"key": "locale", "label": "Locale validity", "score": loc["valid_pct"],
                       "summary": f"{loc['valid_pct']:.1f}% of records locale-valid ({loc['n_invalid_rows']} failing)"})
    if rules:
        con = check_constraints(synthetic, rules)
        details["constraints"] = con
        vcomps.append({"key": "constraints", "label": "Business constraints", "score": con["pass_pct"],
                       "summary": f"{con['rows_violating']} of {con['n_rows']} rows break a rule ({con['n_rules']} rules)"})
    if dsl_rules:
        from sdp.rules import Evaluator, compile_rules, single_table_graph
        g = single_table_graph("data", synthetic)
        compiled = compile_rules(dsl_rules, g, "data")
        ev = Evaluator({"data": synthetic}, g)
        mask = np.zeros(len(synthetic), dtype=bool)
        for r in compiled:
            mask |= ev.check(r).mask
        details["rules"] = {"rules": [r.describe() for r in compiled], "rows_violating": int(mask.sum())}
        vcomps.append({"key": "rules", "label": "Your rules", "score": 100.0 * (1 - mask.mean()) if len(mask) else 100.0,
                       "summary": f"{int(mask.sum())} of {len(synthetic)} rows break a rule ({len(compiled)} rules)"})
    if integrity is not None:
        total, rows = integrity["total_violations"], max(1, integrity.get("rows_checked", 1))
        details["integrity"] = integrity
        vcomps.append({"key": "integrity", "label": "Referential integrity", "score": 100.0 * max(0.0, 1 - total / rows),
                       "summary": f"{total} violations (orphans, duplicate keys, null keys)"})
    cov = edge_case_coverage(real, synthetic)
    details["coverage"] = cov
    vcomps.append({"key": "coverage", "label": "Edge-case coverage", "score": cov["score"],
                   "summary": f"{cov['covered']} of {cov['total']} real edge cases reproduced"})
    for c in vcomps:
        c["weight"] = 1 / len(vcomps)
    validity = _mean([c["score"] for c in vcomps])

    def comp_list(d: dict[str, dict[str, Any]], labels: dict[str, str]) -> list[dict[str, Any]]:
        return [{"key": k, "label": labels.get(k, k), "score": v["score"], "weight": v["weight"], "summary": v["summary"]}
                for k, v in d.items() if v["score"] is not None]

    subs = [
        {"key": "fidelity", "label": LABELS["fidelity"], "score": fid["score"], "weight": WEIGHTS["fidelity"],
         "components": comp_list(fid["components"], {"marginals": "Column distributions", "correlations": "Correlations",
                                                    "utility": "ML utility (TSTR)", "relational": "Relational structure"})},
        {"key": "privacy", "label": LABELS["privacy"], "score": prv["score"], "weight": WEIGHTS["privacy"],
         "components": comp_list(prv["components"], {"dcr": "Distance to real records", "exact_match": "Exact copies",
                                                    "membership_inference": "Membership inference"})},
        {"key": "validity", "label": LABELS["validity"], "score": validity, "weight": WEIGHTS["validity"], "components": vcomps},
    ]
    trust = sum(s["score"] * s["weight"] for s in subs)

    gates: list[str] = []
    if prv["metrics"]["exact_matches"] > 0:
        gates.append(f"{prv['metrics']['exact_matches']} synthetic rows are exact copies of real records.")
    if integrity is not None and integrity["total_violations"] > 0:
        gates.append(f"{integrity['total_violations']} referential-integrity violations.")
    if prv["near_duplicates"]["count"] > prv["metrics"]["exact_matches"]:
        gates.append(f"{prv['near_duplicates']['count']} near-duplicates of real records flagged.")

    label = _band(trust)
    weakest = min(subs, key=lambda s: s["score"])
    worst_comp = min(weakest["components"], key=lambda c: c["score"])
    verdict = f"{label}: {trust:.0f}/100. "
    if trust >= 85 and not gates:
        verdict += "This data closely matches the real data's shape, keeps individuals private and follows the rules you set."
    else:
        verdict += (f"The weakest area is {weakest['label'].lower()} ({weakest['score']:.0f}), mainly "
                    f"{worst_comp['label'].lower()}: {worst_comp['summary']}.")
    if gates:
        verdict += " Review before sharing: " + " ".join(gates)

    return {
        "title": title, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trust_score": float(trust), "label": label, "verdict": verdict, "gates": gates,
        "sub_scores": subs, "weights": WEIGHTS,
        "details": {"fidelity_columns": fid["columns"], "privacy": prv["metrics"], "near_duplicates": prv["near_duplicates"],
                    **details},
        "rows": {"real": len(real), "synthetic": len(synthetic)},
    }


# ------------------------------------------------------------ multi-table
RELATION_BLEND = 0.35  # share of relational Fidelity that comes from relation preservation (R5) rather than per-table stats


def build_trust_report_relational(real: dict[str, pd.DataFrame], synthetic: dict[str, pd.DataFrame], graph: Any, *,
                                  relation: dict | None = None, rules_report: dict | None = None, integrity: dict | None = None,
                                  title: str = "Relational dataset", seed: int = 0) -> dict[str, Any]:
    """Trust Score for a multi-table dataset. Same three sub-scores and weights as the single-table card.

    Fidelity  = (1-0.35) * row-weighted per-table fidelity (key columns excluded) + 0.35 * relation-preservation score
    Privacy   = row-weighted per-table privacy (no holdout, so no membership-inference term)
    Validity  = mean of: cross-table rule reconciliation, referential integrity, edge-case coverage
    """
    def attrs(name: str, df: pd.DataFrame) -> pd.DataFrame:
        t = graph.table(name)
        keys = set(t.primary_key) | {c for f in graph.fks_of(name) for c in f.child_columns}
        cols = [c for c in df.columns if c not in keys]
        return df[cols] if cols else df

    per_table, weights = {}, {}
    for name in real:
        r, s = attrs(name, real[name]), attrs(name, synthetic[name])
        if r.shape[1] == 0 or len(r) < 5:
            continue
        per_table[name] = (fidelity_score(r, s), privacy_score(r, s, seed=seed), edge_case_coverage(r, s))
        weights[name] = len(real[name])
    total_w = sum(weights.values()) or 1

    def wmean(idx: int, key: str = "score") -> float:
        return sum((v[idx][key] if isinstance(v[idx], dict) else v[idx]) * weights[k] for k, v in per_table.items()) / total_w

    fid_tables = wmean(0)
    fid_cols = {f"{t}.{c}": v for t, (f, _, _) in per_table.items() for c, v in f["columns"].items()}
    fid_comps = [{"key": "tables", "label": "Table statistics", "score": fid_tables, "weight": 1.0, "summary": f"{len(per_table)} tables, columns compared"}]
    fidelity = fid_tables
    if relation is not None:
        fidelity = (1 - RELATION_BLEND) * fid_tables + RELATION_BLEND * relation["score"]
        fid_comps = [{**fid_comps[0], "weight": 1 - RELATION_BLEND},
                     {"key": "relations", "label": "Relation preservation", "score": relation["score"], "weight": RELATION_BLEND,
                      "summary": "cardinality, join sizes, cross-table effects, FK coverage, JOIN queries"}]

    priv = wmean(1)
    exact = sum(v[1]["metrics"]["exact_matches"] for v in per_table.values())
    near = sum(v[1]["near_duplicates"]["count"] for v in per_table.values())
    priv_comps = [
        {"key": "dcr", "label": "Distance to real records", "score": sum(v[1]["components"]["dcr"]["score"] * weights[k] for k, v in per_table.items()) / total_w,
         "weight": 0.57, "summary": "nearest-neighbour distance, per table"},
        {"key": "exact_match", "label": "Exact copies", "score": sum(v[1]["components"]["exact_match"]["score"] * weights[k] for k, v in per_table.items()) / total_w,
         "weight": 0.43, "summary": f"{exact} synthetic rows exactly copy a real record"}]

    vcomps: list[dict[str, Any]] = []
    if rules_report:
        vcomps.append({"key": "rules", "label": "Cross-table rules", "score": rules_report["reconciliation_pass_rate"],
                       "summary": f"{rules_report['violations_after']} violations after enforcement "
                                  f"(naive generation: {rules_report['pass_rate_before']:.0f}% pass)"})
    if integrity is not None:
        rows = max(1, integrity.get("rows_checked", 1))
        vcomps.append({"key": "integrity", "label": "Referential integrity", "score": 100.0 * max(0.0, 1 - integrity["total_violations"] / rows),
                       "summary": f"{integrity['total_violations']} violations (orphans, duplicate keys, null keys)"})
    cov = wmean(2)
    vcomps.append({"key": "coverage", "label": "Edge-case coverage", "score": cov, "summary": "real edge cases reproduced, averaged over tables"})
    for c in vcomps:
        c["weight"] = 1 / len(vcomps)
    validity = _mean([c["score"] for c in vcomps])

    subs = [{"key": "fidelity", "label": LABELS["fidelity"], "score": fidelity, "weight": WEIGHTS["fidelity"], "components": fid_comps},
            {"key": "privacy", "label": LABELS["privacy"], "score": priv, "weight": WEIGHTS["privacy"], "components": priv_comps},
            {"key": "validity", "label": LABELS["validity"], "score": validity, "weight": WEIGHTS["validity"], "components": vcomps}]
    trust = sum(s["score"] * s["weight"] for s in subs)
    gates: list[str] = []
    if exact:
        gates.append(f"{exact} synthetic rows are exact copies of real records.")
    if integrity is not None and integrity["total_violations"]:
        gates.append(f"{integrity['total_violations']} referential-integrity violations.")
    if rules_report and rules_report["violations_after"]:
        gates.append(f"{rules_report['violations_after']} business-rule violations remain after enforcement.")
    label = _band(trust)
    weakest = min(subs, key=lambda s: s["score"])
    worst = min(weakest["components"], key=lambda c: c["score"])
    verdict = f"{label}: {trust:.0f}/100. " + (
        "The tables look like the real ones, the relationships between them hold, and no real rows leak."
        if trust >= 85 and not gates else
        f"The weakest area is {weakest['label'].lower()} ({weakest['score']:.0f}), mainly {worst['label'].lower()}: {worst['summary']}.")
    if gates:
        verdict += " Review before sharing: " + " ".join(gates)
    return {
        "title": title, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trust_score": float(trust), "label": label, "verdict": verdict, "gates": gates, "sub_scores": subs, "weights": WEIGHTS,
        "details": {"fidelity_columns": fid_cols, "privacy": {"exact_matches": exact, "near_duplicates": near},
                    "relation": relation, "rules": rules_report, "integrity": integrity},
        "rows": {"real": sum(len(v) for v in real.values()), "synthetic": sum(len(v) for v in synthetic.values())},
    }
