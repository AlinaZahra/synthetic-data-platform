"""Q1. Fidelity sub-score (0-100) with per-column drill-down.

Components (weights renormalised over the ones available):
  marginals     0.40  per column: numeric/datetime -> 100*(1-KS statistic); categorical -> 100*(1-TVD of frequencies)
  correlations  0.25  100*(1 - mean|Δ Spearman| / 0.30), floored at 0
  utility       0.20  TSTR: 100 - mean primary-metric gap % vs the real baseline (gap<=0 -> 100)
  relational    0.15  mean children-per-parent match (relational scorecard) * 100
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from sdp.common import is_identifier_like
from sdp.tabular.quality import correlation_gap

WEIGHTS = {"marginals": 0.40, "correlations": 0.25, "utility": 0.20, "relational": 0.15}
CORR_TOLERANCE = 0.30


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def column_scores(real: pd.DataFrame, synth: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for c in real.columns:
        r, s = real[c], synth[c]
        entry: dict[str, Any] = {
            "dtype_real": str(r.dtype), "dtype_synthetic": str(s.dtype),
            "null_rate_real": float(r.isna().mean()), "null_rate_synthetic": float(s.isna().mean()),
        }
        if is_identifier_like(r):  # names/ids: new values are the point, so distribution distance is not meaningful
            entry.update(kind="identifier", test="none", score=None)
        elif _is_numeric(r) or pd.api.types.is_datetime64_any_dtype(r):
            a, b = _num(r), _num(s)
            if len(a) == 0 or len(b) == 0:
                entry.update(kind="numeric", test="ks", score=0.0)
            else:
                res = stats.ks_2samp(a, b)
                entry.update(kind="numeric", test="ks", statistic=float(res.statistic),
                             p_value=float(res.pvalue), score=100.0 * (1.0 - float(res.statistic)))
        else:
            fr = r.dropna().astype(str).value_counts(normalize=True)
            fs = s.dropna().astype(str).value_counts(normalize=True)
            idx = fr.index.union(fs.index)
            tvd = 0.5 * float(np.abs(fr.reindex(idx, fill_value=0) - fs.reindex(idx, fill_value=0)).sum())
            entry.update(kind="categorical", test="tvd", statistic=tvd, score=100.0 * (1.0 - tvd),
                         missing_categories=sorted(set(fr.index) - set(fs.index))[:10])
        out[c] = entry
    return out


def _num(s: pd.Series) -> np.ndarray:
    s = s.dropna()
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.astype("datetime64[ns]").astype("int64").to_numpy(dtype=float)
    return s.to_numpy(dtype=float)


def fidelity_score(real: pd.DataFrame, synth: pd.DataFrame, tstr: dict | None = None,
                   relational: dict | None = None) -> dict[str, Any]:
    cols = column_scores(real, synth)
    scored = {k: v for k, v in cols.items() if v["score"] is not None}
    comps: dict[str, dict[str, Any]] = {
        "marginals": {"score": float(np.mean([v["score"] for v in scored.values()])) if scored else 100.0,
                      "summary": f"{len(scored)} columns compared; worst: {min(scored, key=lambda k: scored[k]['score'])}" if scored else "no comparable columns"}
    }
    num = [c for c in real.columns if _is_numeric(real[c])]
    if len(num) >= 2:
        gap = correlation_gap(real, synth)
        comps["correlations"] = {"score": float(max(0.0, 100.0 * (1 - gap / CORR_TOLERANCE))),
                                 "summary": f"mean |Δ Spearman| = {gap:.3f}", "mean_abs_gap": gap}
    if tstr and tstr.get("summary", {}).get("mean_gap_pct") is not None:
        g = tstr["summary"]["mean_gap_pct"]
        comps["utility"] = {"score": float(np.clip(100.0 - max(0.0, g), 0, 100)),
                            "summary": f"TSTR {tstr['summary']['primary_metric']} gap {g:.1f}% vs real baseline"}
    if relational and relational.get("score") is not None:  # full R5 relation-preservation metrics
        comps["relational"] = {"score": float(relational["score"]), "summary": f"relation-preservation score {relational['score']:.0f}/100"}
    elif relational and relational.get("mean_cardinality_match") is not None:
        m = relational["mean_cardinality_match"]
        comps["relational"] = {"score": float(100.0 * m), "summary": f"children-per-parent match {m * 100:.0f}%"}
    total_w = sum(WEIGHTS[k] for k in comps)
    for k, v in comps.items():
        v["weight"] = WEIGHTS[k] / total_w
    score = sum(v["score"] * v["weight"] for v in comps.values())
    return {"score": float(score), "components": comps, "columns": cols}
