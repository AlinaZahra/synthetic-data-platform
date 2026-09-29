"""Fidelity metrics shared by tests and the scorecard: KS (numeric), chi-square (categorical), correlation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from sdp.common import is_identifier_like


def ks_report(real: pd.DataFrame, synth: pd.DataFrame) -> dict[str, dict[str, float]]:
    out = {}
    for c in real.columns:
        if pd.api.types.is_numeric_dtype(real[c]) and not pd.api.types.is_bool_dtype(real[c]):
            r, s = real[c].dropna().astype(float), synth[c].dropna().astype(float)
            res = stats.ks_2samp(r, s)
            out[c] = {"statistic": float(res.statistic), "p_value": float(res.pvalue)}
    return out


def chi_square_report(real: pd.DataFrame, synth: pd.DataFrame, max_categories: int = 50) -> dict[str, dict[str, float]]:
    """Goodness of fit of synthetic counts against the real category frequencies."""
    out = {}
    for c in real.columns:
        if pd.api.types.is_numeric_dtype(real[c]) and not pd.api.types.is_bool_dtype(real[c]) \
                and real[c].nunique() > max_categories:
            continue
        if pd.api.types.is_datetime64_any_dtype(real[c]):
            continue
        if pd.api.types.is_float_dtype(real[c]):
            continue
        freq = real[c].value_counts(normalize=True)
        if not 1 < len(freq) <= max_categories:
            continue
        obs = synth[c].value_counts().reindex(freq.index, fill_value=0).to_numpy(dtype=float)
        exp = freq.to_numpy() * obs.sum()
        stat, p = stats.chisquare(obs, exp * obs.sum() / exp.sum())
        out[c] = {"statistic": float(stat), "p_value": float(p),
                  "max_abs_freq_diff": float(np.max(np.abs(obs / obs.sum() - freq.to_numpy())))}
    return out


def correlation_gap(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    """Mean absolute difference of Spearman correlations over numeric columns (0 = identical)."""
    cols = [c for c in real.columns if pd.api.types.is_numeric_dtype(real[c]) and not pd.api.types.is_bool_dtype(real[c])]
    if len(cols) < 2:
        return 0.0
    a = real[cols].astype(float).corr("spearman").to_numpy()
    b = synth[cols].astype(float).corr("spearman").to_numpy()
    iu = np.triu_indices(len(cols), 1)
    return float(np.nanmean(np.abs(a[iu] - b[iu])))


def fidelity_report(real: pd.DataFrame, synth: pd.DataFrame) -> dict:
    ks = ks_report(real, synth)
    chi = chi_square_report(real, synth)
    return {
        "ks": ks,
        "chi_square": chi,
        "correlation_gap": correlation_gap(real, synth),
        "mean_ks_statistic": float(np.mean([v["statistic"] for v in ks.values()])) if ks else None,
        "dtypes_match": all(str(real[c].dtype) == str(synth[c].dtype) for c in real.columns),
    }


def overlay_data(real: pd.DataFrame, synth: pd.DataFrame, bins: int = 20, max_categories: int = 10) -> dict[str, dict]:
    """Per-column distributions for the real-vs-synthetic overlay charts (fractions summing to 1 per series)."""
    out: dict[str, dict] = {}
    for c in real.columns:
        r, s = real[c].dropna(), synth[c].dropna()
        if r.empty or s.empty or is_identifier_like(real[c]):
            continue
        dt = pd.api.types.is_datetime64_any_dtype(r)
        if dt or (pd.api.types.is_numeric_dtype(r) and not pd.api.types.is_bool_dtype(r) and r.nunique() > 10):
            rv = r.astype("datetime64[ns]").astype("int64").to_numpy(dtype=float) if dt else r.to_numpy(dtype=float)
            sv = s.astype("datetime64[ns]").astype("int64").to_numpy(dtype=float) if dt else s.to_numpy(dtype=float)
            lo, hi = float(rv.min()), float(rv.max())
            if hi == lo:
                hi = lo + 1.0
            edges = np.linspace(lo, hi, bins + 1)
            hr = np.histogram(np.clip(rv, lo, hi), edges)[0] / len(rv)
            hs = np.histogram(np.clip(sv, lo, hi), edges)[0] / len(sv)
            labels = [str(pd.Timestamp(int(e)).date()) for e in edges] if dt else [float(e) for e in edges]
            out[c] = {"kind": "datetime" if dt else "numeric", "edges": labels, "real": hr.tolist(), "synthetic": hs.tolist()}
        else:
            fr = r.astype(str).value_counts(normalize=True)
            top = list(fr.index[:max_categories])
            fs = s.astype(str).value_counts(normalize=True)
            real_v = [float(fr[k]) for k in top] + ([float(fr.iloc[max_categories:].sum())] if len(fr) > max_categories else [])
            syn_v = [float(fs.get(k, 0.0)) for k in top] + ([float(fs[~fs.index.isin(top)].sum())] if len(fr) > max_categories else [])
            out[c] = {"kind": "categorical", "categories": top + (["other"] if len(fr) > max_categories else []), "real": real_v, "synthetic": syn_v}
    return out
