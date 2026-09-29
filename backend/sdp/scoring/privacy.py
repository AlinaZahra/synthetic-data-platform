"""Q2. Privacy sub-score: DCR, exact matches, near-duplicate flags, membership-inference AUC.

DCR  = distance from each synthetic record to its closest real training record, in a space where numeric
       columns are standardised on the real data and categoricals are one-hot.
Baseline = how far *unseen real* records (holdout) sit from the training records; without a holdout, the
       leave-one-out nearest-neighbour distance inside the training data.
Membership inference: score = -(distance to nearest synthetic record); AUC of separating training records
       from holdout records. 0.5 = no leakage, 1.0 = every member identified. Needs a holdout.

Sub-score = 0.40*DCR + 0.30*exact-match + 0.30*MIA   (renormalised without MIA when no holdout)
  DCR score   100*min(1, median DCR / median baseline)
  exact score 100*max(0, 1 - 20*exact_match_rate)      (5% exact copies -> 0)
  MIA score   100*(1 - 2*max(0, AUC-0.5))
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from sdp.common import is_identifier_like


class _Encoder:
    def __init__(self, real: pd.DataFrame) -> None:
        self.num, self.cat, self.stats, self.dummies = [], [], {}, []
        for c in real.columns:
            s = real[c]
            if pd.api.types.is_datetime64_any_dtype(s) or (pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)):
                v = self._numeric(s)
                self.num.append(c)
                self.stats[c] = (float(np.nanmedian(v)), float(np.nanstd(v)) or 1.0)
            elif not is_identifier_like(s):  # identifier-like text has no meaningful distance
                self.cat.append(c)
        self.levels = {c: sorted(real[c].dropna().astype(str).unique()) for c in self.cat}

    @staticmethod
    def _numeric(s: pd.Series) -> np.ndarray:
        if pd.api.types.is_datetime64_any_dtype(s):
            t = s.astype("datetime64[ns]")
            return np.where(t.isna(), np.nan, t.astype("int64").to_numpy(dtype=float))
        return s.to_numpy(dtype=float, na_value=np.nan)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        for c in self.num:
            med, sd = self.stats[c]
            v = self._numeric(df[c])
            parts.append(((np.where(np.isnan(v), med, v) - med) / sd)[:, None])
        for c in self.cat:
            vals = df[c].astype(object).where(df[c].notna(), None).map(lambda x: None if x is None else str(x))
            for lv in self.levels[c]:
                parts.append((vals == lv).to_numpy(dtype=float)[:, None])
        return np.hstack(parts) if parts else np.zeros((len(df), 1))


def _row_keys(df: pd.DataFrame) -> pd.Series:
    def norm(s: pd.Series) -> pd.Series:
        return s.round(6) if pd.api.types.is_float_dtype(s) else s
    return pd.util.hash_pandas_object(pd.DataFrame({c: norm(df[c]) for c in df.columns}), index=False)


def privacy_score(real_train: pd.DataFrame, synthetic: pd.DataFrame, real_holdout: pd.DataFrame | None = None,
                  seed: int = 0, near_dup_ratio: float = 0.05, max_rows: int = 5000) -> dict[str, Any]:
    cols = list(real_train.columns)
    rng = np.random.default_rng(seed)

    def cap(df: pd.DataFrame) -> pd.DataFrame:
        return df if len(df) <= max_rows else df.iloc[np.sort(rng.choice(len(df), max_rows, replace=False))]

    train, synth = cap(real_train[cols]).reset_index(drop=True), cap(synthetic[cols]).reset_index(drop=True)
    enc = _Encoder(train)
    X_tr, X_sy = enc.transform(train), enc.transform(synth)

    nn = NearestNeighbors(n_neighbors=1).fit(X_tr)
    d_full, nearest = (a[:, 0] for a in nn.kneighbors(X_sy))  # used to flag individual near-duplicates

    # Baseline: how far do *unseen real* records sit from the reference set? A synthetic record falls inside gaps between
    # real ones, so the fair comparison is real-not-in-reference -> reference, never real -> itself (which would be biased).
    if real_holdout is not None and len(real_holdout):
        hold = cap(real_holdout[cols]).reset_index(drop=True)
        dcr = d_full
        baseline = nn.kneighbors(enc.transform(hold))[0][:, 0]
        baseline_kind = "holdout_to_train"
    else:
        hold = None
        perm = rng.permutation(len(train))
        ref, rest = np.sort(perm[: len(perm) // 2]), perm[len(perm) // 2:]
        nn_ref = NearestNeighbors(n_neighbors=1).fit(X_tr[ref])
        dcr = nn_ref.kneighbors(X_sy)[0][:, 0]           # same-sized reference set as the baseline uses
        baseline = nn_ref.kneighbors(X_tr[rest])[0][:, 0]
        baseline_kind = "train_half_split"
    base_med = float(np.median(baseline))
    # If unseen real rows sit at distance 0 from the training data (many duplicate rows) there is no distance signal
    ratio = float(np.median(dcr)) / base_med if base_med > 0 else 1.0
    base_med = base_med or 1e-9

    # Exact copies are judged against chance. In a low-entropy table (say two dates and a 3-value carrier) synthetic rows
    # coincide with real rows by accident; the real data's own duplicate rate is the yardstick for that. Only the excess
    # over it counts as leakage, and only copies of rows that are UNIQUE in the real data can be flagged individually.
    train_keys = _row_keys(train)
    freq = train_keys.map(train_keys.value_counts()).to_numpy()
    synth_keys = _row_keys(synth).to_numpy()
    matches_any = np.isin(synth_keys, train_keys.to_numpy())
    exact = matches_any & np.isin(synth_keys, train_keys.to_numpy()[freq == 1])
    chance = float((freq > 1).mean())
    exact_rate = max(0.0, float(matches_any.mean()) - chance)
    exact_count = int(round(exact_rate * len(synth)))
    tau = near_dup_ratio * base_med
    near = np.flatnonzero((d_full <= tau) & (freq[nearest] == 1))
    flagged = [{"synthetic_row": int(i), "nearest_real_row": int(nearest[i]), "distance": float(d_full[i]),
                "exact_match": bool(exact[i])} for i in near[:25]]

    mia_auc = None
    if hold is not None:
        n = min(len(train), len(hold))
        mem = X_tr[np.sort(rng.choice(len(train), n, replace=False))]
        non = enc.transform(hold.iloc[np.sort(rng.choice(len(hold), n, replace=False))])
        sy_nn = NearestNeighbors(n_neighbors=1).fit(X_sy)
        d_mem, d_non = sy_nn.kneighbors(mem)[0][:, 0], sy_nn.kneighbors(non)[0][:, 0]
        mia_auc = float(roc_auc_score(np.r_[np.ones(n), np.zeros(n)], -np.r_[d_mem, d_non]))

    comps: dict[str, dict[str, Any]] = {
        "dcr": {"score": 100.0 * min(1.0, ratio), "weight": 0.40,
                "summary": f"median DCR {np.median(dcr):.3f} vs baseline {base_med:.3f} (ratio {ratio:.2f})"},
        "exact_match": {"score": 100.0 * max(0.0, 1.0 - 20.0 * exact_rate), "weight": 0.30,
                        "summary": f"{exact_count} synthetic rows exactly copy a real record (beyond chance)"},
    }
    if mia_auc is not None:
        comps["membership_inference"] = {"score": 100.0 * (1.0 - 2.0 * max(0.0, mia_auc - 0.5)), "weight": 0.30,
                                         "summary": f"attack AUC {mia_auc:.3f} (0.5 = no leakage)"}
    else:
        comps["membership_inference"] = {"score": None, "weight": 0.0, "summary": "skipped: no holdout provided"}
    total_w = sum(c["weight"] for c in comps.values())
    for c in comps.values():
        c["weight"] = c["weight"] / total_w
    score = sum(c["score"] * c["weight"] for c in comps.values() if c["score"] is not None)
    return {
        "score": float(score), "components": comps,
        "metrics": {"median_dcr": float(np.median(dcr)), "min_dcr": float(d_full.min()), "baseline_median": base_med,
                    "baseline_kind": baseline_kind, "dcr_ratio": ratio, "exact_matches": exact_count, "chance_duplicate_rate": chance,
                    "exact_match_rate": exact_rate, "mia_auc": mia_auc, "near_duplicate_threshold": float(tau)},
        "near_duplicates": {"count": int(len(near)), "flagged": flagged},
    }
