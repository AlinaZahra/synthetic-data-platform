"""Children-per-parent distributions: learn from real data or set Poisson / Zipf; compare afterwards."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats


class CardinalityConfig(BaseModel):
    """How many child rows each parent gets.

    learned  resample per-parent counts from the real data (preserves the distribution)
    poisson  Poisson(lam)
    zipf     Zipf(a), a > 1: many parents with 1 child, a heavy tail of very busy parents
    fixed    exactly `value` children per parent
    Counts are clipped to [min_children, max_children].
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["learned", "poisson", "zipf", "fixed"] = "learned"
    lam: float = Field(3.0, gt=0)
    a: float = Field(2.0, gt=1)
    value: int = Field(1, ge=0)
    min_children: int = Field(0, ge=0)
    max_children: int | None = Field(None, ge=1)


def sample_counts(cfg: CardinalityConfig, learned: np.ndarray | None, n_parents: int,
                  rng: np.random.Generator) -> np.ndarray:
    if cfg.kind == "learned":
        if learned is None or len(learned) == 0:
            raise ValueError("kind='learned' needs a fitted generator (no real counts available)")
        counts = rng.choice(learned, size=n_parents, replace=True)
    elif cfg.kind == "poisson":
        counts = rng.poisson(cfg.lam, size=n_parents)
    elif cfg.kind == "zipf":
        counts = rng.zipf(cfg.a, size=n_parents)
    else:
        counts = np.full(n_parents, cfg.value)
    hi = cfg.max_children if cfg.max_children is not None else (10_000 if cfg.kind == "zipf" else None)
    return np.clip(counts, cfg.min_children, hi).astype(int)


def fit_to_total(counts: np.ndarray, total: int, rng: np.random.Generator) -> np.ndarray:
    """Nudge per-parent counts (by +/-1 at random parents) until they sum to `total`."""
    counts = counts.copy()
    diff = int(total - counts.sum())
    while diff != 0 and len(counts):
        if diff > 0:
            np.add.at(counts, rng.integers(0, len(counts), size=diff), 1)
        else:
            eligible = np.flatnonzero(counts > 0)
            if len(eligible) == 0:
                break
            pick = rng.choice(eligible, size=min(-diff, len(eligible)), replace=False)
            counts[pick] -= 1
        diff = int(total - counts.sum())
    return counts


def compare_distributions(real: np.ndarray, synth: np.ndarray) -> dict[str, float]:
    """How closely the synthetic children-per-parent histogram matches the real one.

    match_score = 1 - total-variation distance between the two count histograms (1.0 = identical);
    exact per-count histogram for narrow supports, real-quantile bins for wide ones.
    """
    if len(real) == 0 or len(synth) == 0:
        return {"match_score": 0.0}
    if len(np.unique(real)) <= 20:
        k = int(max(real.max(), synth.max())) + 1
        pr = np.bincount(real, minlength=k) / len(real)
        ps = np.bincount(synth, minlength=k) / len(synth)
    else:  # wide support: compare mass in quantile bins of the real distribution (per-value pmf is just noise)
        edges = np.unique(np.quantile(real, np.linspace(0, 1, 11)))[1:-1]
        pr = np.bincount(np.digitize(real, edges, right=True), minlength=len(edges) + 1) / len(real)
        ps = np.bincount(np.digitize(synth, edges, right=True), minlength=len(edges) + 1) / len(synth)
    tvd = 0.5 * float(np.abs(pr - ps).sum())
    return {
        "match_score": 1.0 - tvd,
        "tvd": tvd,
        "ks_statistic": float(stats.ks_2samp(real, synth).statistic),
        "mean_real": float(real.mean()), "mean_synth": float(synth.mean()),
        "std_real": float(real.std()), "std_synth": float(synth.std()),
        "max_real": int(real.max()), "max_synth": int(synth.max()),
        "zero_frac_real": float((real == 0).mean()), "zero_frac_synth": float((synth == 0).mean()),
    }
