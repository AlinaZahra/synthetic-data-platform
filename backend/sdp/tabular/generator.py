"""Tabular generator: Gaussian copula by default, CTGAN optional.

Gaussian copula recipe
  fit:    each column -> uniform scores via its empirical CDF (categoricals occupy an interval
          of [0,1] proportional to their frequency); uniforms -> normal scores; the dependence
          is the (PSD-repaired) correlation matrix of those scores.
  sample: correlated normals -> uniforms -> inverse marginals (quantile function / category
          intervals) -> original dtypes restored.
Marginals, category frequencies and rank correlations are preserved by construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from sdp.common import is_identifier_like, resolve_seed, spawn_rngs
from sdp.tabular.config import GenConfig
from sdp.tabular.postprocess import InjectionRecord, inject

logger = logging.getLogger("sdp.tabular")

Method = Literal["gaussian_copula", "ctgan"]
_MAX_GRID = 1000
_EPS = 1e-6


@dataclass
class ColumnModel:
    name: str
    kind: Literal["numeric", "category", "datetime", "identifier"]
    dtype: Any
    null_rate: float = 0.0
    integer: bool = False
    grid: np.ndarray | None = None  # quantile function at linspace(0, 1, len(grid))
    categories: list | None = None
    probs: np.ndarray | None = None
    id_prefix: str = ""
    id_width: int = 6

    # -- data -> uniform (used at fit time) -----------------------------------
    def to_uniform(self, s: pd.Series, rng: np.random.Generator) -> np.ndarray:
        u = np.full(len(s), np.nan)
        valid = s.notna().to_numpy()
        if self.kind == "identifier":  # independent of every other column
            u[valid] = rng.random(int(valid.sum()))
        elif self.kind == "category":
            index = {c: i for i, c in enumerate(self.categories or [])}
            idx = np.array([index[v] for v in s[valid].tolist()], dtype=int)
            cum = np.cumsum(self.probs)
            u[valid] = (cum - self.probs)[idx] + rng.random(len(idx)) * self.probs[idx]
        else:
            x = _as_float(s[valid], self.kind)
            u[valid] = stats.rankdata(x) / (len(x) + 1)
        return np.clip(u, _EPS, 1 - _EPS)

    # -- uniform -> data (used at sample time) --------------------------------
    def from_uniform(self, u: np.ndarray) -> pd.Series:
        if self.kind == "identifier":  # brand-new values in the real format, never copies of real ones
            n = (np.asarray(u) * 10**self.id_width).astype(np.int64)
            return pd.Series([f"{self.id_prefix}{v:0{self.id_width}d}" for v in n]).astype(self.dtype)
        if self.kind == "category":
            cum = np.cumsum(self.probs)
            idx = np.minimum(np.searchsorted(cum, u, side="right"), len(cum) - 1)
            values = [self.categories[i] for i in idx]
            if isinstance(self.dtype, pd.CategoricalDtype):
                return pd.Series(pd.Categorical(values, dtype=self.dtype))
            return pd.Series(values).astype(self.dtype)
        x = np.interp(u, np.linspace(0, 1, len(self.grid)), self.grid)
        if self.kind == "datetime":
            return pd.Series(pd.to_datetime(np.rint(x).astype("int64"), unit="ns")).astype(self.dtype)
        if self.integer:
            return pd.Series(np.rint(x)).astype(self.dtype)
        return pd.Series(x).astype(self.dtype)


def _as_float(s: pd.Series, kind: str) -> np.ndarray:
    if kind == "datetime":
        return s.astype("datetime64[ns]").astype("int64").to_numpy(dtype=float)
    return s.to_numpy(dtype=float)


def _nearest_corr(c: np.ndarray) -> np.ndarray:
    """Repair to a valid (PSD, unit-diagonal) correlation matrix."""
    c = np.nan_to_num(np.asarray(c, dtype=float))
    np.fill_diagonal(c, 1.0)
    w, v = np.linalg.eigh((c + c.T) / 2)
    c2 = (v * np.clip(w, 1e-6, None)) @ v.T
    d = np.sqrt(np.diag(c2))
    return c2 / np.outer(d, d)


@dataclass
class GenerationResult:
    data: pd.DataFrame
    log: list[InjectionRecord]
    seed: int
    rules: dict | None = None
    edge: dict | None = None

    def log_dicts(self) -> list[dict]:
        return [r.to_dict() for r in self.log]


class TabularGenerator:
    """fit(df) / sample(n, seed) / generate(config); also buildable from a schema."""

    def __init__(self, method: Method = "gaussian_copula", max_discrete_int: int = 10,
                 ctgan_epochs: int = 100) -> None:
        if method not in ("gaussian_copula", "ctgan"):
            raise ValueError(f"unknown method {method!r}")
        self.method = method
        self.max_discrete_int = max_discrete_int
        self.ctgan_epochs = ctgan_epochs
        self.models_: list[ColumnModel] = []
        self.corr_: np.ndarray | None = None
        self._chol: np.ndarray | None = None
        self._ctgan: Any = None
        self._orig_dtypes: dict[str, Any] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame, seed: int = 0) -> "TabularGenerator":
        if df.empty or df.shape[1] == 0:
            raise ValueError("cannot fit on an empty DataFrame")
        self.models_ = [self._profile(df[c]) for c in df.columns]
        self._orig_dtypes = {c: df[c].dtype for c in df.columns}
        if self.method == "ctgan":
            self._fit_ctgan(df, seed)
            return self
        rng = np.random.default_rng(seed)
        z = pd.DataFrame(
            {m.name: stats.norm.ppf(m.to_uniform(df[m.name], rng)) for m in self.models_}
        ).where(df.notna().to_numpy())
        self._set_corr(z.corr(min_periods=2).to_numpy())
        return self

    def _profile(self, s: pd.Series) -> ColumnModel:
        valid = s.dropna()
        if valid.empty:
            raise ValueError(f"column {s.name!r} is entirely null")
        null_rate = float(s.isna().mean())
        common = dict(name=str(s.name), dtype=s.dtype, null_rate=null_rate)
        if pd.api.types.is_datetime64_any_dtype(s):
            return ColumnModel(kind="datetime", grid=self._grid(_as_float(valid, "datetime")), **common)
        is_bool = pd.api.types.is_bool_dtype(s)
        is_int = pd.api.types.is_integer_dtype(s)
        if pd.api.types.is_numeric_dtype(s) and not is_bool:
            if not (is_int and valid.nunique() <= self.max_discrete_int):
                return ColumnModel(kind="numeric", integer=is_int,
                                   grid=self._grid(valid.to_numpy(dtype=float)), **common)
        if is_identifier_like(s):
            vals = valid.astype(str)
            m = vals.str.extract(r"^(.*?)(\d+)$")
            if m[1].notna().mean() >= 0.9:
                prefix, width = str(m[0].mode().iloc[0]), int(m[1].dropna().str.len().mode().iloc[0])
            else:
                prefix, width = "id-", 8
            return ColumnModel(kind="identifier", id_prefix=prefix, id_width=width, **common)
        freq = valid.value_counts(sort=True)
        order = sorted(freq.index.tolist(), key=lambda k: (-freq[k], str(k)))
        probs = np.array([freq[k] for k in order], dtype=float)
        return ColumnModel(kind="category", categories=order, probs=probs / probs.sum(), **common)

    @staticmethod
    def _grid(x: np.ndarray) -> np.ndarray:
        return np.quantile(x, np.linspace(0, 1, min(len(x), _MAX_GRID)))

    def _set_corr(self, corr: np.ndarray) -> None:
        self.corr_ = _nearest_corr(corr)
        self._chol = np.linalg.cholesky(self.corr_ + 1e-10 * np.eye(len(self.corr_)))

    # --------------------------------------------------------------- schema
    @classmethod
    def from_schema(cls, schema: dict[str, dict],
                    correlations: dict[tuple[str, str], float] | None = None) -> "TabularGenerator":
        """Build an already-fitted generator from column specs, no data needed.

        spec examples:
          {"type": "int", "min": 18, "max": 90, "mean": 40, "std": 12}   (mean/std optional -> uniform)
          {"type": "float", "min": 0, "max": 1e5, "mean": 5e4, "std": 2e4}
          {"type": "category", "categories": {"basic": 0.6, "pro": 0.4}}  (or a plain list -> uniform)
          {"type": "datetime", "start": "2020-01-01", "end": "2024-12-31"}
        """
        gen = cls()
        grid_p = np.linspace(0, 1, _MAX_GRID)
        for name, spec in schema.items():
            t = spec["type"]
            if t in ("int", "float"):
                lo, hi = spec.get("min"), spec.get("max")
                if spec.get("mean") is not None and spec.get("std"):
                    grid = spec["mean"] + spec["std"] * stats.norm.ppf(np.clip(grid_p, 0.001, 0.999))
                elif lo is not None and hi is not None:
                    grid = lo + (hi - lo) * grid_p
                else:
                    raise ValueError(f"{name}: need min+max or mean+std")
                grid = np.clip(grid, lo if lo is not None else -np.inf, hi if hi is not None else np.inf)
                dtype = np.dtype("int64") if t == "int" else np.dtype("float64")
                m = ColumnModel(name, "numeric", dtype, integer=t == "int", grid=grid)
            elif t == "category":
                cats = spec["categories"]
                if not isinstance(cats, dict):
                    cats = {c: 1.0 for c in cats}
                p = np.array(list(cats.values()), dtype=float)
                m = ColumnModel(name, "category", object, categories=list(cats), probs=p / p.sum())
            elif t == "datetime":
                a = pd.Timestamp(spec["start"]).value
                b = pd.Timestamp(spec["end"]).value
                m = ColumnModel(name, "datetime", np.dtype("datetime64[ns]"), grid=a + (b - a) * grid_p)
            else:
                raise ValueError(f"{name}: unknown type {t!r}")
            gen.models_.append(m)
        names = [m.name for m in gen.models_]
        corr = np.eye(len(names))
        for (a, b), r in (correlations or {}).items():
            i, j = names.index(a), names.index(b)
            corr[i, j] = corr[j, i] = r
        gen._orig_dtypes = {m.name: m.dtype for m in gen.models_}
        gen._set_corr(corr)
        return gen

    # --------------------------------------------------------------- sample
    def sample(self, n: int, seed: int | None = 0) -> pd.DataFrame:
        """Clean sample (no injected nulls/outliers). Same seed => identical output."""
        if not self.models_:
            raise RuntimeError("call fit() first")
        if self.method == "ctgan":
            return self._sample_ctgan(n, resolve_seed(seed))
        return self._sample_copula(n, np.random.default_rng(resolve_seed(seed)))

    def _sample_copula(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        z = rng.standard_normal((n, len(self.models_))) @ self._chol.T
        u = np.clip(stats.norm.cdf(z), _EPS, 1 - _EPS)
        return pd.DataFrame({m.name: m.from_uniform(u[:, i]) for i, m in enumerate(self.models_)})

    def sample_conditional(self, n: int, fixed: pd.DataFrame, seed: int | None = 0) -> pd.DataFrame:
        """Sample every column NOT in `fixed`, conditioned on the given values (Gaussian-copula conditional).

        `fixed` has n rows. Null or unseen values are left unconditioned for that row. Used to make child rows
        depend on their parent attributes (e.g. order value on customer segment).
        """
        if self.method != "gaussian_copula":
            raise NotImplementedError("conditional sampling needs the Gaussian copula engine")
        rng = np.random.default_rng(resolve_seed(seed))
        names = [m.name for m in self.models_]
        fi = [names.index(c) for c in fixed.columns]
        ci = [i for i in range(len(names)) if i not in fi]
        if not ci:
            return pd.DataFrame(index=range(n))
        zf = np.empty((n, len(fi)))
        for j, i in enumerate(fi):
            m, col = self.models_[i], fixed.iloc[:, j].reset_index(drop=True)
            if m.kind == "category":
                col = col.where(col.isin(m.categories or []))
            u = m.to_uniform(col, rng)
            z = stats.norm.ppf(u)
            bad = np.isnan(z)
            z[bad] = rng.standard_normal(int(bad.sum()))
            zf[:, j] = z
        S = self.corr_
        Sff = S[np.ix_(fi, fi)] + 1e-6 * np.eye(len(fi))
        Scf = S[np.ix_(ci, fi)]
        gain = np.linalg.solve(Sff, Scf.T).T  # |c| x |f|
        cond_cov = S[np.ix_(ci, ci)] - gain @ Scf.T
        cond_cov = (cond_cov + cond_cov.T) / 2 + 1e-8 * np.eye(len(ci))
        L = np.linalg.cholesky(cond_cov)
        zc = zf @ gain.T + rng.standard_normal((n, len(ci))) @ L.T
        u = np.clip(stats.norm.cdf(zc), _EPS, 1 - _EPS)
        return pd.DataFrame({self.models_[i].name: self.models_[i].from_uniform(u[:, k]) for k, i in enumerate(ci)})

    def generate(self, cfg: GenConfig, rules: list[str] | None = None, rule_mode: str = "hybrid") -> GenerationResult:
        """Sample cfg.rows rows (enforcing `rules` if given), then inject nulls/outliers with an independent RNG stream.

        Rules are enforced on the clean sample; nulls/outliers are injected afterwards on purpose, so injected
        defects can break rules (they are what you asked for)."""
        seed = resolve_seed(cfg.seed)
        rng_inject = spawn_rngs(seed, 3)[1]
        rules_report = None
        if rules:
            from sdp.rules.tabular import sample_with_rules
            rs = sample_with_rules(self, cfg.rows, seed, rules, rule_mode)  # type: ignore[arg-type]
            clean = rs.data
            rules_report = {**rs.report.to_dict(), "mode": rule_mode, "rows_dropped": rs.rows_dropped,
                            "naive_violations": rs.naive_violations, "compiled": rs.compiled}
        else:
            clean = self.sample(cfg.rows, seed)
        data, log = inject(clean, cfg, rng_inject)
        edge_report = None
        if cfg.edge_cases:
            from sdp.edgecases import inject_edge_cases
            res = inject_edge_cases(data, cfg.edge_cases, seed=int(spawn_rngs(seed, 3)[2].integers(0, 2**31 - 1)))
            data, edge_report = res.data, res.report
        return GenerationResult(data=data, log=log, seed=seed, rules=rules_report, edge=edge_report)

    # ---------------------------------------------------------------- CTGAN
    def _fit_ctgan(self, df: pd.DataFrame, seed: int) -> None:
        try:
            from ctgan import CTGAN  # type: ignore
            import torch  # type: ignore
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("method='ctgan' needs `pip install ctgan`") from e
        torch.manual_seed(seed)
        discrete = [m.name for m in self.models_ if m.kind == "category"]
        self._ctgan = CTGAN(epochs=self.ctgan_epochs, verbose=False)
        self._ctgan.fit(df.dropna(), discrete_columns=discrete)

    def _sample_ctgan(self, n: int, seed: int) -> pd.DataFrame:  # pragma: no cover
        import torch  # type: ignore

        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))
        out = self._ctgan.sample(n)
        for c, dt in self._orig_dtypes.items():
            out[c] = out[c].astype(dt) if not pd.api.types.is_integer_dtype(dt) else np.rint(out[c]).astype(dt)
        return out[list(self._orig_dtypes)]

    def null_rates(self) -> dict[str, float]:
        """Null rates seen in the training data (handy input for GenConfig.null_rate)."""
        return {m.name: m.null_rate for m in self.models_ if m.null_rate > 0}
