"""Visualize: small, aggregated JSON for charts. NEVER row-level data.

`resolve(dataset_id)` turns a built-in dataset name or a saved job id into a `Bundle` (real tables, synthetic tables, relationship graph). Each
`payload(...)` view then returns only summaries: histogram bins, category shares, correlation matrices, orphan counts, pass rates. Guards:
identifier-like columns are refused, rare categories (< MIN_CELL real rows) are merged into "other", and sizes are capped (bins, categories,
matrix size, rows), so a response is a few KB whatever the dataset size.

Views: distribution, categorical, correlation, tstr, cardinality, privacy_distance, locale_validity, batch_summary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from sdp.common import is_identifier_like

TYPES = ("distribution", "categorical", "correlation", "tstr", "cardinality", "privacy_distance", "locale_validity", "batch_summary")
MIN_CELL = 5          # categories with fewer real rows are merged into "other" (small counts can identify people)
MAX_BINS, MAX_CATS, MAX_MATRIX, MAX_ROWS = 60, 25, 15, 20_000
JOB_ID = re.compile(r"^\d{14}-[0-9a-f]{6}$")
NL_DOMAINS = ("bank_customers", "ecommerce_customers")
LOCALE_LABELS = {"phone": "Phone numbers", "national_id": "National ID numbers", "name_script": "Names in the right script", "currency": "Currency amounts",
                 "date": "Dates in local format", "unicode_nfc": "Text encoding (NFC)"}


class VisualizeError(ValueError):
    """The request cannot be answered (bad column, wrong dataset kind...). Mapped to HTTP 422."""


class DatasetNotFound(KeyError):
    pass


class NotReady(RuntimeError):
    """The dataset is a saved job that has not finished yet (HTTP 409)."""


@dataclass
class Bundle:
    id: str
    kind: str                                  # tabular | relational | nl | documents
    title: str
    source: str                                # built-in | job
    synth: dict[str, pd.DataFrame] = field(default_factory=dict)
    real: dict[str, pd.DataFrame] = field(default_factory=dict)
    graph: Any = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def has_real(self) -> bool:
        return bool(self.real)

    @property
    def types(self) -> list[str]:
        if self.kind == "tabular":
            return ["distribution", "categorical", "correlation", "tstr", "privacy_distance"]
        if self.kind == "relational":
            return ["distribution", "categorical", "correlation", "cardinality"]
        if self.kind == "nl":
            return ["distribution", "categorical", "cardinality", "locale_validity"]
        return ["batch_summary"]


# ------------------------------------------------------------ resolving
def _clip_rows(df: pd.DataFrame, n: int = MAX_ROWS) -> pd.DataFrame:
    return df if len(df) <= n else df.iloc[np.linspace(0, len(df) - 1, n).astype(int)].reset_index(drop=True)


@lru_cache(maxsize=8)
def _tabular_builtin(name: str, rows: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sdp.datasets import TABULAR_SAMPLES
    from sdp.tabular import TabularGenerator
    real = TABULAR_SAMPLES[name][0](2000, seed=0)
    return real, TabularGenerator().fit(real).sample(rows, seed=seed)


@lru_cache(maxsize=4)
def _relational_builtin(name: str, seed: int) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], Any]:
    from sdp.datasets import SHOP_FULL_RULES, make_shop, make_shop_full
    from sdp.relational import RelationalGenerator, infer_graph
    real = make_shop_full(400, seed=0) if name == "shop_full" else make_shop(300, seed=0)
    graph = infer_graph(real)
    res = RelationalGenerator(graph).fit(real, seed=seed).generate(seed=seed, rules=SHOP_FULL_RULES if name == "shop_full" else None)
    return real, res.tables, graph


@lru_cache(maxsize=4)
def _nl_builtin(domain: str, locale: str, rows: int, seed: int) -> tuple[dict[str, pd.DataFrame], Any, dict[str, Any]]:
    from sdp.nl import DatasetConfig, generate_dataset
    from sdp.nl.parser import domains
    dom = domains()[domain]
    cfg = DatasetConfig(domain=domain, locale=locale, rows=rows, seed=seed, flag={"name": dom["flag"]["name"], "rate": 0.03},
                        history_months=6 if "history" in dom else None, schema_columns=dom["columns"], rules=dom["rules"])
    ds = generate_dataset(cfg)
    return ds.tables, ds.graph, ds.validate()


def _read_csv(store: Any, jid: str, name: str) -> pd.DataFrame:
    from sdp.service import _read_csv as rc
    return rc(store, jid, name)


def resolve(dataset_id: str, store: Any = None, rows: int = 1000, seed: int = 0, locale: str = "ur-PK") -> Bundle:
    rows = int(min(max(rows, 100), MAX_ROWS))
    from sdp.datasets import TABULAR_SAMPLES
    if dataset_id in TABULAR_SAMPLES:
        real, synth = _tabular_builtin(dataset_id, rows, seed)
        return Bundle(dataset_id, "tabular", TABULAR_SAMPLES[dataset_id][2], "built-in", {"data": synth}, {"data": real},
                      meta={"target": TABULAR_SAMPLES[dataset_id][1], "seed": seed})
    if dataset_id in ("shop", "shop_full"):
        real, synth, graph = _relational_builtin(dataset_id, seed)
        return Bundle(dataset_id, "relational", "Sample shop (linked tables)", "built-in", synth, real, graph, {"seed": seed})
    if dataset_id in NL_DOMAINS:
        tables, graph, val = _nl_builtin(dataset_id, locale, rows, seed)
        return Bundle(dataset_id, "nl", f"Described: {dataset_id.replace('_', ' ')} ({locale})", "built-in", tables, {}, graph, {"validation": val, "locale": locale})
    if dataset_id == "documents-demo":
        return Bundle(dataset_id, "documents", "Sample batch of 40 documents (2 deliberately invalid)", "built-in", meta={"report": _documents_demo()})
    if store is not None and JOB_ID.match(dataset_id):
        return _from_job(store, dataset_id)
    raise DatasetNotFound(dataset_id)


def _from_job(store: Any, jid: str) -> Bundle:
    import json
    from sdp import service
    from sdp.lineage import JobNotFound
    try:
        m = store.read(jid)
    except JobNotFound as e:
        raise DatasetNotFound(jid) from e
    if m["status"] in ("queued", "running", "cancelling"):
        raise NotReady(f"job {jid} is still {m['status']}; try again in a moment")
    if m["status"] != "succeeded":
        raise VisualizeError(f"job {jid} is {m['status']}; there is nothing to visualize")
    kind, title = m["kind"], f"Saved run {jid} ({m['kind']})"
    if kind in ("tabular", "large"):
        p = (service.TabularParams if kind == "tabular" else service.LargeParams).model_validate(m["request"])
        real = service._real_tabular(store, jid, p)
        name = "synthetic.csv" if kind == "tabular" else next(f["name"] for f in m["outputs"]["files"] if f["name"].startswith("synthetic."))
        if not name.endswith(".csv"):
            raise VisualizeError("only CSV outputs of large runs can be visualized")
        synth = service._read_csv(store, jid, name) if kind == "tabular" else _dates(pd.read_csv(store.path(jid) / name, nrows=MAX_ROWS))
        synth = synth[[c for c in synth.columns if not str(c).startswith("_")]]
        keep = [c for c in real.columns if c in synth.columns]
        return Bundle(jid, "tabular", title, "job", {"data": _clip_rows(synth[keep])}, {"data": real[keep]}, meta={"target": None, "seed": p.seed, "job": True})
    if kind == "relational":
        p = service.RelationalParams.model_validate(m["request"])
        real = service._real_relational(store, jid, p)
        from sdp.relational import infer_graph
        synth = {n: _read_csv(store, jid, f"{n}.csv") for n in real}
        return Bundle(jid, "relational", title, "job", synth, real, infer_graph(real), {"job": True})
    if kind == "nl":
        val = json.loads(store.artifact(jid, "validation.json"))
        cfg = json.loads(store.artifact(jid, "config.json"))
        names = [f["name"][:-4] for f in m["outputs"]["files"] if f["name"].endswith(".csv")]
        synth = {n: _read_csv(store, jid, f"{n}.csv") for n in names}
        from sdp.nl import DatasetConfig, generate_dataset
        graph = generate_dataset(DatasetConfig.model_validate({**cfg, "rows": min(cfg["rows"], 100)})).graph      # only for the relationship structure
        return Bundle(jid, "nl", title, "job", synth, {}, graph, {"validation": val, "locale": cfg.get("locale")})
    if kind == "document":
        return Bundle(jid, "documents", title, "job", meta={"report": json.loads(store.artifact(jid, "report.json"))})
    raise VisualizeError(f"jobs of kind {kind!r} have no charts")


def _dates(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) in ("str", "string"):
            try:
                df[c] = pd.to_datetime(df[c], errors="raise", format="ISO8601")
            except (ValueError, TypeError):
                pass
    return df


@lru_cache(maxsize=2)
def _documents_demo() -> dict[str, Any]:
    from sdp.documents.pipeline import DocumentPipeline
    def spec(i: int) -> dict[str, Any]:
        t = ("invoice", "statement", "payslip", "retail_receipt")[i % 4]
        loc = ("en-US", "en-GB", "fr", "es")[(i // 4) % 4]
        extra = {"n_transactions": 15} if t == "statement" else {} if t == "payslip" else {"n_lines": 4}
        return {"doc_type": t, "seed": i, "locale": loc, **extra}
    specs = [spec(i) for i in range(38)]
    specs += [{"doc_type": "invoice", "seed": 100, "locale": "xx"}, {"doc_type": "invoice", "seed": 101, "n_lines": -3}]   # two deliberately invalid: shows failures are isolated
    rep = DocumentPipeline(workers=4, batch_size=20).run(specs).to_dict()
    rep.pop("started_at", None)
    return rep


# ------------------------------------------------------------- helpers
def _pick_table(b: Bundle, table: str | None) -> str:
    if table is None:
        table = next(iter(b.synth))
    if table not in b.synth:
        raise VisualizeError(f"unknown table {table!r}; available: {sorted(b.synth)}")
    return table


def _frames(b: Bundle, table: str | None, usable: Any = None) -> tuple[str, pd.DataFrame | None, pd.DataFrame]:
    """(table name, real frame, synthetic frame). With no table given, the first table for which `usable(df)` holds is chosen (so a shop opens on
    a table that actually has numbers to chart, not on a table of names)."""
    if table is None and usable is not None:
        table = next((n for n, df in b.synth.items() if usable(df)), None)
    t = _pick_table(b, table)
    real = b.real.get(t)
    return t, real, b.synth[t]


def _chartable_numeric(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if _numericish(df[c]) and not is_identifier_like(df[c]) and df[c].nunique() > 10 and not str(c).lower().endswith("id")
            and not (pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() == df[c].notna().sum())]


def _correlatable(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]) and not is_identifier_like(df[c])
            and df[c].nunique() > 2 and not str(c).lower().endswith("_id")]


def _numericish(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s) or (pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s))


def _as_float(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dropna().astype("datetime64[ns]").astype("int64").to_numpy(dtype=float)
    return s.dropna().to_numpy(dtype=float)


def _fmt(x: float, dt: bool) -> str:
    if dt:
        return str(pd.Timestamp(int(x)).date())
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.3g}"


def _describe(v: np.ndarray, dt: bool) -> dict[str, Any]:
    if not len(v):
        return {}
    f = (lambda x: str(pd.Timestamp(int(x)).date())) if dt else (lambda x: round(float(x), 4))
    return {"n": int(len(v)), "min": f(v.min()), "median": f(np.median(v)), "mean": f(v.mean()), "max": f(v.max())} | ({} if dt else {"std": round(float(v.std()), 4)})


def _envelope(b: Bundle, typ: str, **body: Any) -> dict[str, Any]:
    return {"dataset_id": b.id, "type": typ, "kind": b.kind, "title": b.title, "source": b.source, "has_real": b.has_real, **body}


def _tvd(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


# -------------------------------------------------------------- views
def distribution(b: Bundle, column: str | None = None, table: str | None = None, bins: int = 20) -> dict[str, Any]:
    t, real, synth = _frames(b, table, lambda df: bool(_chartable_numeric(df)))
    bins = int(min(max(bins, 5), MAX_BINS))
    cols = [c for c in synth.columns if _numericish(synth[c]) and not is_identifier_like(synth[c]) and synth[c].nunique() > 10
            and not str(c).lower().endswith("id")
            and not (pd.api.types.is_integer_dtype(synth[c]) and synth[c].nunique() == synth[c].notna().sum())]     # not keys: an ID has no distribution worth drawing
    if not cols:
        raise VisualizeError("this table has no numeric or date column with enough distinct values for a histogram")
    column = column or cols[0]
    if column not in synth.columns:
        raise VisualizeError(f"unknown column {column!r}; available: {cols}")
    if column not in cols:
        raise VisualizeError(f"{column!r} is not a numeric/date column with enough distinct values; use type=categorical instead")
    dt = pd.api.types.is_datetime64_any_dtype(synth[column])
    sv = _as_float(synth[column])
    rv = _as_float(real[column]) if real is not None and column in real.columns else None
    allv = sv if rv is None else np.concatenate([sv, rv])
    lo, hi = float(allv.min()), float(allv.max())
    hi = hi if hi > lo else lo + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    hs = np.histogram(sv, edges)[0] / max(len(sv), 1)
    hr = np.histogram(rv, edges)[0] / max(len(rv), 1) if rv is not None else None
    rows = [{"x0": _fmt(edges[i], dt), "x1": _fmt(edges[i + 1], dt), "label": _fmt((edges[i] + edges[i + 1]) / 2, dt),
             "real": None if hr is None else round(float(hr[i]), 5), "synthetic": round(float(hs[i]), 5)} for i in range(bins)]
    overlap = None if hr is None else round(100 * (1 - _tvd(hr, hs)), 1)
    cap = (f"Each bar shows the share of rows in a range of {column}. Where the two colours line up, the synthetic data has the same shape as the real data"
           + (f" (about {overlap:.0f}% overlap)." if overlap is not None else ".") if hr is not None else
           f"Each bar shows the share of rows in a range of {column}. There is no real data to compare with for this dataset.")
    return _envelope(b, "distribution", table=t, column=column, columns=cols, tables=sorted(b.synth), kind_of_column="date" if dt else "number",
                     bins=rows, overlap_pct=overlap, stats={"real": _describe(rv, dt) if rv is not None else None, "synthetic": _describe(sv, dt)}, caption=cap)


def categorical(b: Bundle, column: str | None = None, table: str | None = None, limit: int = 12) -> dict[str, Any]:
    t, real, synth = _frames(b, table, lambda df: any((not _numericish(df[c]) or df[c].nunique() <= 10) and not is_identifier_like(df[c]) and df[c].nunique() <= 500 for c in df.columns))
    limit = int(min(max(limit, 3), MAX_CATS))
    ref = real if real is not None else synth
    cols = [c for c in synth.columns if not _numericish(synth[c]) or synth[c].nunique() <= 10]
    cols = [c for c in cols if c in ref.columns and not is_identifier_like(ref[c]) and ref[c].nunique() <= 500]
    if not cols:
        raise VisualizeError("this table has no category-like column (identifier-like columns are never charted)")
    column = column or cols[0]
    if column not in synth.columns:
        raise VisualizeError(f"unknown column {column!r}; available: {cols}")
    if column not in cols:
        raise VisualizeError(f"{column!r} looks like an identifier or has too many distinct values; it is not charted")
    sc = synth[column].dropna().astype(str).value_counts()
    rc = real[column].dropna().astype(str).value_counts() if real is not None and column in real.columns else None
    base = rc if rc is not None else sc
    keep = [k for k in base.index[:limit] if base[k] >= MIN_CELL]
    ns, nr = int(sc.sum()), int(rc.sum()) if rc is not None else 0
    cats = [{"label": k, "real": None if rc is None else round(float(rc.get(k, 0)) / nr, 5), "synthetic": round(float(sc.get(k, 0)) / ns, 5),
             "real_count": None if rc is None else int(rc.get(k, 0)), "synthetic_count": int(sc.get(k, 0))} for k in keep]
    rest_s, rest_r = int(sc[~sc.index.isin(keep)].sum()), int(rc[~rc.index.isin(keep)].sum()) if rc is not None else 0
    if rest_s or rest_r:
        cats.append({"label": "other (rare values)", "real": None if rc is None else round(rest_r / nr, 5), "synthetic": round(rest_s / ns, 5),
                     "real_count": None if rc is None else rest_r, "synthetic_count": rest_s})
    tvd = None
    if rc is not None:
        allk = sorted(set(rc.index) | set(sc.index))
        tvd = round(100 * (1 - _tvd(np.array([rc.get(k, 0) / nr for k in allk]), np.array([sc.get(k, 0) / ns for k in allk]))), 1)
    cap = (f"Each bar is how common a value of {column} is. Bars of the two colours that match mean the synthetic data keeps the same mix"
           + (f" (about {tvd:.0f}% match)." if tvd is not None else ".") + f" Values seen in fewer than {MIN_CELL} real rows are grouped as 'other'.")
    return _envelope(b, "categorical", table=t, column=column, columns=cols, tables=sorted(b.synth), categories=cats, match_pct=tvd, caption=cap)


def correlation(b: Bundle, table: str | None = None) -> dict[str, Any]:
    t, real, synth = _frames(b, table, lambda df: len(_correlatable(df)) >= 2)
    if real is None:
        raise VisualizeError("correlation needs real data to compare with; this dataset was described in words")
    cols = [c for c in real.columns if c in synth.columns and pd.api.types.is_numeric_dtype(real[c]) and not pd.api.types.is_bool_dtype(real[c])
            and not is_identifier_like(real[c]) and real[c].nunique() > 2 and not str(c).lower().endswith("_id")]
    cols = cols[:MAX_MATRIX]
    if len(cols) < 2:
        raise VisualizeError("need at least two numeric columns to compare correlations")

    def mat(df: pd.DataFrame) -> np.ndarray:
        return np.nan_to_num(df[cols].corr(method="spearman").to_numpy(dtype=float))
    r, s = mat(real), mat(synth)
    d = s - r
    iu = np.triu_indices(len(cols), 1)
    worst = sorted(zip(np.abs(d[iu]), iu[0], iu[1]), reverse=True)[:3]
    mad = float(np.abs(d[iu]).mean())
    rnd = lambda m: [[round(float(x), 3) for x in row] for row in m]  # noqa: E731
    cap = ("Each square is how strongly two columns move together (1 = together, -1 = opposite). The third grid shows real minus synthetic: pale means the "
           f"relationship was kept. On average the gap is {mad:.2f}" + (f"; the biggest is {cols[worst[0][1]]} with {cols[worst[0][2]]}." if worst else "."))
    return _envelope(b, "correlation", table=t, tables=sorted(b.synth), columns=cols, real_matrix=rnd(r), synthetic_matrix=rnd(s), diff_matrix=rnd(d),
                     mean_abs_diff=round(mad, 4), max_abs_diff=round(float(np.abs(d[iu]).max()), 4),
                     worst_pairs=[{"a": cols[i], "b": cols[j], "real": round(float(r[i, j]), 3), "synthetic": round(float(s[i, j]), 3)} for _, i, j in worst], caption=cap)


def tstr(b: Bundle, target: str | None = None, seed: int = 0) -> dict[str, Any]:
    from sklearn.model_selection import train_test_split
    from sdp.evaluation import evaluate_tstr
    from sdp.tabular import TabularGenerator
    if b.kind != "tabular":
        raise VisualizeError("the utility (TSTR) chart is for single tables")
    real, synth = b.real["data"], b.synth["data"]
    target = target or b.meta.get("target") or real.columns[-1]
    if target not in real.columns:
        raise VisualizeError(f"unknown target {target!r}; available: {list(real.columns)}")
    tr, te = train_test_split(real, test_size=0.3, random_state=seed)
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
    syn = TabularGenerator().fit(tr).sample(len(tr), seed=seed) if b.source == "built-in" else synth
    res = evaluate_tstr(tr, te, syn, target, seed=seed)
    prim = res["summary"]["primary_metric"]
    models = [{"model": n, "real": m["baseline_real"].get(prim), "synthetic": m["synthetic"].get(prim), "gap_pct": m["gap_pct"].get(prim)}
              for n, m in res["models"].items()]
    gap = res["summary"]["mean_gap_pct"]
    metric = {"auc": "AUC (1 is perfect, 0.5 is guessing)", "r2": "R² (1 is perfect)", "f1_macro": "F1 score"}.get(prim, prim)
    cap = (f"A model is trained twice: once on real data, once on synthetic data, then both are tested on real data it never saw. Score: {metric}. "
           + ("Bars of similar height mean the synthetic data teaches a model almost as well as the real data" + (f" (average gap {gap:.1f}%)." if gap is not None else "."))
           if gap is not None else "The models could not be compared for this target.")
    return _envelope(b, "tstr", target=target, targets=list(real.columns), task=res["task"], metric=prim, metric_label=metric, models=models,
                     mean_gap_pct=None if gap is None else round(float(gap), 2), n_train_real=res["n_train_real"], n_test_real=res["n_test_real"], caption=cap)


def cardinality(b: Bundle, fk: str | None = None) -> dict[str, Any]:
    from sdp.relational.inference import children_per_parent
    from sdp.relational.integrity import check_integrity
    g = b.graph
    fks = [f for f in (g.foreign_keys if g else []) if f.child_table != f.parent_table]
    if not fks:
        raise VisualizeError("this dataset has no relationships between tables")
    chosen = next((f for f in fks if f.key == fk), None) if fk else fks[0]
    if chosen is None:
        raise VisualizeError(f"unknown relationship {fk!r}; available: {[f.key for f in fks]}")
    cap_k = 10

    def hist(tables: dict[str, pd.DataFrame]) -> tuple[np.ndarray, float, int]:
        counts = children_per_parent(tables[chosen.child_table], chosen, tables[chosen.parent_table])
        h = np.bincount(np.minimum(counts, cap_k), minlength=cap_k + 1) / max(len(counts), 1)
        child = tables[chosen.child_table][chosen.child_columns[0]].dropna()
        orphans = int((~child.isin(set(tables[chosen.parent_table][chosen.parent_columns[0]]))).sum())
        return h, float(counts.mean()) if len(counts) else 0.0, orphans
    hs, ms, os_ = hist(b.synth)
    hr, mr, or_ = hist(b.real) if b.has_real and chosen.child_table in b.real else (None, None, None)
    bins = [{"k": k, "label": f"{k}" if k < cap_k else f"{cap_k}+", "real": None if hr is None else round(float(hr[k]), 5), "synthetic": round(float(hs[k]), 5)}
            for k in range(cap_k + 1)]
    integ = check_integrity(g, b.synth).to_dict()
    per_fk = []
    for f in g.foreign_keys:
        child = b.synth[f.child_table][f.child_columns[0]].dropna()
        per_fk.append({"key": f.key, "child": f.child_table, "parent": f.parent_table, "orphans": int((~child.isin(set(b.synth[f.parent_table][f.parent_columns[0]]))).sum())})
    nodes = [{"id": t.name, "rows": int(len(b.synth[t.name])) if t.name in b.synth else 0, "columns": len(t.columns), "primary_key": list(t.primary_key)} for t in g.tables]
    edges = [{"id": f.key, "source": f.parent_table, "target": f.child_table, "label": f"{f.child_columns[0]}", "cardinality": f.cardinality,
              "orphans": next(x["orphans"] for x in per_fk if x["key"] == f.key)} for f in g.foreign_keys]
    cap = (f"For each {chosen.parent_table.rstrip('s')}, how many {chosen.child_table} rows point to it. Matching bars mean the synthetic data has the same "
           f"pattern as the real one (average {ms:.1f} real vs {mr:.1f} synthetic)." if hr is not None else
           f"For each {chosen.parent_table.rstrip('s')}, how many {chosen.child_table} rows point to it (average {ms:.1f}).")
    return _envelope(b, "cardinality", fk=chosen.key, fks=[f.key for f in fks], child=chosen.child_table, parent=chosen.parent_table, bins=bins,
                     mean_children={"real": None if mr is None else round(mr, 3), "synthetic": round(ms, 3)}, orphans={"real": or_, "synthetic": os_},
                     integrity={"ok": bool(integ["ok"]), "total_violations": int(integ["total_violations"]), "by_kind": integ["by_kind"], "rows_checked": int(integ["rows_checked"])},
                     graph={"nodes": nodes, "edges": edges}, caption=cap)


def privacy_distance(b: Bundle, seed: int = 0, bins: int = 20) -> dict[str, Any]:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.model_selection import train_test_split
    from sdp.scoring.privacy import _Encoder, _row_keys
    if b.kind != "tabular" or not b.has_real:
        raise VisualizeError("the privacy chart compares synthetic rows with real rows, so it needs a table with real data")
    real, synth = b.real["data"], b.synth["data"]
    cols = [c for c in real.columns if c in synth.columns]
    tr, ho = train_test_split(real[cols], test_size=0.3, random_state=seed)
    rng = np.random.default_rng(seed)
    cap = lambda df, n=2000: df if len(df) <= n else df.iloc[np.sort(rng.choice(len(df), n, replace=False))]  # noqa: E731
    tr, ho, sy = cap(tr).reset_index(drop=True), cap(ho).reset_index(drop=True), cap(synth[cols]).reset_index(drop=True)
    enc = _Encoder(tr)
    X = enc.transform(tr)
    nn = NearestNeighbors(n_neighbors=1).fit(X)
    d_syn, d_hold = nn.kneighbors(enc.transform(sy))[0][:, 0], nn.kneighbors(enc.transform(ho))[0][:, 0]
    hi = float(np.percentile(np.concatenate([d_syn, d_hold]), 99)) or 1.0
    edges = np.linspace(0, hi, int(min(max(bins, 5), MAX_BINS)) + 1)
    h_s, h_h = (np.histogram(np.clip(d, 0, hi), edges)[0] / len(d) for d in (d_syn, d_hold))
    tk = _row_keys(tr)
    freq = tk.map(tk.value_counts()).to_numpy()
    exact = int((np.isin(_row_keys(sy).to_numpy(), tk.to_numpy()[freq == 1])).sum())
    med_s, med_h = float(np.median(d_syn)), float(np.median(d_hold))
    ratio = med_s / med_h if med_h > 0 else 1.0
    rows = [{"x0": round(float(edges[i]), 3), "x1": round(float(edges[i + 1]), 3), "label": f"{(edges[i] + edges[i + 1]) / 2:.2f}", "real": round(float(h_h[i]), 5),
             "synthetic": round(float(h_s[i]), 5)} for i in range(len(edges) - 1)]
    verdict = "far enough from real people" if ratio >= 0.9 else "a little close to real records" if ratio >= 0.6 else "too close to real records"
    cap = ("For every row, how far is the closest real record? Blue shows real rows the model never saw (the normal distance); teal shows synthetic rows. "
           f"If teal sits at or to the right of blue, no synthetic row is a copy of a real person. Median distance {med_s:.2f} vs {med_h:.2f}: {verdict}.")
    return _envelope(b, "privacy_distance", bins=rows, median={"real": round(med_h, 4), "synthetic": round(med_s, 4)}, ratio=round(ratio, 3), exact_copies=exact,
                     rows_compared=int(len(sy)), verdict=verdict, caption=cap, series_labels={"real": "Real (unseen)", "synthetic": "Synthetic"})


def locale_validity(b: Bundle) -> dict[str, Any]:
    val = b.meta.get("validation")
    if not val:
        raise VisualizeError("locale checks need a dataset with country-bound columns (phone, national ID...). Try a described dataset such as bank_customers")
    loc = val["locale"]
    checks = [{"check": k, "label": LOCALE_LABELS.get(k, k), "checked": v["checked"], "failed": v["failed"], "pass_pct": round(float(v["pct_valid"]), 2)}
              for k, v in loc["by_check"].items() if v["checked"]]
    cap = (f"Share of values that follow the rules of {b.meta.get('locale', 'the chosen country')}: real-looking phone numbers, ID checksums, names in the right script and so on. "
           f"{loc['n_rows'] - loc['n_invalid_rows']:,} of {loc['n_rows']:,} records pass every check.")
    return _envelope(b, "locale_validity", locale=b.meta.get("locale"), valid_pct=round(float(loc["valid_pct"]), 2), n_rows=int(loc["n_rows"]), n_invalid_rows=int(loc["n_invalid_rows"]),
                     checks=checks, caption=cap)


def batch_summary(b: Bundle) -> dict[str, Any]:
    rep = b.meta.get("report")
    if rep is None:
        raise VisualizeError("batch summary is for document batches")
    total, ok, failed, skipped = int(rep["total"]), int(rep["succeeded"]), int(rep["failed"]), int(rep.get("skipped", 0))
    good = ok + skipped
    rate = round(100.0 * good / total, 2) if total else 100.0
    recon_fail = int(rep.get("by_stage", {}).get("reconcile", 0)) + int(rep.get("by_stage", {}).get("post_validate", 0))
    rec_rate = round(100.0 * (total - recon_fail) / total, 2) if total else 100.0
    fails = [{"index": f.get("index"), "stage": f.get("stage"), "error_type": f.get("error_type"), "message": str(f.get("message"))[:160]} for f in rep.get("failures", [])[:5]]
    dur = float(rep.get("duration_s") or 0)
    cap = (f"{good:,} of {total:,} documents were produced and every total was re-added and matched ({rate:.1f}% success, {rec_rate:.1f}% reconcile exactly)."
           + (f" {failed} failed; each failure is isolated and listed by stage." if failed else " None failed."))
    return _envelope(b, "batch_summary", total=total, succeeded=ok, failed=failed, skipped=skipped, success_rate_pct=rate, reconciliation_rate_pct=rec_rate,
                     by_stage={k: int(v) for k, v in rep.get("by_stage", {}).items()}, by_error={k: int(v) for k, v in rep.get("by_error", {}).items()}, failures=fails,
                     duration_s=round(dur, 2), docs_per_s=round(total / dur, 1) if dur > 0 else None, cancelled=bool(rep.get("cancelled")), caption=cap)


def payload(b: Bundle, typ: str, **p: Any) -> dict[str, Any]:
    if typ not in TYPES:
        raise VisualizeError(f"unknown type {typ!r}; available: {list(TYPES)}")
    if typ not in b.types:
        raise VisualizeError(f"{typ!r} is not available for a {b.kind} dataset; try: {b.types}")
    if typ == "distribution":
        return distribution(b, p.get("column"), p.get("table"), p.get("bins", 20))
    if typ == "categorical":
        return categorical(b, p.get("column"), p.get("table"), p.get("limit", 12))
    if typ == "correlation":
        return correlation(b, p.get("table"))
    if typ == "tstr":
        return tstr(b, p.get("target"), p.get("seed", 0))
    if typ == "cardinality":
        return cardinality(b, p.get("fk"))
    if typ == "privacy_distance":
        return privacy_distance(b, p.get("seed", 0), p.get("bins", 20))
    if typ == "locale_validity":
        return locale_validity(b)
    return batch_summary(b)


def catalog(store: Any = None, limit: int = 30) -> dict[str, Any]:
    """Everything that can be visualized: built-in datasets and finished saved runs."""
    from sdp.datasets import TABULAR_SAMPLES
    builtin = [{"id": k, "title": v[2], "kind": "tabular", "source": "built-in"} for k, v in TABULAR_SAMPLES.items()]
    builtin += [{"id": "shop_full", "title": "Sample shop (linked tables)", "kind": "relational", "source": "built-in"},
                {"id": "bank_customers", "title": "Described: bank customers", "kind": "nl", "source": "built-in"},
                {"id": "ecommerce_customers", "title": "Described: e-commerce customers", "kind": "nl", "source": "built-in"},
                {"id": "documents-demo", "title": "Sample batch of 40 documents (2 deliberately invalid)", "kind": "documents", "source": "built-in"}]
    for x in builtin:
        x["types"] = {"tabular": ["distribution", "categorical", "correlation", "tstr", "privacy_distance"], "relational": ["distribution", "categorical", "correlation", "cardinality"],
                      "nl": ["distribution", "categorical", "cardinality", "locale_validity"], "documents": ["batch_summary"]}[x["kind"]]
    jobs = []
    if store is not None:
        for m in store.list()[: limit * 3]:
            if m["status"] == "succeeded" and m["kind"] in ("tabular", "relational", "nl", "document", "large") and len(jobs) < limit:
                kind = "documents" if m["kind"] == "document" else "tabular" if m["kind"] == "large" else m["kind"]
                jobs.append({"id": m["id"], "title": f"Saved {m['kind']} run · {m['created_at'][:16].replace('T', ' ')}", "kind": kind, "source": "job",
                             "types": {"tabular": ["distribution", "categorical", "correlation", "tstr", "privacy_distance"], "relational": ["distribution", "categorical", "correlation", "cardinality"],
                                       "nl": ["distribution", "categorical", "cardinality", "locale_validity"], "documents": ["batch_summary"]}[kind]})
    return {"types": list(TYPES), "datasets": builtin, "saved_runs": jobs}
