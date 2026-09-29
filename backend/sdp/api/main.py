"""FastAPI surface for the workspace UI. Run: uvicorn sdp.api.main:app --reload --port 8000"""

from __future__ import annotations

import base64
import logging
import os
import json
from contextlib import asynccontextmanager
from decimal import Decimal
from functools import lru_cache
from typing import Any, Literal

from sdp.envfile import load_env

load_env()  # .env in the project root: real environment variables win

import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sklearn.model_selection import train_test_split

from sdp.datasets import SHOP_FULL_RULES, make_customers, make_shop, make_shop_full
from sdp.evaluation import evaluate_tstr
from sdp.relational import CardinalityConfig, RelationalGenerator, RelationshipGraph, infer_graph, parse_ddl
from sdp.relational.inference import annotate_cardinalities
from sdp.documents import InvoiceSpec, generate_invoice, render_invoice
from sdp.documents.pipeline import DocumentPipeline
from sdp.locale import get_locale, registry
from sdp.nl import DatasetConfig, generate_dataset, parse_request
from sdp.scoring import build_trust_report, trust_report_pdf
from sdp.relational.metrics import relation_metrics
from sdp.rules import RuleSyntaxError, compare_naive_vs_enforced
from sdp.scoring import build_trust_report_relational
from sdp.tabular import GenConfig, TabularGenerator, fidelity_report
from sdp.tabular.quality import overlay_data

@asynccontextmanager
async def lifespan(_: FastAPI):
    from sdp import service
    from sdp.lineage import Store
    n = service.recover(Store())  # jobs that were running when the previous process died cannot continue
    if n:
        logging.getLogger("sdp.api").warning("marked %d interrupted job(s) as failed", n)
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Synthetic Data Platform", version="0.1.0",
    description="Generate, score and version synthetic datasets. The public API (/generate, /jobs, /score) needs an API key "
                "(X-API-Key header). /api/* endpoints serve the bundled UI.",
    openapi_tags=[{"name": "public API", "description": "Requires X-API-Key (or Authorization: Bearer)"},
                  {"name": "ui history", "description": "Job history for the UI (no key)"}])
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

PREVIEW_ROWS = 100


def records(df: pd.DataFrame, n: int = PREVIEW_ROWS) -> list[dict[str, Any]]:
    return json.loads(df.head(n).to_json(orient="records", date_format="iso"))


@lru_cache(maxsize=1)
def demo_customers() -> pd.DataFrame:
    return make_customers(2000, seed=0)


@lru_cache(maxsize=1)
def demo_shop() -> dict[str, pd.DataFrame]:
    return make_shop(300, seed=0)


@lru_cache(maxsize=1)
def demo_shop_full() -> dict[str, pd.DataFrame]:
    return make_shop_full(400, seed=0)


@lru_cache(maxsize=4)
def demo_sample(name: str) -> pd.DataFrame:
    """One of the built-in fictional tables (customers, students, employees), 2,000 rows."""
    from sdp.datasets import TABULAR_SAMPLES
    if name not in TABULAR_SAMPLES:
        raise HTTPException(422, f"unknown sample {name!r}; available: {sorted(TABULAR_SAMPLES)}")
    return TABULAR_SAMPLES[name][0](2000, seed=0)


@lru_cache(maxsize=12)
def _fitted_tabular(dataset: str, method: str, sample: str = "customers") -> TabularGenerator:
    if dataset != "demo":
        raise KeyError(dataset)
    return TabularGenerator(method).fit(demo_sample(sample))  # type: ignore[arg-type]


def _frame(data: list[dict] | None) -> pd.DataFrame | None:
    if data is None:
        return None
    df = pd.DataFrame(data)
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == "str":
            try:
                df[c] = pd.to_datetime(df[c], errors="raise", format="ISO8601")
            except (ValueError, TypeError):
                pass
    return df


# ------------------------------------------------------------------ meta
@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/demo/tabular")
def demo_tabular(sample: str = "customers") -> dict:
    from sdp.datasets import TABULAR_SAMPLES
    df = demo_sample(sample)
    return {"dataset": "demo", "sample": sample, "n_rows": len(df), "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()}, "target": TABULAR_SAMPLES[sample][1], "preview": records(df, 50)}


@app.get("/api/demo/tabular/samples")
def demo_tabular_samples() -> list[dict]:
    """The built-in tables the Tabular page can work on."""
    from sdp.datasets import TABULAR_SAMPLES
    return [{"name": k, "title": v[2], "target": v[1]} for k, v in TABULAR_SAMPLES.items()]


@app.get("/api/demo/relational")
def demo_relational() -> dict:
    shop = demo_shop()
    return {"dataset": "demo", "row_counts": {k: len(v) for k, v in shop.items()},
            "columns": {k: list(v.columns) for k, v in shop.items()},
            "preview": {k: records(v, 20) for k, v in shop.items()}}


# --------------------------------------------------------------- tabular
class TabularRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: Literal["demo", "inline"] = "demo"
    sample: str = "customers"      # which built-in table when dataset == "demo" (customers, students, employees)
    data: list[dict] | None = None
    method: Literal["gaussian_copula", "ctgan"] = "gaussian_copula"
    config: GenConfig = Field(default_factory=GenConfig)
    target: str | None = None
    rules: list[str] = Field(default_factory=list)        # DSL or plain language; compiled and enforced during generation
    rule_mode: Literal["repair", "reject", "hybrid"] = "hybrid"


_INLINE_FITS: dict[str, TabularGenerator] = {}
MAX_INLINE_ROWS = 200_000


def _tabular_setup(req: TabularRequest) -> tuple[pd.DataFrame, TabularGenerator]:
    try:
        if req.dataset == "demo":
            return demo_sample(req.sample), _fitted_tabular("demo", req.method, req.sample)
        real = _frame(req.data)
        if real is None or real.empty:
            raise HTTPException(422, "dataset='inline' requires non-empty `data`")
        if len(real) > MAX_INLINE_ROWS:
            raise HTTPException(422, f"inline data is limited to {MAX_INLINE_ROWS:,} rows")
        import hashlib
        key = hashlib.sha256((req.method + json.dumps(req.data, default=str, sort_keys=True)).encode()).hexdigest()
        if key not in _INLINE_FITS:
            if len(_INLINE_FITS) >= 4:
                _INLINE_FITS.pop(next(iter(_INLINE_FITS)))
            _INLINE_FITS[key] = TabularGenerator(req.method).fit(real)
        return real, _INLINE_FITS[key]
    except ImportError as e:
        raise HTTPException(501, str(e)) from e


@app.post("/api/tabular/generate")
def tabular_generate(req: TabularRequest) -> dict:
    real, gen = _tabular_setup(req)
    try:
        res = gen.generate(req.config, req.rules or None, req.rule_mode)
    except (ValueError, RuleSyntaxError) as e:
        raise HTTPException(422, str(e)) from e
    clean = gen.sample(min(len(real), 5000), seed=res.seed)  # fidelity is judged on a clean sample: injected defects are intentional
    from sdp.edgecases import TAG, visible
    shown = visible(res.data)  # hidden bookkeeping columns (_edge_case) never appear as data columns
    tags = res.data[TAG].head(50).astype(object).where(res.data[TAG].head(50).notna(), None).tolist() if TAG in res.data else None
    return {
        "seed": res.seed, "n_rows": len(shown), "columns": list(shown.columns),
        "dtypes": {c: str(t) for c, t in shown.dtypes.items()},
        "preview": records(shown, 50), "injection_log": res.log_dicts(), "rules": res.rules,
        "edge_cases": res.edge, "edge_tags": tags,
        "fidelity": fidelity_report(real, clean),
        "overlay": overlay_data(real, shown.dropna(how="all") if not res.log and not res.edge else clean),
    }


@app.post("/api/tabular/tstr")
def tabular_tstr(req: TabularRequest) -> dict:
    real, gen = _tabular_setup(req)
    if not req.target or req.target not in real.columns:
        raise HTTPException(422, f"`target` must be one of {list(real.columns)}")
    tr, te = train_test_split(real, test_size=0.3, random_state=req.config.seed or 0)
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
    synth = TabularGenerator(req.method).fit(tr).sample(len(tr), seed=req.config.seed)
    return evaluate_tstr(tr, te, synth, req.target, seed=req.config.seed or 0)


# ------------------------------------------------------------ relational
DATASETS = {"shop": demo_shop, "shop_full": demo_shop_full}
MAX_INLINE_ROWS = 200_000


def _tables(dataset: str, inline: dict[str, list[dict]] | None) -> dict[str, pd.DataFrame]:
    if dataset in DATASETS:
        return DATASETS[dataset]()
    if not inline:
        raise HTTPException(422, "dataset='inline' requires `tables` (name -> list of row objects)")
    out = {}
    for name, rows in inline.items():
        df = _frame(rows)
        if df is None or df.empty:
            raise HTTPException(422, f"table {name!r} is empty")
        out[name] = df
    if sum(len(v) for v in out.values()) > MAX_INLINE_ROWS:
        raise HTTPException(422, f"inline data is limited to {MAX_INLINE_ROWS:,} rows in total")
    return out


class InferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: Literal["shop", "shop_full", "inline", "demo"] | None = "shop"
    tables: dict[str, list[dict]] | None = None
    ddl: str | None = None


@app.post("/api/relational/infer")
def relational_infer(req: InferRequest) -> dict:
    ds = "shop" if req.dataset == "demo" else req.dataset  # None = "no data attached" (DDL only)
    try:
        if req.ddl:
            graph = parse_ddl(req.ddl)
            if ds in DATASETS or (ds == "inline" and req.tables):
                annotate_cardinalities(graph, _tables(ds, req.tables))
        else:
            graph = infer_graph(_tables(ds or "shop", req.tables))
    except HTTPException:
        raise
    except Exception as e:  # parser errors -> 422 rather than 500
        raise HTTPException(422, f"could not build graph: {e}") from e
    errs = graph.validate_graph()
    return {"graph": graph.to_dict(), "errors": errs, "order": graph.topological_order() if not errs else [],
            "suggested_rules": SHOP_FULL_RULES if ds == "shop_full" else []}


class GraphRequest(BaseModel):
    graph: dict


def _graph(d: dict) -> RelationshipGraph:
    d = {**d}
    d.pop("many_to_many", None)
    for f in d.get("foreign_keys", []):
        f.pop("key", None)
    return RelationshipGraph.model_validate(d)


@app.post("/api/relational/validate")
def relational_validate(req: GraphRequest) -> dict:
    g = _graph(req.graph)
    errs = g.validate_graph()
    return {"errors": errs, "order": [] if errs else g.topological_order(), "graph": g.to_dict()}


class RelationalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: Literal["shop", "shop_full", "inline"] = "shop"
    tables: dict[str, list[dict]] | None = None
    graph: dict | None = None
    scale: float = Field(1.0, gt=0, le=50)
    rows: dict[str, int] = Field(default_factory=dict)
    seed: int = 0
    cardinality: dict[str, CardinalityConfig] = Field(default_factory=dict)
    rules: list[str] = Field(default_factory=list)
    enforce: bool = True
    condition_on_parents: bool = True
    include_metrics: bool = True
    bom: bool = False


_FITS: dict[str, RelationalGenerator] = {}


def _fitted(req: RelationalRequest, real: dict[str, pd.DataFrame], graph: RelationshipGraph) -> RelationalGenerator:
    import hashlib
    key = hashlib.sha256((req.dataset + graph.to_json(None) + json.dumps({k: v.model_dump() for k, v in req.cardinality.items()}, sort_keys=True)
                          + str(req.condition_on_parents) + str(req.seed) + (json.dumps(req.tables, default=str, sort_keys=True) if req.tables else "")).encode()).hexdigest()
    if key not in _FITS:
        if len(_FITS) >= 6:
            _FITS.pop(next(iter(_FITS)))
        _FITS[key] = RelationalGenerator(graph, req.cardinality, condition_on_parents=req.condition_on_parents).fit(real, seed=req.seed)
    return _FITS[key]


def _run_relational(req: RelationalRequest):
    real = _tables(req.dataset, req.tables)
    graph = _graph(req.graph) if req.graph else infer_graph(real)
    errs = graph.validate_graph()
    if errs:
        raise HTTPException(422, {"errors": errs})
    unknown = set(req.cardinality) - {f.key for f in graph.foreign_keys}
    if unknown:
        raise HTTPException(422, f"unknown foreign keys in cardinality: {sorted(unknown)}")
    try:
        gen = _fitted(req, real, graph)
        res = gen.generate(rows=req.rows, scale=req.scale, seed=req.seed, rules=req.rules or None, enforce=req.enforce)
    except (ValueError, KeyError, RuleSyntaxError) as e:
        raise HTTPException(422, str(e)) from e
    return real, graph, res


@app.post("/api/relational/generate")
def relational_generate(req: RelationalRequest) -> dict:
    real, graph, res = _run_relational(req)
    out = {"scorecard": res.scorecard(), "order": graph.topological_order(), "graph": graph.to_dict(),
           "preview": {k: records(v, 30) for k, v in res.tables.items()},
           "columns": {k: list(v.columns) for k, v in res.tables.items()}}
    if req.include_metrics:
        out["relation_metrics"] = relation_metrics(real, res.tables, graph)
    return out


@app.post("/api/relational/trust")
def relational_trust(req: RelationalRequest) -> dict:
    real, graph, res = _run_relational(req)
    rel = relation_metrics(real, res.tables, graph)
    return build_trust_report_relational(real, res.tables, graph, relation=rel, rules_report=res.rules,
                                         integrity=res.integrity.to_dict(), title=f"Relational dataset ({req.dataset})", seed=req.seed)


@app.get("/api/demo/relational/full")
def demo_relational_full() -> dict:
    shop = demo_shop_full()
    return {"dataset": "shop_full", "row_counts": {k: len(v) for k, v in shop.items()}, "columns": {k: list(v.columns) for k, v in shop.items()},
            "preview": {k: records(v, 20) for k, v in shop.items()}, "rules": SHOP_FULL_RULES}


# ---------------------------------------------------------------- locales
@app.get("/api/locales")
def locales() -> list[dict]:
    import numpy as np
    out = []
    for code in registry().codes():
        p, rng = get_locale(code), np.random.default_rng(0)
        out.append({"code": code, "country": p.data["country"], "script": p.script, "currency": p.data["currency"]["code"],
                    "direction": p.data.get("pdf", {}).get("direction", "ltr"), "font_script": p.data.get("pdf", {}).get("font_script", "latin"),
                    "native_title": p.data.get("labels", {}).get("invoice_title", "INVOICE"),
                    "native_digits": bool(p.data.get("native_digits")), "complex_shaping": bool(p.data.get("pdf", {}).get("complex_shaping")),
                    "tax_label": p.data["tax"]["label"], "date_format": p.data["date"]["format"],
                    "sample": {"name": p.person_name(rng), "phone": p.phone(rng), "national_id": p.national_id(rng),
                               "address": p.address(rng)[0], "amount": p.format_currency(Decimal("1234567.5"))}})
    return out


# -------------------------------------------------- natural language -> data
class NLParseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(max_length=1000)
    seed: int = 0


@app.post("/api/nl/parse")
def nl_parse(req: NLParseRequest) -> dict:
    """Parse only. Nothing is generated until the client posts the confirmed config to /api/nl/generate."""
    return json.loads(parse_request(req.text, req.seed).model_dump_json())


class NLGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config: DatasetConfig
    confirmed: bool = False


MAX_API_ROWS = 200_000


@app.post("/api/nl/generate")
def nl_generate(req: NLGenerateRequest) -> dict:
    if not req.confirmed:
        raise HTTPException(409, "config must be confirmed by the user before generation (send confirmed=true)")
    if req.config.rows > MAX_API_ROWS:
        raise HTTPException(422, f"the API generates at most {MAX_API_ROWS:,} rows per request")
    ds = generate_dataset(req.config)
    val = ds.validate()
    from sdp.api.autosave import autosave
    history_id = autosave("nl", {"config": req.config.model_dump(), "seed": req.config.seed})
    val["locale"]["failures"] = val["locale"]["failures"][:20]
    val["constraints"].pop("rules", None)
    from sdp.service import nl_quality
    return {"history_id": history_id, "quality": nl_quality(val), "config_hash": req.config.config_hash(), "validation": val, "graph": ds.graph.to_dict(),
            "columns": {k: list(v.columns) for k, v in ds.tables.items()},
            "preview": {k: records(v, 25) for k, v in ds.tables.items()}}


# ------------------------------------------------------------ trust score
class TrustRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: Literal["demo", "inline"] = "demo"
    sample: str = "customers"      # built-in table when dataset == "demo"
    data: list[dict] | None = None
    rows: int = Field(1500, ge=100, le=20000)
    seed: int = 0
    include_utility: bool = False
    target: str | None = None
    rules: list[str] = Field(default_factory=list)   # DSL / plain language, evaluated on the synthetic data


DEMO_RULES = [{"type": "range", "column": "age", "min": 18, "max": 90}, {"type": "range", "column": "income", "min": 0},
              {"type": "range", "column": "tenure_months", "min": 0, "max": 120},
              {"type": "in_set", "column": "plan", "values": ["basic", "pro", "enterprise"]},
              {"type": "in_set", "column": "churned", "values": [0, 1]}]


@app.post("/api/trust/report")
def trust_report(req: TrustRequest) -> dict:
    demo = req.dataset == "demo"
    real = demo_sample(req.sample) if demo else _frame(req.data)
    if real is None or len(real) < 30:
        raise HTTPException(422, "need at least 30 rows of real data")
    if len(real) > MAX_INLINE_ROWS:
        raise HTTPException(422, f"inline data is limited to {MAX_INLINE_ROWS:,} rows")
    train, hold = train_test_split(real, test_size=0.3, random_state=req.seed)
    train, hold = train.reset_index(drop=True), hold.reset_index(drop=True)
    try:
        gen = TabularGenerator().fit(train)
        if req.rules:
            synth = gen.generate(GenConfig(rows=req.rows, seed=req.seed), req.rules).data
        else:
            synth = gen.sample(req.rows, seed=req.seed)
    except (ValueError, RuleSyntaxError) as e:
        raise HTTPException(422, str(e)) from e
    from sdp.datasets import TABULAR_SAMPLES
    target = req.target or (TABULAR_SAMPLES[req.sample][1] if demo else None)
    tstr = evaluate_tstr(train, hold, synth, target, seed=req.seed) if req.include_utility and target in train.columns else None
    title = f"{TABULAR_SAMPLES[req.sample][2].split(' (')[0]} table (demo data)" if demo else "Uploaded table"
    return build_trust_report(train, synth, real_holdout=hold, tstr=tstr, rules=DEMO_RULES if demo and req.sample == "customers" else None,
                              dsl_rules=req.rules or None, seed=req.seed, title=title)


class PdfRequest(BaseModel):
    report: dict
    visuals: dict[str, dict] | None = None   # optional Visualize payloads (small JSON summaries) to draw as charts in the PDF


@app.post("/api/trust/pdf")
def trust_pdf(req: PdfRequest) -> Response:
    try:
        pdf = trust_report_pdf(req.report, req.visuals)
    except (KeyError, TypeError, ValueError, IndexError) as e:
        raise HTTPException(422, f"not a trust report or chart payload: {type(e).__name__}: {e}") from e
    return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": 'attachment; filename="trust-score.pdf"'})


# --------------------------------------------------------------- documents
@app.post("/api/documents/invoice/pdf")
def invoice_pdf(spec: InvoiceSpec) -> Response:
    res = DocumentPipeline(workers=1).process_one(spec.model_dump(mode="json"))
    result, art = res
    if art is None:
        raise HTTPException(422, {"stage": result.stage, "error": result.error_type, "message": result.message})
    return Response(art.pdf, media_type="application/pdf", headers={"X-Invoice-Total": art.data["total"],
                    "Content-Disposition": f'attachment; filename="{art.data["invoice_number"]}.pdf"'})


@app.post("/api/documents/invoice/json")
def invoice_json(spec: InvoiceSpec) -> dict:
    return generate_invoice(spec)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    specs: list[Any] | None = None  # malformed entries are reported per-document by the pipeline, not rejected here
    count: int = Field(20, ge=1, le=5000)
    locale: str = "en-US"
    region: str | None = None
    seed: int = 0
    n_lines: int = Field(5, ge=1, le=100)


@app.post("/api/documents/batch")
def invoice_batch(req: BatchRequest) -> dict:
    specs = req.specs if req.specs is not None else [
        {"locale": req.locale, "region": req.region, "seed": req.seed + i, "n_lines": req.n_lines} for i in range(req.count)]
    pipe = DocumentPipeline(workers=4)
    report = pipe.run(specs)
    preview = []
    for i, raw in enumerate(specs[:3]):
        res, art = DocumentPipeline(workers=1).process_one(raw, i)
        if art:
            preview.append({"invoice_number": art.data["invoice_number"], "currency": art.data["currency"], "total": art.data["total"],
                            "lines": len(art.data["lines"]), "pages": res.pages, "tax_summary": art.data["tax_summary"]})
    return {"report": report.to_dict(), "preview": preview}


from sdp.api.public import history as _history, public as _public  # noqa: E402
from sdp.api.v2 import router as _v2  # noqa: E402  (imports main lazily; must be included before the static mount)
from sdp.api.v5 import connect_public as _connect, router as _v5  # noqa: E402
from sdp.api.visualize import public as _viz_public, ui as _viz_ui  # noqa: E402

app.include_router(_v2)
app.include_router(_public)   # /generate, /jobs/{id}, /score/{id}: API key required
app.include_router(_history)  # /api/history: UI, no key
app.include_router(_v5)       # schema inference, content, chat editing, multilingual, exports, contracts (UI); connectors are gated
app.include_router(_connect)  # POST /connect/load: API key required
app.include_router(_viz_ui)      # /api/visualize: chart summaries for the UI
app.include_router(_viz_public)  # GET /visualize/{dataset_id}: same, API key required

# Docker / single-container mode: serve the built frontend from the same origin
_static = os.environ.get("SDP_STATIC_DIR")
if _static and os.path.isdir(_static):
    app.mount("/", StaticFiles(directory=_static, html=True), name="ui")
