"""Second API router: rules (A4), multilingual documents (M2/M6), exports with UTF-8 BOM option.

Handlers import from sdp.api.main lazily (main includes this router at import time).
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sdp.documents import DOC_TYPES, build_layout
from sdp.documents.fonts import describe
from sdp.documents.pipeline import DocumentPipeline
from sdp.api.autosave import autosave, relational_params, tabular_params
from sdp.export import csv_bytes, json_bytes, zip_bytes
from sdp.rules import RuleSyntaxError, compare_naive_vs_enforced, compile_rule, single_table_graph

router = APIRouter()


def _records(df) -> list[dict]:
    import json
    return json.loads(df.to_json(orient="records", force_ascii=False))  # NaN -> null


def _main():
    from sdp.api import main
    return main


def _download(payload: bytes, media: str, name: str) -> Response:
    return Response(payload, media_type=media, headers={"Content-Disposition": f'attachment; filename="{name}"'})


# ------------------------------------------------------------------ rules
class RulesCompileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rules: list[str] = Field(max_length=100)
    dataset: Literal["customers", "shop", "shop_full", "inline"] = "customers"
    tables: dict[str, list[dict]] | None = None
    sample: str | None = None                  # dataset="customers": check against another built-in table (students, employees)
    data: list[dict] | None = Field(None, max_length=50_000)   # dataset="customers": check against these rows (a single uploaded table)


@router.post("/api/rules/compile")
def rules_compile(req: RulesCompileRequest) -> dict:
    """Show what each rule (plain language or DSL) compiles to, before anything is generated."""
    m = _main()
    if req.dataset == "customers":
        df = m._frame(req.data) if req.data else (m.demo_sample(req.sample) if req.sample else m.demo_customers())
        if df is None or df.empty:
            raise HTTPException(422, "no data to check the rules against")
        graph, default, cols = single_table_graph("data", df), "data", list(df.columns)
    else:
        from sdp.relational import infer_graph
        graph, default = infer_graph(m._tables(req.dataset, req.tables)), None
        cols = [f"{t.name}.{c.name}" for t in graph.tables for c in t.columns]
    results = []
    for i, text in enumerate(req.rules):
        try:
            c = compile_rule(text, graph, default, name=f"R{i + 1}")
            results.append({"ok": True, **c.describe()})
        except RuleSyntaxError as e:
            results.append({"ok": False, "input": text, "error": str(e)})
    return {"results": results, "columns": cols}


class RulesCompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rules: list[str] = Field(min_length=1, max_length=50)
    rows: int = Field(2000, ge=100, le=50_000)
    seed: int = 0
    mode: Literal["repair", "reject", "hybrid"] = "hybrid"


@router.post("/api/rules/compare")
def rules_compare(req: RulesCompareRequest) -> dict:
    """Naive sampling (rules ignored) vs our enforced sampling on the demo customers table."""
    m = _main()
    gen = m._fitted_tabular("demo", "gaussian_copula")
    try:
        return compare_naive_vs_enforced(gen, req.rows, req.seed, req.rules, real=m.demo_customers(), mode=req.mode)
    except RuleSyntaxError as e:
        raise HTTPException(422, str(e)) from e


# -------------------------------------------------------------- documents
class DocRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_type: str = "invoice"   # any registered type (built-in or template pack)
    spec: dict[str, Any] = Field(default_factory=dict)
    bom: bool = False


def _spec(req: DocRequest):
    dt = DOC_TYPES.get(req.doc_type)
    if dt is None:
        raise HTTPException(422, f"unknown doc_type {req.doc_type!r}; available: {sorted(DOC_TYPES)}")
    try:
        return dt, dt.spec_cls.model_validate(req.spec)
    except ValidationError as e:
        raise HTTPException(422, "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())) from e


@router.get("/api/documents/types")
def document_types() -> dict:
    return {"types": sorted(DOC_TYPES), "fonts": describe(),
            "details": [{"name": k, "title": v.title or k.replace("_", " ").title(), "engine": v.engine,
                         "params": sorted(v.spec_cls.model_fields.keys() - {"seed", "locale", "region", "font", "native", "native_digits"})}
                        for k, v in sorted(DOC_TYPES.items())]}


@router.post("/api/documents/preview")
def document_preview(req: DocRequest) -> dict:
    """The exact layout the PDF uses (labels, number/date formats, direction), for an HTML preview."""
    dt, spec = _spec(req)
    doc = dt.generate(spec)
    if dt.html is not None:  # template types preview as the HTML that becomes the PDF
        return {"document": doc, "layout": None, "html": dt.html(doc), "engine": "html", "reconciliation_errors": dt.reconcile(doc)}
    return {"document": doc, "layout": build_layout(doc).to_dict(), "html": None, "engine": "reportlab", "reconciliation_errors": dt.reconcile(doc)}


@router.post("/api/documents/pdf")
def document_pdf(req: DocRequest) -> Response:
    _spec(req)
    result, art = DocumentPipeline(workers=1).process_one({"doc_type": req.doc_type, **req.spec})
    if art is None:
        raise HTTPException(422, {"stage": result.stage, "error": result.error_type, "message": result.message})
    autosave("document", {"doc_type": req.doc_type, "spec": {k: v for k, v in req.spec.items() if k != "seed"}, "count": 1, "seed": int(req.spec.get("seed") or 0)})
    return _download(art.pdf, "application/pdf", f"{req.doc_type}-{result.key}.pdf")


@router.post("/api/documents/json")
def document_json(req: DocRequest) -> Response:
    dt, spec = _spec(req)
    doc = dt.generate(spec)
    return _download(json_bytes(doc, bom=req.bom), "application/json; charset=utf-8", f"{req.doc_type}.json")


# ------------------------------------------------------------------ export
class TabularExport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bom: bool = False


@router.post("/api/export/tabular")
def export_tabular(body: dict[str, Any]) -> Response:
    m = _main()
    bom = bool(body.pop("bom", False))
    req = m.TabularRequest.model_validate(body)
    real, gen = m._tabular_setup(req)
    try:
        res = gen.generate(req.config, req.rules or None, req.rule_mode)
    except (ValueError, RuleSyntaxError) as e:
        raise HTTPException(422, str(e)) from e
    autosave("tabular", tabular_params(req, res.seed))
    return _download(csv_bytes(res.data, bom), "text/csv; charset=utf-8", "synthetic.csv")


@router.post("/api/export/relational")
def export_relational(body: dict[str, Any]) -> Response:
    m = _main()
    bom = bool(body.get("bom", False))
    req = m.RelationalRequest.model_validate({k: v for k, v in body.items()})
    real, graph, res = m._run_relational(req)
    autosave("relational", relational_params(req))
    extra = {"scorecard.json": json_bytes(res.scorecard(), bom), "relationships.json": json_bytes(graph.to_dict(), bom)}
    return _download(zip_bytes(res.tables, bom, extra), "application/zip", "synthetic-tables.zip")


@router.post("/api/export/nl")
def export_nl(body: dict[str, Any]) -> Response:
    from sdp.nl import DatasetConfig, generate_dataset
    bom = bool(body.pop("bom", False))
    cfg = DatasetConfig.model_validate(body.get("config", body))
    if cfg.rows > 200_000:
        raise HTTPException(422, "the API generates at most 200,000 rows per request")
    ds = generate_dataset(cfg)
    val = ds.validate()
    autosave("nl", {"config": cfg.model_dump(), "seed": cfg.seed})
    return _download(zip_bytes(ds.tables, bom, {"validation.json": json_bytes(val, bom), "config.json": json_bytes(cfg.model_dump(), bom)}),
                     "application/zip", "synthetic-dataset.zip")


# ------------------------------------------------------ statements (D2)
class QueryParseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(max_length=300)


@router.post("/api/documents/statement/parse")
def statement_parse(req: QueryParseRequest) -> dict:
    """Show how a query such as "last 90 days, balance over $500" was understood, before generating anything."""
    from sdp.documents.statement_query import QuerySyntaxError, parse_statement_query
    try:
        f = parse_statement_query(req.query)
    except QuerySyntaxError as e:
        raise HTTPException(422, str(e)) from e
    return {"ok": not f.unparsed, "filter": f.to_dict(), "unparsed": f.unparsed}


@router.post("/api/documents/csv")
def document_csv(req: DocRequest) -> Response:
    """Machine-readable transactions of a statement (ASCII decimals, ISO dates)."""
    from sdp.documents.types import statement_csv
    if req.doc_type != "statement":
        raise HTTPException(422, "CSV export is available for statements")
    dt, spec = _spec(req)
    return _download(statement_csv(dt.generate(spec), req.bom), "text/csv; charset=utf-8", "statement.csv")


# ------------------------------------------------------------- scan (D4)
class ScanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_type: str = "invoice"   # any registered type (built-in or template pack)
    spec: dict[str, Any] = Field(default_factory=dict)
    preset: str | None = None
    scan: dict[str, Any] = Field(default_factory=dict)


def _scan(req: ScanRequest):
    from sdp.documents.scan import ScanConfig, preset, scan_document
    dt, spec = _spec(DocRequest(doc_type=req.doc_type, spec=req.spec))
    try:
        cfg = preset(req.preset, **req.scan) if req.preset else ScanConfig(**req.scan)
    except (ValidationError, ValueError) as e:
        raise HTTPException(422, str(e)) from e
    doc = dt.generate(spec)
    try:
        return scan_document(doc, cfg, spec.font)
    except Exception as e:  # missing font etc.
        if e.__class__.__name__ in ("MissingFontError", "RenderOverflowError"):
            raise HTTPException(422, str(e)) from e
        raise


@router.get("/api/documents/scan/presets")
def scan_presets() -> dict:
    from sdp.documents.scan import PRESETS, ScanConfig
    return {"presets": PRESETS, "defaults": ScanConfig().model_dump()}


@router.post("/api/documents/scan")
def document_scan(req: ScanRequest) -> dict:
    import base64
    r = _scan(req)
    pages = []
    for img, p in zip(r.images[:3], r.labels["pages"][:3]):   # preview the first pages only; the zip has them all
        pages.append({**p, "image_base64": base64.b64encode(img).decode()})
    return {"mime": r.mime, "n_pages": len(r.images), "pages": pages, "labels_schema": r.labels["schema"]}


@router.post("/api/documents/scan/zip")
def document_scan_zip(req: ScanRequest) -> Response:
    return _download(_scan(req).to_zip(), "application/zip", "scan-with-labels.zip")


# ------------------------------------------------------- locale lab (M3-M5)
class PeopleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int = Field(200, ge=1, le=20000)
    mix: str = "70% ur-PK, 30% en"
    seed: int = 0
    bom: bool = False
    download: bool = False


@router.post("/api/locale/people")
def locale_people(req: PeopleRequest):
    from sdp.locale import LocaleError
    from sdp.locale.coherent import coherence_report, generate_people
    try:
        df = generate_people(req.n, req.mix, req.seed)
    except LocaleError as e:
        raise HTTPException(422, str(e)) from e
    if req.download:
        return _download(csv_bytes(df, req.bom), "text/csv; charset=utf-8", "people.csv")
    rep = coherence_report(df)
    return {"columns": list(df.columns), "preview": _records(df.head(25)), "coherence": {k: rep[k] for k in ("coherent_pct", "by_check", "mix")}}


class EntitiesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int = Field(50, ge=1, le=5000)
    locale: str = "ur-PK"
    variants: int = Field(5, ge=2, le=12)
    confusers: float = Field(0.1, ge=0, le=0.5)
    typos: bool = False
    seed: int = 0
    bom: bool = False
    download: bool = False


@router.post("/api/locale/entities")
def locale_entities(req: EntitiesRequest):
    from sdp.locale import LocaleError
    from sdp.locale.translit import generate_entities, matching_pairs
    try:
        df = generate_entities(req.n, req.locale, req.seed, req.variants, req.confusers, req.typos)
    except LocaleError as e:
        raise HTTPException(422, str(e)) from e
    if req.download:
        return _download(csv_bytes(df, req.bom), "text/csv; charset=utf-8", "entities.csv")
    pairs = matching_pairs(df, seed=req.seed)
    return {"columns": list(df.columns), "preview": _records(df.head(60)), "entities": int(df["entity_id"].nunique()), "records": len(df),
            "matching_pairs": {"positive": int((pairs["label"] == 1).sum()), "negative": int((pairs["label"] == 0).sum())}}


@router.get("/api/locale/codemix/options")
def codemix_options() -> dict:
    from sdp.locale.codemix import pairs, topics
    return {"pairs": [{"name": k, "label": v["label"], "topics": topics(k)} for k, v in pairs().items()]}


class CodeMixRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int = Field(30, ge=1, le=5000)
    pair: str = "roman_urdu"
    topic: str = "banking"
    level: float = Field(0.5, ge=0, le=1)
    seed: int = 0
    bom: bool = False
    download: bool = False


@router.post("/api/locale/codemix")
def locale_codemix(req: CodeMixRequest):
    from sdp.locale.codemix import CodeMixError, english_token_ratio, generate_code_mixed
    try:
        df = generate_code_mixed(req.n, req.pair, req.topic, req.level, req.seed)
    except CodeMixError as e:
        raise HTTPException(422, str(e)) from e
    if req.download:
        flat = df.assign(tokens=df["tokens"].map(lambda t: " ".join(f"{w}/{lang}" for w, lang in t)))
        return _download(csv_bytes(flat, req.bom), "text/csv; charset=utf-8", "code-mixed.csv")
    rows = [{"text": r.text, "realized_mix": None if r.realized_mix != r.realized_mix else round(r.realized_mix, 2), "tokens": r.tokens} for r in df.head(25).itertuples()]
    return {"preview": rows, "mean_english_token_ratio": float(df["tokens"].map(english_token_ratio).mean())}
