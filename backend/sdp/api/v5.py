"""Fifth API router: schema understanding (A1), content synthesis (A2), conversational editing (A6), multilingual prompts (M7),
language edge cases (M8), export formats (P1), database connectors (P3) and data contracts / starter packs (P5).

UI endpoints live under /api (same-origin, no key, like the rest of the UI). Connectors are the exception: they open outbound connections,
so the UI route is OFF unless SDP_CONNECTORS_UI=1, and the always-on route POST /connect/load requires the API key.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sdp.ai import llm
from sdp.ai.chat_edit import ChatEditError, EditState, apply_message
from sdp.ai.content import KINDS as CONTENT_KINDS, ContentError, synthesize
from sdp.ai.schema_infer import SchemaError, SchemaProposal, apply_edits, infer_schema, to_rules
from sdp.api.autosave import autosave
from sdp.api.public import require_api_key
from sdp.connectors import ConnectorError, get_connector
from sdp.contracts import PackError, generate_pack, list_packs, validate_pack, validate_suite
from sdp.contracts.packs import export_suite, get_pack
from sdp.exporters import REGISTRY, ExportError, export_bytes, schema_from
from sdp.locale import LocaleError
from sdp.locale.langedge import PACK_NAMES as LANG_PACKS, language_cases
from sdp.nl.multilingual import SUPPORTED as LANGS, parse_multilingual

router = APIRouter(tags=["v5"])
connect_public = APIRouter(tags=["public API"], dependencies=[Depends(require_api_key)])
MAX_ROWS_TOTAL = 200_000


def _records(df: pd.DataFrame, n: int | None = None) -> list[dict]:
    d = df.head(n) if n else df
    return json.loads(d.to_json(orient="records", date_format="iso", force_ascii=False))


def _frame(records: list[dict]) -> pd.DataFrame:
    from sdp.service import _frame as f
    return f(records)


def _tables_from(inline: dict[str, list[dict]]) -> dict[str, pd.DataFrame]:
    if sum(len(v) for v in inline.values()) > MAX_ROWS_TOTAL:
        raise HTTPException(422, f"inline data is limited to {MAX_ROWS_TOTAL:,} rows")
    if not inline:
        raise HTTPException(422, "no tables supplied")
    return {k: _frame(v) for k, v in inline.items()}


# --------------------------------------------------------------- data sources
class Source(BaseModel):
    """Where the tables come from: a starter pack, a built-in demo dataset, or inline rows."""
    model_config = ConfigDict(extra="forbid")
    kind: Literal["pack", "demo", "inline"] = "pack"
    name: str = "banking"
    rows: int | None = Field(None, ge=10, le=200_000)
    seed: int = 0
    locale: str | None = None
    tables: dict[str, list[dict]] | None = None


def resolve_source(src: Source) -> dict[str, pd.DataFrame]:
    try:
        if src.kind == "pack":
            return generate_pack(src.name, src.rows, src.seed, src.locale)
        if src.kind == "demo":
            from sdp.datasets import make_customers, make_shop, make_shop_full
            if src.name == "customers":
                return {"customers": make_customers(src.rows or 500, seed=src.seed)}
            if src.name == "shop":
                return make_shop(src.rows or 300, seed=src.seed)
            if src.name == "shop_full":
                return make_shop_full(src.rows or 300, seed=src.seed)
            raise HTTPException(422, "demo datasets: customers, shop, shop_full")
        return _tables_from(src.tables or {})
    except (PackError, LocaleError, ValueError) as e:
        raise HTTPException(422, str(e)) from e


def _graph(tables: dict[str, pd.DataFrame]):
    if len(tables) < 2:
        return None
    from sdp.relational import infer_graph
    try:
        return infer_graph(tables)
    except Exception:  # noqa: BLE001 - keys are optional for exports
        return None


# ------------------------------------------------------------------ LLM status
@router.get("/api/llm/status")
def llm_status() -> dict[str, Any]:
    c = llm.get_client()
    return {"available": c is not None, "model": getattr(c, "name", None),
            "note": "Set ANTHROPIC_API_KEY to enable Claude. Everything works offline with local fallbacks."}


# ------------------------------------------------------------------------ A1
class InferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tables: dict[str, list[dict]] | None = None
    records: list[dict] | None = None
    name: str = "table"
    use_llm: bool = Field(False, description="send column names and up to 50 sample rows to Claude to refine the local guess")


@router.post("/api/schema/infer")
def schema_infer(req: InferRequest) -> dict[str, Any]:
    tables = _tables_from(req.tables or ({req.name: req.records} if req.records else {}))
    try:
        p = infer_schema(tables, client=True if req.use_llm else None)
    except SchemaError as e:
        raise HTTPException(422, str(e)) from e
    return {"proposal": json.loads(p.model_dump_json()), "rules": to_rules(p), "llm_available": llm.get_client() is not None}


class EditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    proposal: dict[str, Any]
    edits: list[dict[str, Any]]


@router.post("/api/schema/edit")
def schema_edit(req: EditRequest) -> dict[str, Any]:
    try:
        p = apply_edits(SchemaProposal.model_validate(req.proposal), req.edits)
    except (SchemaError, ValidationError, KeyError) as e:
        raise HTTPException(422, str(e)[:500]) from e
    return {"proposal": json.loads(p.model_dump_json()), "rules": to_rules(p)}


# ------------------------------------------------------------------------ A2
class ContentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str = "review"
    n: int = Field(10, ge=1, le=2000)
    locale: str = "en-US"
    contexts: list[dict[str, Any]] | None = None
    seed: int = 0
    use_llm: bool = False


@router.post("/api/content/synthesize")
def content_synthesize(req: ContentRequest) -> dict[str, Any]:
    if req.kind not in CONTENT_KINDS:
        raise HTTPException(422, f"kind must be one of {list(CONTENT_KINDS)}")
    try:
        r = synthesize(req.kind, req.n, req.locale, req.contexts, req.seed, client=True if req.use_llm else None)
    except (ContentError, LocaleError) as e:
        raise HTTPException(422, str(e)) from e
    return {"values": r.values, "sources": r.sources, "stats": r.stats, "warnings": r.warnings}


# ------------------------------------------------------------------------ A6
class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: dict[str, Any]
    text: str = Field(min_length=1, max_length=500)
    language: str | None = None
    use_llm: bool = False


@router.post("/api/chat/edit")
def chat_edit(req: ChatRequest) -> dict[str, Any]:
    try:
        state = EditState.model_validate(req.state)
        if state.config.rows > 100_000:
            raise HTTPException(422, "conversational editing works on datasets up to 100,000 rows")
        out = apply_message(state, req.text, req.language, client=True if req.use_llm else None)
    except (ChatEditError, ValidationError, ValueError) as e:
        raise HTTPException(422, str(e)[:500]) from e
    return out


# ------------------------------------------------------------------------ M7
class MLRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(max_length=1000)
    seed: int = 0
    language: str | None = None


@router.post("/api/nl/parse-multilingual")
def nl_parse_multilingual(req: MLRequest) -> dict[str, Any]:
    try:
        out = parse_multilingual(req.text, req.seed, req.language)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    out.pop("parse", None)
    out["supported"] = LANGS
    return out


# ------------------------------------------------------------------------ M8
@router.get("/api/locale/language-cases")
def language_cases_endpoint(packs: str | None = None, per_case: int = 2, seed: int = 0) -> dict[str, Any]:
    try:
        df = language_cases([p for p in packs.split(",") if p] if packs else None, max(1, min(per_case, 10)), seed)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return {"packs": LANG_PACKS, "rows": _records(df)}


# ------------------------------------------------------------------------ P1
@router.get("/api/export/formats")
def export_formats() -> list[dict[str, str]]:
    return [{"name": e.name, "extension": e.extension, "mime": e.mime} for e in REGISTRY.values()]


class ExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Source = Field(default_factory=Source)
    format: str = "csv"
    options: dict[str, Any] = Field(default_factory=dict)


@router.post("/api/export")
def export(req: ExportRequest) -> Response:
    tables = resolve_source(req.source)
    try:
        opts = {k: v for k, v in req.options.items() if k in ("dialect", "bom", "drop", "formats", "preview_rows", "title")}
        data, _ = export_bytes(tables, req.format, schema_from(tables, _graph(tables)), **opts)
        ext = REGISTRY[req.format].extension
        mime = REGISTRY[req.format].mime
    except ExportError as e:
        raise HTTPException(422, str(e)) from e
    if req.source.kind == "pack":
        autosave("pack", {"pack": req.source.name, "rows": req.source.rows, "seed": req.source.seed, "locale": req.source.locale})
    name = f"{req.source.name}.{'zip' if req.format == 'csv' and len(tables) > 1 else ext}"
    return Response(data, media_type="application/zip" if name.endswith(".zip") else mime, headers={"Content-Disposition": f'attachment; filename="{name}"'})


# ------------------------------------------------------------------------ P3
class LoadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(description="postgresql://, mysql://, mongodb://... or sqlite:///relative.db (inside the data directory)")
    source: Source = Field(default_factory=Source)
    dry_run: bool = True
    mode: Literal["create", "append", "replace"] = "create"


def _load(req: LoadRequest) -> dict[str, Any]:
    tables = resolve_source(req.source)
    try:
        res = get_connector(req.url).load(tables, schema_from(tables, _graph(tables)), dry_run=req.dry_run, mode=req.mode)
    except ConnectorError as e:
        raise HTTPException(422, str(e)) from e
    return res.to_dict()


@connect_public.post("/connect/load", summary="Load generated tables into a database (dry run by default; API key required)")
def connect_load(req: LoadRequest) -> dict[str, Any]:
    return _load(req)


@router.post("/api/connect/load")
def ui_connect_load(req: LoadRequest) -> dict[str, Any]:
    if os.environ.get("SDP_CONNECTORS_UI") != "1":
        raise HTTPException(403, "database loading from the UI is disabled; set SDP_CONNECTORS_UI=1, or use POST /connect/load with an API key")
    return _load(req)


@router.get("/api/connect/status")
def connect_status() -> dict[str, Any]:
    from sdp.connectors.base import allowed_hosts
    return {"ui_enabled": os.environ.get("SDP_CONNECTORS_UI") == "1", "allowed_hosts": sorted(allowed_hosts()), "default": "dry run (transaction is rolled back)"}


# ------------------------------------------------------------------------ P5
@router.get("/api/contracts/packs")
def contracts_packs() -> list[dict[str, Any]]:
    return list_packs()


class PackRun(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pack: str
    rows: int | None = Field(None, ge=10, le=100_000)
    seed: int = 0
    locale: str | None = None
    preview_rows: int = Field(8, ge=0, le=50)


@router.post("/api/contracts/run")
def contracts_run(req: PackRun) -> dict[str, Any]:
    try:
        tables = generate_pack(req.pack, req.rows, req.seed, req.locale)
        report = validate_pack(req.pack, tables)
    except (PackError, LocaleError) as e:
        raise HTTPException(422, str(e)) from e
    hid = autosave("pack", {"pack": req.pack, "rows": req.rows, "seed": req.seed, "locale": req.locale})
    return {"history_id": hid, "pack": req.pack, "rows": {k: len(v) for k, v in tables.items()}, "report": report,
            "preview": {k: _records(v, req.preview_rows) for k, v in tables.items()}}


class Validate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pack: str | None = None
    suite: list[dict[str, Any]] | None = None
    tables: dict[str, list[dict]]


@router.post("/api/contracts/validate")
def contracts_validate(req: Validate) -> dict[str, Any]:
    tables = _tables_from(req.tables)
    try:
        if req.suite:
            return validate_suite(tables, req.suite, "custom")
        if req.pack:
            return validate_pack(req.pack, tables)
    except PackError as e:
        raise HTTPException(422, str(e)) from e
    raise HTTPException(422, "send `pack` or `suite`")


@router.get("/api/contracts/{pack}/great-expectations")
def contracts_ge(pack: str) -> dict[str, Any]:
    try:
        get_pack(pack)
    except PackError as e:
        raise HTTPException(404, str(e)) from e
    return export_suite(pack)
