"""Job service: run generation jobs, record manifests, compute scores, rerun from a manifest.

Used by the API-key protected REST endpoints (/generate, /jobs, /score), the UI history endpoints and the CLI, so all three
behave identically. Jobs run on a small thread pool; `wait=True` runs inline (CLI, tests).
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import threading
import time
import traceback
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from sdp import lineage
from sdp.lineage import JobNotFound, Store

EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sdp-job")
MAX_INLINE_ROWS = 200_000
KINDS = ("tabular", "relational", "nl", "document", "large", "pack")
TERMINAL = ("succeeded", "failed", "cancelled")

_EVENTS: dict[str, threading.Event] = {}      # job id -> cancel flag (cooperative)
_FUTURES: dict[str, Future] = {}
_LAST_WRITE: dict[str, float] = {}


class Cancelled(Exception):
    """Raised inside a worker when the job was cancelled; carries whatever partial output was already produced."""

    def __init__(self, files: list[dict[str, Any]] | None = None, schema: dict[str, Any] | None = None, dataset: dict[str, Any] | None = None) -> None:
        super().__init__("cancelled")
        self.files, self.schema, self.dataset = files or [], schema, dataset


def _cancelled(job_id: str) -> bool:
    ev = _EVENTS.get(job_id)
    return bool(ev and ev.is_set())


def _progress(store: Store, job_id: str, done: int, total: int | None, phase: str, failed: int = 0, force: bool = False) -> None:
    """Persist progress (throttled to ~2 writes/second so a 20,000-document batch does not thrash the disk)."""
    now = time.monotonic()
    if not force and now - _LAST_WRITE.get(job_id, 0) < 0.4 and done != total:
        return
    _LAST_WRITE[job_id] = now
    m = store.read(job_id)
    prog = {**m.get("progress", {}), "done": done, "total": total, "phase": phase, "failed": failed, "updated_at": lineage.now()}
    store.update(job_id, progress=prog)


def _step(store: Store, job_id: str, phase: str, done: int, total: int) -> None:
    """A coarse checkpoint for fast job kinds: report progress and honour a pending cancel."""
    if _cancelled(job_id):
        raise Cancelled()
    _progress(store, job_id, done, total, phase, force=True)


# ------------------------------------------------------------- parameters
class TabularParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: Literal["demo", "inline"] = "demo"
    sample: str = "customers"      # built-in table when dataset == "demo": customers, students, employees
    data: list[dict] | None = None
    data_ref: str | None = None
    rows: int = Field(500, ge=1, le=1_000_000)
    seed: int = 0
    method: Literal["gaussian_copula", "ctgan"] = "gaussian_copula"
    rules: list[str] = Field(default_factory=list)
    rule_mode: Literal["repair", "reject", "hybrid"] = "hybrid"
    null_rate: dict[str, float] = Field(default_factory=dict)
    outlier_rate: dict[str, float] = Field(default_factory=dict)
    outlier_method: Literal["iqr", "zscore", "scale"] = "iqr"
    edge_cases: dict[str, float] = Field(default_factory=dict)
    include_hidden: bool = False   # include the hidden _edge_case tag column in the CSV


class RelationalParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: Literal["shop", "shop_full", "inline"] = "shop_full"
    tables: dict[str, list[dict]] | None = None
    tables_ref: list[str] | None = None
    seed: int = 0
    scale: float = Field(1.0, gt=0, le=50)
    rows: dict[str, int] = Field(default_factory=dict)
    rules: list[str] = Field(default_factory=list)
    enforce: bool = True
    condition_on_parents: bool = True


class NLParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str | None = Field(None, max_length=1000)
    config: dict[str, Any] | None = None
    seed: int = 0


class DocumentParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_type: str = "invoice"                      # any registered type: invoice, receipt, statement, payslip, purchase_order, ...
    spec: dict[str, Any] = Field(default_factory=dict)
    count: int = Field(1, ge=1, le=20_000)
    seed: int = 0
    specs: list[Any] | None = Field(None, max_length=20_000, description="explicit per-document specs (may mix types); overrides doc_type/spec/count")
    workers: int = Field(4, ge=1, le=8)


class LargeParams(BaseModel):
    """P6: millions of rows, chunked and streamed to one file. Memory stays flat; the output file is identical for any worker count."""
    model_config = ConfigDict(extra="forbid")
    dataset: Literal["demo", "inline"] = "demo"
    data: list[dict] | None = None
    data_ref: str | None = None
    rows: int = Field(1_000_000, ge=1, le=100_000_000)
    seed: int = 0
    format: Literal["csv", "jsonl", "json", "sql"] = "csv"
    chunk_rows: int = Field(100_000, ge=100, le=1_000_000)
    workers: int = Field(4, ge=1, le=16)
    executor: Literal["auto", "thread", "process"] = "auto"
    compress: bool = False
    dialect: Literal["postgres", "mysql", "sqlite", "ansi"] = "postgres"    # for format=sql
    null_rate: dict[str, float] = Field(default_factory=dict)
    outlier_rate: dict[str, float] = Field(default_factory=dict)
    sequence_columns: list[str] | None = None


class PackParams(BaseModel):
    """A starter pack (banking, ecommerce, healthcare) generated and checked against its data contract."""
    model_config = ConfigDict(extra="forbid")
    pack: str
    rows: int | None = Field(None, ge=10, le=100_000)
    seed: int = 0
    locale: str | None = None


PARAMS: dict[str, type[BaseModel]] = {"tabular": TabularParams, "relational": RelationalParams, "nl": NLParams, "document": DocumentParams, "large": LargeParams,
                                      "pack": PackParams}


def validate_params(kind: str, params: dict[str, Any]) -> dict[str, Any]:
    if kind not in PARAMS:
        raise ValueError(f"unknown kind {kind!r}; available: {list(KINDS)}")
    m = PARAMS[kind].model_validate(params)
    if kind == "nl" and not (m.text or m.config):  # type: ignore[attr-defined]
        raise ValueError("nl jobs need `text` or `config`")
    if kind == "document":
        from sdp.documents import DOC_TYPES
        if not m.specs:  # type: ignore[attr-defined]
            if m.doc_type not in DOC_TYPES:  # type: ignore[attr-defined]
                raise ValueError(f"unknown doc_type {m.doc_type!r}; available: {sorted(DOC_TYPES)}")  # type: ignore[attr-defined]
            DOC_TYPES[m.doc_type].spec_cls.model_validate({**m.spec, "seed": m.seed})  # type: ignore[attr-defined]  # fail fast, not per document
    return m.model_dump(exclude_none=True)


# ------------------------------------------------------------------ inputs
def _frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == "str":
            try:
                df[c] = pd.to_datetime(df[c], errors="raise", format="ISO8601")
            except (ValueError, TypeError):
                pass
    return df


def _read_csv(store: Store, job_id: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(store.artifact(job_id, name)))
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype) == "str":
            try:
                df[c] = pd.to_datetime(df[c], errors="raise", format="ISO8601")
            except (ValueError, TypeError):
                pass
    return df


def _materialize(store: Store, job_id: str, request: dict[str, Any], from_job: str | None) -> dict[str, Any]:
    """Inline data goes into the job directory (input files), the manifest keeps only references. A rerun copies them."""
    r = dict(request)
    if r.get("data") is not None:
        df = _frame(r["data"])
        if len(df) > MAX_INLINE_ROWS:
            raise ValueError(f"inline data is limited to {MAX_INLINE_ROWS:,} rows")
        store.save_artifact(job_id, "input.csv", df.to_csv(index=False).encode())
        r.pop("data")
        r["data_ref"] = "input.csv"
    elif r.get("data_ref") and from_job:
        store.save_artifact(job_id, r["data_ref"], store.artifact(from_job, r["data_ref"]))
    if r.get("tables") is not None:
        refs = []
        for name, rows in r.pop("tables").items():
            store.save_artifact(job_id, f"input_{name}.csv", _frame(rows).to_csv(index=False).encode())
            refs.append(name)
        r["tables_ref"] = refs
    elif r.get("tables_ref") and from_job:
        for name in r["tables_ref"]:
            store.save_artifact(job_id, f"input_{name}.csv", store.artifact(from_job, f"input_{name}.csv"))
    return r


def _real_tabular(store: Store, job_id: str, p: Any) -> pd.DataFrame:
    if p.dataset == "demo":
        from sdp.datasets import TABULAR_SAMPLES
        sample = getattr(p, "sample", "customers")
        if sample not in TABULAR_SAMPLES:
            raise ValueError(f"unknown sample {sample!r}; available: {sorted(TABULAR_SAMPLES)}")
        return TABULAR_SAMPLES[sample][0](2000, seed=0)
    return _read_csv(store, job_id, p.data_ref or "input.csv")


def _real_relational(store: Store, job_id: str, p: RelationalParams) -> dict[str, pd.DataFrame]:
    from sdp.datasets import make_shop, make_shop_full
    if p.dataset == "shop":
        return make_shop(300, seed=0)
    if p.dataset == "shop_full":
        return make_shop_full(400, seed=0)
    return {n: _read_csv(store, job_id, f"input_{n}.csv") for n in (p.tables_ref or [])}


def _schema(df: pd.DataFrame) -> dict[str, str]:
    return {c: str(t) for c, t in df.dtypes.items()}


# ---------------------------------------------------------------- generation
def _generate(store: Store, m: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Returns (output files, schema, dataset info). Deterministic given the request."""
    jid, kind = m["id"], m["kind"]
    files: list[dict[str, Any]] = []
    save = lambda name, b: files.append(store.save_artifact(jid, name, b))  # noqa: E731
    if kind == "tabular":
        from sdp.edgecases import visible
        from sdp.tabular import GenConfig, TabularGenerator
        p = TabularParams.model_validate(m["request"])
        _step(store, jid, "fitting", 0, 3)
        real = _real_tabular(store, jid, p)
        gen = TabularGenerator(p.method).fit(real)
        _step(store, jid, "sampling", 1, 3)
        cfg = GenConfig(rows=p.rows, seed=p.seed, null_rate=p.null_rate, outlier_rate=p.outlier_rate, outlier_method=p.outlier_method,
                        edge_cases=p.edge_cases)
        res = gen.generate(cfg, p.rules or None, p.rule_mode)
        _step(store, jid, "writing", 2, 3)
        out = res.data if p.include_hidden else visible(res.data)
        save("synthetic.csv", out.to_csv(index=False).encode())
        save("report.json", lineage.canonical({"injection_log": res.log_dicts(), "rules": res.rules, "edge_cases": res.edge, "seed": res.seed}).encode())
        return files, {"columns": _schema(out)}, {"source": p.dataset, "rows": len(real), "sha256": lineage.sha256_bytes(real.to_csv(index=False).encode())}
    if kind == "relational":
        from sdp.relational import RelationalGenerator, infer_graph
        p = RelationalParams.model_validate(m["request"])
        _step(store, jid, "fitting", 0, 3)
        real = _real_relational(store, jid, p)
        graph = infer_graph(real)
        gen = RelationalGenerator(graph, condition_on_parents=p.condition_on_parents).fit(real, seed=p.seed)
        _step(store, jid, "generating", 1, 3)
        res = gen.generate(rows=p.rows, scale=p.scale, seed=p.seed, rules=p.rules or None, enforce=p.enforce)
        _step(store, jid, "writing", 2, 3)
        for name, df in res.tables.items():
            save(f"{name}.csv", df.to_csv(index=False).encode())
        save("scorecard.json", lineage.canonical(res.scorecard()).encode())
        save("relationships.json", lineage.canonical(graph.to_dict()).encode())
        h = lineage.sha256_bytes(lineage.canonical({k: v.to_csv(index=False) for k, v in sorted(real.items())}).encode())
        return files, {"graph": graph.to_dict(), "tables": {k: _schema(v) for k, v in res.tables.items()}}, \
            {"source": p.dataset, "rows": {k: len(v) for k, v in real.items()}, "sha256": h}
    if kind == "nl":
        from sdp.nl import DatasetConfig, generate_dataset, parse_request
        p = NLParams.model_validate(m["request"])
        if p.config:
            cfg = DatasetConfig.model_validate({**p.config, "seed": p.seed})
        else:
            parsed = parse_request(p.text or "", p.seed)
            if not parsed.ok or parsed.config is None:
                raise ValueError("; ".join(parsed.errors))
            cfg = parsed.config
        ds = generate_dataset(cfg)
        for name, df in ds.tables.items():
            save(f"{name}.csv", df.to_csv(index=False).encode())
        save("validation.json", lineage.canonical(ds.validate()).encode())
        save("config.json", lineage.canonical(cfg.model_dump()).encode())
        return files, {"tables": {k: _schema(v) for k, v in ds.tables.items()}, "config": cfg.model_dump()}, \
            {"source": "generated", "rows": cfg.rows, "sha256": cfg.config_hash()}
    if kind == "document":
        from sdp.documents.pipeline import DocumentPipeline
        p = DocumentParams.model_validate(m["request"])
        specs = p.specs if p.specs else [{"doc_type": p.doc_type, **p.spec, "seed": p.seed + i} for i in range(p.count)]
        out_dir = store.path(jid) / "docs"
        pipe = DocumentPipeline(out_dir=out_dir, workers=p.workers, batch_size=25)
        _progress(store, jid, 0, len(specs), "rendering", force=True)
        report = pipe.run(specs, on_progress=lambda done, rep: _progress(store, jid, done, len(specs), "rendering", rep.failed),
                          should_cancel=lambda: _cancelled(jid))
        # index of everything produced (hashes make the batch reproducible); timing is left out so the hash is stable
        entries = []
        for f in sorted(out_dir.glob("*.json")):
            if f.name.endswith(".report.json"):
                continue
            pdf = out_dir / f"{f.stem}.pdf"
            if pdf.exists():
                entries.append({"doc_id": f.stem, "json_sha256": lineage.sha256_bytes(f.read_bytes()), "pdf_sha256": lineage.sha256_bytes(pdf.read_bytes())})
        rep = report.to_dict()
        for k in ("duration_s", "started_at"):
            rep.pop(k, None)
        save("index.json", lineage.canonical(entries).encode())
        save("report.json", lineage.canonical(rep).encode())
        info = ({"doc_types": sorted({(x.get("doc_type", "invoice") if isinstance(x, dict) else "?") for x in specs})}, {"source": "generated", "rows": len(specs), "sha256": None})
        if report.cancelled:
            raise Cancelled(files, {"documents": len(entries), **info[0]}, info[1])
        if report.total and report.succeeded == 0 and report.failed:
            first = report.failures[0]
            raise ValueError(f"all {report.failed} documents failed; first: {first['stage']}: {first['message']}")
        return files, {"documents": len(entries), **info[0]}, info[1]
    if kind == "pack":
        from sdp.contracts import generate_pack, validate_pack
        p = PackParams.model_validate(m["request"])
        _step(store, jid, "generating", 0, 2)
        tables = generate_pack(p.pack, p.rows, p.seed, p.locale)
        _step(store, jid, "checking the data contract", 1, 2)
        report = validate_pack(p.pack, tables)
        for name, df in tables.items():
            save(f"{name}.csv", df.to_csv(index=False).encode())
        rep = {k: v for k, v in report.items() if k != "meta"}          # run time is left out so the output hash is reproducible
        save("contract_report.json", lineage.canonical(rep).encode())
        return files, {"tables": {k: _schema(v) for k, v in tables.items()}}, {"source": p.pack, "rows": {k: len(v) for k, v in tables.items()}, "sha256": None}
    if kind == "large":
        from sdp.scale import generate_large
        from sdp.tabular import TabularGenerator
        p = LargeParams.model_validate(m["request"])
        _step(store, jid, "fitting", 0, p.rows)
        real = _real_tabular(store, jid, p)
        gen = TabularGenerator().fit(real)
        out = store.path(jid) / f"synthetic.{p.format}"
        rep = generate_large(gen, p.rows, out, seed=p.seed, fmt=p.format, chunk_rows=p.chunk_rows, workers=p.workers, executor=p.executor, compress=p.compress,
                             sequence_columns=p.sequence_columns, null_rate=p.null_rate, outlier_rate=p.outlier_rate,
                             export_options={"dialect": p.dialect} if p.format == "sql" else None,
                             on_progress=lambda done, total: _progress(store, jid, done, total, "generating"), should_cancel=lambda: _cancelled(jid))
        files.append({"name": rep.file, "sha256": rep.sha256, "bytes": rep.bytes})
        info = {"source": p.dataset, "rows": len(real), "sha256": lineage.sha256_bytes(real.to_csv(index=False).encode()),
                "throughput": {k: v for k, v in rep.to_dict().items() if k != "per_chunk_seconds"}}
        if rep.cancelled:
            raise Cancelled(files, {"rows_written": rep.rows}, info)
        return files, {"rows_written": rep.rows, "format": p.format}, info
    raise ValueError(kind)


def _execute(store: Store, job_id: str) -> None:
    if _cancelled(job_id):  # cancelled while still queued
        store.update(job_id, status="cancelled", finished_at=lineage.now(), progress={**store.read(job_id)["progress"], "phase": "cancelled"})
        return
    store.update(job_id, status="running", started_at=lineage.now())
    m = store.read(job_id)
    try:
        files, schema, dataset = _generate(store, m)
        oh = lineage.output_hash(files)
        parent_id = m["lineage"]["parent_id"]
        lin: dict[str, Any] = {}
        if parent_id:
            parent = store.read(parent_id)
            same_request = parent["request_hash"] == m["request_hash"]
            if same_request and parent["outputs"]["output_hash"] and parent["status"] == "succeeded":
                lin = {"reproduced": parent["outputs"]["output_hash"] == oh,
                       "code_changed": parent["model"]["code_fingerprint"] != m["model"]["code_fingerprint"]}
        prog = store.read(job_id)["progress"]
        store.update(job_id, status="succeeded", finished_at=lineage.now(), schema=schema, dataset=dataset,
                     outputs={"files": files, "output_hash": oh}, lineage=lin,
                     progress={**prog, "done": prog["total"] if prog.get("total") is not None else prog.get("done"), "phase": "done"})
    except Cancelled as c:
        prog = store.read(job_id)["progress"]
        store.update(job_id, status="cancelled", finished_at=lineage.now(), schema=c.schema, dataset=c.dataset,
                     outputs={"files": c.files, "output_hash": None}, progress={**prog, "phase": "cancelled"})
    except Exception as e:  # noqa: BLE001 - a failed job is recorded, never raised into the worker thread
        store.update(job_id, status="failed", finished_at=lineage.now(), error=f"{type(e).__name__}: {e}",
                     debug=traceback.format_exc(limit=4), progress={**store.read(job_id)["progress"], "phase": "failed"})
    finally:
        _FUTURES.pop(job_id, None)
        _EVENTS.pop(job_id, None)
        _LAST_WRITE.pop(job_id, None)


# --------------------------------------------------------------------- API
def submit(store: Store, kind: str, params: dict[str, Any], parent_id: str | None = None, wait: bool = False) -> dict[str, Any]:
    request = validate_params(kind, params)
    m = store.create(kind, request, parent_id)
    request = _materialize(store, m["id"], request, parent_id)
    m = store.update(m["id"], request=request, request_hash=lineage.sha256_bytes(lineage.canonical(request).encode())[:16], seed=request.get("seed"))
    if kind == "large":
        m = store.update(m["id"], progress={**m["progress"], "total": request["rows"]})
    if kind == "document":  # a queued bulk job already knows how big it is
        n = len(request["specs"]) if request.get("specs") else request.get("count", 1)
        m = store.update(m["id"], progress={**m["progress"], "total": n})
    _EVENTS[m["id"]] = threading.Event()
    if wait:
        _execute(store, m["id"])
    else:
        _FUTURES[m["id"]] = EXECUTOR.submit(_execute, store, m["id"])
    return store.read(m["id"])


def cancel(store: Store, job_id: str) -> dict[str, Any]:
    """Cooperative cancel: a queued job never starts; a running job stops at its next checkpoint and keeps what it finished."""
    m = store.read(job_id)
    if m["status"] in TERMINAL:
        raise ValueError(f"job is already {m['status']}")
    store.update(job_id, cancel_requested=True, status="cancelling" if m["status"] == "running" else m["status"])
    ev = _EVENTS.get(job_id)
    if ev is not None:
        ev.set()
    fut = _FUTURES.get(job_id)
    if fut is not None and fut.cancel():  # was still waiting in the queue
        store.update(job_id, status="cancelled", finished_at=lineage.now(), progress={**store.read(job_id)["progress"], "phase": "cancelled"})
        _FUTURES.pop(job_id, None)
        _EVENTS.pop(job_id, None)
    elif ev is None:  # not owned by this process (server restarted): nothing is running it
        store.update(job_id, status="cancelled", finished_at=lineage.now(), progress={**store.read(job_id)["progress"], "phase": "cancelled"})
    return store.read(job_id)


def progress(store: Store, job_id: str) -> dict[str, Any]:
    m = store.read(job_id)
    p = m.get("progress") or {}
    done, total = p.get("done") or 0, p.get("total")
    started = m.get("started_at")
    elapsed = None
    if started:
        from datetime import datetime, timezone
        end = m["finished_at"] or lineage.now()
        elapsed = max(0.0, (datetime.fromisoformat(end) - datetime.fromisoformat(started)).total_seconds())
    rate = done / elapsed if elapsed and done else None
    return {"job_id": job_id, "status": m["status"], "phase": p.get("phase"), "done": done, "total": total,
            "percent": (100.0 * done / total) if total else (100.0 if m["status"] == "succeeded" else 0.0),
            "failed": p.get("failed", 0), "elapsed_s": elapsed, "items_per_s": rate,
            "eta_s": ((total - done) / rate) if rate and total and m["status"] in ("running", "cancelling") else None,
            "cancel_requested": m.get("cancel_requested", False), "finished": m["status"] in TERMINAL}


def zip_job(store: Store, job_id: str) -> Path:
    """ZIP of a finished (or cancelled) job's outputs plus its manifest. Private input files are excluded. Caller deletes the file."""
    m = store.read(job_id)
    if m["status"] not in TERMINAL:
        raise ValueError(f"job is {m['status']}; download is available once it has finished")
    fd, name = tempfile.mkstemp(suffix=".zip", prefix=f"sdp-{job_id}-")
    os.close(fd)  # keep only the path: an open handle would stop Windows from deleting the file after the download
    tmp = Path(name)
    root = store.path(job_id)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(root.rglob("*")):
            rel = f.relative_to(root).as_posix()
            if f.is_file() and not rel.startswith("input") and not f.name.startswith(".") and not f.suffix == ".tmp":
                z.write(f, rel)
    return tmp


def recover(store: Store) -> int:
    """At startup: jobs that were queued/running when the process died cannot continue; mark them so the UI does not wait forever."""
    n = 0
    for m in store.list():
        if m["status"] in ("queued", "running", "cancelling"):
            store.update(m["id"], status="failed", finished_at=lineage.now(), error="interrupted: the server restarted while this job was running",
                         progress={**m.get("progress", {}), "phase": "interrupted"})
            n += 1
    return n


def rerun(store: Store, job_id: str, wait: bool = False, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """New job from a manifest. Without overrides it is a reproducibility check (manifest.lineage.reproduced)."""
    src = store.read(job_id)
    params = {**src["request"], **(overrides or {})}
    return submit(store, src["kind"], params, parent_id=job_id, wait=wait)


def score(store: Store, job_id: str, refresh: bool = False) -> dict[str, Any]:
    m = store.read(job_id)
    if m["status"] != "succeeded":
        raise ValueError(f"job is {m['status']}; scores exist only for succeeded jobs")
    if m["scores"] and not refresh:
        return m["scores"]
    result = _score(store, m)
    store.update(job_id, scores=result)
    return result


def _label(s: float) -> str:
    return "Strong" if s >= 85 else "Good, with caveats" if s >= 70 else "Needs work" if s >= 50 else "Not recommended"


def nl_quality(val: dict[str, Any]) -> dict[str, Any]:
    """Quality score for a dataset described in words. There is no real data to compare with, so fidelity and privacy do not apply: the score
    covers what can be checked (valid values for the locale, rules, keys) and reports whether the requested flag rate was met."""
    loc, con = float(val["locale"]["valid_pct"]), float(val["constraints"]["pass_pct"])
    viol = int(val["integrity"]["total_violations"])
    integ = 100.0 if viol == 0 else 0.0
    score = 0.4 * loc + 0.3 * con + 0.3 * integ
    comps = [
        {"key": "locale", "label": "Values valid for the country", "score": loc, "weight": 0.4, "summary": f"{val['locale']['n_rows']:,} records checked (names, phones, IDs, dates)"},
        {"key": "rules", "label": "Rules and constraints", "score": con, "weight": 0.3, "summary": f"{val['constraints']['n_rules']} rules checked on every row"},
        {"key": "integrity", "label": "Keys and links between tables", "score": integ, "weight": 0.3, "summary": "every link points to a real record" if viol == 0 else f"{viol} broken links"},
    ]
    flag = val.get("flag")
    if flag:
        want, got = float(flag["requested"]), float(flag["achieved"])
        acc = 100.0 if want == 0 and got == 0 else max(0.0, 100.0 - abs(got - want) / max(want, 1e-9) * 100.0)
        comps.append({"key": "flag", "label": "Requested rate met", "score": acc, "weight": 0.0,
                      "summary": f"{flag['name']}: asked {want:.1%}, got {got:.1%} ({flag['count']} rows). Shown for information, not scored"})
    return {"score": score, "label": _label(score), "components": comps,
            "note": "Described datasets have no real data to compare with, so this is a quality score (validity of the data), not a Trust Score with fidelity and privacy."}


def _score(store: Store, m: dict[str, Any]) -> dict[str, Any]:
    jid, kind = m["id"], m["kind"]
    if kind == "tabular":
        from sdp.scoring import build_trust_report
        p = TabularParams.model_validate(m["request"])
        real = _real_tabular(store, jid, p)
        synth = _read_csv(store, jid, "synthetic.csv")
        synth = synth[[c for c in synth.columns if not c.startswith("_")]]
        keep = [c for c in real.columns if c in synth.columns]
        rep = build_trust_report(real[keep], synth[keep], dsl_rules=p.rules or None, seed=p.seed, title=f"job {jid}")
        return {"score": rep["trust_score"], "label": rep["label"], "kind": "trust", "report": rep}
    if kind == "relational":
        from sdp.relational import infer_graph
        from sdp.relational.metrics import relation_metrics
        from sdp.scoring import build_trust_report_relational
        p = RelationalParams.model_validate(m["request"])
        real = _real_relational(store, jid, p)
        graph = infer_graph(real)
        synth = {n: _read_csv(store, jid, f"{n}.csv") for n in real}
        card = json.loads(store.artifact(jid, "scorecard.json"))
        rel = relation_metrics(real, synth, graph)
        rep = build_trust_report_relational(real, synth, graph, relation=rel, rules_report=card.get("rules"),
                                            integrity=card["integrity"], title=f"job {jid}", seed=p.seed)
        return {"score": rep["trust_score"], "label": rep["label"], "kind": "trust", "report": rep}
    if kind == "nl":
        val = json.loads(store.artifact(jid, "validation.json"))
        q = nl_quality(val)
        return {"score": q["score"], "label": q["label"], "kind": "validation",
                "report": {"locale_valid_pct": val["locale"]["valid_pct"], "constraints_pass_pct": val["constraints"]["pass_pct"],
                           "integrity_violations": val["integrity"]["total_violations"], "flag": val.get("flag"), "components": q["components"], "note": q["note"]}}
    if kind == "pack":
        rep = json.loads(store.artifact(jid, "contract_report.json"))
        s = float(rep["statistics"]["success_percent"])
        return {"score": s, "label": _label(s), "kind": "contract",
                "report": {"passed": rep["success"], "checks": rep["statistics"]["evaluated_expectations"], "failed": rep["statistics"]["unsuccessful_expectations"],
                           "note": "share of data-contract checks the generated data meets"}}
    if kind == "large":
        p = LargeParams.model_validate(m["request"])
        if p.format != "csv" or p.compress:
            raise ValueError("scoring a large job needs uncompressed csv output (it scores the first 20,000 rows)")
        from sdp.scoring import build_trust_report
        real = _real_tabular(store, jid, p)
        synth = pd.read_csv(store.path(jid) / "synthetic.csv", nrows=20_000)
        for c in synth.columns:                      # same date recovery as _read_csv
            if synth[c].dtype == object or str(synth[c].dtype) == "str":
                try:
                    synth[c] = pd.to_datetime(synth[c], errors="raise", format="ISO8601")
                except (ValueError, TypeError):
                    pass
        keep = [c for c in real.columns if c in synth.columns]
        rep = build_trust_report(real[keep], synth[keep], seed=p.seed, title=f"job {jid} (first {len(synth):,} rows)")
        return {"score": rep["trust_score"], "label": rep["label"], "kind": "trust", "report": rep}
    if kind == "document":
        rep = json.loads(store.artifact(jid, "report.json"))
        ok = rep["succeeded"] + rep["skipped"]
        s = 100.0 * ok / rep["total"] if rep["total"] else 100.0
        return {"score": s, "label": _label(s), "kind": "reconciliation",
                "report": {"documents": rep["total"], "succeeded": rep["succeeded"], "failed": rep["failed"], "by_stage": rep["by_stage"],
                           "note": "every produced document reconciled exactly; failures are isolated and listed in report.json"}}
    raise ValueError(kind)


def copy_outputs(store: Store, job_id: str, out_dir: str) -> list[str]:
    m = store.read(job_id)
    dest = __import__("pathlib").Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    names = []
    for f in m["outputs"]["files"]:
        shutil.copyfile(store.path(job_id) / f["name"], dest / f["name"])
        names.append(f["name"])
    shutil.copyfile(store.path(job_id) / "manifest.json", dest / "manifest.json")
    return names


__all__ = ["JobNotFound", "Store", "submit", "rerun", "score", "validate_params", "copy_outputs", "KINDS", "cancel", "progress", "zip_job", "recover"]
