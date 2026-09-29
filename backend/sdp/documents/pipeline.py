"""D5. Robust document pipeline.

Stages per document:  validate -> generate -> render -> post_validate -> export
Guarantees
  * per-document isolation: any exception is caught, recorded and the batch continues
  * retries with exponential backoff (+ jitter) for *transient* failures only (TransientError, OSError, TimeoutError);
    validation/rendering/rule failures are deterministic, so retrying them would only waste time
  * idempotent: job_id = hash of the specs; doc_id = hash of the validated spec; existing outputs are skipped on
    re-run, duplicates inside a batch are skipped, files are written atomically (tmp + rename)
  * structured JSON-lines logs (logger "sdp.pipeline") and a failure report grouped by stage and error type
  * batching: specs are processed in chunks of `batch_size`, each chunk on a thread pool
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from sdp.documents.render import MissingFontError, RenderResult, count_pdf_pages, render_document, resolve_font
from sdp.documents.types import DOC_TYPES

logger = logging.getLogger("sdp.pipeline")
STAGES = ("validate", "generate", "render", "post_validate", "export")


class TransientError(Exception):
    """Raise for failures worth retrying (I/O hiccups, timeouts, rate limits)."""


class PostValidationError(Exception):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


class StageFailure(Exception):
    def __init__(self, stage: str, cause: BaseException, attempts: int) -> None:
        super().__init__(f"{stage}: {cause}")
        self.stage, self.cause, self.attempts = stage, cause, attempts


@dataclass
class RetryPolicy:
    retries: int = 2
    base_delay: float = 0.05
    factor: float = 2.0
    max_delay: float = 2.0
    jitter: float = 0.1

    def delay(self, attempt: int) -> float:
        d = min(self.max_delay, self.base_delay * self.factor ** (attempt - 1))
        return d * (1 + random.uniform(-self.jitter, self.jitter))


@dataclass
class DocResult:
    index: int
    key: str                       # doc_id when the spec validated, else a hash of the raw input
    status: str                    # ok | failed | skipped
    stage: str | None = None       # failing stage
    error_type: str | None = None
    message: str | None = None
    attempts: int = 1
    warnings: list[str] = field(default_factory=list)
    pages: int | None = None
    total: str | None = None
    duration_ms: float = 0.0
    reason: str | None = None      # for skipped: existing | duplicate


@dataclass
class BatchReport:
    job_id: str
    total: int
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    warnings: int = 0
    duration_s: float = 0.0
    started_at: str = ""
    cancelled: bool = False
    failures: list[dict[str, Any]] = field(default_factory=list)
    by_stage: dict[str, int] = field(default_factory=dict)
    by_error: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def job_id_for(specs: list[Any]) -> str:
    return "job-" + hashlib.sha256(canonical(specs).encode()).hexdigest()[:16]


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
                   "level": record.levelname, "msg": record.getMessage(), **getattr(record, "fields", {})}
        return json.dumps(payload, default=str)


def structured_logging(level: int = logging.INFO, stream=None) -> None:
    h = logging.StreamHandler(stream)
    h.setFormatter(JsonLogFormatter())
    logger.handlers[:] = [h]
    logger.setLevel(level)
    logger.propagate = False


def _log(level: int, msg: str, **fields: Any) -> None:
    logger.log(level, msg, extra={"fields": fields})


@dataclass
class Artifacts:
    data: dict[str, Any]
    pdf: bytes


class DocumentPipeline:
    def __init__(self, out_dir: str | Path | None = None, retry: RetryPolicy | None = None, workers: int = 4,
                 batch_size: int = 100, max_pages: int = 10, fallback_font: str | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 generate_fn: Callable[[Any], dict] | None = None,
                 render_fn: Callable[..., RenderResult] = render_document) -> None:
        self.out_dir = Path(out_dir) if out_dir else None
        self.retry, self.workers, self.batch_size, self.max_pages = retry or RetryPolicy(), workers, batch_size, max_pages
        self.fallback_font, self.sleep = fallback_font, sleep
        self.generate_fn, self.render_fn = generate_fn, render_fn
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------ batch
    def run(self, specs: list[Any], force: bool = False, on_progress: Callable[[int, BatchReport], None] | None = None,
            should_cancel: Callable[[], bool] | None = None) -> BatchReport:
        """on_progress(done, report_so_far) is called after every chunk; should_cancel() is polled before each chunk, so a
        cancel stops within one chunk (batch_size documents) and everything finished so far is kept."""
        t0 = time.perf_counter()
        job = job_id_for(specs)
        report = BatchReport(job_id=job, total=len(specs), started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self._seen = set()
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        _log(logging.INFO, "batch started", job_id=job, total=len(specs), batch_size=self.batch_size, workers=self.workers)
        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as pool:
            for start in range(0, len(specs), self.batch_size):
                if should_cancel is not None and should_cancel():
                    report.cancelled = True
                    _log(logging.INFO, "batch cancelled", job_id=job, done=start, total=len(specs))
                    break
                chunk = list(enumerate(specs[start:start + self.batch_size], start))
                for res in pool.map(lambda iv: self.process_one(iv[1], iv[0], job, force)[0], chunk):
                    self._tally(report, res)
                _log(logging.INFO, "chunk done", job_id=job, done=min(start + self.batch_size, len(specs)), total=len(specs))
                if on_progress is not None:
                    on_progress(min(start + self.batch_size, len(specs)), report)
        report.duration_s = round(time.perf_counter() - t0, 3)
        _log(logging.INFO, "batch finished", job_id=job, ok=report.succeeded, failed=report.failed, skipped=report.skipped,
             duration_s=report.duration_s)
        if self.out_dir:
            self._write_atomic(self.out_dir / f"{job}.report.json", report.to_json().encode())
        return report

    @staticmethod
    def _tally(report: BatchReport, r: DocResult) -> None:
        report.warnings += len(r.warnings)
        if r.status == "ok":
            report.succeeded += 1
        elif r.status == "skipped":
            report.skipped += 1
        else:
            report.failed += 1
            report.by_stage[r.stage or "unknown"] = report.by_stage.get(r.stage or "unknown", 0) + 1
            report.by_error[r.error_type or "unknown"] = report.by_error.get(r.error_type or "unknown", 0) + 1
            report.failures.append({"index": r.index, "doc": r.key, "stage": r.stage, "error_type": r.error_type,
                                    "message": r.message, "attempts": r.attempts})

    # ------------------------------------------------------- one document
    def process_one(self, raw: Any, index: int = 0, job_id: str = "adhoc", force: bool = False) -> tuple[DocResult, Artifacts | None]:
        t0 = time.perf_counter()
        key = "raw-" + hashlib.sha256(canonical(raw).encode()).hexdigest()[:12]
        attempts_total, warnings = 1, []
        try:
            (doc_type, spec), _ = self._stage("validate", lambda: self._validate(raw))
            key = spec.doc_id()
            with self._lock:
                dup = key in self._seen
                self._seen.add(key)
            if dup:
                return DocResult(index, key, "skipped", reason="duplicate", duration_ms=_ms(t0)), None
            if self.out_dir and not force and (self.out_dir / f"{key}.json").exists() and (self.out_dir / f"{key}.pdf").exists():
                _log(logging.INFO, "skipped existing", job_id=job_id, doc=key)
                return DocResult(index, key, "skipped", reason="existing", duration_ms=_ms(t0)), None
            data, a = self._stage("generate", lambda: (self.generate_fn or DOC_TYPES[doc_type].generate)(spec))
            attempts_total = max(attempts_total, a)
            rendered, a = self._stage("render", lambda: self._render(data, spec, warnings))
            attempts_total = max(attempts_total, a)
            self._stage("post_validate", lambda: self._post_validate(data, rendered, doc_type))
            data = {**data, "doc_id": key}
            if self.out_dir:
                _, a = self._stage("export", lambda: self._export(key, data, rendered.pdf))
                attempts_total = max(attempts_total, a)
            warnings += rendered.warnings
            res = DocResult(index, key, "ok", attempts=attempts_total, warnings=warnings, pages=rendered.pages,
                            total=data.get("total", data.get("closing_balance", data.get("net_pay"))), duration_ms=_ms(t0))
            _log(logging.DEBUG, "document ok", job_id=job_id, doc=key, pages=rendered.pages)
            return res, Artifacts(data, rendered.pdf)
        except StageFailure as f:
            _log(logging.WARNING, "document failed", job_id=job_id, doc=key, stage=f.stage,
                 error=type(f.cause).__name__, message=str(f.cause)[:300], attempts=f.attempts)
            return DocResult(index, key, "failed", f.stage, type(f.cause).__name__, str(f.cause)[:500], f.attempts,
                             warnings, duration_ms=_ms(t0)), None
        except Exception as e:  # isolation of last resort: nothing may escape one document
            _log(logging.ERROR, "unexpected error", job_id=job_id, doc=key, error=type(e).__name__, message=str(e)[:300])
            return DocResult(index, key, "failed", "unexpected", type(e).__name__, str(e)[:500], duration_ms=_ms(t0)), None

    # ------------------------------------------------------------ stages
    def _stage(self, name: str, fn: Callable[[], Any]) -> tuple[Any, int]:
        attempt = 0
        while True:
            attempt += 1
            try:
                return fn(), attempt
            except (TransientError, OSError, TimeoutError) as e:
                if attempt > self.retry.retries:
                    raise StageFailure(name, e, attempt) from e
                delay = self.retry.delay(attempt)
                _log(logging.INFO, "retrying", stage=name, attempt=attempt, delay_s=round(delay, 3), error=type(e).__name__)
                self.sleep(delay)
            except Exception as e:
                raise StageFailure(name, e, attempt) from e

    @staticmethod
    def _validate(raw: Any) -> tuple[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError(f"spec must be an object, got {type(raw).__name__}")
        raw = dict(raw)
        doc_type = raw.pop("doc_type", "invoice")
        if doc_type not in DOC_TYPES:
            raise ValueError(f"unknown doc_type {doc_type!r}; available: {sorted(DOC_TYPES)}")
        try:
            return doc_type, DOC_TYPES[doc_type].spec_cls.model_validate(raw)
        except ValidationError as e:
            raise ValueError("; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())) from e

    def _render(self, data: dict, spec: Any, warnings: list[str]) -> RenderResult:
        font = spec.font
        try:
            if font not in ("Helvetica", "auto"):
                resolve_font(font)
            return self.render_fn(data, font=font)
        except MissingFontError:
            if not self.fallback_font:
                raise
            warnings.append(f"font {font!r} unavailable; used {self.fallback_font!r}")
            return self.render_fn(data, font=self.fallback_font, strict=True)

    def _post_validate(self, data: dict, r: RenderResult, doc_type: str = "invoice") -> None:
        dt = DOC_TYPES[doc_type]
        def get(path: str) -> Any:
            cur: Any = data
            for part in path.split("."):
                cur = cur.get(part) if isinstance(cur, dict) else None
            return cur
        problems = [f"missing required field {k}" for k in dt.required if get(k) in (None, "")]
        problems += [f"missing {k}" for k in ("lines",) if doc_type in ("invoice", "receipt") and not data.get(k)]
        if doc_type == "statement" and not data.get("transactions") and not data.get("filter"):
            problems.append("no transactions")  # an empty *filtered* statement is a valid answer to the query
        if not problems:
            problems += [f"totals: {e}" for e in dt.reconcile(data)]
        if r.unrenderable:
            problems.append(f"font cannot render characters {r.unrenderable}")
        if (data.get("presentation") or {}).get("native") and not r.embedded_font:
            problems.append("native-script document must embed a Unicode font file")
        if not (r.pdf.startswith(b"%PDF") and r.pdf.rstrip().endswith(b"%%EOF")):
            problems.append("PDF header/trailer malformed")
        real_pages = count_pdf_pages(r.pdf)
        if real_pages != r.pages:
            problems.append(f"page count mismatch: renderer says {r.pages}, PDF has {real_pages}")
        if not 1 <= r.pages <= self.max_pages:
            problems.append(f"page count {r.pages} outside 1..{self.max_pages}")
        if problems:
            raise PostValidationError(problems)

    def _export(self, key: str, data: dict, pdf: bytes) -> None:
        assert self.out_dir is not None
        self._write_atomic(self.out_dir / f"{key}.pdf", pdf)
        self._write_atomic(self.out_dir / f"{key}.json", json.dumps(data, indent=2).encode())  # json last: marks completion

    @staticmethod
    def _write_atomic(path: Path, payload: bytes) -> None:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, path)


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)
