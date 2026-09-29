"""P6. Large-scale generation: chunked, parallel, streaming, memory-bounded, with throughput reporting.

The fitted model is sampled chunk by chunk. Each chunk has its own seed derived from (seed, chunk index), so the output is identical for any
worker count. Chunks are generated on a thread pool with a bounded in-flight window and written IN ORDER straight to disk through an exporter,
so memory is `window x chunk_rows` rows regardless of the total (millions of rows stay flat). Cancel stops at a chunk boundary and leaves a
valid, complete file of what was produced.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import re
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from sdp.edgecases import visible
from sdp.exporters import get_exporter, schema_from
from sdp.exporters.base import ExportError
from sdp.tabular import GenConfig, TabularGenerator

MAX_ROWS = 100_000_000
STREAM_FORMATS = ("csv", "jsonl", "json", "sql")


class ScaleError(ValueError):
    pass


@dataclass
class ScaleReport:
    rows: int = 0
    requested: int = 0
    chunks: int = 0
    chunk_rows: int = 0
    workers: int = 0
    executor: str = "thread"
    format: str = "csv"
    file: str = ""
    bytes: int = 0
    sha256: str = ""
    seconds: float = 0.0
    rows_per_second: float = 0.0
    mb_per_second: float = 0.0
    peak_rss_mb: float | None = None
    max_rows_in_memory: int = 0
    cancelled: bool = False
    per_chunk_seconds: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class _HashingSink:
    """Counts bytes and hashes the content as it is written, so the checksum needs no second pass."""

    def __init__(self, raw: Any) -> None:
        self.raw, self.n, self.h = raw, 0, hashlib.sha256()

    def write(self, b: bytes) -> int:
        self.raw.write(b)
        self.h.update(b)
        self.n += len(b)
        return len(b)

    def flush(self) -> None:
        self.raw.flush()


def _rss_mb() -> float | None:
    try:
        import psutil  # type: ignore
        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        pass
    try:
        import resource  # type: ignore
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1e6 if os.uname().sysname == "Darwin" else r / 1e3   # noqa: PLW0108
    except (ImportError, AttributeError):
        pass
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            ctypes.windll.kernel32.K32GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
            return pmc.PeakWorkingSetSize / 1e6
        except Exception:  # noqa: BLE001
            return None
    return None


def chunk_seed(seed: int, index: int) -> int:
    return int(np.random.SeedSequence([int(seed), int(index)]).generate_state(1)[0] & 0x7FFFFFFF)


def plan(rows: int, chunk_rows: int) -> list[int]:
    if rows < 1 or rows > MAX_ROWS:
        raise ScaleError(f"rows must be between 1 and {MAX_ROWS:,}")
    if chunk_rows < 100:
        raise ScaleError("chunk_rows must be at least 100")
    full, rest = divmod(rows, chunk_rows)
    return [chunk_rows] * full + ([rest] if rest else [])


def generate_chunk(gen: TabularGenerator, n: int, seed: int, index: int, offset: int, cfg_extra: dict[str, Any], sequence_columns: list[str] | None) -> pd.DataFrame:
    cfg = GenConfig(rows=n, seed=chunk_seed(seed, index), **cfg_extra)
    df = visible(gen.generate(cfg).data).reset_index(drop=True)
    for c in sequence_columns or []:
        if c in df.columns:
            df[c] = np.arange(offset + 1, offset + n + 1)      # keys must stay unique across chunks
    return df


_PROC_GEN: TabularGenerator | None = None


def _proc_init(gen: TabularGenerator) -> None:
    global _PROC_GEN
    _PROC_GEN = gen


def _work(gen: TabularGenerator | None, n: int, seed: int, index: int, offset: int, extra: dict[str, Any], seq: list[str] | None,
          fmt: str, schema: dict[str, Any], opts: dict[str, Any]) -> tuple[bytes, int, float]:
    """Generate one chunk AND serialise it, so both run in the worker (serialisation is as expensive as generation)."""
    t = time.perf_counter()
    df = generate_chunk(gen or _PROC_GEN, n, seed, index, offset, extra, seq)  # type: ignore[arg-type]
    data = get_exporter(fmt).encode_chunk(df, schema, index, **opts)
    return data, len(df), time.perf_counter() - t


def generate_large(gen: TabularGenerator, rows: int, out_path: str | Path, seed: int = 0, fmt: str = "csv", chunk_rows: int = 100_000, workers: int = 4,
                   compress: bool = False, sequence_columns: list[str] | None = None, null_rate: dict[str, float] | None = None,
                   outlier_rate: dict[str, float] | None = None, outlier_method: str = "iqr", table_name: str = "synthetic",
                   on_progress: Callable[[int, int], None] | None = None, should_cancel: Callable[[], bool] | None = None,
                   export_options: dict[str, Any] | None = None, executor: str = "auto") -> ScaleReport:
    """Stream `rows` synthetic rows to `out_path` (`.gz` appended when compress). Returns throughput and memory figures.

    executor="process" sidesteps the GIL (each worker gets a copy of the fitted model) and is ~2.4x faster from a few million rows, but pays a
    multi-second start-up; "thread" has none. "auto" picks process from 1,000,000 rows. Output is identical either way."""
    if fmt not in STREAM_FORMATS:
        raise ScaleError(f"streaming formats: {list(STREAM_FORMATS)}")
    if not 1 <= workers <= 16:
        raise ScaleError("workers must be between 1 and 16")
    if executor not in ("auto", "thread", "process"):
        raise ScaleError("executor must be 'auto', 'thread' or 'process'")
    if executor == "auto":
        executor = "process" if rows >= 1_000_000 and workers > 1 else "thread"
    sizes = plan(rows, chunk_rows)
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int).tolist()
    extra = {"null_rate": null_rate or {}, "outlier_rate": outlier_rate or {}, "outlier_method": outlier_method}
    opts = dict(export_options or {})
    window = max(2, workers * 2)
    exporter = get_exporter(fmt)
    if not hasattr(exporter, "encode_chunk"):
        raise ScaleError(f"format {fmt!r} cannot be streamed in chunks")
    out_path = Path(out_path)
    if compress:
        out_path = out_path.with_name(out_path.name + ".gz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rep = ScaleReport(executor=executor, requested=rows, chunk_rows=chunk_rows, workers=workers, format=fmt, file=out_path.name, max_rows_in_memory=window * chunk_rows)
    t0 = time.perf_counter()
    peak = _rss_mb() or 0.0

    # chunk 0 is produced up front: it fixes the schema and (when not given) which integer columns are keys renumbered globally
    df0 = generate_chunk(gen, sizes[0], seed, 0, 0, extra, sequence_columns)
    seq_cols = sequence_columns
    if seq_cols is None:
        seq_cols = [c for c in df0.columns if pd.api.types.is_integer_dtype(df0[c]) and len(df0) > 1 and (df0[c].is_unique or re.search(r"(^|_)id$", str(c).lower()))]
        for c in seq_cols:
            df0[c] = np.arange(1, len(df0) + 1)
    schema = _schema(df0, table_name, set(null_rate or {}))
    try:
        head = exporter.stream_begin(schema, **opts)
    except ExportError as e:
        raise ScaleError(str(e)) from e
    first_bytes = exporter.encode_chunk(df0, schema, 0, **opts)
    first_dt = time.perf_counter() - t0

    file_obj = open(out_path, "wb")
    raw = gzip.GzipFile(filename="", fileobj=file_obj, mode="wb", compresslevel=5, mtime=0) if compress else file_obj   # mtime=0: reproducible bytes
    sink = _HashingSink(raw)
    done = 0
    if executor == "process":
        ex: Any = ProcessPoolExecutor(max_workers=workers, initializer=_proc_init, initargs=(gen,))
        wgen = None
    else:
        ex = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sdp-chunk")
        wgen = gen
    pending: deque = deque()
    try:
        sink.write(head)
        sink.write(first_bytes)
        done += len(df0)
        rep.chunks = 1
        rep.per_chunk_seconds["0"] = round(first_dt, 4)
        if on_progress:
            on_progress(done, rows)
        nxt = 1
        cancelled = bool(should_cancel and should_cancel())
        while (nxt < len(sizes) or pending) and not cancelled:
            while nxt < len(sizes) and len(pending) < window:
                pending.append((nxt, ex.submit(_work, wgen, sizes[nxt], seed, nxt, offsets[nxt], extra, seq_cols, fmt, schema, opts)))
                nxt += 1
            i, fut = pending.popleft()
            data, n, dt = fut.result()
            sink.write(data)                                   # strictly in chunk order: the file does not depend on worker count
            done += n
            rep.chunks += 1
            rep.per_chunk_seconds[str(i)] = round(dt, 4)
            p = _rss_mb()
            if p:
                peak = max(peak, p)
            if on_progress:
                on_progress(done, rows)
            cancelled = bool(should_cancel and should_cancel())
        if cancelled:
            rep.cancelled = True
        sink.write(exporter.stream_end(**opts))                # a cancelled file is still well-formed (JSON array closed, SQL committed)
    finally:
        for _, f in pending:
            f.cancel()
        ex.shutdown(wait=True, cancel_futures=True)
        raw.close()
        file_obj.close()
    rep.rows = done
    rep.bytes, rep.sha256 = sink.n, sink.h.hexdigest()
    rep.seconds = round(time.perf_counter() - t0, 3)
    rep.rows_per_second = round(rep.rows / rep.seconds, 1) if rep.seconds else 0.0
    rep.mb_per_second = round(sink.n / 1e6 / rep.seconds, 2) if rep.seconds else 0.0
    rep.peak_rss_mb = round(peak, 1) if peak else None
    return rep


def _schema(df: pd.DataFrame, name: str, nullable: set[str]) -> dict[str, Any]:
    s = schema_from({name: df})
    for c in s[name].columns:
        if c.kind == "str":
            c.max_len = 1000            # later chunks may hold longer strings than the first: TEXT is the safe type
        if c.name in nullable:
            c.nullable = True
    return s
