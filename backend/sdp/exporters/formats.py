"""Built-in exporters: csv, json, jsonl, sql, pdf, zip."""

from __future__ import annotations

import io
import json
import math
import zipfile
from datetime import date, datetime
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

from sdp.exporters.base import ExportError, TableData, TableSchema, chunks, get_exporter, register_exporter, visible_columns

BOM = b"\xef\xbb\xbf"
DIALECTS = ("postgres", "mysql", "sqlite", "ansi")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df[visible_columns(df)]


def _single(tables: dict[str, TableData]) -> tuple[str, TableData]:
    if len(tables) != 1:
        raise ExportError(f"this format holds one table; got {len(tables)} (use zip, json or sql)")
    return next(iter(tables.items()))


def _jsonable(v: Any) -> Any:
    if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NaT or v is pd.NA:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if math.isnan(float(v)) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return v.isoformat()
    return v


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    cols = list(df.columns)
    return [{c: _jsonable(v) for c, v in zip(cols, row)} for row in df.itertuples(index=False, name=None)]


@register_exporter
class CsvExporter:
    name, extension, mime = "csv", "csv", "text/csv"

    def export(self, tables: dict[str, TableData], schema: dict[str, TableSchema], sink: BinaryIO, bom: bool = False, **_: Any) -> dict[str, Any]:
        if len(tables) > 1:                                  # several tables -> one CSV each inside a zip
            return get_exporter("zip").export(tables, schema, sink, formats=["csv"], bom=bom)
        _, data = _single(tables)
        rows, first = 0, True
        if bom:
            sink.write(BOM)
        for ch in chunks(data):
            sink.write(_clean(ch).to_csv(index=False, header=first, lineterminator="\n").encode("utf-8"))
            first, rows = False, rows + len(ch)
        return {"rows": rows}

    # chunk-level protocol used by sdp.scale so workers can serialise in parallel; ordered concatenation == export()
    def stream_begin(self, schema: dict[str, TableSchema], bom: bool = False, **_: Any) -> bytes:
        return BOM if bom else b""

    def encode_chunk(self, df: pd.DataFrame, schema: dict[str, TableSchema], index: int, **_: Any) -> bytes:
        return _clean(df).to_csv(index=False, header=index == 0, lineterminator="\n").encode("utf-8")

    def stream_end(self, **_: Any) -> bytes:
        return b""


@register_exporter
class JsonExporter:
    name, extension, mime = "json", "json", "application/json"

    def export(self, tables: dict[str, TableData], schema: dict[str, TableSchema], sink: BinaryIO, bom: bool = False, **_: Any) -> dict[str, Any]:
        """{"table": [rows...]} streamed table by table; a single table is written as a plain array."""
        class _W:                                           # only .write is needed, so any sink with write(bytes) works (hashing, gzip)
            write = staticmethod(lambda t: sink.write(t.encode("utf-8")))
        w = _W
        counts: dict[str, int] = {}
        if bom:
            sink.write(BOM)
        multi = len(tables) > 1
        w.write("{" if multi else "")
        for ti, (name, data) in enumerate(tables.items()):
            w.write((("," if ti else "") + json.dumps(name, ensure_ascii=False) + ":") if multi else "")
            w.write("[")
            n = 0
            for ch in chunks(data):
                for r in _records(_clean(ch)):
                    w.write(("," if n else "") + json.dumps(r, ensure_ascii=False))
                    n += 1
            w.write("]")
            counts[name] = n
        w.write("}" if multi else "")
        return {"rows": counts}

    def stream_begin(self, schema: dict[str, TableSchema], bom: bool = False, **_: Any) -> bytes:
        return (BOM if bom else b"") + b"["

    def encode_chunk(self, df: pd.DataFrame, schema: dict[str, TableSchema], index: int, **_: Any) -> bytes:
        body = ",".join(json.dumps(r, ensure_ascii=False) for r in _records(_clean(df)))
        return (("," if index else "") + body).encode("utf-8")

    def stream_end(self, **_: Any) -> bytes:
        return b"]"


@register_exporter
class JsonlExporter:
    name, extension, mime = "jsonl", "jsonl", "application/x-ndjson"

    def export(self, tables: dict[str, TableData], schema: dict[str, TableSchema], sink: BinaryIO, bom: bool = False, **_: Any) -> dict[str, Any]:
        _, data = _single(tables)
        n = 0
        if bom:
            sink.write(BOM)
        for ch in chunks(data):
            for r in _records(_clean(ch)):
                sink.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))
                n += 1
        return {"rows": n}

    def stream_begin(self, schema: dict[str, TableSchema], bom: bool = False, **_: Any) -> bytes:
        return BOM if bom else b""

    def encode_chunk(self, df: pd.DataFrame, schema: dict[str, TableSchema], index: int, **_: Any) -> bytes:
        return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in _records(_clean(df))).encode("utf-8")

    def stream_end(self, **_: Any) -> bytes:
        return b""


# ------------------------------------------------------------------- SQL
def quote_ident(name: str, dialect: str) -> str:
    return "`" + name.replace("`", "``") + "`" if dialect == "mysql" else '"' + name.replace('"', '""') + '"'


def sql_type(c: Any, dialect: str) -> str:
    k = c.kind
    if k == "int":
        return "INTEGER" if dialect == "sqlite" else "BIGINT"
    if k == "float":
        return {"postgres": "DOUBLE PRECISION", "mysql": "DOUBLE", "sqlite": "REAL"}.get(dialect, "DOUBLE PRECISION")
    if k == "bool":
        return "TINYINT(1)" if dialect == "mysql" else ("INTEGER" if dialect == "sqlite" else "BOOLEAN")
    if k == "date":
        return "DATE" if dialect != "sqlite" else "TEXT"
    if k == "datetime":
        return {"postgres": "TIMESTAMP", "mysql": "DATETIME", "sqlite": "TEXT"}.get(dialect, "TIMESTAMP")
    n = c.max_len or 1
    return "TEXT" if n > 255 or dialect == "sqlite" else f"VARCHAR({max(16, -(-n // 16) * 16)})"


def sql_literal(v: Any, kind: str, dialect: str) -> str:
    if v is None or v is pd.NA or v is pd.NaT or (isinstance(v, float) and math.isnan(v)):
        return "NULL"
    if kind == "bool":
        b = bool(v)
        return ("1" if b else "0") if dialect in ("mysql", "sqlite") else ("TRUE" if b else "FALSE")
    if kind == "int":
        return str(int(v))
    if kind == "float":
        f = float(v)
        return "NULL" if math.isnan(f) or math.isinf(f) else repr(f)
    if isinstance(v, (pd.Timestamp, datetime, date)):
        v = v.strftime("%Y-%m-%d") if kind == "date" else pd.Timestamp(v).strftime("%Y-%m-%d %H:%M:%S")
    s = str(v).replace("\x00", "")
    if dialect == "mysql":
        s = s.replace("\\", "\\\\")
    return "'" + s.replace("'", "''") + "'"


def create_table_sql(t: TableSchema, dialect: str, drop: bool = False) -> str:
    q = lambda n: quote_ident(n, dialect)  # noqa: E731
    lines = [f"  {q(c.name)} {sql_type(c, dialect)}{'' if c.nullable and c.name not in t.primary_key else ' NOT NULL'}{' UNIQUE' if c.unique else ''}" for c in t.columns]
    if t.primary_key:
        lines.append(f"  PRIMARY KEY ({', '.join(q(c) for c in t.primary_key)})")
    for fk in t.foreign_keys:
        lines.append(f"  FOREIGN KEY ({', '.join(q(c) for c in fk.columns)}) REFERENCES {q(fk.ref_table)} ({', '.join(q(c) for c in fk.ref_columns)})")
    stmt = f"CREATE TABLE {q(t.name)} (\n" + ",\n".join(lines) + "\n);\n"
    return (f"DROP TABLE IF EXISTS {q(t.name)};\n" if drop else "") + stmt


def order_tables(schema: dict[str, TableSchema], names: list[str]) -> list[str]:
    """Parents before children (self references ignored)."""
    deps = {n: {f.ref_table for f in schema[n].foreign_keys if f.ref_table != n and f.ref_table in names} for n in names}
    order: list[str] = []
    left = list(names)
    while left:
        ready = [n for n in left if deps[n] <= set(order)]
        if not ready:
            raise ExportError(f"cyclic foreign keys among {left}")
        order += ready
        left = [n for n in left if n not in ready]
    return order


def insert_statements(t: TableSchema, data: TableData, dialect: str, batch: int = 500):
    q = lambda n: quote_ident(n, dialect)  # noqa: E731
    cols = [c.name for c in t.columns]
    kinds = {c.name: c.kind for c in t.columns}
    head = f"INSERT INTO {q(t.name)} ({', '.join(q(c) for c in cols)}) VALUES\n"
    buf: list[str] = []
    for ch in chunks(data):                                   # rows are re-batched across chunks: output does not depend on chunk size
        ch = ch[[c for c in cols if c in ch.columns]]
        names = list(ch.columns)
        for r in ch.itertuples(index=False, name=None):
            buf.append("  (" + ", ".join(sql_literal(v, kinds[c], dialect) for c, v in zip(names, r)) + ")")
            if len(buf) >= batch:
                yield head + ",\n".join(buf) + ";\n", len(buf)
                buf = []
    if buf:
        yield head + ",\n".join(buf) + ";\n", len(buf)


@register_exporter
class SqlExporter:
    """Schema-preserving dump: CREATE TABLE with types, NOT NULL, primary and foreign keys, parents first, batched INSERTs, one transaction."""
    name, extension, mime = "sql", "sql", "application/sql"

    def export(self, tables: dict[str, TableData], schema: dict[str, TableSchema], sink: BinaryIO, dialect: str = "postgres", drop: bool = False,
               batch: int = 500, header: str | None = None, **_: Any) -> dict[str, Any]:
        if dialect not in DIALECTS:
            raise ExportError(f"unknown SQL dialect {dialect!r}; available: {list(DIALECTS)}")
        w = lambda s: sink.write(s.encode("utf-8"))  # noqa: E731
        w(f"-- Synthetic data ({dialect}); all values are fictional\n" + (f"-- {header}\n" if header else ""))
        if dialect == "mysql":
            w("SET NAMES utf8mb4;\nSET FOREIGN_KEY_CHECKS = 0;\nSTART TRANSACTION;\n\n")
        else:
            w("BEGIN;\n\n")
        order = order_tables(schema, list(tables))
        if drop:
            for n in reversed(order):
                w(f"DROP TABLE IF EXISTS {quote_ident(n, dialect)}{' CASCADE' if dialect == 'postgres' else ''};\n")
            w("\n")
        counts: dict[str, int] = {}
        for n in order:
            w(create_table_sql(schema[n], dialect) + "\n")
            counts[n] = 0
            for stmt, k in insert_statements(schema[n], tables[n], dialect, batch):
                w(stmt)
                counts[n] += k
            w("\n")
        w("COMMIT;\n" + ("SET FOREIGN_KEY_CHECKS = 1;\n" if dialect == "mysql" else ""))
        return {"rows": counts, "dialect": dialect}

    def stream_begin(self, schema: dict[str, TableSchema], dialect: str = "postgres", batch: int = 500, **_: Any) -> bytes:
        if dialect not in DIALECTS:
            raise ExportError(f"unknown SQL dialect {dialect!r}; available: {list(DIALECTS)}")
        (t,) = schema.values()
        head = f"-- Synthetic data ({dialect}); all values are fictional\n" + ("SET NAMES utf8mb4;\nSET FOREIGN_KEY_CHECKS = 0;\nSTART TRANSACTION;\n\n" if dialect == "mysql" else "BEGIN;\n\n")
        return (head + create_table_sql(t, dialect) + "\n").encode("utf-8")

    def encode_chunk(self, df: pd.DataFrame, schema: dict[str, TableSchema], index: int, dialect: str = "postgres", batch: int = 500, **_: Any) -> bytes:
        (t,) = schema.values()
        return "".join(stmt for stmt, _k in insert_statements(t, df, dialect, batch)).encode("utf-8")

    def stream_end(self, dialect: str = "postgres", **_: Any) -> bytes:
        return ("\nCOMMIT;\n" + ("SET FOREIGN_KEY_CHECKS = 1;\n" if dialect == "mysql" else "")).encode("utf-8")


# --------------------------------------------------------------------- PDF
@register_exporter
class PdfExporter:
    """A human-readable report: schema and the first `preview_rows` rows per table (not a bulk format)."""
    name, extension, mime = "pdf", "pdf", "application/pdf"

    def export(self, tables: dict[str, TableData], schema: dict[str, TableSchema], sink: BinaryIO, preview_rows: int = 25, title: str = "Synthetic dataset", **_: Any) -> dict[str, Any]:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        from xml.sax.saxutils import escape

        def safe(v: Any) -> str:                             # built-in PDF fonts are WinAnsi only
            s = "" if _jsonable(v) is None else str(v)
            return escape(s.encode("cp1252", "replace").decode("cp1252"))[:40]

        st = getSampleStyleSheet()
        doc = SimpleDocTemplate(sink, pagesize=landscape(A4), title=title, invariant=True, leftMargin=24, rightMargin=24, topMargin=24, bottomMargin=24)
        story: list[Any] = [Paragraph(escape(title), st["Title"]), Paragraph("All values are synthetic. Preview only; use CSV, JSON or SQL for the full data.", st["Normal"])]
        counts: dict[str, int] = {}
        for name, data in tables.items():
            frames, n = [], 0
            for ch in chunks(data):
                n += len(ch)
                if sum(len(f) for f in frames) < preview_rows:
                    frames.append(_clean(ch).head(preview_rows))
            df = pd.concat(frames).head(preview_rows) if frames else pd.DataFrame()
            counts[name] = n
            story += [Spacer(1, 12), Paragraph(f"{escape(name)} ({n:,} rows)", st["Heading2"])]
            ts = schema.get(name)
            if ts:
                story.append(Paragraph(escape(", ".join(f"{c.name}: {c.kind}" + ("*" if c.name in ts.primary_key else "") for c in ts.columns)), st["Italic"]))
            if len(df.columns):
                cols = list(df.columns)[:10]
                rows = [[safe(c) for c in cols]] + [[safe(v) for v in r] for r in df[cols].itertuples(index=False, name=None)]
                tbl = Table(rows, repeatRows=1)
                tbl.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey)]))
                story.append(tbl)
        doc.build(story)
        return {"rows": counts, "preview_rows": preview_rows}


# --------------------------------------------------------------------- ZIP
@register_exporter
class ZipExporter:
    """A bundle: one file per table per requested format (default csv), streamed entry by entry."""
    name, extension, mime = "zip", "zip", "application/zip"

    def export(self, tables: dict[str, TableData], schema: dict[str, TableSchema], sink: BinaryIO, formats: list[str] | None = None, **opts: Any) -> dict[str, Any]:
        formats = formats or ["csv"]
        if "zip" in formats:
            raise ExportError("a zip cannot contain a zip")
        files: list[str] = []
        with zipfile.ZipFile(sink, "w", zipfile.ZIP_DEFLATED) as z:
            for fmt in formats:
                ex = get_exporter(fmt)
                if fmt in ("sql", "pdf"):                    # whole-dataset formats: one file
                    with z.open(f"dataset.{ex.extension}", "w") as f:
                        ex.export(tables, schema, f, **opts)
                    files.append(f"dataset.{ex.extension}")
                    continue
                for name, data in tables.items():
                    with z.open(f"{name}.{ex.extension}", "w", force_zip64=True) as f:
                        ex.export({name: data}, schema, f, **opts)
                    files.append(f"{name}.{ex.extension}")
        return {"files": files}
