import io
import json
import sqlite3
import zipfile

import numpy as np
import pandas as pd
import pytest

from sdp.datasets import make_shop_full
from sdp.exporters import ExportError, export_bytes, register_exporter, schema_from
from sdp.exporters.base import REGISTRY
from sdp.relational import infer_graph


@pytest.fixture(scope="module")
def shop():
    t = make_shop_full(60)
    return t, infer_graph(t)


def test_registry_lists_all_formats():
    assert {"csv", "json", "jsonl", "sql", "pdf", "zip"} <= set(REGISTRY)
    with pytest.raises(ExportError):
        export_bytes({"a": pd.DataFrame({"x": [1]})}, "xml")


def test_sql_dump_loads_into_sqlite_and_preserves_schema_and_data(shop):
    tables, graph = shop
    data, info = export_bytes(tables, "sql", graph=graph, dialect="sqlite")
    con = sqlite3.connect(":memory:")
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(data.decode("utf-8"))
    for name, df in tables.items():
        assert con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0] == len(df) == info["rows"][name]
    # keys survived
    for t in graph.tables:
        pk = [r[1] for r in con.execute(f'PRAGMA table_info("{t.name}")') if r[5]]
        assert sorted(pk) == sorted(t.primary_key)
    fks = {(r[2], r[3], r[4]) for t in graph.tables for r in con.execute(f'PRAGMA foreign_key_list("{t.name}")')}
    assert {(f.parent_table, f.child_columns[0], f.parent_columns[0]) for f in graph.foreign_keys} <= fks
    assert con.execute("PRAGMA foreign_key_check").fetchall() == []
    # FK enforcement is real: an orphan is rejected
    child = next(f for f in graph.foreign_keys)
    with pytest.raises(sqlite3.IntegrityError):
        con.execute(f'INSERT INTO "{child.child_table}" ({", ".join(chr(34) + c.name + chr(34) for c in graph.table(child.child_table).columns)}) VALUES ({", ".join("-999" if c.name == child.child_columns[0] else "NULL" for c in graph.table(child.child_table).columns)})')


def test_parents_come_before_children(shop):
    tables, graph = shop
    text = export_bytes(tables, "sql", graph=graph, dialect="postgres")[0].decode()
    for fk in graph.foreign_keys:
        if fk.parent_table != fk.child_table:
            assert text.index(f'CREATE TABLE "{fk.parent_table}"') < text.index(f'CREATE TABLE "{fk.child_table}"')


def test_dialect_specific_syntax_and_escaping():
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["O'Brien", "back\\slash", "لاہور"], "ok": [True, False, True],
                       "amt": [1.5, np.nan, 3.0], "d": pd.to_datetime(["2024-01-01", "2024-02-29", None])})
    tables = {"t": df}
    my = export_bytes(tables, "sql", dialect="mysql")[0].decode()
    pg = export_bytes(tables, "sql", dialect="postgres")[0].decode()
    assert "`name` VARCHAR(16)" in my and "'back\\\\slash'" in my and "SET NAMES utf8mb4" in my and "TINYINT(1)" in my and "START TRANSACTION" in my
    assert '"name" VARCHAR(16)' in pg and "'back\\slash'" in pg and "BOOLEAN" in pg and "TRUE" in pg and "BEGIN;" in pg and "DOUBLE PRECISION" in pg
    assert "'O''Brien'" in pg and "لاہور" in pg and "NULL" in pg and "'2024-02-29'" in pg and "DATE" in pg
    with pytest.raises(ExportError):
        export_bytes(tables, "sql", dialect="oracle")
    dr = export_bytes(tables, "sql", dialect="postgres", drop=True)[0].decode()
    assert 'DROP TABLE IF EXISTS "t" CASCADE' in dr


def test_injection_attempt_in_values_stays_data():
    df = pd.DataFrame({"id": [1], "n": ["Robert'); DROP TABLE t;--"]})
    con = sqlite3.connect(":memory:")
    con.executescript(export_bytes({"t": df}, "sql", dialect="sqlite")[0].decode())
    assert con.execute("SELECT n FROM t").fetchone()[0] == "Robert'); DROP TABLE t;--"
    assert con.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1


def test_hidden_columns_never_exported():
    df = pd.DataFrame({"a": [1, 2], "_edge_case": ["x", None]})
    for fmt in ("csv", "json", "jsonl", "sql"):
        assert "_edge_case" not in export_bytes({"t": df}, fmt)[0].decode()


def test_csv_json_jsonl_roundtrip_and_bom():
    df = pd.DataFrame({"a": [1, 2], "b": ["x,y", "لاہور"], "c": [1.5, np.nan], "d": pd.to_datetime(["2024-01-01", None])})
    csv = export_bytes({"t": df}, "csv", bom=True)[0]
    assert csv.startswith(b"\xef\xbb\xbf") and pd.read_csv(io.BytesIO(csv), encoding="utf-8-sig")["b"].tolist() == ["x,y", "لاہور"]
    js = json.loads(export_bytes({"t": df}, "json")[0])
    assert js[1]["c"] is None and js[0]["d"].startswith("2024-01-01") and js[1]["b"] == "لاہور"
    multi = json.loads(export_bytes({"t": df, "u": df.head(1)}, "json")[0])
    assert set(multi) == {"t", "u"} and len(multi["u"]) == 1
    lines = export_bytes({"t": df}, "jsonl")[0].decode().strip().split("\n")
    assert len(lines) == 2 and json.loads(lines[0])["a"] == 1
    assert json.loads(export_bytes({"t": df}, "json", bom=True)[0][3:])            # BOM then valid JSON


def test_zip_bundles_formats_and_multi_table_csv_is_a_zip(shop):
    tables, graph = shop
    z = zipfile.ZipFile(io.BytesIO(export_bytes(tables, "zip", graph=graph, formats=["csv", "json", "sql"], dialect="sqlite")[0]))
    names = set(z.namelist())
    assert {f"{t}.csv" for t in tables} | {f"{t}.json" for t in tables} | {"dataset.sql"} == names
    z2 = zipfile.ZipFile(io.BytesIO(export_bytes(tables, "csv", graph=graph)[0]))
    assert {f"{t}.csv" for t in tables} == set(z2.namelist())
    with pytest.raises(ExportError):
        export_bytes(tables, "zip", formats=["zip"])
    with pytest.raises(ExportError):
        export_bytes(tables, "jsonl")


def test_pdf_report_is_a_valid_pdf_and_deterministic(shop):
    tables, graph = shop
    a = export_bytes(tables, "pdf", graph=graph)[0]
    b = export_bytes(tables, "pdf", graph=graph)[0]
    assert a.startswith(b"%PDF") and a == b and len(a) > 2000


def test_streamed_chunks_produce_the_same_output_as_a_frame():
    df = pd.DataFrame({"id": range(1050), "v": np.linspace(0, 1, 1050)})
    schema = schema_from({"t": df})
    parts = (df.iloc[i:i + 100] for i in range(0, len(df), 100))
    streamed = export_bytes({"t": parts}, "sql", schema=schema, dialect="sqlite")[0]
    assert streamed == export_bytes({"t": df}, "sql", schema=schema, dialect="sqlite")[0]
    csv = export_bytes({"t": (df.iloc[i:i + 100] for i in range(0, len(df), 100))}, "csv", schema=schema)[0]
    assert csv == export_bytes({"t": df}, "csv")[0]
    with pytest.raises(ExportError):
        export_bytes({"t": iter([df])}, "csv")             # streamed chunks need an explicit schema


def test_custom_exporter_plugs_in():
    @register_exporter
    class Tsv:
        name, extension, mime = "tsv", "tsv", "text/tab-separated-values"

        def export(self, tables, schema, sink, **opts):
            (n, df), = tables.items()
            sink.write(df.to_csv(sep="\t", index=False, lineterminator="\n").encode())
            return {"rows": len(df)}

    try:
        data, info = export_bytes({"t": pd.DataFrame({"a": [1]})}, "tsv")
        assert data.startswith(b"a\n1") and info["rows"] == 1
    finally:
        REGISTRY.pop("tsv")
