import io
import json
import sqlite3
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from sdp.ai import llm
from sdp.api.main import app

KEY = "test-key-v5"
H = {"X-API-Key": KEY}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SDP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SDP_API_KEYS", KEY)
    monkeypatch.delenv("SDP_CONNECTORS_UI", raising=False)
    llm.set_client(None)
    yield TestClient(app)
    llm.set_client(False)


def rows(n=30):
    return [{"cust_id": i + 1, "email": f"u{i}@example.com", "signup": f"{1 + i % 28:02d}/03/2024", "tier": ["gold", "silver", "basic"][i % 3], "age": 20 + i % 40} for i in range(n)]


def test_schema_infer_and_edit_roundtrip(client):
    r = client.post("/api/schema/infer", json={"records": rows()})
    assert r.status_code == 200
    body = r.json()
    prop = body["proposal"]
    cols = {c["name"]: c for c in prop["tables"]["table"]}
    assert cols["email"]["semantic_type"] == "email" and cols["signup"]["date_format"] == "%d/%m/%Y" and prop["used_llm"] is False
    assert body["rules"]["table"] and body["llm_available"] is False
    e = client.post("/api/schema/edit", json={"proposal": prop, "edits": [{"action": "set_type", "table": "table", "column": "tier", "value": "other"}, {"action": "confirm"}]})
    assert e.status_code == 200 and e.json()["proposal"]["confirmed"] is True
    assert client.post("/api/schema/edit", json={"proposal": prop, "edits": [{"action": "set_type", "table": "x", "column": "y", "value": "email"}]}).status_code == 422
    assert client.post("/api/schema/infer", json={"records": rows(2)}).status_code == 422
    assert client.post("/api/schema/infer", json={}).status_code == 422


def test_content_synthesis_endpoint(client):
    r = client.post("/api/content/synthesize", json={"kind": "review", "n": 5, "contexts": [{"rating": 1, "product": "kettle"}] * 5, "seed": 1})
    assert r.status_code == 200 and len(r.json()["values"]) == 5 and set(r.json()["sources"]) == {"fallback"}
    assert client.post("/api/content/synthesize", json={"kind": "poem"}).status_code == 422
    assert client.post("/api/content/synthesize", json={"kind": "name", "locale": "xx"}).status_code == 422


def test_chat_edit_and_multilingual(client):
    p = client.post("/api/nl/parse-multilingual", json={"text": "500 clientes bancarios pakistaníes, 4% de fraude, 6 meses de historial"}).json()
    assert p["language"] == "es" and p["result"]["ok"] and p["reply"]["lines"][0].startswith("Filas") and "parse" not in p
    state = {"config": p["result"]["config"], "ops": []}
    r = client.post("/api/chat/edit", json={"state": state, "text": "double customers from Lahore"})
    assert r.status_code == 200
    out = r.json()
    assert out["applied"] == 1 and out["after"]["rows"] > out["before"]["rows"] and out["preview"]["rows"]
    r2 = client.post("/api/chat/edit", json={"state": out["state"], "text": "fraud 8%"}).json()
    assert r2["after"]["flag_rate"] == pytest.approx(0.08, abs=0.003) and len(r2["state"]["ops"]) == 2
    assert client.post("/api/chat/edit", json={"state": {"config": {"bogus": 1}}, "text": "x"}).status_code == 422
    assert client.post("/api/nl/parse-multilingual", json={"text": "x", "language": "xx"}).status_code == 422


def test_language_cases_and_llm_status(client):
    r = client.get("/api/locale/language-cases?packs=apostrophes,long_names&per_case=1").json()
    assert set(r["packs"]) >= {"apostrophes", "long_names"} and {x["pack"] for x in r["rows"]} == {"apostrophes", "long_names"}
    assert client.get("/api/locale/language-cases?packs=klingon").status_code == 422
    assert client.get("/api/llm/status").json()["available"] is False


def test_export_endpoint_all_formats(client):
    fmts = {f["name"] for f in client.get("/api/export/formats").json()}
    assert {"csv", "json", "jsonl", "sql", "pdf", "zip"} <= fmts
    src = {"kind": "demo", "name": "shop", "rows": 40}
    sql = client.post("/api/export", json={"source": src, "format": "sql", "options": {"dialect": "sqlite"}})
    assert sql.status_code == 200 and "attachment" in sql.headers["content-disposition"]
    con = sqlite3.connect(":memory:")
    con.executescript(sql.text)
    assert con.execute("SELECT COUNT(*) FROM customers").fetchone()[0] > 0
    z = client.post("/api/export", json={"source": src, "format": "csv"})
    assert zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    assert client.post("/api/export", json={"source": {"kind": "demo", "name": "customers", "rows": 30}, "format": "pdf"}).content.startswith(b"%PDF")
    assert client.post("/api/export", json={"source": src, "format": "xml"}).status_code == 422
    assert client.post("/api/export", json={"source": {"kind": "inline", "tables": {"t": rows(5)}}, "format": "jsonl"}).text.count("\n") == 5


def test_connectors_are_gated_and_dry_run_by_default(client):
    body = {"url": "sqlite:///api.db", "source": {"kind": "demo", "name": "shop", "rows": 40}}
    assert client.post("/api/connect/load", json=body).status_code == 403
    assert client.post("/connect/load", json=body).status_code == 401
    r = client.post("/connect/load", json=body, headers=H)
    assert r.status_code == 200 and r.json()["dry_run"] and r.json()["rolled_back"] and r.json()["ok"]
    real = client.post("/connect/load", json={**body, "dry_run": False}, headers=H).json()
    assert real["committed"]
    again = client.post("/connect/load", json={**body, "dry_run": False}, headers=H).json()
    assert not again["ok"] and "already exists" in again["errors"][0]
    assert client.post("/connect/load", json={"url": "postgresql://u:p@evil.example.com/db"}, headers=H).status_code == 422
    assert client.post("/connect/load", json={"url": "sqlite:////etc/passwd"}, headers=H).status_code == 422
    assert client.get("/api/connect/status").json()["ui_enabled"] is False


def test_connectors_ui_route_when_enabled(client, monkeypatch):
    monkeypatch.setenv("SDP_CONNECTORS_UI", "1")
    r = client.post("/api/connect/load", json={"url": "sqlite:///ui.db", "source": {"kind": "pack", "name": "healthcare", "rows": 30}})
    assert r.status_code == 200 and r.json()["ok"] and r.json()["dry_run"]


def test_contracts_endpoints(client):
    packs = client.get("/api/contracts/packs").json()
    assert {p["name"] for p in packs} == {"banking", "ecommerce", "healthcare"}
    r = client.post("/api/contracts/run", json={"pack": "banking", "rows": 60, "seed": 2}).json()
    assert r["report"]["success"] and r["rows"]["customers"] == 60 and len(r["preview"]["customers"]) == 8
    bad = client.post("/api/contracts/validate", json={"pack": "healthcare", "tables": {"patients": [{"patient_id": 1, "name": "x", "sex": "Q", "date_of_birth": "2000-01-01", "blood_type": "O+", "phone": "1"}] * 2,
                                                                                           "encounters": [], "claims": []}}).json()
    assert not bad["success"] and any(x["expectation_config"]["expectation_type"] == "expect_column_values_to_be_unique" and not x["success"] for x in bad["results"])
    custom = client.post("/api/contracts/validate", json={"suite": [{"expectation_type": "expect_column_values_to_be_between", "table": "t", "kwargs": {"column": "age", "min_value": 0, "max_value": 30}}], "tables": {"t": rows(30)}}).json()
    assert not custom["success"] and custom["results"][0]["result"]["unexpected_count"] > 0
    assert client.get("/api/contracts/banking/great-expectations").json()["customers"]["expectation_suite_name"] == "banking.customers"
    assert client.get("/api/contracts/nope/great-expectations").status_code == 404
    assert client.post("/api/contracts/run", json={"pack": "nope"}).status_code == 422
    assert client.post("/api/contracts/validate", json={"tables": {"t": rows(5)}}).status_code == 422


def wait(client, jid, timeout=180):
    t = time.time()
    while time.time() - t < timeout:
        p = client.get(f"/jobs/{jid}/progress", headers=H).json()
        if p["status"] in ("succeeded", "failed", "cancelled"):
            return p
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_large_job_streams_reports_throughput_and_reproduces(client):
    body = {"kind": "large", "params": {"rows": 30_000, "chunk_rows": 5_000, "workers": 2, "seed": 3, "format": "csv"}}
    r = client.post("/generate", json=body, headers=H)
    assert r.status_code == 202
    jid = r.json()["id"]
    p = wait(client, jid)
    assert p["status"] == "succeeded" and p["done"] == p["total"] == 30_000
    m = client.get(f"/jobs/{jid}", headers=H).json()
    tp = client.get(f"/jobs/{jid}/manifest", headers=H).json()["dataset"]["throughput"]
    assert tp["rows"] == 30_000 and tp["rows_per_second"] > 0 and tp["chunks"] == 6
    csv = client.get(f"/jobs/{jid}/files/synthetic.csv", headers=H)
    assert csv.status_code == 200 and csv.text.count("\n") == 30_001
    z = client.get(f"/jobs/{jid}/download.zip", headers=H)
    assert "synthetic.csv" in zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    rr = client.post(f"/jobs/{jid}/rerun", headers=H, json={"wait": True}).json()
    assert rr["status"] == "succeeded" and rr["reproduced"] is True
    sc = client.get(f"/score/{jid}", headers=H)
    assert sc.status_code == 200 and sc.json()["kind"] == "trust"
    assert m["kind"] == "large"


def test_large_job_can_be_cancelled_and_keeps_a_valid_file(client):
    jid = client.post("/generate", json={"kind": "large", "params": {"rows": 5_000_000, "chunk_rows": 20_000, "workers": 2, "format": "jsonl", "executor": "thread"}}, headers=H).json()["id"]
    t = time.time()
    while client.get(f"/jobs/{jid}/progress", headers=H).json()["done"] < 40_000 and time.time() - t < 120:
        time.sleep(0.2)
    assert client.post(f"/jobs/{jid}/cancel", headers=H).status_code == 200
    p = wait(client, jid)
    assert p["status"] == "cancelled" and 0 < p["done"] < 5_000_000
    body = client.get(f"/jobs/{jid}/files/synthetic.jsonl", headers=H).text
    lines = body.splitlines()
    assert len(lines) >= 40_000 and all(json.loads(x) for x in lines[:50]) and json.loads(lines[-1])
    assert client.get(f"/score/{jid}", headers=H).status_code == 409


def test_large_job_validation(client):
    assert client.post("/generate", json={"kind": "large", "params": {"rows": 10, "chunk_rows": 5}}, headers=H).status_code == 422
    assert client.post("/generate", json={"kind": "large", "params": {"format": "pdf"}}, headers=H).status_code == 422
