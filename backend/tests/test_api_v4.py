import base64
import io
import json
import zipfile

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from sdp.api.main import app
from sdp.export import BOM

client = TestClient(app)


def test_statement_query_parse_preview_and_csv():
    p = client.post("/api/documents/statement/parse", json={"query": "last 90 days, balance over $500"}).json()
    assert p["ok"] and p["filter"]["days"] == 90 and p["filter"]["min_balance"] == "500.01"
    bad = client.post("/api/documents/statement/parse", json={"query": "last 30 days pizza"}).json()
    assert bad["ok"] is False and "pizza" in bad["unparsed"]
    assert client.post("/api/documents/statement/parse", json={"query": "category zzz"}).status_code == 422
    spec = {"seed": 3, "n_transactions": 200, "days": 365, "query": "last 60 days, debits only"}
    r = client.post("/api/documents/preview", json={"doc_type": "statement", "spec": spec}).json()
    assert r["reconciliation_errors"] == [] and r["document"]["filter"]["matched"] == len(r["layout"]["rows"]) < 200
    assert any("last 60 days" in n for n in r["layout"]["notes"])
    csv = client.post("/api/documents/csv", json={"doc_type": "statement", "spec": spec, "bom": True})
    assert csv.content.startswith(BOM) and len(pd.read_csv(io.BytesIO(csv.content), encoding="utf-8-sig")) == r["document"]["filter"]["matched"]
    assert client.post("/api/documents/csv", json={"doc_type": "invoice"}).status_code == 422
    assert client.post("/api/documents/preview", json={"doc_type": "statement", "spec": {"query": "nonsense words"}}).status_code == 422


def test_scan_endpoints():
    presets = client.get("/api/documents/scan/presets").json()
    assert "bad_fax" in presets["presets"] and presets["defaults"]["dpi"] == 150
    body = {"doc_type": "invoice", "spec": {"seed": 1, "n_lines": 4}, "preset": "office_scan", "scan": {"seed": 2, "dpi": 100}}
    r = client.post("/api/documents/scan", json=body).json()
    page = r["pages"][0]
    im = Image.open(io.BytesIO(base64.b64decode(page["image_base64"])))
    assert im.size == (page["width"], page["height"]) == (827, 1170) and r["mime"] == "image/png"
    assert {f["key"] for f in page["fields"]} >= {"total", "invoice_number", "stamp", "handwritten_signature"}
    z = zipfile.ZipFile(io.BytesIO(client.post("/api/documents/scan/zip", json=body).content))
    assert {"page-1.png", "labels.json", "labels.jsonl"} <= set(z.namelist()) and json.loads(z.read("labels.json"))["config"]["seed"] == 2
    assert client.post("/api/documents/scan", json={**body, "scan": {"dpi": 5}}).status_code == 422
    assert client.post("/api/documents/scan", json={**body, "preset": "nope"}).status_code == 422


def test_tabular_generate_reports_edge_cases_and_hides_the_tag_column():
    r = client.post("/api/tabular/generate", json={"config": {"rows": 300, "seed": 1, "edge_cases": {"typos": 0.1, "boundary_values": 0.1}}}).json()
    assert "_edge_case" not in r["columns"] and all("_edge_case" not in row for row in r["preview"])
    assert r["edge_cases"]["rows_tagged"] > 0 and 0 < r["edge_cases"]["coverage_pct"] <= 100 and len(r["edge_tags"]) == 50
    assert set(r["edge_cases"]["packs"]) == {"typos", "boundary_values"}
    assert client.post("/api/tabular/generate", json={"config": {"rows": 50, "edge_cases": {"nope": 0.1}}}).status_code == 422
    plain = client.post("/api/tabular/generate", json={"config": {"rows": 50, "seed": 1}}).json()
    assert plain["edge_cases"] is None and plain["edge_tags"] is None


def test_locale_lab_endpoints():
    p = client.post("/api/locale/people", json={"n": 300, "mix": "70% ur-PK, 30% en", "seed": 1}).json()
    assert p["coherence"]["coherent_pct"] == 100.0 and 0.6 < p["coherence"]["mix"]["ur-PK"] < 0.8 and len(p["preview"]) == 25
    csv = client.post("/api/locale/people", json={"n": 20, "mix": "50% ar, 50% fr", "download": True, "bom": True})
    df = pd.read_csv(io.BytesIO(csv.content), encoding="utf-8-sig")
    assert csv.content.startswith(BOM) and set(df["locale"]) <= {"ar", "fr"} and len(df) == 20
    assert client.post("/api/locale/people", json={"mix": "lots"}).status_code == 422

    e = client.post("/api/locale/entities", json={"n": 30, "locale": "ur-PK", "seed": 2}).json()
    assert e["entities"] >= 30 and e["records"] > e["entities"] and e["matching_pairs"]["positive"] > 0
    names = {r["name"] for r in client.post("/api/locale/entities", json={"n": 200, "locale": "ur-PK", "variants": 8, "seed": 1}).json()["preview"]}
    assert len(names) > 10
    assert client.post("/api/locale/entities", json={"locale": "xx"}).status_code == 422

    o = client.get("/api/locale/codemix/options").json()
    assert {x["name"] for x in o["pairs"]} == {"roman_urdu", "hinglish", "spanglish"}
    lo = client.post("/api/locale/codemix", json={"pair": "hinglish", "topic": "support", "level": 0.0, "n": 40, "seed": 1}).json()
    hi = client.post("/api/locale/codemix", json={"pair": "hinglish", "topic": "support", "level": 1.0, "n": 40, "seed": 1}).json()
    assert lo["mean_english_token_ratio"] < hi["mean_english_token_ratio"] and hi["preview"][0]["tokens"]
    assert client.post("/api/locale/codemix", json={"pair": "roman_urdu", "topic": "cooking"}).status_code == 422
    dl = client.post("/api/locale/codemix", json={"n": 5, "download": True})
    assert dl.headers["content-type"].startswith("text/csv") and b"/base" in dl.content or b"/en" in dl.content


def test_nl_mixed_locale_through_the_api():
    p = client.post("/api/nl/parse", json={"text": "200 bank customers, 70% ur-PK, 30% en, 2% fraud"}).json()
    assert p["ok"] and p["config"]["locale_mix"] == {"ur-PK": pytest.approx(0.7), "en-US": pytest.approx(0.3)}
    r = client.post("/api/nl/generate", json={"config": p["config"], "confirmed": True}).json()
    assert r["validation"]["locale"]["valid_pct"] == 100 and r["validation"]["integrity"]["total_violations"] == 0
    assert "locale" in r["columns"]["customers"] and {x["currency"] for x in r["preview"]["customers"]} <= {"PKR", "USD"}
