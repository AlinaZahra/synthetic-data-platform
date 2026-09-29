import io
import json
import re
import zipfile

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from sdp.api.main import app
from sdp.datasets import SHOP_FULL_RULES
from sdp.documents import fonts
from sdp.export import BOM

client = TestClient(app)
needs_arabic = pytest.mark.skipif(fonts.find_font("arabic") is None, reason="no Arabic-capable TrueType font")


# ------------------------------------------------------------------- rules
def test_rules_compile_shows_dsl_and_errors():
    r = client.post("/api/rules/compile", json={"rules": ["age must be at least 30", "income <= 100000", "plan must be one of basic, pro",
                                                      "make it nicer", "nope > 1"]}).json()
    ok = [x for x in r["results"] if x["ok"]]
    assert [x["dsl"] for x in ok] == ["age >= 30", "income <= 100000", "plan IN ('basic', 'pro')"]
    assert ok[0]["translated_from_plain_language"] is True and ok[1]["translated_from_plain_language"] is False
    assert [x["ok"] for x in r["results"]] == [True, True, True, False, False] and "age" in r["columns"]
    rel = client.post("/api/rules/compile", json={"rules": SHOP_FULL_RULES, "dataset": "shop_full"}).json()
    assert all(x["ok"] for x in rel["results"]) and {x["owner"] for x in rel["results"]} == {"orders", "order_items", "shipments"}


def test_rules_compare_naive_vs_ours():
    r = client.post("/api/rules/compare", json={"rules": ["age >= 40", "income <= 50000"], "rows": 800, "seed": 1}).json()
    assert r["naive"]["pass_rate"] < 90 and r["enforced"]["pass_rate"] == 100.0 and r["enforced"]["rows_returned"] == 800
    assert client.post("/api/rules/compare", json={"rules": ["zzz > 1"]}).status_code == 422


def test_tabular_generate_enforces_rules_and_returns_overlay():
    body = {"config": {"rows": 400, "seed": 2, "null_rate": {"income": 0.05}}, "rules": ["age must be at least 35", "plan must be one of basic, pro"]}
    r = client.post("/api/tabular/generate", json=body).json()
    assert r["rules"]["violations_after"] == 0 and sum(r["rules"]["naive_violations"].values()) > 0
    assert min(x["age"] for x in r["preview"]) >= 35 and {x["plan"] for x in r["preview"]} <= {"basic", "pro"}
    assert {"age", "plan", "signup_date"} <= set(r["overlay"]) and r["overlay"]["age"]["kind"] == "numeric"
    assert sum(r["overlay"]["plan"]["real"]) == pytest.approx(1)
    assert client.post("/api/tabular/generate", json={"config": {"rows": 10}, "rules": ["what"]}).status_code == 422


# -------------------------------------------------------------- relational
def test_relational_generate_with_rules_metrics_and_trust():
    body = {"dataset": "shop_full", "seed": 1, "rules": SHOP_FULL_RULES}
    r = client.post("/api/relational/generate", json=body).json()
    rules = r["scorecard"]["rules"]
    assert rules["reconciliation_pass_rate"] == 100.0 and rules["derived_pass_rate_before"] < 50
    assert r["scorecard"]["integrity"]["total_violations"] == 0
    rm = r["relation_metrics"]
    assert 60 < rm["score"] <= 100 and set(rm["components"]) == {"cardinality", "join_size", "cross_table", "fk_coverage", "queries"}
    assert {t["name"] for t in r["graph"]["tables"]} == {"customers", "products", "orders", "order_items", "shipments"}
    naive = client.post("/api/relational/generate", json={**body, "enforce": False, "include_metrics": False}).json()
    assert naive["scorecard"]["rules"]["violations_after"] > 100 and "relation_metrics" not in naive

    t = client.post("/api/relational/trust", json=body).json()
    assert [s["key"] for s in t["sub_scores"]] == ["fidelity", "privacy", "validity"] and t["trust_score"] > 70


def test_relational_inline_tables_and_errors():
    customers = [{"customer_id": i, "segment": "a" if i % 2 else "b"} for i in range(1, 61)]
    orders = [{"order_id": i, "customer_id": (i % 60) + 1, "total": float(i)} for i in range(1, 241)]
    inf = client.post("/api/relational/infer", json={"dataset": "inline", "tables": {"customers": customers, "orders": orders}}).json()
    assert [f["key"] for f in inf["graph"]["foreign_keys"]] == ["orders(customer_id)->customers"]
    r = client.post("/api/relational/generate", json={"dataset": "inline", "tables": {"customers": customers, "orders": orders}, "include_metrics": False})
    assert r.status_code == 200 and r.json()["scorecard"]["integrity"]["total_violations"] == 0
    assert client.post("/api/relational/generate", json={"dataset": "inline"}).status_code == 422
    assert client.post("/api/relational/generate", json={"dataset": "shop_full", "rules": ["orders.nope > 1"]}).status_code == 422


# --------------------------------------------------------------- documents
def test_document_preview_layout_matches_locale():
    r = client.post("/api/documents/preview", json={"doc_type": "invoice", "spec": {"locale": "es", "native": True, "seed": 3, "n_lines": 3}}).json()
    assert r["layout"]["title"] == "FACTURA" and r["layout"]["direction"] == "ltr" and r["reconciliation_errors"] == []
    assert any("IVA" in t["label"] for t in r["layout"]["totals"]) and r["layout"]["rows"]
    rtl = client.post("/api/documents/preview", json={"doc_type": "receipt", "spec": {"locale": "ur-PK", "native": True, "native_digits": True}}).json()
    assert rtl["layout"]["direction"] == "rtl" and re.search("[۰-۹]", rtl["layout"]["totals"][-3]["value"])
    st = client.post("/api/documents/preview", json={"doc_type": "statement", "spec": {"locale": "en-GB", "n_transactions": 8}}).json()
    assert len(st["layout"]["rows"]) == 8 and st["reconciliation_errors"] == []
    assert client.post("/api/documents/preview", json={"doc_type": "invoice", "spec": {"locale": "xx"}}).status_code == 422
    assert client.post("/api/documents/preview", json={"doc_type": "contract"}).status_code == 422


@needs_arabic
def test_arabic_pdf_endpoint_and_json_bom():
    pdf = client.post("/api/documents/pdf", json={"doc_type": "invoice", "spec": {"locale": "ar", "native": True, "seed": 1}})
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF") and b"FontFile2" in pdf.content
    plain = client.post("/api/documents/json", json={"doc_type": "invoice", "spec": {"locale": "ar", "native": True, "seed": 1}})
    bom = client.post("/api/documents/json", json={"doc_type": "invoice", "spec": {"locale": "ar", "native": True, "seed": 1}, "bom": True})
    assert not plain.content.startswith(BOM) and bom.content.startswith(BOM) and bom.content[3:] == plain.content
    doc = json.loads(bom.content.decode("utf-8-sig"))
    assert any("؀" <= ch <= "ۿ" for ch in doc["customer"]["name"])       # native script, not \u-escaped
    assert "charset=utf-8" in bom.headers["content-type"]


def test_document_pdf_error_is_structured():
    r = client.post("/api/documents/pdf", json={"doc_type": "invoice", "spec": {"font": "Nope"}})
    assert r.status_code == 422 and r.json()["detail"]["stage"] == "render"


def test_types_fonts_and_locale_metadata():
    t = client.get("/api/documents/types").json()
    assert {"invoice", "receipt", "statement", "payslip", "purchase_order", "retail_receipt"} <= set(t["types"]) and set(t["fonts"]) == {"latin", "arabic", "devanagari", "cjk"}
    loc = {x["code"]: x for x in client.get("/api/locales").json()}
    assert loc["ar"]["direction"] == "rtl" and loc["ur-PK"]["native_digits"] and loc["hi"]["complex_shaping"] and loc["zh"]["native_title"] == "发票"


# ------------------------------------------------------------------ export
def test_export_tabular_csv_with_and_without_bom():
    body = {"config": {"rows": 120, "seed": 1}, "rules": ["age >= 30"]}
    plain = client.post("/api/export/tabular", json=body)
    bom = client.post("/api/export/tabular", json={**body, "bom": True})
    assert plain.headers["content-type"].startswith("text/csv") and bom.content.startswith(BOM) and not plain.content.startswith(BOM)
    df = pd.read_csv(io.BytesIO(bom.content), encoding="utf-8-sig")
    assert len(df) == 120 and df["age"].min() >= 30


def test_export_relational_zip_and_nl_zip():
    z = zipfile.ZipFile(io.BytesIO(client.post("/api/export/relational", json={"dataset": "shop_full", "rules": SHOP_FULL_RULES, "bom": True}).content))
    assert {"orders.csv", "shipments.csv", "scorecard.json", "relationships.json"} <= set(z.namelist())
    assert all(z.read(n).startswith(BOM) for n in z.namelist())
    orders = pd.read_csv(io.BytesIO(z.read("orders.csv")), encoding="utf-8-sig")
    assert (orders["discount"] <= 0.3).all()
    cfg = client.post("/api/nl/parse", json={"text": "60 Pakistani bank customers, 5% fraud, 3 months of history"}).json()["config"]
    z = zipfile.ZipFile(io.BytesIO(client.post("/api/export/nl", json={"config": cfg, "bom": True}).content))
    names = pd.read_csv(io.BytesIO(z.read("customers.csv")), encoding="utf-8-sig")["name"]
    assert len(names) == 60 and all("؀" <= ch <= "ۿ" or ch == " " for ch in names.iloc[0]) and "validation.json" in z.namelist()
