from fastapi.testclient import TestClient

from sdp.api.main import app

client = TestClient(app)


def test_locales_endpoint():
    r = client.get("/api/locales").json()
    assert {x["code"] for x in r} >= {"en-US", "en-GB", "ur-PK", "ar", "hi", "es", "fr", "zh"}
    hi = next(x for x in r if x["code"] == "hi")
    assert hi["sample"]["amount"] == "₹12,34,567.50" and hi["currency"] == "INR"


def test_nl_parse_then_confirm_flow():
    p = client.post("/api/nl/parse", json={"text": "300 Pakistani bank customers, 3% fraud, 6 months of history"}).json()
    assert p["ok"] and p["confirmation_required"] and p["config"]["locale"] == "ur-PK" and p["config_hash"]
    unconfirmed = client.post("/api/nl/generate", json={"config": p["config"]})
    assert unconfirmed.status_code == 409  # never generates without confirmation
    r = client.post("/api/nl/generate", json={"config": p["config"], "confirmed": True}).json()
    v = r["validation"]
    assert v["rows"] == {"customers": 300, "monthly_activity": 1800}
    assert v["locale"]["valid_pct"] == 100 and v["integrity"]["total_violations"] == 0 and v["flag"]["count"] == 9
    assert r["config_hash"] == p["config_hash"] and len(r["preview"]["customers"]) == 25


def test_nl_errors_and_limits():
    assert client.post("/api/nl/parse", json={"text": "hello"}).json()["ok"] is False
    p = client.post("/api/nl/parse", json={"text": "500000 bank customers"}).json()
    assert p["ok"]
    assert client.post("/api/nl/generate", json={"config": p["config"], "confirmed": True}).status_code == 422
    assert client.post("/api/nl/generate", json={"config": {"domain": "x", "locale": "en-US", "rows": 5}, "confirmed": True}).status_code == 422


def test_trust_report_and_pdf():
    rep = client.post("/api/trust/report", json={"rows": 400, "seed": 1}).json()
    assert [s["key"] for s in rep["sub_scores"]] == ["fidelity", "privacy", "validity"] and 0 <= rep["trust_score"] <= 100
    pdf = client.post("/api/trust/pdf", json={"report": rep})
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF") and pdf.headers["content-type"] == "application/pdf"
    assert client.post("/api/trust/pdf", json={"report": {}}).status_code == 422


def test_invoice_endpoints():
    spec = {"locale": "en-US", "region": "CA", "seed": 4, "n_lines": 3}
    j = client.post("/api/documents/invoice/json", json=spec).json()
    pdf = client.post("/api/documents/invoice/pdf", json=spec)
    assert pdf.content.startswith(b"%PDF") and pdf.headers["x-invoice-total"] == j["total"]
    assert client.post("/api/documents/invoice/json", json={"locale": "xx"}).status_code == 422
    bad_font = client.post("/api/documents/invoice/pdf", json={**spec, "font": "Nope"})
    assert bad_font.status_code == 422 and bad_font.json()["detail"]["stage"] == "render"


def test_batch_endpoint_isolates_failures():
    specs = [{"seed": 1}, {"locale": "xx"}, {"seed": 2, "locale": "fr"}, "junk"]
    r = client.post("/api/documents/batch", json={"specs": specs}).json()
    assert (r["report"]["succeeded"], r["report"]["failed"]) == (2, 2) and r["report"]["by_stage"] == {"validate": 2}
    assert len(r["preview"]) == 2
    big = client.post("/api/documents/batch", json={"count": 300, "locale": "hi", "region": "KA"}).json()
    assert big["report"]["succeeded"] == 300 and big["report"]["failed"] == 0
