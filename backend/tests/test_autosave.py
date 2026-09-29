import time

import pytest
from fastapi.testclient import TestClient

from sdp.api.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SDP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SDP_AUTOSAVE", "1")
    return TestClient(app)


def jobs(client):
    return client.get("/api/history").json()


def wait_all(client, n, timeout=120):
    t = time.time()
    while time.time() - t < timeout:
        j = jobs(client)
        if len(j) >= n and all(x["status"] in ("succeeded", "failed", "cancelled") for x in j):
            return j
        time.sleep(0.3)
    raise AssertionError(jobs(client))


CFG = {"domain": "bank_customers", "locale": "ur-PK", "rows": 60, "seed": 4, "flag": {"name": "is_fraud", "rate": 0.05}, "history_months": 3}


def test_live_previews_and_scores_are_not_saved(client):
    assert client.post("/api/tabular/generate", json={"config": {"rows": 50, "seed": 1}}).status_code == 200
    assert client.post("/api/documents/preview", json={"doc_type": "invoice", "spec": {"seed": 1}}).status_code == 200
    assert client.post("/api/trust/report", json={"rows": 200, "seed": 1}).status_code == 200
    assert jobs(client) == []


def test_every_explicit_generate_or_download_is_recorded(client):
    r = client.post("/api/nl/generate", json={"config": CFG, "confirmed": True}).json()
    assert r["history_id"]
    assert client.post("/api/export/tabular", json={"config": {"rows": 80, "seed": 2}, "rules": ["age must be at least 25"]}).status_code == 200
    assert client.post("/api/export/relational", json={"dataset": "shop", "seed": 1}).status_code == 200
    assert client.post("/api/export/nl", json={"config": CFG}).status_code == 200
    assert client.post("/api/documents/pdf", json={"doc_type": "payslip", "spec": {"seed": 3}}).status_code == 200
    run = client.post("/api/contracts/run", json={"pack": "healthcare", "rows": 40, "seed": 1}).json()
    assert run["history_id"]
    j = wait_all(client, 6)
    kinds = sorted(x["kind"] for x in j)
    assert kinds == ["document", "nl", "nl", "pack", "relational", "tabular"]
    assert all(x["status"] == "succeeded" for x in j), [(x["kind"], x["error"]) for x in j]
    pack = next(x for x in j if x["kind"] == "pack")
    assert client.get(f"/api/history/{pack['id']}/score").json()["kind"] == "contract"


def test_generating_twice_records_twice_and_recorded_runs_reproduce(client):
    for _ in range(2):
        client.post("/api/nl/generate", json={"config": CFG, "confirmed": True})
    j = wait_all(client, 2)
    assert len(j) == 2 and j[0]["output_hash"] == j[1]["output_hash"]


def test_can_be_switched_off_and_never_breaks_the_request(client, monkeypatch):
    monkeypatch.setenv("SDP_AUTOSAVE", "0")
    assert client.post("/api/nl/generate", json={"config": CFG, "confirmed": True}).json()["history_id"] is None
    assert jobs(client) == []
    monkeypatch.setenv("SDP_AUTOSAVE", "1")
    from sdp.api import autosave as a
    monkeypatch.setattr(a.service, "submit", lambda *x, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
    assert client.post("/api/nl/generate", json={"config": CFG, "confirmed": True}).status_code == 200      # the user still gets their data
