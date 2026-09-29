import io
import json
import time
import zipfile

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

from sdp import lineage, service
from sdp.api.main import app
from sdp.lineage import Store

KEY = "queue-test-key"
H = {"X-API-Key": KEY}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SDP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SDP_API_KEYS", KEY)
    return tmp_path


@pytest.fixture()
def client(env):
    c = TestClient(app)
    yield c
    # never leave workers busy for the next test
    for j in c.get("/jobs", headers=H).json():
        if j["status"] in ("queued", "running", "cancelling"):
            c.post(f"/jobs/{j['id']}/cancel", headers=H)
    deadline = time.time() + 60
    while time.time() < deadline and any(j["status"] in ("queued", "running", "cancelling") for j in c.get("/jobs", headers=H).json()):
        time.sleep(0.2)


def start(client, params, kind="document"):
    r = client.post("/generate", headers=H, json={"kind": kind, "params": params})
    assert r.status_code == 202, r.text
    return r.json()["id"]


def wait(client, jid, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        p = client.get(f"/jobs/{jid}/progress", headers=H).json()
        if p["finished"]:
            return p
        time.sleep(0.15)
    raise AssertionError("timed out")


def wait_for(client, jid, cond, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        p = client.get(f"/jobs/{jid}/progress", headers=H).json()
        if cond(p):
            return p
        time.sleep(0.05)
    raise AssertionError(f"condition not met: {p}")


# ----------------------------------------------------------------- non-blocking + progress
def test_submit_returns_immediately_and_progress_is_monotonic(client):
    t0 = time.time()
    jid = start(client, {"doc_type": "invoice", "count": 600, "spec": {"n_lines": 3}})
    assert time.time() - t0 < 2.0                              # the request does not wait for 600 PDFs
    seen, last = [], -1
    while True:
        p = client.get(f"/jobs/{jid}/progress", headers=H).json()
        assert set(p) >= {"job_id", "status", "phase", "done", "total", "percent", "failed", "elapsed_s", "items_per_s", "eta_s", "finished"}
        assert p["done"] >= last and 0 <= p["percent"] <= 100 and p["total"] == 600
        last = p["done"]
        seen.append((p["status"], p["done"]))
        if p["finished"]:
            break
        time.sleep(0.1)
    assert p["status"] == "succeeded" and p["done"] == 600 and p["percent"] == 100.0 and p["phase"] == "done"
    assert len({d for _, d in seen}) > 2                       # progress really advanced in steps, not 0 -> done
    assert any(s == "running" for s, _ in seen)


def test_zip_contains_every_document_and_the_manifest(client):
    jid = start(client, {"count": 40, "doc_type": "receipt", "spec": {"locale": "fr"}})
    wait(client, jid)
    r = client.get(f"/jobs/{jid}/download.zip", headers=H)
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(z.namelist())
    assert {"manifest.json", "index.json", "report.json"} <= names
    pdfs = [n for n in names if n.startswith("docs/") and n.endswith(".pdf")]
    jsons = [n for n in names if n.startswith("docs/") and n.endswith(".json") and not n.endswith(".report.json")]
    assert len(pdfs) == len(jsons) == 40
    assert z.read(pdfs[0]).startswith(b"%PDF") and PdfReader(io.BytesIO(z.read(pdfs[0]))).pages
    idx = json.loads(z.read("index.json"))
    assert len(idx) == 40 and {e["doc_id"] for e in idx} == {n[5:-5] for n in jsons}
    assert json.loads(z.read("manifest.json"))["id"] == jid and not any(n.startswith("input") for n in names)
    assert z.testzip() is None


def test_zip_is_refused_while_running_and_files_are_safe(client):
    jid = start(client, {"count": 3000, "doc_type": "invoice"})
    wait_for(client, jid, lambda p: p["status"] == "running")
    assert client.get(f"/jobs/{jid}/download.zip", headers=H).status_code == 409
    client.post(f"/jobs/{jid}/cancel", headers=H)
    wait(client, jid)
    for bad in ("docs/../manifest.json", "../x", "docs/.hidden"):
        assert client.get(f"/jobs/{jid}/files/{bad}", headers=H).status_code == 404


def test_mixed_types_and_isolated_failures_in_one_bulk_job(client):
    specs = [{"doc_type": t, "seed": i, "locale": ["en-US", "fr", "hi"][i % 3]} for i, t in enumerate(["invoice", "payslip", "purchase_order", "retail_receipt", "statement"] * 8)]
    specs[3] = {"doc_type": "payslip", "locale": "nope"}          # malformed: fails alone
    specs[10] = "garbage"
    jid = start(client, {"specs": specs})
    p = wait(client, jid)
    assert p["status"] == "succeeded" and p["done"] == 40 and p["failed"] == 2
    rep = json.loads(client.get(f"/jobs/{jid}/files/report.json", headers=H).text)
    assert (rep["succeeded"], rep["failed"]) == (38, 2) and rep["by_stage"] == {"validate": 2}
    sc = client.get(f"/score/{jid}", headers=H).json()
    assert sc["score"] == pytest.approx(95.0) and sc["kind"] == "reconciliation" and sc["report"]["failed"] == 2
    assert client.get(f"/jobs/{jid}", headers=H).json()["files"] == ["index.json", "report.json"]


def test_a_job_where_everything_fails_is_failed_not_succeeded(client):
    jid = start(client, {"specs": [{"doc_type": "invoice", "locale": "zz"}] * 5})
    p = wait(client, jid)
    j = client.get(f"/jobs/{jid}", headers=H).json()
    assert p["status"] == "failed" and "all 5 documents failed" in j["error"]


def test_invalid_document_params_are_rejected_up_front(client):
    for params in ({"doc_type": "contract"}, {"doc_type": "payslip", "spec": {"locale": "zz"}}, {"count": 0}, {"count": 99999}, {"doc_type": "invoice", "spec": {"n_lines": 0}}):
        r = client.post("/generate", headers=H, json={"kind": "document", "params": params})
        assert r.status_code == 422, params


# ------------------------------------------------------------------------ cancel
def test_cancel_a_running_bulk_job_keeps_finished_documents(client):
    jid = start(client, {"count": 6000, "doc_type": "invoice"})
    wait_for(client, jid, lambda p: p["done"] >= 100)
    r = client.post(f"/jobs/{jid}/cancel", headers=H)
    assert r.status_code == 200 and r.json()["cancel_requested"] is True
    p = wait(client, jid)
    assert p["status"] == "cancelled" and 100 <= p["done"] < 6000 and p["phase"] == "cancelled"
    j = client.get(f"/jobs/{jid}", headers=H).json()
    assert j["files"] == ["index.json", "report.json"] and j["output_hash"] is None and j["status"] == "cancelled"
    idx = json.loads(client.get(f"/jobs/{jid}/files/index.json", headers=H).text)
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/jobs/{jid}/download.zip", headers=H).content))
    assert len(idx) >= 100 and len([n for n in z.namelist() if n.endswith(".pdf")]) == len(idx)   # partial work is downloadable and consistent
    rep = json.loads(z.read("report.json"))
    assert rep["cancelled"] is True and rep["total"] == 6000
    assert client.post(f"/jobs/{jid}/cancel", headers=H).status_code == 409                       # already finished
    assert client.get(f"/score/{jid}", headers=H).status_code == 409                              # no score for a cancelled job


def test_cancel_a_queued_job_before_it_starts(client):
    busy = [start(client, {"count": 8000, "doc_type": "invoice"}) for _ in range(2)]        # occupy both workers
    for b in busy:
        wait_for(client, b, lambda p: p["status"] == "running")
    queued = start(client, {"count": 50, "doc_type": "invoice"})
    assert client.get(f"/jobs/{queued}/progress", headers=H).json()["status"] == "queued"
    r = client.post(f"/jobs/{queued}/cancel", headers=H)
    assert r.status_code == 200 and r.json()["status"] == "cancelled"
    p = client.get(f"/jobs/{queued}/progress", headers=H).json()
    assert p["status"] == "cancelled" and p["done"] == 0 and p["finished"]
    assert client.get(f"/jobs/{queued}", headers=H).json()["files"] == []
    for b in busy:
        client.post(f"/jobs/{b}/cancel", headers=H)


def test_cancel_reaches_short_jobs_at_their_checkpoints(env):
    deadline = time.time() + 120
    while service._FUTURES and time.time() < deadline:      # let earlier tests' cancelled jobs finish winding down
        time.sleep(0.2)
    store = Store()
    m = service.submit(store, "tabular", {"rows": 100, "seed": 1}, wait=False)
    service.cancel(store, m["id"]) if store.read(m["id"])["status"] in ("queued", "running") else None
    deadline = time.time() + 30
    while time.time() < deadline and store.read(m["id"])["status"] in ("queued", "running", "cancelling"):
        time.sleep(0.1)
    assert store.read(m["id"])["status"] in ("cancelled", "succeeded")     # either it was stopped, or it had already finished
    with pytest.raises(ValueError):
        service.cancel(store, m["id"])


# ----------------------------------------------------- other kinds report progress
def test_tabular_and_relational_jobs_report_phases_and_finish(client):
    jid = start(client, {"rows": 300, "seed": 1}, kind="tabular")
    p = wait(client, jid)
    assert p["status"] == "succeeded" and p["done"] == p["total"] == 3 and p["phase"] == "done"
    jid = start(client, {"dataset": "shop_full", "seed": 1}, kind="relational")
    assert wait(client, jid)["status"] == "succeeded"
    assert client.get("/jobs/nope/progress", headers=H).status_code == 404 and client.post("/jobs/nope/cancel", headers=H).status_code == 404


# -------------------------------------------------------------------- reproducibility
def test_bulk_job_reruns_reproduce_the_same_hash_including_template_pdfs(client):
    params = {"specs": [{"doc_type": t, "seed": i} for i, t in enumerate(["payslip", "purchase_order", "invoice"] * 4)]}
    jid = start(client, params)
    wait(client, jid)
    rr = client.post(f"/jobs/{jid}/rerun", headers=H, json={"wait": False}).json()
    wait(client, rr["id"])
    j2 = client.get(f"/jobs/{rr['id']}", headers=H).json()
    assert j2["lineage"]["reproduced"] is True and j2["output_hash"] == client.get(f"/jobs/{jid}", headers=H).json()["output_hash"]


# ------------------------------------------------------------------------ recovery
def test_restart_recovery_marks_orphaned_jobs(env):
    store = Store()
    m = store.create("document", {"count": 5})
    store.update(m["id"], status="running", started_at=lineage.now())
    q = store.create("document", {"count": 5})
    done = store.create("tabular", {"rows": 5})
    store.update(done["id"], status="succeeded")
    assert service.recover(store) == 2
    assert store.read(m["id"])["status"] == "failed" and "interrupted" in store.read(m["id"])["error"] and store.read(q["id"])["status"] == "failed"
    assert store.read(done["id"])["status"] == "succeeded"
    with TestClient(app) as c:                                  # startup hook runs the same recovery
        store.update(m["id"], status="running")
        assert c.get("/api/history").status_code == 200
    with TestClient(app):
        pass
    assert store.read(m["id"])["status"] == "failed"


# ------------------------------------------------------------------------------ UI
def test_ui_history_endpoints_expose_progress_cancel_and_zip(env):
    c = TestClient(app)
    j = c.post("/api/history/generate", json={"kind": "document", "params": {"count": 30, "doc_type": "purchase_order"}}).json()
    t0 = time.time()
    while time.time() - t0 < 60:
        p = c.get(f"/api/history/{j['id']}/progress").json()
        if p["finished"]:
            break
        time.sleep(0.1)
    assert p["status"] == "succeeded" and p["done"] == 30
    z = zipfile.ZipFile(io.BytesIO(c.get(f"/api/history/{j['id']}/download.zip").content))
    assert len([n for n in z.namelist() if n.endswith(".pdf")]) == 30
    row = c.get("/api/history").json()[0]
    assert row["progress"]["done"] == 30 and row["kind"] == "document"
    assert c.post(f"/api/history/{j['id']}/cancel").status_code == 409
