import json
import socket
import threading
import time

import pytest
import uvicorn
from fastapi.testclient import TestClient

from sdp import lineage, service
from sdp.api.main import app
from sdp.cli import EXIT_ERROR, EXIT_GATE, EXIT_OK, run
from sdp.lineage import JobNotFound, Store

KEY = "test-key-123"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SDP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SDP_API_KEYS", f"{KEY},other-key")
    return tmp_path


@pytest.fixture()
def store(env):
    return Store()


TAB = {"rows": 150, "seed": 3, "rules": ["age must be at least 30"], "edge_cases": {"typos": 0.05}}


def wait_done(client, jid, headers, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/jobs/{jid}", headers=headers).json()
        if j["status"] in ("succeeded", "failed"):
            return j
        time.sleep(0.2)
    raise AssertionError("job did not finish")


# ------------------------------------------------------------------ lineage
def test_manifest_records_everything_needed_to_reproduce(store):
    m = service.submit(store, "tabular", TAB, wait=True)
    assert m["status"] == "succeeded" and m["error"] is None
    assert m["request"]["seed"] == 3 and m["seed"] == 3 and m["request"]["rules"] == ["age must be at least 30"]
    assert set(m["schema"]["columns"]) >= {"age", "income", "plan"} and "_edge_case" not in m["schema"]["columns"]   # hidden column stays hidden
    assert m["model"]["engine"] == "gaussian_copula" and m["model"]["code_fingerprint"] == lineage.code_fingerprint() and m["model"]["package_version"]
    assert m["dataset"]["source"] == "demo" and len(m["dataset"]["sha256"]) == 64
    assert {f["name"] for f in m["outputs"]["files"]} == {"synthetic.csv", "report.json"} and all(len(f["sha256"]) == 64 for f in m["outputs"]["files"])
    assert m["outputs"]["output_hash"] and m["lineage"] == {"root_id": m["id"], "parent_id": None, "version": 1, "reproduced": None, "code_changed": None}
    s = service.score(store, m["id"])
    assert 0 <= s["score"] <= 100 and store.read(m["id"])["scores"]["score"] == s["score"]           # score saved into the manifest


def test_rerun_from_manifest_reproduces_bit_for_bit(store):
    m = service.submit(store, "tabular", TAB, wait=True)
    r = service.rerun(store, m["id"], wait=True)
    assert r["lineage"]["parent_id"] == m["id"] and r["lineage"]["root_id"] == m["id"] and r["lineage"]["version"] == 2
    assert r["lineage"]["reproduced"] is True and r["lineage"]["code_changed"] is False
    assert r["outputs"]["output_hash"] == m["outputs"]["output_hash"] and r["id"] != m["id"]
    assert store.artifact(r["id"], "synthetic.csv") == store.artifact(m["id"], "synthetic.csv")


def test_overrides_make_a_new_version_not_a_reproducibility_claim(store):
    m = service.submit(store, "tabular", TAB, wait=True)
    v2 = service.rerun(store, m["id"], wait=True, overrides={"seed": 4})
    assert v2["seed"] == 4 and v2["lineage"]["reproduced"] is None and v2["outputs"]["output_hash"] != m["outputs"]["output_hash"]
    v3 = service.rerun(store, v2["id"], wait=True)
    assert [x["lineage"]["version"] for x in store.versions(m["id"])] == [1, 2, 3] and v3["lineage"]["reproduced"] is True
    assert {x["id"] for x in store.versions(v3["id"])} == {m["id"], v2["id"], v3["id"]}


def test_code_change_is_detected(store, monkeypatch):
    m = service.submit(store, "tabular", TAB, wait=True)
    monkeypatch.setattr(lineage, "code_fingerprint", lambda: "deadbeefdeadbeef")
    r = service.rerun(store, m["id"], wait=True)
    assert r["lineage"]["code_changed"] is True and r["model"]["code_fingerprint"] == "deadbeefdeadbeef"
    assert r["lineage"]["reproduced"] is True          # same output despite a different fingerprint: reported honestly, not assumed


@pytest.mark.parametrize("kind,params", [
    ("relational", {"dataset": "shop_full", "seed": 2, "rules": ["orders.discount <= 0.3"]}),
    ("nl", {"text": "80 Pakistani bank customers, 5% fraud, 2 months of history", "seed": 4}),
    ("document", {"doc_type": "statement", "count": 2, "seed": 5, "spec": {"locale": "en-GB", "n_transactions": 15}}),
])
def test_every_job_kind_reproduces(store, kind, params):
    m = service.submit(store, kind, params, wait=True)
    assert m["status"] == "succeeded", m["error"]
    r = service.rerun(store, m["id"], wait=True)
    assert r["lineage"]["reproduced"] is True
    assert service.score(store, m["id"])["score"] > 50


def test_inline_data_is_stored_with_the_job_and_survives_rerun(store):
    rows = [{"age": 20 + i % 50, "spend": float(i), "tier": "abc"[i % 3]} for i in range(120)]
    m = service.submit(store, "tabular", {"dataset": "inline", "data": rows, "rows": 50, "seed": 1}, wait=True)
    assert m["status"] == "succeeded" and m["request"].get("data") is None and m["request"]["data_ref"] == "input.csv"
    assert "input.csv" in {p.name for p in store.path(m["id"]).iterdir()}
    r = service.rerun(store, m["id"], wait=True)
    assert r["lineage"]["reproduced"] is True


def test_failed_jobs_are_recorded_not_raised(store):
    m = service.submit(store, "nl", {"text": "gibberish nobody understands"}, wait=True)
    assert m["status"] == "failed" and m["error"] and m["outputs"]["files"] == []
    with pytest.raises(ValueError, match="failed"):
        service.score(store, m["id"])
    for bad in (("tabular", {"rows": -1}), ("nope", {}), ("nl", {})):
        with pytest.raises(Exception):
            service.submit(store, *bad, wait=True)


def test_store_rejects_path_tricks(store):
    m = service.submit(store, "tabular", {"rows": 50, "seed": 1}, wait=True)
    for name in ("../x", "manifest.json", ".hidden", "a/b"):
        with pytest.raises(JobNotFound):
            store.artifact(m["id"], name)
    with pytest.raises(JobNotFound):
        store.read("../../etc")


# ------------------------------------------------------------------ REST API
def test_api_key_is_required_everywhere(env):
    c = TestClient(app)
    for method, path in (("post", "/generate"), ("get", "/jobs"), ("get", "/jobs/x"), ("get", "/score/x"), ("post", "/jobs/x/rerun")):
        r = getattr(c, method)(path, **({"json": {"kind": "tabular"}} if method == "post" else {}))
        assert r.status_code == 401 and r.headers["www-authenticate"] == "ApiKey", path
    assert c.get("/jobs", headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/jobs", headers={"X-API-Key": KEY}).status_code == 200 and c.get("/jobs", headers={"X-API-Key": "other-key"}).status_code == 200
    assert c.get("/jobs", headers={"Authorization": f"Bearer {KEY}"}).status_code == 200
    assert c.get("/api/health").status_code == 200                                # UI endpoints stay open


def test_generate_poll_score_download_flow(env):
    c, h = TestClient(app), {"X-API-Key": KEY}
    r = c.post("/generate", headers=h, json={"kind": "tabular", "params": TAB})
    assert r.status_code == 202 and set(r.json()["links"]) == {"self", "score", "manifest"}
    jid = r.json()["id"]
    j = wait_done(c, jid, h)
    assert j["status"] == "succeeded" and j["files"] == ["synthetic.csv", "report.json"] and j["lineage"]["version"] == 1
    sc = c.get(f"/score/{jid}", headers=h).json()
    assert sc["job_id"] == jid and 0 <= sc["score"] <= 100 and sc["kind"] == "trust" and "report" in sc
    csv = c.get(f"/jobs/{jid}/files/synthetic.csv", headers=h)
    assert csv.status_code == 200 and csv.text.splitlines()[0].startswith("age,") and len(csv.text.splitlines()) == 151
    man = c.get(f"/jobs/{jid}/manifest", headers=h).json()
    assert man["scores"]["score"] == sc["score"] and man["request"]["seed"] == 3
    rr = c.post(f"/jobs/{jid}/rerun", headers=h, json={"wait": True}).json()
    assert rr["reproduced"] is True and rr["parent_id"] == jid and rr["version"] == 2
    assert [v["version"] for v in c.get(f"/jobs/{jid}/versions", headers=h).json()] == [1, 2]
    assert {x["id"] for x in c.get("/jobs?kind=tabular", headers=h).json()} >= {jid, rr["id"]}


def test_wait_true_and_errors(env):
    c, h = TestClient(app), {"X-API-Key": KEY}
    r = c.post("/generate", headers=h, json={"kind": "document", "params": {"doc_type": "receipt", "count": 2}, "wait": True})
    assert r.status_code == 200 and r.json()["status"] == "succeeded" and r.json()["files"] == ["index.json", "report.json"]
    idx = json.loads(c.get(f"/jobs/{r.json()['id']}/files/index.json", headers=h).text)
    assert len(idx) == 2 and c.get(f"/jobs/{r.json()['id']}/files/docs/{idx[0]['doc_id']}.pdf", headers=h).content.startswith(b"%PDF")
    assert c.post("/generate", headers=h, json={"kind": "tabular", "params": {"rows": -5}}).status_code == 422
    assert c.post("/generate", headers=h, json={"kind": "spreadsheet", "params": {}}).status_code == 422
    assert c.post("/generate", headers=h, json={"kind": "tabular", "params": {"bogus": 1}}).status_code == 422
    for path in ("/jobs/nope", "/jobs/nope/manifest", "/score/nope", "/jobs/nope/files/x"):
        assert c.get(path, headers=h).status_code == 404
    bad = c.post("/generate", headers=h, json={"kind": "nl", "params": {"text": "what"}, "wait": True}).json()
    assert bad["status"] == "failed" and c.get(f"/score/{bad['id']}", headers=h).status_code == 409


def test_concurrent_jobs_all_finish(env):
    c, h = TestClient(app), {"X-API-Key": KEY}
    ids = [c.post("/generate", headers=h, json={"kind": "tabular", "params": {"rows": 80, "seed": i}}).json()["id"] for i in range(4)]
    assert len(set(ids)) == 4
    assert all(wait_done(c, i, h)["status"] == "succeeded" for i in ids)


def test_openapi_documents_the_public_api(env):
    spec = TestClient(app).get("/openapi.json").json()
    for path in ("/generate", "/jobs/{job_id}", "/score/{job_id}"):
        assert path in spec["paths"]
    assert spec["paths"]["/generate"]["post"]["tags"] == ["public API"]
    assert any(s.get("type") == "apiKey" and s.get("name") == "X-API-Key" for s in spec["components"]["securitySchemes"].values())
    assert TestClient(app).get("/docs").status_code == 200


def test_ui_history_endpoints_need_no_key_and_support_rerun(env):
    c = TestClient(app)
    r = c.post("/api/history/generate", json={"kind": "tabular", "params": {"rows": 60, "seed": 2}, "wait": True}).json()
    assert r["status"] == "succeeded" and r["links"]["self"].startswith("/api/history")
    assert [x["id"] for x in c.get("/api/history").json()][0] == r["id"]
    rr = c.post(f"/api/history/{r['id']}/rerun", json={"wait": True}).json()
    assert rr["reproduced"] is True and c.get(f"/api/history/{r['id']}").json()["outputs"]["output_hash"] == rr["output_hash"]
    assert c.get(f"/api/history/{r['id']}/score").status_code == 200 and len(c.get(f"/api/history/{r['id']}/versions").json()) == 2


# ------------------------------------------------------------------------ CLI
def cli(env, *args, capsys):
    code = run(["--data-dir", str(env / "data"), *args])
    out = capsys.readouterr()
    return code, (json.loads(out.out) if out.out.strip() else None), out.err


def test_cli_generate_score_gate_and_outputs(env, capsys):
    out_dir = env / "out"
    code, o, _ = cli(env, "generate", "--kind", "tabular", "--params", json.dumps(TAB), "--score", "--min-score", "50", "--out", str(out_dir), capsys=capsys)
    assert code == EXIT_OK and o["job"]["status"] == "succeeded" and o["score"]["score"] >= 50
    assert (out_dir / "synthetic.csv").exists() and json.loads((out_dir / "manifest.json").read_text())["id"] == o["job"]["id"]
    code, o2, err = cli(env, "generate", "--kind", "tabular", "--params", json.dumps(TAB), "--min-score", "99.99", capsys=capsys)
    assert code == EXIT_GATE and "below --min-score" in o2["gate"]


def test_cli_jobs_rerun_versions_manifest_download(env, capsys):
    _, o, _ = cli(env, "generate", "--kind", "tabular", "--params", '{"rows": 60, "seed": 5}', capsys=capsys)
    jid = o["job"]["id"]
    code, jobs, _ = cli(env, "jobs", capsys=capsys)
    assert code == EXIT_OK and jobs[0]["id"] == jid
    assert cli(env, "jobs", "get", jid, capsys=capsys)[1]["id"] == jid
    code, r, _ = cli(env, "rerun", jid, capsys=capsys)
    assert code == EXIT_OK and r["reproduced"] is True
    code, r2, _ = cli(env, "rerun", jid, "--override", '{"seed": 6}', capsys=capsys)
    assert code == EXIT_OK and r2["seed"] == 6 and r2["reproduced"] is None
    assert [v["version"] for v in cli(env, "versions", jid, capsys=capsys)[1]] == [1, 2, 3]
    assert cli(env, "manifest", jid, capsys=capsys)[1]["model"]["engine"] == "gaussian_copula"
    dest = env / "f.csv"
    assert cli(env, "download", jid, "synthetic.csv", "--out", str(dest), capsys=capsys)[0] == EXIT_OK and dest.read_text().startswith("age,")
    code, s, _ = cli(env, "score", jid, "--min-score", "10", capsys=capsys)
    assert code == EXIT_OK and s["job_id"] == jid


def test_cli_errors_exit_1_with_message(env, capsys):
    code, _, err = cli(env, "generate", "--kind", "tabular", "--params", '{"rows": -3}', capsys=capsys)
    assert code == EXIT_ERROR and "rows" in err
    code, _, err = cli(env, "score", "nope", capsys=capsys)
    assert code == EXIT_ERROR and "not found" in err
    code, _, err = cli(env, "jobs", "get", capsys=capsys)
    assert code == EXIT_ERROR


def test_cli_failed_job_exits_2(env, capsys):
    code, o, _ = cli(env, "generate", "--kind", "nl", "--params", '{"text": "nothing useful"}', capsys=capsys)
    assert code == EXIT_GATE and o["job"]["status"] == "failed"


def test_cli_remote_mode_against_a_live_server(env, capsys):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    try:
        base = ["--url", f"http://127.0.0.1:{port}"]
        code = run([*base, "--api-key", "wrong", "jobs"])
        assert code == EXIT_ERROR and "401" in capsys.readouterr().err
        code = run([*base, "--api-key", KEY, "generate", "--kind", "tabular", "--params", '{"rows": 70, "seed": 2}', "--score", "--out", str(env / "remote")])
        o = json.loads(capsys.readouterr().out)
        assert code == EXIT_OK and o["job"]["status"] == "succeeded" and "score" in o and (env / "remote" / "synthetic.csv").exists()
        code = run([*base, "--api-key", KEY, "rerun", o["job"]["id"]])
        assert code == EXIT_OK and json.loads(capsys.readouterr().out)["reproduced"] is True
    finally:
        server.should_exit = True
        th.join(timeout=10)
