import pytest
from fastapi.testclient import TestClient

from sdp.api.main import app

client = TestClient(app)


def test_health_and_demo():
    assert client.get("/api/health").json() == {"status": "ok"}
    d = client.get("/api/demo/tabular").json()
    assert "churned" in d["columns"] and len(d["preview"]) == 50
    r = client.get("/api/demo/relational").json()
    assert set(r["row_counts"]) == {"customers", "products", "orders", "order_items"}


def test_tabular_generate_is_reproducible_and_logged():
    body = {"config": {"rows": 300, "seed": 11, "null_rate": {"income": 0.1}, "outlier_rate": {"age": 0.05}}}
    a = client.post("/api/tabular/generate", json=body).json()
    b = client.post("/api/tabular/generate", json=body).json()
    assert a["preview"] == b["preview"] and a["n_rows"] == 300
    assert {x["action"] for x in a["injection_log"]} == {"null", "outlier"}
    assert a["fidelity"]["correlation_gap"] < 0.1
    assert a["dtypes"]["age"] == "int64" and a["dtypes"]["income"] == "float64"


def test_tabular_bad_config_is_422():
    assert client.post("/api/tabular/generate", json={"config": {"rows": 10, "null_rate": {"nope": 0.1}}}).status_code == 422
    assert client.post("/api/tabular/generate", json={"config": {"rows": 10, "null_rate": {"age": 2}}}).status_code == 422


@pytest.mark.slow
def test_tabular_tstr_endpoint():
    r = client.post("/api/tabular/tstr", json={"target": "churned", "config": {"rows": 500, "seed": 1}})
    assert r.status_code == 200 and r.json()["task"] == "classification"
    assert client.post("/api/tabular/tstr", json={"config": {"rows": 5}}).status_code == 422


def test_relational_infer_edit_generate_roundtrip():
    inf = client.post("/api/relational/infer", json={}).json()
    assert inf["errors"] == [] and inf["order"].index("orders") > inf["order"].index("customers")
    graph = inf["graph"]
    assert len(graph["foreign_keys"]) == 3 and graph["many_to_many"][0]["junction"] == "order_items"

    graph["foreign_keys"] = [f for f in graph["foreign_keys"] if f["parent_table"] != "products"]  # user edit
    v = client.post("/api/relational/validate", json={"graph": graph}).json()
    assert v["errors"] == []

    key = "orders(customer_id)->customers"
    r = client.post("/api/relational/generate", json={
        "graph": graph, "rows": {"customers": 60}, "seed": 3,
        "cardinality": {key: {"kind": "poisson", "lam": 4}}}).json()
    assert r["scorecard"]["integrity"]["total_violations"] == 0
    assert r["scorecard"]["cardinality"][key]["target"] == "poisson"


def test_relational_ddl_and_errors():
    ddl = "CREATE TABLE a (id INT PRIMARY KEY); CREATE TABLE b (id INT PRIMARY KEY, a_id INT REFERENCES a(id));"
    r = client.post("/api/relational/infer", json={"ddl": ddl, "dataset": None}).json()
    assert [f["key"] for f in r["graph"]["foreign_keys"]] == ["b(a_id)->a"]
    bad = client.post("/api/relational/generate", json={"cardinality": {"x(y)->z": {"kind": "fixed"}}})
    assert bad.status_code == 422
    assert client.post("/api/relational/infer", json={"ddl": "CREATE TABLE x (a INT REFERENCES nope(id));"}).status_code == 422
