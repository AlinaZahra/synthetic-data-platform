import pytest
from fastapi.testclient import TestClient

from sdp.api.main import app
from sdp.datasets import TABULAR_SAMPLES

client = TestClient(app)


@pytest.mark.parametrize("name", list(TABULAR_SAMPLES))
def test_every_sample_is_deterministic_and_has_a_usable_target(name):
    build, target, title = TABULAR_SAMPLES[name]
    a, b = build(500, 3), build(500, 3)
    assert a.equals(b) and not a.equals(build(500, 4)) and title
    assert set(a[target].unique()) == {0, 1} and 0.05 < a[target].mean() < 0.95


def test_samples_endpoint_and_per_sample_metadata():
    s = client.get("/api/demo/tabular/samples").json()
    assert {x["name"] for x in s} == {"customers", "students", "employees"}
    d = client.get("/api/demo/tabular?sample=students").json()
    assert "prior_gpa" in d["columns"] and d["target"] == "passed" and d["sample"] == "students"
    assert client.get("/api/demo/tabular").json()["target"] == "churned"       # default is unchanged
    assert client.get("/api/demo/tabular?sample=nope").status_code == 422


def test_generate_rules_tstr_and_export_follow_the_chosen_sample():
    body = {"sample": "employees", "config": {"rows": 120, "seed": 2}, "rules": ["salary must be at least 40000"]}
    g = client.post("/api/tabular/generate", json=body).json()
    assert "salary" in g["columns"] and "income" not in g["columns"] and len(g["preview"]) > 0 and g["overlay"]
    assert all(r["salary"] >= 40000 for r in g["preview"])
    t = client.post("/api/tabular/tstr", json={"sample": "students", "target": "passed", "config": {"rows": 300, "seed": 1}})
    assert t.status_code == 200 and t.json()["target"] == "passed"
    csv = client.post("/api/export/tabular", json={"sample": "students", "config": {"rows": 40, "seed": 1}}).text
    assert csv.splitlines()[0].startswith("age,major,year_of_study")
    assert client.post("/api/tabular/generate", json={"sample": "nope", "config": {"rows": 10}}).status_code == 422


def test_rules_compile_against_a_sample_or_uploaded_rows():
    ok = client.post("/api/rules/compile", json={"rules": ["attendance_pct >= 50"], "dataset": "customers", "sample": "students"}).json()
    assert ok["results"][0]["ok"] and "attendance_pct" in ok["columns"]
    bad = client.post("/api/rules/compile", json={"rules": ["attendance_pct >= 50"], "dataset": "customers"}).json()   # customers has no such column
    assert not bad["results"][0]["ok"]
    rows = [{"score": 40 + i % 50, "grade": "ab"[i % 2]} for i in range(60)]
    up = client.post("/api/rules/compile", json={"rules": ["score must be at least 45"], "dataset": "customers", "data": rows}).json()
    assert up["results"][0]["ok"] and up["columns"] == ["score", "grade"]


def test_uploaded_table_works_end_to_end():
    rows = [{"score": 40 + (i * 7) % 55, "hours": (i % 9) + 1.5, "grade": "ABC"[i % 3]} for i in range(150)]
    g = client.post("/api/tabular/generate", json={"dataset": "inline", "data": rows, "config": {"rows": 60, "seed": 1}, "rules": ["score must be at least 50"]})
    assert g.status_code == 200 and g.json()["columns"] == ["score", "hours", "grade"] and all(r["score"] >= 50 for r in g.json()["preview"])
    t = client.post("/api/tabular/tstr", json={"dataset": "inline", "data": rows, "target": "grade", "config": {"rows": 100, "seed": 1}})
    assert t.status_code == 200
