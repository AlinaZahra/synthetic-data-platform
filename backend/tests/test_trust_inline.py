import numpy as np
import pytest
from fastapi.testclient import TestClient

from sdp.api.main import app

client = TestClient(app)


def rows(n=300):
    rng = np.random.default_rng(0)
    return [{"age": int(a), "spend": float(round(s, 2)), "tier": t} for a, s, t in
            zip(rng.integers(18, 80, n), rng.gamma(3, 40, n), rng.choice(["a", "b", "c"], n))]


def test_trust_report_for_uploaded_data_with_dsl_rules():
    r = client.post("/api/trust/report", json={"dataset": "inline", "data": rows(), "rows": 300, "seed": 1,
                                              "rules": ["age must be at least 30", "spend <= 400"]})
    assert r.status_code == 200
    rep = r.json()
    assert rep["title"] == "Uploaded table" and 0 <= rep["trust_score"] <= 100
    comps = {c["key"]: c for c in rep["sub_scores"][2]["components"]}
    assert comps["rules"]["score"] == 100.0 and "Your rules" == comps["rules"]["label"]   # enforced during generation
    assert "coverage" in comps


def test_trust_report_input_validation():
    assert client.post("/api/trust/report", json={"dataset": "inline", "data": rows(10)}).status_code == 422
    assert client.post("/api/trust/report", json={"dataset": "inline", "data": rows(), "rules": ["zzz > 1"]}).status_code == 422
