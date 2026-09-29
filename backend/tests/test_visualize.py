import io
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

from sdp import visualize as viz
from sdp.api.main import app
from sdp.datasets import TABULAR_SAMPLES
from sdp.scoring import build_trust_report, trust_report_pdf

KEY = "viz-key"
H = {"X-API-Key": KEY}
MAX_BYTES = 30_000            # every chart payload is a few KB; 30 KB is a generous ceiling


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SDP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SDP_API_KEYS", KEY)
    return TestClient(app)


def get(client, dataset, **params):
    return client.get(f"/api/visualize/{dataset}", params=params)


def small(r):
    assert r.status_code == 200, r.text
    assert len(r.content) < MAX_BYTES, f"{len(r.content)} bytes"
    return r.json()


# ------------------------------------------------------------ catalog
def test_catalog_lists_datasets_types_and_saved_runs(client):
    j = client.post("/generate", json={"kind": "tabular", "params": {"rows": 200, "seed": 1}, "wait": True}, headers=H).json()
    c = client.get("/api/visualize").json()
    assert c["types"] == list(viz.TYPES) and {d["id"] for d in c["datasets"]} >= {"customers", "students", "employees", "shop_full", "bank_customers", "documents-demo"}
    assert all(d["types"] for d in c["datasets"]) and any(r["id"] == j["id"] for r in c["saved_runs"])


# ------------------------------------------------------- distribution
def test_distribution_overlays_real_and_synthetic_with_shared_bins(client):
    d = small(get(client, "customers", type="distribution", column="income", bins=15))
    assert d["column"] == "income" and len(d["bins"]) == 15 and "age" in d["columns"] and d["kind_of_column"] == "number"
    assert abs(sum(b["real"] for b in d["bins"]) - 1) < 1e-3 and abs(sum(b["synthetic"] for b in d["bins"]) - 1) < 1e-3
    assert d["overlap_pct"] > 70 and d["stats"]["real"]["n"] == 2000 and d["stats"]["synthetic"]["n"] == 1000 and "line up" in d["caption"]
    # a date column bins over days and is labelled in dates
    dt = small(get(client, "customers", type="distribution", column="signup_date"))
    assert dt["kind_of_column"] == "date" and dt["bins"][0]["x0"].startswith("20")


def test_distribution_defaults_and_errors(client):
    assert small(get(client, "students", type="distribution"))["column"] in small(get(client, "students", type="distribution"))["columns"]
    r = get(client, "customers", type="distribution", column="plan")
    assert r.status_code == 422 and "categorical" in r.json()["detail"]
    assert get(client, "customers", type="distribution", column="nope").status_code == 422
    assert get(client, "customers", type="distribution", bins=3).status_code == 422 and get(client, "customers", type="distribution", bins=500).status_code == 422


# -------------------------------------------------------- categorical
def test_categorical_shares_counts_and_small_cell_suppression(client):
    d = small(get(client, "customers", type="categorical", column="plan"))
    assert {c["label"] for c in d["categories"]} == {"basic", "pro", "enterprise"} and d["match_pct"] > 80
    assert abs(sum(c["real"] for c in d["categories"]) - 1) < 1e-3 and all(c["real_count"] >= viz.MIN_CELL for c in d["categories"])
    b = viz.Bundle("t", "tabular", "t", "built-in", {"data": pd.DataFrame({"c": ["a"] * 200 + ["b"] * 100 + ["rare1"] * 2 + ["rare2"] * 3})},
                   {"data": pd.DataFrame({"c": ["a"] * 200 + ["b"] * 100 + ["rare1"] * 2 + ["rare2"] * 3})})
    out = viz.categorical(b, "c")
    assert [c["label"] for c in out["categories"]] == ["a", "b", "other (rare values)"] and out["categories"][-1]["real_count"] == 5


def test_identifier_like_columns_are_never_charted(client):
    real = pd.DataFrame({"name": [f"person {i}" for i in range(300)], "city": ["x", "y", "z"] * 100, "amount": np.linspace(1, 100, 300)})
    b = viz.Bundle("t", "tabular", "t", "built-in", {"data": real}, {"data": real})
    with pytest.raises(viz.VisualizeError, match="identifier"):
        viz.categorical(b, "name")
    assert "name" not in viz.categorical(b)["columns"]
    unique_numbers = pd.DataFrame({"row_id": np.arange(300), "v": np.random.default_rng(0).normal(size=300)})
    b2 = viz.Bundle("t", "tabular", "t", "built-in", {"data": unique_numbers}, {"data": unique_numbers})
    assert viz.distribution(b2)["column"] == "v"


# -------------------------------------------------------- correlation
def test_correlation_matrices_are_square_symmetric_and_bounded(client):
    d = small(get(client, "customers", type="correlation"))
    n = len(d["columns"])
    for k in ("real_matrix", "synthetic_matrix", "diff_matrix"):
        m = np.array(d[k])
        assert m.shape == (n, n) and np.allclose(m, m.T, atol=1e-3) and np.abs(m).max() <= 2.0
    assert np.allclose(np.diag(d["real_matrix"]), 1.0) and np.allclose(np.array(d["diff_matrix"]), np.array(d["synthetic_matrix"]) - np.array(d["real_matrix"]), atol=2e-3)
    assert 0 <= d["mean_abs_diff"] < 0.3 and len(d["worst_pairs"]) == 3 and n <= viz.MAX_MATRIX
    assert small(get(client, "shop_full", type="correlation", table="order_items"))["table"] == "order_items"


# --------------------------------------------------------------- tstr
def test_tstr_compares_real_and_synthetic_trained_models(client):
    d = small(get(client, "students", type="tstr"))
    assert d["target"] == "passed" and d["metric"] in ("auc", "f1_macro") and len(d["models"]) >= 2 and "passed" in d["targets"]
    for m in d["models"]:
        assert 0.4 <= m["real"] <= 1 and 0.3 <= m["synthetic"] <= 1
    assert d["mean_gap_pct"] is not None and abs(d["mean_gap_pct"]) < 40 and "never saw" in d["caption"]
    assert small(get(client, "customers", type="tstr", target="is_active"))["target"] == "is_active"
    assert get(client, "customers", type="tstr", target="zzz").status_code == 422


# -------------------------------------------------------- cardinality
def test_cardinality_histogram_er_graph_and_integrity_counter(client):
    d = small(get(client, "shop_full", type="cardinality"))
    assert d["child"] == "orders" and d["parent"] == "customers" and len(d["bins"]) == 11 and d["bins"][-1]["label"] == "10+"
    assert abs(sum(b["real"] for b in d["bins"]) - 1) < 1e-3 and abs(sum(b["synthetic"] for b in d["bins"]) - 1) < 1e-3
    assert d["orphans"] == {"real": 0, "synthetic": 0} and d["integrity"]["ok"] and d["integrity"]["total_violations"] == 0
    names = {n["id"] for n in d["graph"]["nodes"]}
    assert names == {"customers", "products", "orders", "order_items", "shipments"} and all(n["rows"] > 0 for n in d["graph"]["nodes"])
    assert all(e["source"] in names and e["target"] in names and e["orphans"] == 0 for e in d["graph"]["edges"]) and len(d["graph"]["edges"]) >= 4
    other = small(get(client, "shop_full", type="cardinality", fk=d["fks"][-1]))
    assert other["fk"] == d["fks"][-1] and get(client, "shop_full", type="cardinality", fk="nope").status_code == 422


def test_orphans_are_counted_when_a_link_is_broken():
    from sdp.datasets import make_shop
    from sdp.relational import infer_graph
    real = make_shop(120, seed=0)
    graph = infer_graph(real)
    broken = {k: v.copy() for k, v in real.items()}
    broken["orders"].loc[:4, "customer_id"] = 999_999
    b = viz.Bundle("t", "relational", "t", "built-in", broken, real, graph)
    d = viz.cardinality(b, "orders(customer_id)->customers")
    assert d["orphans"] == {"real": 0, "synthetic": 5} and not d["integrity"]["ok"] and d["integrity"]["total_violations"] >= 5
    assert any(e["orphans"] == 5 for e in d["graph"]["edges"])


# ---------------------------------------------------- privacy distance
def test_privacy_distance_compares_with_unseen_real_rows(client):
    d = small(get(client, "customers", type="privacy_distance"))
    assert len(d["bins"]) == 20 and d["median"]["synthetic"] > 0 and d["ratio"] > 0.5 and d["exact_copies"] == 0 and d["rows_compared"] <= 2000
    assert abs(sum(b["real"] for b in d["bins"]) - 1) < 1e-2 and "Real (unseen)" == d["series_labels"]["real"] and "closest real record" in d["caption"]


def test_privacy_distance_flags_copies():
    real = TABULAR_SAMPLES["customers"][0](600, seed=0)
    b = viz.Bundle("t", "tabular", "t", "job", {"data": real.head(200).copy()}, {"data": real})            # "synthetic" rows ARE real rows
    d = viz.privacy_distance(b)
    assert d["exact_copies"] > 0 and d["ratio"] < 0.6 and "too close" in d["verdict"]


# ------------------------------------------------------ locale validity
def test_locale_validity_pass_rates_per_check(client):
    d = small(get(client, "bank_customers", type="locale_validity", locale="ur-PK", rows=300))
    assert d["locale"] == "ur-PK" and d["valid_pct"] == 100 and {c["check"] for c in d["checks"]} >= {"phone", "national_id", "name_script"}
    assert all(0 <= c["pass_pct"] <= 100 and c["checked"] > 0 for c in d["checks"]) and "follow the rules" in d["caption"]
    fr = small(get(client, "ecommerce_customers", type="locale_validity", locale="fr", rows=200))
    assert fr["locale"] == "fr"
    r = get(client, "customers", type="locale_validity")
    assert r.status_code == 422


# -------------------------------------------------------- batch summary
def test_batch_summary_counts_and_reconciliation(client):
    d = small(get(client, "documents-demo", type="batch_summary"))
    assert d["total"] == 40 and d["succeeded"] == 38 and d["failed"] == 2 and d["by_stage"] == {"validate": 2}
    assert d["success_rate_pct"] == 95.0 and d["reconciliation_rate_pct"] == 100.0 and len(d["failures"]) == 2 and all(len(f["message"]) <= 160 for f in d["failures"])
    assert "isolated" in d["caption"]


def test_batch_summary_from_a_saved_bulk_job(client):
    j = client.post("/generate", json={"kind": "document", "params": {"doc_type": "payslip", "count": 6, "seed": 1}, "wait": True}, headers=H).json()
    d = small(get(client, j["id"], type="batch_summary"))
    assert d["total"] == 6 and d["succeeded"] == 6 and d["failed"] == 0 and "None failed" in d["caption"]


# ------------------------------------------------------------- saved runs
def test_saved_tabular_and_relational_runs_can_be_visualized(client):
    t = client.post("/generate", json={"kind": "tabular", "params": {"rows": 400, "seed": 3}, "wait": True}, headers=H).json()
    d = small(get(client, t["id"], type="distribution", column="income"))
    assert d["source"] == "job" and d["stats"]["synthetic"]["n"] == 400 and d["stats"]["real"]["n"] == 2000
    assert small(get(client, t["id"], type="correlation"))["mean_abs_diff"] < 0.35
    r = client.post("/generate", json={"kind": "relational", "params": {"dataset": "shop", "seed": 1}, "wait": True}, headers=H).json()
    c = small(get(client, r["id"], type="cardinality"))
    assert c["child"] and c["integrity"]["ok"] and c["source"] == "job"
    n = client.post("/generate", json={"kind": "nl", "params": {"text": "200 Pakistani bank customers, 3% fraud, 3 months of history"}, "wait": True}, headers=H).json()
    assert small(get(client, n["id"], type="locale_validity"))["valid_pct"] == 100
    no_real = small(get(client, n["id"], type="distribution", table="customers"))
    assert no_real["has_real"] is False and all(b["real"] is None for b in no_real["bins"]) and "no real data" in no_real["caption"]
    assert get(client, n["id"], type="correlation", table="customers").status_code == 422


def test_unfinished_and_unknown_datasets(client):
    assert get(client, "nope", type="distribution").status_code == 404
    assert get(client, "20250101000000-abcdef", type="distribution").status_code == 404
    from sdp.lineage import Store
    st = Store()
    m = st.create("tabular", {"rows": 10}, None)
    assert get(client, m["id"], type="distribution").status_code == 409                # queued: not ready yet
    assert get(client, "customers", type="pie").status_code == 422 and client.get("/api/visualize/customers").status_code == 422


# ------------------------------------------------- privacy of the output
def test_responses_contain_only_aggregates_never_rows(client):
    real = TABULAR_SAMPLES["students"][0](2000, seed=0)
    blob = json.dumps([small(get(client, "students", type=t)) for t in ("distribution", "categorical", "correlation", "privacy_distance")])
    assert len(blob) < 60_000
    for v in real["enrolled_date"].astype(str).head(20):        # no raw date values appear beyond bin edges
        assert v not in blob
    assert all(len(small(get(client, "customers", type="distribution", rows=20000))["bins"]) <= viz.MAX_BINS for _ in range(1))
    big = small(get(client, "customers", type="distribution", rows=20000, bins=60))
    assert len(big["bins"]) == 60 and len(json.dumps(big)) < 12_000                                  # size does not grow with the data


def test_payload_is_deterministic_and_seeded(client):
    a = get(client, "customers", type="distribution", column="age", seed=4).json()
    assert a == get(client, "customers", type="distribution", column="age", seed=4).json()
    assert a != get(client, "customers", type="distribution", column="age", seed=5).json()


# --------------------------------------------------------------- API key
def test_public_route_needs_a_key(client):
    assert client.get("/visualize/customers", params={"type": "categorical"}).status_code == 401
    r = client.get("/visualize/customers", params={"type": "categorical", "column": "region"}, headers=H)
    assert r.status_code == 200 and r.json()["column"] == "region"


# ------------------------------------------------------------------ PDF
def test_scorecard_pdf_embeds_the_charts(client):
    real = TABULAR_SAMPLES["customers"][0](1200, seed=0)
    from sdp.tabular import TabularGenerator
    rep = build_trust_report(real, TabularGenerator().fit(real).sample(800, seed=1), seed=1, title="demo")
    vis = {t: small(get(client, "customers", type=t)) for t in ("distribution", "categorical", "correlation", "tstr", "privacy_distance")}
    vis["cardinality"] = small(get(client, "shop_full", type="cardinality"))
    vis["locale_validity"] = small(get(client, "bank_customers", type="locale_validity"))
    vis["batch_summary"] = small(get(client, "documents-demo", type="batch_summary"))
    plain = trust_report_pdf(rep)
    rich = client.post("/api/trust/pdf", json={"report": rep, "visuals": vis})
    assert rich.status_code == 200 and rich.content.startswith(b"%PDF") and len(rich.content) > len(plain)
    pages = PdfReader(io.BytesIO(rich.content)).pages
    assert len(pages) > len(PdfReader(io.BytesIO(plain)).pages)
    text = " ".join(p.extract_text() for p in pages[2:])
    assert "Charts" in text and "Correlations" in text and "Category mix: plan" in text and "Rows per parent" in text and "Valid for the country" in text and "Document batch" in text
    assert "line up" in text and "never saw" in text                                                   # captions are printed under the charts
    assert client.post("/api/trust/pdf", json={"report": rep}).status_code == 200                       # visuals stay optional
    assert client.post("/api/trust/pdf", json={"report": rep, "visuals": {"distribution": {"type": "distribution"}}}).status_code == 422
