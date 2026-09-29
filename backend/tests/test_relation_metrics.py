import numpy as np
import pandas as pd
import pytest

from sdp.datasets import SHOP_FULL_RULES, make_shop_full
from sdp.relational import RelationalGenerator, infer_graph
from sdp.relational.metrics import WEIGHTS, auto_queries, relation_metrics
from sdp.scoring import fidelity_score


@pytest.fixture(scope="module")
def world():
    real = make_shop_full(400, seed=1)
    g = infer_graph(real)
    ours = RelationalGenerator(g).fit(real).generate(seed=2, rules=SHOP_FULL_RULES).tables
    indep = RelationalGenerator(g, condition_on_parents=False).fit(real).generate(seed=2, rules=SHOP_FULL_RULES).tables
    return real, g, ours, indep


def comp(m):
    return {k: v["score"] for k, v in m["components"].items()}


def test_identical_data_scores_perfect(world):
    real, g, *_ = world
    m = relation_metrics(real, {k: v.copy() for k, v in real.items()}, g)
    assert m["score"] == pytest.approx(100.0, abs=1e-6)
    assert all(v == pytest.approx(100.0, abs=1e-6) for v in comp(m).values() if v is not None)


def test_score_ranks_conditioned_over_independent_over_broken(world):
    real, g, ours, indep = world
    s_ours, s_indep = relation_metrics(real, ours, g), relation_metrics(real, indep, g)
    broken = {k: v.copy() for k, v in ours.items()}
    rng = np.random.default_rng(0)
    broken["orders"]["customer_id"] = rng.permutation(broken["orders"]["customer_id"].to_numpy())  # relations destroyed
    broken["order_items"]["order_id"] = rng.permutation(broken["order_items"]["order_id"].to_numpy())
    s_broken = relation_metrics(real, broken, g)
    assert s_ours["score"] > s_indep["score"] + 3
    assert comp(s_ours)["cross_table"] > comp(s_indep)["cross_table"] + 10   # conditioning is what preserves segment -> value
    assert s_broken["score"] < s_ours["score"] - 8 and comp(s_broken)["cross_table"] < 30
    assert s_ours["score"] > 70


def test_components_weights_and_details(world):
    real, g, ours, _ = world
    m = relation_metrics(real, ours, g)
    assert set(m["components"]) == set(WEIGHTS)
    assert sum(v["weight"] for v in m["components"].values()) == pytest.approx(1)
    assert 0 <= m["score"] <= 100
    card = m["components"]["cardinality"]["details"]
    assert set(card) == {f.key for f in g.foreign_keys}
    assert any("segment" in f"{d['anc_col']}" for d in m["components"]["cross_table"]["details"])
    assert "customers <- order_items" in m["components"]["join_size"]["details"]  # two-hop join path
    assert all(0 <= d["score"] <= 100 for d in m["components"]["fk_coverage"]["details"].values())


def test_fk_coverage_detects_orphans(world):
    real, g, ours, _ = world
    bad = {k: v.copy() for k, v in ours.items()}
    bad["orders"]["customer_id"] = 10_000_000 + bad["orders"]["customer_id"]
    m = relation_metrics(real, bad, g)
    assert m["components"]["fk_coverage"]["details"]["orders(customer_id)->customers"]["synthetic"]["resolves"] == 0
    assert comp(m)["fk_coverage"] < comp(relation_metrics(real, ours, g))["fk_coverage"] - 10


def test_auto_queries_are_real_sql_on_both_databases(world):
    real, g, ours, _ = world
    qs = auto_queries(real, g)
    assert 1 <= len(qs) <= 6 and all(q.upper().startswith("SELECT") and "JOIN" in q for q in qs)
    assert any(q.count("JOIN") == 2 for q in qs)  # multi-table (two-hop) query present
    m = relation_metrics(real, ours, g)
    d = m["components"]["queries"]["details"]
    assert all("error" not in x and x["score"] > 40 for x in d)
    first = d[0]
    assert first["real_rows"] and first["synthetic_rows"] and "grp" in first["real_rows"][0]


def test_custom_queries_and_guards(world):
    real, g, ours, _ = world
    sql = ('SELECT c.segment AS grp, COUNT(*) AS n_orders, AVG(o.total) AS avg_total FROM orders o '
           'JOIN customers c ON o.customer_id = c.customer_id GROUP BY c.segment')
    m = relation_metrics(real, ours, g, queries=[sql, "DROP TABLE orders", "SELECT nope FROM orders"])
    d = m["components"]["queries"]["details"]
    assert d[0]["score"] > 60 and d[0]["columns"]["n_orders"]["kind"] == "share" and d[0]["columns"]["avg_total"]["kind"] == "level"
    assert "only SELECT" in d[1]["error"] and "error" in d[2]
    assert "orders" in relation_metrics(real, real, g, queries=["SELECT 1 AS grp, COUNT(*) AS n FROM orders"])["components"]["queries"]["details"][0]["sql"]


def test_feeds_the_fidelity_score(world):
    real, g, ours, _ = world
    m = relation_metrics(real, ours, g)
    f = fidelity_score(real["orders"], ours["orders"], relational=m)
    assert f["components"]["relational"]["score"] == pytest.approx(m["score"])
    assert "relation-preservation" in f["components"]["relational"]["summary"]


def test_works_on_single_table_and_no_relationships():
    df = pd.DataFrame({"a": range(100), "b": np.random.default_rng(0).normal(size=100)})
    g = infer_graph({"t": df})
    m = relation_metrics({"t": df}, {"t": df}, g)
    assert m["components"]["cardinality"]["score"] is None and m["components"]["queries"]["score"] is None
    assert m["score"] == 0.0 or 0 <= m["score"] <= 100
