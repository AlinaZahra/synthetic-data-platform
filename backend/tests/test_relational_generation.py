import numpy as np
import pandas as pd
import pytest

from sdp.datasets import make_shop
from sdp.relational import CardinalityConfig, RelationalGenerator, RelationshipGraph, check_integrity, infer_graph
from sdp.relational.inference import children_per_parent

CUST_KEY = "orders(customer_id)->customers"


@pytest.fixture(scope="module")
def shop():
    return make_shop(300, seed=1)


@pytest.fixture(scope="module")
def graph(shop):
    return infer_graph(shop)


@pytest.fixture(scope="module")
def fitted(shop, graph):
    return RelationalGenerator(graph).fit(shop)


def test_real_data_passes_checker(shop, graph):
    assert check_integrity(graph, shop).ok


def test_zero_violations_and_topological_generation(fitted):
    res = fitted.generate(seed=5)
    assert res.integrity.ok, res.integrity.to_dict()
    assert res.integrity.total == 0
    assert set(res.tables) == {"customers", "products", "orders", "order_items"}
    assert all(len(t) > 0 for t in res.tables.values())


def test_scale_and_explicit_rows(fitted):
    res = fitted.generate(rows={"customers": 50, "products": 10}, seed=1)
    assert len(res.tables["customers"]) == 50 and len(res.tables["products"]) == 10
    assert res.integrity.ok
    big = fitted.generate(scale=2, seed=1)
    assert len(big.tables["customers"]) == 600 and big.integrity.ok


def test_child_total_can_be_pinned(fitted):
    res = fitted.generate(rows={"customers": 100, "orders": 500}, seed=2)
    assert len(res.tables["orders"]) == 500 and res.integrity.ok


def test_dtypes_and_columns_follow_schema(fitted, shop):
    res = fitted.generate(seed=0)
    for name, real in shop.items():
        assert list(res.tables[name].columns) == list(real.columns)
        for k in [c for c in real.columns if c.endswith("_id")]:
            assert pd.api.types.is_integer_dtype(res.tables[name][k]), (name, k)


def test_reproducible(fitted):
    a, b = fitted.generate(seed=9), fitted.generate(seed=9)
    for name in a.tables:
        pd.testing.assert_frame_equal(a.tables[name], b.tables[name])
    assert not a.tables["orders"].equals(fitted.generate(seed=10).tables["orders"])


def test_learned_cardinality_matches_real(fitted):
    res = fitted.generate(scale=3, seed=3)
    rep = res.cardinality[CUST_KEY]
    assert rep["match_score"] > 0.85
    assert abs(rep["mean_synth"] - rep["mean_real"]) / rep["mean_real"] < 0.1
    assert rep["ks_statistic"] < 0.1
    assert res.scorecard()["mean_cardinality_match"] > 0.8


def test_poisson_and_zipf_targets(shop, graph):
    gen = RelationalGenerator(graph, {CUST_KEY: CardinalityConfig(kind="poisson", lam=6)}).fit(shop)
    res = gen.generate(rows={"customers": 2000}, seed=0)
    counts = children_per_parent(res.tables["orders"], graph.fk(CUST_KEY), res.tables["customers"])
    assert counts.mean() == pytest.approx(6, rel=0.05) and counts.var() == pytest.approx(6, rel=0.15)
    assert res.integrity.ok

    gen = RelationalGenerator(graph, {CUST_KEY: CardinalityConfig(kind="zipf", a=1.8, max_children=200)}).fit(shop)
    res = gen.generate(rows={"customers": 2000}, seed=0)
    counts = children_per_parent(res.tables["orders"], graph.fk(CUST_KEY), res.tables["customers"])
    assert (counts == 1).mean() > 0.4 and counts.max() > 20  # heavy tail
    assert res.integrity.ok


def test_fixed_cardinality(shop, graph):
    res = RelationalGenerator(graph, {CUST_KEY: CardinalityConfig(kind="fixed", value=2)}).fit(shop).generate(
        rows={"customers": 40}, seed=0)
    assert len(res.tables["orders"]) == 80


def test_many_to_many_via_junction_and_unique_pairs():
    rng = np.random.default_rng(1)
    students = pd.DataFrame({"student_id": range(1, 101), "grade": rng.integers(1, 5, 100)})
    courses = pd.DataFrame({"course_id": range(1, 21), "credits": [3, 4] * 10})
    pairs = {(s, c) for s in range(1, 101) for c in rng.choice(np.arange(1, 21), rng.integers(1, 6), replace=False)}
    enroll = pd.DataFrame(sorted(pairs), columns=["student_id", "course_id"])
    enroll["score"] = rng.normal(75, 10, len(enroll))
    tables = {"students": students, "courses": courses, "enrollments": enroll}
    g = infer_graph(tables)
    assert g.table("enrollments").primary_key == ["student_id", "course_id"]
    assert g.many_to_many() == [{"junction": "enrollments", "left": "courses", "right": "students"}]
    res = RelationalGenerator(g).fit(tables).generate(scale=3, seed=4)
    out = res.tables["enrollments"]
    assert res.integrity.ok and not out.duplicated(["student_id", "course_id"]).any()
    assert res.cardinality["enrollments(student_id)->students"]["match_score"] > 0.8
    assert out["score"].between(enroll["score"].min(), enroll["score"].max()).all()


def test_self_reference_and_composite_keys():
    rng = np.random.default_rng(0)
    n = 200
    mgr = [None] + [int(rng.integers(1, i)) for i in range(2, n + 1)]
    emp = pd.DataFrame({"employee_id": range(1, n + 1), "manager_id": mgr, "salary": rng.normal(6e4, 9e3, n).round()})
    orders = pd.DataFrame({"order_id": np.repeat(np.arange(1, 61), 3), "line_no": np.tile([1, 2, 3], 60),
                           "qty": rng.integers(1, 9, 180)})
    tables = {"employees": emp, "orders": orders,
              "shipments": pd.DataFrame({"sid": range(100), "order_id": np.repeat(np.arange(1, 51), 2),
                                         "line_no": np.tile([1, 2], 50)})}
    g = infer_graph(tables)
    assert g.table("orders").primary_key == ["order_id", "line_no"]
    assert g.fk("shipments(order_id,line_no)->orders").parent_columns == ["order_id", "line_no"]
    assert g.fk("employees(manager_id)->employees").is_self_reference
    res = RelationalGenerator(g).fit(tables).generate(scale=2, seed=7)
    e = res.tables["employees"]
    assert res.integrity.ok, res.integrity.to_dict()
    assert e["manager_id"].isna().sum() >= 1
    ids, mg = e["employee_id"].to_numpy(), e["manager_id"]
    assert (mg.dropna().astype(int).to_numpy() < ids[mg.notna().to_numpy()]).all()  # only earlier rows: acyclic
    o = res.tables["orders"]
    assert not o.duplicated(["order_id", "line_no"]).any()
    ship = res.tables["shipments"]
    assert ship.merge(o, on=["order_id", "line_no"]).shape[0] == len(ship)  # composite FK resolves


def test_non_nullable_self_reference_points_to_self_for_roots():
    emp = pd.DataFrame({"employee_id": [1, 2, 3, 4], "manager_id": [1, 1, 2, 3], "salary": [4.0, 3.0, 2.0, 1.0]})
    g = infer_graph({"employees": emp})
    assert not g.fks_of("employees")[0].nullable
    res = RelationalGenerator(g).fit({"employees": emp}).generate(rows={"employees": 50}, seed=0)
    assert res.integrity.ok and res.tables["employees"]["manager_id"].notna().all()


def test_nullable_non_driver_fk_gets_learned_null_rate(shop):
    items = shop["order_items"].copy()
    items["product_id"] = items["product_id"].astype("Int64")
    items.loc[items.sample(frac=0.2, random_state=0).index, "product_id"] = pd.NA
    tables = {**shop, "order_items": items}
    g = infer_graph(tables)
    assert g.fk("order_items(product_id)->products").nullable
    res = RelationalGenerator(g).fit(tables).generate(seed=0)
    rate = res.tables["order_items"]["product_id"].isna().mean()
    assert 0.15 < rate < 0.25 and res.integrity.ok


def test_generator_rejects_invalid_graph(shop, graph):
    bad = RelationshipGraph.from_json(graph.to_json())
    bad.set_primary_key("customers", ["nope"])
    with pytest.raises(ValueError, match="invalid relationship graph"):
        RelationalGenerator(bad).fit(shop)


def test_learned_kind_requires_fit(graph):
    with pytest.raises(RuntimeError, match="fit"):
        RelationalGenerator(graph).generate(rows={"customers": 5, "products": 5})
