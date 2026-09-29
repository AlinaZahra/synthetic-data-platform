import numpy as np
import pandas as pd
import pytest

from sdp.datasets import SHOP_FULL_RULES, make_customers, make_shop_full
from sdp.relational import RelationalGenerator, check_integrity, infer_graph
from sdp.rules import (Evaluator, RuleSyntaxError, compare_naive_vs_enforced, compile_rule, compile_rules, enforce,
                       evaluate_rules, parse, plain_to_dsl, sample_with_rules, single_table_graph, to_dsl)
from sdp.rules.dsl import Bin
from sdp.tabular import TabularGenerator


# ------------------------------------------------------------------- DSL
def test_parser_precedence_and_literals():
    a = parse("a + b * 2 > 10 AND NOT c IS NULL OR d IN (1, 2, 'x')")
    assert isinstance(a, Bin) and a.op == "or"
    assert parse("discount <= 30%").right.value == pytest.approx(0.3)
    assert parse("x BETWEEN 1 AND 5").hi.value == 5
    assert parse("a => b").op == "implies" and parse("a IMPLIES b").op == "implies"
    assert parse("name = 'O''Brien'").right.value == "O'Brien"


@pytest.mark.parametrize("bad", ["", "a >", "a > > b", "SUM(", "foo(a)", "a $ b", "a IN 1", "(a > 1", "a > 1)"])
def test_syntax_errors(bad):
    with pytest.raises(RuleSyntaxError):
        parse(bad)


@pytest.mark.parametrize("text,dsl", [
    ("delivery date must be after order date", "delivery_date > order_date"),
    ("delivery_date is on or after order_date", "delivery_date >= order_date"),
    ("discount must be at most 30%", "discount <= 30%"),
    ("the discount should be no more than 0.3", "discount <= 0.3"),
    ("age must be at least 18", "age >= 18"),
    ("age is between 18 and 90", "age BETWEEN 18 AND 90"),
    ("plan must be one of basic, pro, enterprise", "plan IN ('basic', 'pro', 'enterprise')"),
    ("email is required", "email IS NOT NULL"),
    ("customer id must be unique", "UNIQUE(customer_id)"),
    ("income greater than 0", "income > 0"),
])
def test_plain_language_translation(text, dsl):
    cols = ["delivery_date", "order_date", "discount", "age", "plan", "email", "customer_id", "income", "status"]
    out = plain_to_dsl(text, cols)
    assert out == dsl
    parse(out)


def test_conditional_plain_language():
    cols = ["status", "discount"]
    out = plain_to_dsl("if discount is at least 0.2 then status is one of paid, shipped", cols)
    assert out == "(discount >= 0.2) => (status IN ('paid', 'shipped'))"
    with pytest.raises(RuleSyntaxError):  # a bare word is not a value: fail loudly rather than guess
        plain_to_dsl("if status is cancelled then discount equals 0", cols)


def test_dsl_passthrough_and_untranslatable():
    assert to_dsl("age > 18", ["age"]) == ("age > 18", False)
    assert to_dsl("age must be over 18", ["age"])[1] is True
    with pytest.raises(RuleSyntaxError):
        to_dsl("make the data nicer", ["age"])
    with pytest.raises(RuleSyntaxError, match="no known column"):
        to_dsl("foo must be at least 2", ["age"])


# --------------------------------------------------------------- engine
@pytest.fixture(scope="module")
def shop():
    t = make_shop_full(120, seed=2)
    return t, infer_graph(t)


def test_compile_owner_and_kind(shop):
    _, g = shop
    r = {x.text: x for x in compile_rules(SHOP_FULL_RULES, g)}
    assert [(x.owner, x.kind) for x in r.values()] == [
        ("orders", "derive"), ("order_items", "derive"), ("shipments", "bound"), ("shipments", "bound"),
        ("orders", "bound"), ("orders", "conditional"), ("orders", "conditional"), ("shipments", "conditional")]
    assert compile_rule("total = SUM(order_items.quantity * order_items.unit_price)", g, "orders").owner == "orders"
    assert compile_rule("orders.total > 0", g).dsl == "orders.total > 0"


def test_compile_errors(shop):
    _, g = shop
    for bad in ("nope > 1", "orders.nope > 1", "ghost_col > 1", "customers.name = products.category", "SUM(quantity) > 1",
                "orders.total = SUM(customers.name)"):
        with pytest.raises(RuleSyntaxError):
            compile_rule(bad, g)


def test_real_data_satisfies_all_rules(shop):
    t, g = shop
    ev = evaluate_rules(t, g, compile_rules(SHOP_FULL_RULES, g))
    assert all(e["violations"] == 0 for e in ev) and {e["owner"] for e in ev} >= {"orders", "shipments"}


def test_null_semantics_and_three_valued_logic():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [2.0, 2.0, None], "s": ["x", None, "y"]})
    g = single_table_graph("t", df)
    ev = Evaluator({"t": df}, g)
    chk = lambda text: ev.check(compile_rule(text, g, "t"))  # noqa: E731
    c = chk("a < b")
    assert c.violations == 0 and c.unknown == 2                     # unknown is not a violation
    assert chk("a > b").violations == 1
    assert chk("a IS NOT NULL").violations == 1 and chk("s IN ('x', 'y')").violations == 0
    assert chk("a IS NULL => b IS NULL").violations == 1            # a null, b not null
    assert chk("a > 100 OR b IS NULL").violations == 1              # row 0 false OR false (row 2: b null -> true)
    assert chk("ABS(a - b) <= 1").violations == 0                   # row 0 |1-2|=1 ok; rows 1-2 unknown
    assert chk("UNIQUE(s)").violations == 0


def test_aggregates_and_cross_table_evaluation(shop):
    t, g = shop
    ev = Evaluator(t, g)
    # tamper: an order total that disagrees with its items, and a shipment before its order
    t2 = {k: v.copy() for k, v in t.items()}
    t2["orders"].loc[0, "total"] += 5
    t2["shipments"].loc[0, "shipped_date"] = t2["orders"]["order_date"].min() - pd.Timedelta(days=30)
    ev = Evaluator(t2, g)
    assert ev.check(compile_rule("orders.total = SUM(order_items.quantity * order_items.unit_price)", g)).violations == 1
    assert ev.check(compile_rule("shipments.shipped_date >= orders.order_date", g)).violations == 1
    assert ev.check(compile_rule("COUNT(order_items) >= 1", g)).violations == 0            # owner inferred: orders
    assert ev.check(compile_rule("AVG(order_items.quantity) >= 1", g)).violations == 0
    assert ev.check(compile_rule("orders.status = 'cancelled' => COUNT(shipments) = 0", g)).violations == 0


# ------------------------------------------------------------ enforcement
def broken_shop():
    t = make_shop_full(150, seed=5)
    rng = np.random.default_rng(0)
    o, s, i = t["orders"], t["shipments"], t["order_items"]
    o.loc[o.sample(30, random_state=1).index, "total"] *= 1.1
    o.loc[o.sample(20, random_state=2).index, "discount"] = 0.5
    s.loc[s.sample(25, random_state=3).index, "shipped_date"] -= pd.Timedelta(days=90)
    o.loc[o.sample(25, random_state=4).index, "status"] = "cancelled"
    i.loc[i.sample(40, random_state=5).index, "unit_price"] += 1.0
    return t, infer_graph(t)


def test_enforcement_repairs_every_rule_kind_and_reports_pass_rates():
    t, g = broken_shop()
    rules = compile_rules(SHOP_FULL_RULES, g)
    before = sum(e["violations"] for e in evaluate_rules(t, g, rules))
    assert before > 100
    rep = enforce(t, g, rules, seed=1)
    d = rep.to_dict()
    assert d["violations_after"] == 0 and d["reconciliation_pass_rate"] == 100.0 and d["pass_rate_before"] < 100
    assert d["derived_pass_rate"] == 100.0 and d["derived_pass_rate_before"] < 100
    strategies = {r["strategy"] for r in d["rules"] if r["violations_before"]}
    assert {"derived value assigned", "bound repair"} <= strategies
    assert check_integrity(g, t).ok  # repairs never touch keys
    assert (t["orders"]["discount"] <= 0.3).all()
    assert (t["shipments"]["shipped_date"] >= t["shipments"].merge(t["orders"], on="order_id")["order_date"].to_numpy()).all()


def test_key_columns_are_protected():
    t, g = broken_shop()
    rules = compile_rules(["orders.customer_id = 1"], g)   # would require editing an FK
    rep = enforce(t, g, rules)
    assert rep.rules[0].violations_after > 0 and "key column" in rep.rules[0].unrepairable


def test_date_repair_keeps_nulls_and_orders_dates():
    df = pd.DataFrame({"start": pd.to_datetime(["2024-01-10", "2024-01-05", "2024-02-01", None]),
                       "end": pd.to_datetime(["2024-01-01", "2024-01-20", None, "2024-03-01"])})
    g = single_table_graph("t", df)
    tables = {"t": df.copy()}
    enforce(tables, g, compile_rules(["end > start"], g, "t"), seed=3)
    out = tables["t"]
    assert pd.isna(out.loc[2, "end"]) and pd.isna(out.loc[3, "start"])
    assert (out["end"] > out["start"]).where(out["end"].notna() & out["start"].notna(), True).all()


def test_numeric_bounds_reflect_not_pile_up():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"discount": np.round(rng.uniform(0, 0.5, 4000), 2)})
    g = single_table_graph("t", df)
    tables = {"t": df.copy()}
    enforce(tables, g, compile_rules(["discount <= 0.3"], g, "t"))
    out = tables["t"]["discount"]
    assert out.max() <= 0.3 and (out == 0.3).mean() < 0.08  # reflection spreads mass instead of a spike at the bound


# ---------------------------------------------------- tabular (A4) modes
@pytest.fixture(scope="module")
def gen():
    real = make_customers(3000, seed=8)
    return real, TabularGenerator().fit(real)


RULES = ["age >= 25", "income <= 60000", "plan IN ('basic', 'pro')", "tenure_months <= age"]


@pytest.mark.parametrize("mode", ["repair", "reject", "hybrid"])
def test_all_modes_reach_zero_violations(gen, mode):
    real, g = gen
    res = sample_with_rules(g, 1500, 4, RULES, mode)
    assert len(res.data) == 1500
    graph, rules = compile_rules_for(res.data)
    ev = Evaluator({"data": res.data}, graph)
    assert sum(ev.check(r).violations for r in rules) == 0, mode
    assert sum(res.naive_violations.values()) > 0
    assert list(res.data.columns) == list(real.columns)


def compile_rules_for(df):
    g = single_table_graph("data", df)
    return g, compile_rules(RULES, g, "data")


def test_reject_mode_drops_rows_repair_mode_keeps_them(gen):
    _, g = gen
    assert sample_with_rules(g, 1000, 1, RULES, "reject").rows_dropped > 0
    assert sample_with_rules(g, 1000, 1, RULES, "repair").rows_dropped == 0


def test_naive_vs_ours_comparison(gen):
    real, g = gen
    cmp = compare_naive_vs_enforced(g, 2000, 3, RULES, real=real)
    assert cmp["naive"]["pass_rate"] < 80 and cmp["enforced"]["pass_rate"] == 100.0
    assert all(r["enforced_violations"] == 0 and r["naive_violations"] >= 0 for r in cmp["rules"])
    assert sum(r["naive_violations"] for r in cmp["rules"]) > 300
    # honesty check: enforcement changes the data, but must not wreck it
    assert cmp["enforced"]["mean_ks_vs_real"] < 0.35
    assert cmp["enforced"]["rows_returned"] == 2000


def test_plain_language_rules_drive_generation(gen):
    _, g = gen
    res = sample_with_rules(g, 800, 2, ["age must be at least 30", "plan must be one of basic, pro"], "hybrid")
    assert res.data["age"].min() >= 30 and set(res.data["plan"]) <= {"basic", "pro"}


def test_reproducible(gen):
    _, g = gen
    a, b = sample_with_rules(g, 500, 9, RULES), sample_with_rules(g, 500, 9, RULES)
    pd.testing.assert_frame_equal(a.data, b.data)


# ----------------------------------------------- relational generation (R4)
@pytest.fixture(scope="module")
def full():
    t = make_shop_full(300, seed=1)
    g = infer_graph(t)
    return t, g, RelationalGenerator(g).fit(t)


def test_generation_enforces_cross_table_rules_and_reports(full):
    t, g, gen = full
    naive = gen.generate(seed=3, rules=SHOP_FULL_RULES, enforce=False).rules
    ours = gen.generate(seed=3, rules=SHOP_FULL_RULES).rules
    assert naive["pass_rate_before"] < 80 and naive["violations_after"] == naive["violations_before"]
    assert ours["violations_after"] == 0 and ours["reconciliation_pass_rate"] == 100.0
    assert ours["derived_pass_rate"] == 100.0 and ours["derived_pass_rate_before"] < 50


def test_generated_totals_reconcile_exactly(full):
    _, _, gen = full
    res = gen.generate(seed=6, rules=SHOP_FULL_RULES)
    o, i = res.tables["orders"], res.tables["order_items"]
    calc = (i["quantity"] * i["unit_price"]).groupby(i["order_id"]).sum().reindex(o["order_id"]).fillna(0).to_numpy()
    assert np.abs(o["total"].to_numpy() - calc).max() <= 0.005 + 1e-9
    prod = res.tables["products"].set_index("product_id")["price"]
    assert (i["unit_price"].to_numpy() == prod.loc[i["product_id"]].to_numpy()).all()
    assert res.integrity.ok


def test_status_consistency_and_date_order_after_generation(full):
    _, _, gen = full
    r = gen.generate(seed=7, rules=SHOP_FULL_RULES)
    o, s = r.tables["orders"], r.tables["shipments"]
    n_ship = s.groupby("order_id").size().reindex(o["order_id"]).fillna(0).to_numpy()
    assert (n_ship[o["status"].to_numpy() == "cancelled"] == 0).all()
    assert (n_ship[o["status"].isin(["shipped", "delivered"]).to_numpy()] >= 1).all()
    j = s.merge(o[["order_id", "order_date", "status"]], on="order_id")
    assert (j["shipped_date"] >= j["order_date"]).all()
    assert (j.loc[j["delivered_date"].notna(), "status"] == "delivered").all()
    assert (j["delivered_date"].dropna() >= j.loc[j["delivered_date"].notna(), "shipped_date"]).all()


def test_child_rows_depend_on_parent_attributes(full):
    t, g, gen = full
    real_means = t["orders"].merge(t["customers"], on="customer_id").groupby("segment")["total"].mean()
    res = gen.generate(scale=2, seed=1, rules=SHOP_FULL_RULES)
    syn = res.tables["orders"].merge(res.tables["customers"], on="customer_id").groupby("segment")["total"].mean()
    assert real_means["corp"] > 2 * real_means["retail"]                       # the relation exists in the real data
    assert syn["corp"] > 1.5 * syn["retail"]                                     # and survives in the synthetic data
    indep = RelationalGenerator(g, condition_on_parents=False).fit(t).generate(scale=2, seed=1, rules=SHOP_FULL_RULES)
    flat = indep.tables["orders"].merge(indep.tables["customers"], on="customer_id").groupby("segment")["total"].mean()
    assert (syn["corp"] / syn["retail"]) > (flat["corp"] / flat["retail"])      # conditioning is what preserves it


def test_bad_rule_is_rejected_before_generation(full):
    _, _, gen = full
    with pytest.raises(RuleSyntaxError):
        gen.generate(seed=1, rules=["orders.nonexistent > 1"])
