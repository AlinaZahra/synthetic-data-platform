import pandas as pd
import pytest

from sdp.datasets import make_shop
from sdp.relational.ddl import parse_ddl
from sdp.relational.inference import annotate_cardinalities, infer_graph
from sdp.relational.schema import RelationshipGraph


@pytest.fixture(scope="module")
def shop():
    return make_shop(200, seed=1)


def edges(g):
    return {(f.child_table, tuple(f.child_columns), f.parent_table, tuple(f.parent_columns)) for f in g.foreign_keys}


def test_customers_orders_items_example():
    s = make_shop(200, seed=1)
    three = {"customers": s["customers"], "orders": s["orders"],
             "items": s["order_items"].drop(columns="product_id").rename(columns={"item_id": "id"})}
    g = infer_graph(three)
    assert {t.name: t.primary_key for t in g.tables} == {
        "customers": ["customer_id"], "orders": ["order_id"], "items": ["id"]}
    assert edges(g) == {("orders", ("customer_id",), "customers", ("customer_id",)),
                        ("items", ("order_id",), "orders", ("order_id",))}
    assert {f.key: f.cardinality for f in g.foreign_keys} == {
        "orders(customer_id)->customers": "1:N", "items(order_id)->orders": "1:N"}


def test_four_table_shop_finds_junction_and_no_spurious_fks(shop):
    g = infer_graph(shop)
    assert edges(g) == {
        ("orders", ("customer_id",), "customers", ("customer_id",)),
        ("order_items", ("order_id",), "orders", ("order_id",)),
        ("order_items", ("product_id",), "products", ("product_id",)),
    }  # quantity (small ints) must NOT be matched to any key
    assert g.many_to_many() == [{"junction": "order_items", "left": "orders", "right": "products"}]
    assert g.topological_order().index("customers") < g.topological_order().index("orders") < g.topological_order().index("order_items")
    assert g.validate_graph() == []


def test_stats_and_cardinality(shop):
    g = infer_graph(shop)
    fk = g.fk("orders(customer_id)->customers")
    assert fk.stats["orphan_rate"] == 0
    assert fk.stats["mean_children"] == pytest.approx(len(shop["orders"]) / len(shop["customers"]))
    assert fk.stats["zero_child_fraction"] > 0  # some customers never order
    assert fk.confidence == 1.0


def test_one_to_one_detected():
    users = pd.DataFrame({"user_id": [1, 2, 3], "n": [1, 2, 3]})
    prof = pd.DataFrame({"pid": [10, 11, 12], "user_id": [1, 2, 3], "bio": list("abc")})
    g = infer_graph({"users": users, "profiles": prof})
    assert g.fk("profiles(user_id)->users").cardinality == "1:1"


def test_orphans_reduce_confidence_but_small_rate_tolerated():
    parent = pd.DataFrame({"customer_id": range(1, 201)})
    child = pd.DataFrame({"oid": range(500), "customer_id": [(i % 200) + 1 for i in range(499)] + [9999]})
    g = infer_graph({"customers": parent, "orders": child})
    fk = g.fk("orders(customer_id)->customers")
    assert 0 < fk.stats["orphan_rate"] < 0.01


def test_no_fk_when_values_dont_overlap():
    a = pd.DataFrame({"a_id": range(100)})
    b = pd.DataFrame({"b_id": range(100), "a_id": range(1000, 1100)})
    assert infer_graph({"a": a, "b": b}).foreign_keys == []


def test_composite_pk_and_fk():
    parent = pd.DataFrame({"region_id": [1, 1, 2, 2], "store_id": [1, 2, 1, 2], "name": list("abcd")})
    child = pd.DataFrame({"sale_id": range(8), "region_id": [1, 1, 2, 2] * 2, "store_id": [1, 2, 1, 2] * 2})
    g = infer_graph({"stores": parent, "sales": child})
    assert g.table("stores").primary_key == ["region_id", "store_id"]
    fk = g.fk("sales(region_id,store_id)->stores")
    assert fk.parent_columns == ["region_id", "store_id"]


def test_self_reference():
    emp = pd.DataFrame({"employee_id": range(1, 9), "manager_id": [None, 1, 1, 2, 2, 3, 3, 4],
                        "dept_id": [1, 2, 3, 1, 2, 3, 1, 2]})
    g = infer_graph({"employees": emp})
    assert edges(g) == {("employees", ("manager_id",), "employees", ("employee_id",))}
    assert g.fks_of("employees")[0].nullable
    assert g.topological_order() == ["employees"]


def test_json_roundtrip_and_editing(shop):
    g = infer_graph(shop)
    g2 = RelationshipGraph.from_json(g.to_json())
    assert g2.to_json() == g.to_json()
    g2.remove_foreign_key("order_items(product_id)->products")
    assert len(g2.foreign_keys) == 2
    fk = g2.add_foreign_key("order_items", ["product_id"], "products")
    assert fk.source == "user" and g2.validate_graph() == []
    g2.update_foreign_key(fk.key, nullable=False, cardinality="1:N")
    assert g2.fk(fk.key).nullable is False
    with pytest.raises(ValueError):
        g2.add_foreign_key("order_items", ["product_id"], "products")
    g2.set_primary_key("products", ["nope"])
    assert any("does not exist" in e for e in g2.validate_graph())


def test_cycle_detection():
    g = RelationshipGraph.model_validate({
        "tables": [{"name": "a", "columns": [{"name": "id", "dtype": "int"}, {"name": "b_id", "dtype": "int"}], "primary_key": ["id"]},
                   {"name": "b", "columns": [{"name": "id", "dtype": "int"}, {"name": "a_id", "dtype": "int"}], "primary_key": ["id"]}],
        "foreign_keys": [{"child_table": "a", "child_columns": ["b_id"], "parent_table": "b", "parent_columns": ["id"]},
                         {"child_table": "b", "child_columns": ["a_id"], "parent_table": "a", "parent_columns": ["id"]}]})
    with pytest.raises(ValueError, match="cyclic"):
        g.topological_order()


def test_add_many_to_many_helper():
    g = RelationshipGraph.model_validate({"tables": [
        {"name": "students", "columns": [{"name": "id", "dtype": "int"}], "primary_key": ["id"]},
        {"name": "courses", "columns": [{"name": "id", "dtype": "int"}], "primary_key": ["id"]}]})
    j = g.add_many_to_many("students", "courses", "enrollments")
    assert j.primary_key == ["student_id", "course_id"]
    assert g.many_to_many() == [{"junction": "enrollments", "left": "courses", "right": "students"}]
    assert g.validate_graph() == []


DDL = """
-- shop schema
CREATE TABLE customers (
  customer_id INTEGER PRIMARY KEY,
  email VARCHAR(100) UNIQUE NOT NULL,
  referred_by INTEGER REFERENCES customers(customer_id)
);
CREATE TABLE "orders" (
  order_id SERIAL,
  customer_id INT NOT NULL,
  placed_at TIMESTAMP,
  PRIMARY KEY (order_id),
  CONSTRAINT fk_c FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);
CREATE TABLE items (
  order_id INT NOT NULL,
  line_no INT NOT NULL,
  sku VARCHAR(20),
  price DECIMAL(10,2),
  PRIMARY KEY (order_id, line_no)
);
ALTER TABLE items ADD CONSTRAINT fk_o FOREIGN KEY (order_id) REFERENCES orders(order_id);
"""


def test_ddl_parsing():
    g = parse_ddl(DDL)
    assert [t.name for t in g.tables] == ["customers", "orders", "items"]
    assert g.table("items").primary_key == ["order_id", "line_no"]
    assert g.table("orders").column("customer_id").nullable is False
    assert g.table("customers").column("email").dtype == "str"
    assert g.table("items").column("price").dtype == "float"
    assert {f.key for f in g.foreign_keys} == {
        "customers(referred_by)->customers", "orders(customer_id)->customers", "items(order_id)->orders"}
    assert g.fk("orders(customer_id)->customers").nullable is False
    assert g.fk("customers(referred_by)->customers").is_self_reference
    assert g.validate_graph() == []
    assert g.topological_order() == ["customers", "orders", "items"]


def test_ddl_then_annotate_with_data(shop):
    ddl = """CREATE TABLE customers (customer_id INT PRIMARY KEY, name TEXT, country TEXT, segment TEXT);
             CREATE TABLE orders (order_id INT PRIMARY KEY, customer_id INT REFERENCES customers(customer_id),
                                  order_date DATE, status TEXT);"""
    g = parse_ddl(ddl)
    annotate_cardinalities(g, shop)
    assert g.fk("orders(customer_id)->customers").stats["mean_children"] > 1
