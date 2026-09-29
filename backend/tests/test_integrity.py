import pandas as pd

from sdp.relational import check_integrity, infer_graph
from sdp.relational.schema import RelationshipGraph


def make():
    tables = {
        "parents": pd.DataFrame({"pid": [1, 2, 3], "x": [1.0, 2.0, 3.0]}),
        "kids": pd.DataFrame({"kid_id": [10, 11, 12, 13], "pid": [1, 2, 3, 3]}),
    }
    g = infer_graph(tables)
    return g, tables


def test_clean_data_no_violations():
    g, t = make()
    r = check_integrity(g, t)
    assert r.ok and r.total == 0 and r.by_kind()["orphan_fk"] == 0


def test_detects_every_violation_type():
    g, t = make()
    g.fk("kids(pid)->parents").nullable = False
    t["kids"] = pd.DataFrame({"kid_id": [10, 10, 12, None, 14], "pid": [1, 99, None, 3, 3]})
    t["parents"] = pd.DataFrame({"pid": [1, 2, 3, 3], "x": [1.0, 2.0, 3.0, 4.0]})
    r = check_integrity(g, t)
    kinds = r.by_kind()
    assert kinds["orphan_fk"] == 1          # pid=99
    assert kinds["null_fk"] == 1            # pid=None on a NOT NULL FK
    assert kinds["duplicate_pk"] == 2       # parents.pid=3 twice, kids.kid_id=10 twice
    assert kinds["null_pk"] == 1
    assert not r.ok and r.violations[0].examples is not None


def test_nullable_fk_null_is_fine_but_partial_composite_null_is_not():
    g = RelationshipGraph.model_validate({
        "tables": [{"name": "p", "columns": [{"name": "a", "dtype": "int"}, {"name": "b", "dtype": "int"}], "primary_key": ["a", "b"]},
                   {"name": "c", "columns": [{"name": "id", "dtype": "int"}, {"name": "a", "dtype": "int"}, {"name": "b", "dtype": "int"}], "primary_key": ["id"]}],
        "foreign_keys": [{"child_table": "c", "child_columns": ["a", "b"], "parent_table": "p", "parent_columns": ["a", "b"], "nullable": True}]})
    t = {"p": pd.DataFrame({"a": [1, 1], "b": [1, 2]}),
         "c": pd.DataFrame({"id": [1, 2, 3, 4], "a": [1, None, 1, 2], "b": [1, None, None, 1]})}
    r = check_integrity(g, t)
    assert r.by_kind()["partial_null_fk"] == 1 and r.by_kind()["orphan_fk"] == 1 and r.by_kind()["null_fk"] == 0
