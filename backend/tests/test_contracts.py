import pandas as pd
import pytest

from sdp.contracts import PackError, generate_pack, list_packs, validate_pack, validate_suite
from sdp.contracts.expectations import report_html
from sdp.contracts.healthcare import make_healthcare
from sdp.contracts.packs import export_suite, get_pack


@pytest.mark.parametrize("name", ["banking", "ecommerce", "healthcare"])
def test_every_starter_pack_generates_data_that_meets_its_own_contract(name):
    t = generate_pack(name, 200, seed=4)
    rep = validate_pack(name, t)
    failed = [(r["expectation_config"]["expectation_type"], r["expectation_config"]["table"], r["result"]) for r in rep["results"] if not r["success"]]
    assert rep["success"], failed
    assert rep["statistics"]["evaluated_expectations"] == len(get_pack(name)["expectations"]) >= 15
    assert set(rep["statistics"]["by_table"]) == set(get_pack(name)["tables"])
    assert set(t) == set(get_pack(name)["tables"])


def test_packs_are_deterministic_and_listed():
    a, b = generate_pack("healthcare", 100, seed=2), generate_pack("healthcare", 100, seed=2)
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k])
    assert not a["patients"].equals(generate_pack("healthcare", 100, seed=3)["patients"])
    assert {p["name"] for p in list_packs()} == {"banking", "ecommerce", "healthcare"}
    with pytest.raises(PackError):
        get_pack("astrology")
    with pytest.raises(PackError):
        generate_pack("banking", 3)


def test_contract_catches_broken_data_with_counts_and_examples():
    t = generate_pack("healthcare", 200, seed=1)
    t["claims"].loc[0, "approved_amount"] = t["claims"].loc[0, "billed_amount"] + 500          # pays more than billed
    t["encounters"].loc[1, "patient_id"] = 999999                                                # orphan
    t["patients"].loc[2, "sex"] = "X"
    t["patients"].loc[3, "patient_id"] = t["patients"].loc[4, "patient_id"]                      # duplicate key
    t["encounters"].loc[5, "charge"] = -5
    rep = validate_pack("healthcare", t)
    assert not rep["success"]

    def key(r):
        c = r["expectation_config"]
        return c["expectation_type"], c["table"], c["kwargs"].get("column", c["kwargs"].get("column_A"))

    bad = {key(r): r for r in rep["results"] if not r["success"]}
    assert bad[("expect_column_pair_values_a_to_be_greater_than_b", "claims", "billed_amount")]["result"]["unexpected_count"] == 1
    assert bad[("expect_column_values_to_be_in_other_table", "encounters", "patient_id")]["result"]["partial_unexpected_list"][0] == 999999   # (patient 3 also lost its parent row)
    assert bad[("expect_column_values_to_be_in_set", "patients", "sex")]["result"]["partial_unexpected_list"] == ["X"]
    assert bad[("expect_column_values_to_be_unique", "patients", "patient_id")]["result"]["unexpected_count"] == 2
    assert bad[("expect_column_values_to_be_between", "encounters", "charge")]["result"]["unexpected_percent"] > 0
    assert rep["statistics"]["unsuccessful_expectations"] == len(bad)
    assert "FAILED" in report_html(rep)


def test_mostly_tolerates_a_fraction_and_broken_expectations_become_errors():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, -1]})
    s = [{"expectation_type": "expect_column_values_to_be_between", "table": "t", "kwargs": {"column": "a", "min_value": 0, "mostly": 0.9}},
         {"expectation_type": "expect_column_values_to_be_between", "table": "t", "kwargs": {"column": "a", "min_value": 0, "mostly": 0.95}},
         {"expectation_type": "expect_column_values_to_be_between", "table": "t", "kwargs": {"column": "nope", "min_value": 0}},
         {"expectation_type": "expect_something_else", "table": "t", "kwargs": {}},
         {"expectation_type": "expect_column_values_to_be_unique", "table": "ghost", "kwargs": {"column": "a"}},
         {"expectation_type": "expect_column_values_to_match_regex", "table": "t", "kwargs": {"column": "a", "regex": "("}}]
    r = validate_suite({"t": df}, s)
    assert [x["success"] for x in r["results"]] == [True, False, False, False, False, False]
    msgs = [x["exception_info"]["exception_message"] for x in r["results"][2:]]
    assert "does not exist" in msgs[0] and "unknown expectation" in msgs[1] and "not found" in msgs[2] and "regex" in msgs[3]


def test_type_null_and_row_count_expectations():
    df = pd.DataFrame({"i": [1, 2], "f": [1.5, None], "s": ["a", None], "d": pd.to_datetime(["2024-01-01", None])})

    def run(t, **kw):
        return validate_suite({"t": df}, [{"expectation_type": t, "table": "t", "kwargs": kw}])["results"][0]["success"]

    assert run("expect_column_values_to_be_of_type", column="i", type_="int") and not run("expect_column_values_to_be_of_type", column="f", type_="int")
    assert run("expect_column_values_to_be_of_type", column="s", type_="string") and run("expect_column_values_to_be_of_type", column="d", type_="datetime")
    assert not run("expect_column_values_to_not_be_null", column="f") and run("expect_column_values_to_not_be_null", column="f", mostly=0.5)
    assert run("expect_table_row_count_to_be_between", min_value=2, max_value=2) and not run("expect_table_row_count_to_be_between", min_value=3)
    assert run("expect_table_columns_to_match_set", column_set=["i"], exact_match=False) and not run("expect_table_columns_to_match_set", column_set=["i"])
    assert run("expect_column_to_exist", column="i") and not run("expect_column_to_exist", column="z")


def test_great_expectations_export_shape():
    ge = export_suite("healthcare")
    assert set(ge) == {"patients", "encounters", "claims"}
    s = ge["claims"]
    assert s["expectation_suite_name"] == "healthcare.claims" and all({"expectation_type", "kwargs", "meta"} <= set(e) for e in s["expectations"])
    assert any(e["meta"].get("custom") for e in ge["encounters"]["expectations"])          # in_other_table is not a GE built-in


def test_healthcare_generator_is_internally_consistent():
    t = make_healthcare(400, seed=5, locale="fr")
    e, p = t["encounters"], t["patients"]
    assert e["patient_id"].isin(p["patient_id"]).all() and t["claims"]["encounter_id"].isin(e["encounter_id"]).all()
    dob = p.set_index("patient_id")["date_of_birth"].reindex(e["patient_id"]).to_numpy()
    assert (e["encounter_date"].to_numpy() > dob).all()
    assert (t["claims"]["status"].eq("pending") == t["claims"]["approved_amount"].isna()).all()
    assert (e.loc[e.encounter_type != "inpatient", "length_of_stay_days"] == 0).all()
