import numpy as np
import pandas as pd
import pytest

from sdp.datasets import make_customers
from sdp.edgecases import ALL_PACKS, TAG, EdgeCaseError, inject_edge_cases, visible
from sdp.tabular import GenConfig, TabularGenerator


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(0)
    d = make_customers(1000, seed=3)
    d["balance"] = np.round(rng.gamma(3, 500, len(d)), 2)
    d["name"] = [f"Person {i:04d}" for i in range(len(d))]
    return d


def tags(res, pack):
    return res.data[res.data[TAG].fillna("").str.contains(f"{pack}:")]


def test_rate_tagging_and_hidden_column(df):
    res = inject_edge_cases(df, {"typos": 0.1}, seed=1)
    assert len(res.data) == len(df) and TAG in res.data.columns
    assert res.data[TAG].notna().sum() == 100 == res.report["rows_tagged"] == res.report["packs"]["typos"]["rows_affected"]
    assert TAG not in visible(res.data).columns and list(visible(res.data).columns) == list(df.columns)
    assert res.data[TAG].dropna().str.startswith("typos:").all()
    untouched = res.data[res.data[TAG].isna()].drop(columns=TAG)
    pd.testing.assert_frame_equal(untouched, df.loc[untouched.index], check_dtype=False)  # only tagged rows changed
    assert inject_edge_cases(df, {"typos": 0.0}).report["rows_tagged"] == 0


def test_deterministic(df):
    a = inject_edge_cases(df, {p: 0.05 for p in ALL_PACKS}, seed=7)
    b = inject_edge_cases(df, {p: 0.05 for p in ALL_PACKS}, seed=7)
    pd.testing.assert_frame_equal(a.data, b.data)
    assert not a.data.equals(inject_edge_cases(df, {p: 0.05 for p in ALL_PACKS}, seed=8).data)


def test_boundary_values(df):
    res = inject_edge_cases(df, {"boundary_values": 0.2}, seed=2)
    d = res.data
    assert (d["age"] == np.iinfo(np.int64).max).any() and (d["age"] == 0).any() and (d["age"] == np.iinfo(np.int64).min).any()
    assert (d["region"] == "").any() and (d["region"] == " ").any() and d["region"].str.len().max() == 1024
    assert (d["signup_date"] == pd.Timestamp("1970-01-01")).any() and (d["signup_date"] == pd.Timestamp("2262-04-11")).any()
    assert res.report["packs"]["boundary_values"]["coverage_pct"] == 100.0
    assert d["age"].dtype == df["age"].dtype


def test_typos_change_strings_by_one_edit(df):
    res = inject_edge_cases(df, {"typos": 0.3}, seed=3)
    hit = tags(res, "typos")
    changed = 0
    for i in hit.index:
        for c in ("name", "region", "plan"):
            if df.at[i, c] != res.data.at[i, c]:
                changed += 1
                a, b = df.at[i, c], res.data.at[i, c]
                assert abs(len(a) - len(b)) <= 1
    assert changed >= len(hit) * 0.9
    assert set(res.report["packs"]["typos"]["cases_exercised"]) == {"swap_adjacent", "drop_char", "double_char", "keyboard_neighbor", "case_flip", "trailing_space"}


def test_negative_balances_only_on_money_like_columns(df):
    res = inject_edge_cases(df, {"negative_balances": 0.2}, seed=4)
    assert set(res.report["packs"]["negative_balances"]["columns"]) <= {"income", "balance"}
    assert (res.data["balance"] < 0).sum() + (res.data["income"] < 0).sum() >= res.report["rows_tagged"] * 0.95
    assert (res.data["age"] >= 0).all() and (res.data["tenure_months"] >= 0).all()
    assert res.data["income"].dtype == df["income"].dtype


def test_leap_year_and_timezone_packs(df):
    d = inject_edge_cases(df, {"leap_year_dates": 0.3}, seed=5).data
    assert ((d["signup_date"].dt.month == 2) & (d["signup_date"].dt.day == 29) & d["signup_date"].dt.year.isin([2000, 2020, 2024])).any()
    assert (d["signup_date"].dt.year == 1900).any() and ((d["signup_date"].dt.month == 3) & (d["signup_date"].dt.day == 1) & (d["signup_date"].dt.year == 2023)).any()
    t = inject_edge_cases(df, {"timezone_shifts": 0.3}, seed=6)
    d2 = t.data
    assert (d2["signup_date"] == pd.Timestamp("2024-03-10 02:30:00")).any() and (d2["signup_date"] == pd.Timestamp("2024-11-03 01:30:00")).any()
    shifted = (d2["signup_date"] - df["signup_date"]).abs()
    assert (shifted.between(pd.Timedelta(hours=1), pd.Timedelta(hours=15))).any()
    assert t.report["packs"]["timezone_shifts"]["coverage_pct"] == 100.0


def test_duplicates_add_rows_and_tag_both_copies(df):
    res = inject_edge_cases(df, {"duplicates": 0.1}, seed=8)
    assert len(res.data) == len(df) + 100 and res.report["packs"]["duplicates"]["rows_added"] == 100
    assert res.data[TAG].notna().sum() == 200                       # originals and copies are both tagged
    exact = res.data.drop(columns=TAG).duplicated(keep=False).sum()
    assert exact >= 2 * len([1 for t in res.data[TAG].dropna() if t == "duplicates:exact_duplicate"]) // 2
    assert set(res.report["packs"]["duplicates"]["cases_exercised"]) == {"exact_duplicate", "near_duplicate_whitespace", "near_duplicate_case"}


def test_coverage_reporting_and_inapplicable_packs():
    only_ints = pd.DataFrame({"a": np.arange(200), "b": np.arange(200) * 2})
    res = inject_edge_cases(only_ints, {"typos": 0.5, "leap_year_dates": 0.5, "boundary_values": 0.5}, seed=1)
    assert set(res.report["not_applicable"]) == {"typos", "leap_year_dates"}
    assert res.report["packs"]["typos"]["coverage_pct"] is None and res.report["packs"]["boundary_values"]["coverage_pct"] == 100.0
    assert res.report["coverage_pct"] == 100.0                        # measured over applicable cases only
    tiny = inject_edge_cases(pd.DataFrame({"s": ["abc"] * 5}), {"typos": 0.2}, seed=1)   # 1 row, 6 cases
    assert tiny.report["packs"]["typos"]["coverage_pct"] == pytest.approx(100 / 6)
    assert tiny.report["coverage_pct"] < 20


def test_protected_columns_and_validation(df):
    res = inject_edge_cases(df, {"boundary_values": 0.5, "typos": 0.5}, seed=1, protect=["age", "name", "region", "plan"])
    assert (res.data["age"] == df["age"]).all() and (res.data["name"] == df["name"]).all()
    with pytest.raises(EdgeCaseError):
        inject_edge_cases(df, {"nope": 0.1})
    with pytest.raises(EdgeCaseError):
        inject_edge_cases(df, {"typos": 2})


def test_generator_integration_keeps_sampling_stream_and_reports_coverage():
    real = make_customers(1500, seed=5)
    gen = TabularGenerator().fit(real)
    base = gen.generate(GenConfig(rows=400, seed=9, null_rate={"income": 0.1}))
    withedge = gen.generate(GenConfig(rows=400, seed=9, null_rate={"income": 0.1}, edge_cases={"typos": 0.1, "boundary_values": 0.1}))
    assert TAG not in base.data.columns and base.edge is None
    assert TAG in withedge.data.columns and withedge.edge["rows_tagged"] > 0 and 0 < withedge.edge["coverage_pct"] <= 100
    pd.testing.assert_frame_equal(base.data[base.data.columns], visible(withedge.data).where(withedge.data[TAG].isna())
                                  .fillna(base.data), check_dtype=False)            # untouched rows identical to the normal run
    again = gen.generate(GenConfig(rows=400, seed=9, null_rate={"income": 0.1}, edge_cases={"typos": 0.1, "boundary_values": 0.1}))
    pd.testing.assert_frame_equal(withedge.data, again.data)
    with pytest.raises(ValueError):
        GenConfig(edge_cases={"nope": 0.1})
