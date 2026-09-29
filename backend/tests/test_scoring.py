import json
import unicodedata
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from sdp.datasets import make_customers
from sdp.locale import get_locale
from sdp.scoring import (build_trust_report, check_constraints, edge_case_coverage, fidelity_score,
                         locale_validity, privacy_score, trust_report_pdf)
from sdp.tabular import TabularGenerator


@pytest.fixture(scope="module")
def data():
    real = make_customers(3000, seed=4)
    tr, ho = train_test_split(real, test_size=0.3, random_state=0)
    tr, ho = tr.reset_index(drop=True), ho.reset_index(drop=True)
    synth = TabularGenerator().fit(tr).sample(len(tr), seed=1)
    return tr, ho, synth


# ---------------------------------------------------------------- fidelity
def test_fidelity_good_vs_junk(data):
    tr, _, synth = data
    good = fidelity_score(tr, synth)
    rng = np.random.default_rng(0)
    junk = pd.DataFrame({c: rng.permutation(tr[c].to_numpy()[::-1]) if c in ("plan", "region") else
                         (tr[c] * 3 if pd.api.types.is_numeric_dtype(tr[c]) and c != "is_active" else tr[c]) for c in tr.columns})
    bad = fidelity_score(tr, junk)
    assert good["score"] > 90 and bad["score"] < good["score"] - 10
    assert set(good["columns"]) == set(tr.columns)
    assert all(0 <= v["score"] <= 100 for v in good["columns"].values())
    assert good["columns"]["income"]["kind"] == "numeric" and good["columns"]["plan"]["kind"] == "categorical"
    assert sum(c["weight"] for c in good["components"].values()) == pytest.approx(1)


def test_fidelity_aggregates_tstr_and_relational(data):
    tr, _, synth = data
    tstr = {"summary": {"primary_metric": "auc", "mean_gap_pct": 20.0}}
    rel = {"mean_cardinality_match": 0.9}
    f = fidelity_score(tr, synth, tstr=tstr, relational=rel)
    assert set(f["components"]) == {"marginals", "correlations", "utility", "relational"}
    assert f["components"]["utility"]["score"] == 80 and f["components"]["relational"]["score"] == pytest.approx(90)
    assert sum(c["weight"] for c in f["components"].values()) == pytest.approx(1)
    assert fidelity_score(tr, synth, tstr={"summary": {"primary_metric": "auc", "mean_gap_pct": -5.0}})["components"]["utility"]["score"] == 100


# ----------------------------------------------------------------- privacy
def test_privacy_good_synthetic(data):
    tr, ho, synth = data
    p = privacy_score(tr, synth, ho)
    assert p["metrics"]["exact_matches"] == 0 and p["near_duplicates"]["count"] == 0
    assert p["metrics"]["mia_auc"] < 0.6
    assert p["score"] > 70


def test_privacy_detects_leak_and_flags_near_duplicates(data):
    tr, ho, _ = data
    leaked = tr.sample(frac=1, random_state=0).reset_index(drop=True)  # the whole training set, shuffled
    p = privacy_score(tr, leaked, ho)
    assert p["metrics"]["exact_matches"] == len(tr) and p["metrics"]["min_dcr"] == 0
    assert p["metrics"]["mia_auc"] > 0.9
    assert p["near_duplicates"]["count"] == len(tr) and p["near_duplicates"]["flagged"][0]["exact_match"]
    partial = privacy_score(tr, tr.sample(400, random_state=0).reset_index(drop=True), ho)  # partial leak -> weaker attack
    assert 0.55 < partial["metrics"]["mia_auc"] < p["metrics"]["mia_auc"]
    assert p["score"] < 30


def test_near_duplicate_with_tiny_noise_is_flagged_but_not_exact(data):
    tr, ho, _ = data
    near = tr.sample(50, random_state=1).reset_index(drop=True)
    near["income"] = near["income"] * 1.0001
    p = privacy_score(tr, near, ho)
    assert p["metrics"]["exact_matches"] == 0 and p["near_duplicates"]["count"] == 50


def test_privacy_without_holdout_skips_mia(data):
    tr, _, synth = data
    p = privacy_score(tr, synth)
    assert p["metrics"]["mia_auc"] is None and p["metrics"]["baseline_kind"] == "train_half_split"
    assert sum(c["weight"] for c in p["components"].values()) == pytest.approx(1)


# ---------------------------------------------------------- locale validity
def locale_frame(code: str, n: int = 120) -> pd.DataFrame:
    p, rng = get_locale(code), np.random.default_rng(3)
    return pd.DataFrame({
        "name": [p.person_name(rng) for _ in range(n)], "phone": [p.phone(rng) for _ in range(n)],
        "nid": [p.national_id(rng) for _ in range(n)], "dob": [p.format_date(p.birth_date(rng)) for _ in range(n)],
        "balance": [p.format_currency(Decimal(int(rng.integers(1, 10**7))) / 100) for _ in range(n)]})


SPEC = {"phone": "phone", "national_id": "nid", "name": ["name"], "currency": ["balance"], "date": ["dob"]}


@pytest.mark.parametrize("code", ["en-US", "en-GB", "ur-PK", "ar", "hi", "es", "fr", "zh"])
def test_clean_locale_data_is_fully_valid(code):
    r = locale_validity(locale_frame(code), locale=code, columns=SPEC)
    assert r["valid_pct"] == 100.0 and r["failures"] == []
    assert set(r["by_check"]) == {"phone", "national_id", "name_script", "currency", "date", "unicode_nfc"}


def test_locale_failures_are_reported_with_reasons():
    df = locale_frame("hi")
    df.loc[0, "phone"] = "+91 12345 67890"                                  # bad prefix digit
    df.loc[1, "nid"] = df.loc[1, "nid"][:-1] + str((int(df.loc[1, "nid"][-1]) + 1) % 10)  # broken Verhoeff
    df.loc[2, "name"] = "Rahul Sharma"                                        # Latin in a Devanagari locale
    df.loc[3, "balance"] = "12,345.00"                                        # missing symbol
    df.loc[4, "dob"] = "1990/01/31"
    r = locale_validity(df, locale="hi", columns=SPEC)
    assert r["n_invalid_rows"] == 5 and r["valid_pct"] == pytest.approx(100 * (1 - 5 / 120))
    by = {(f["row"], f["check"]): f["reason"] for f in r["failures"]}
    assert "check digit" in by[(1, "national_id")]
    assert {k[1] for k in by} == {"phone", "national_id", "name_script", "currency", "date"}
    assert all(v for v in by.values())


def test_unicode_normalization_check():
    df = pd.DataFrame({"name": ["José García", unicodedata.normalize("NFD", "José García")]})
    r = locale_validity(df, locale="es", columns={"name": ["name"]})
    assert r["by_check"]["unicode_nfc"]["failed"] == 1 and r["failures"][0]["row"] == 1


def test_per_row_locale_column_and_errors():
    a, b = locale_frame("es", 5), locale_frame("fr", 5)
    df = pd.concat([a, b], ignore_index=True)
    df["loc"] = ["es"] * 5 + ["fr"] * 5
    assert locale_validity(df, locale_column="loc", columns=SPEC)["valid_pct"] == 100.0
    df["loc"] = "es"  # French rows checked against Spanish rules must fail
    assert locale_validity(df, locale_column="loc", columns=SPEC)["n_invalid_rows"] == 5
    with pytest.raises(ValueError):
        locale_validity(df)
    with pytest.raises(ValueError, match="not in DataFrame"):
        locale_validity(df, locale="es", columns={"phone": "nope"})


# ------------------------------------------------- constraints and coverage
def test_constraints():
    df = pd.DataFrame({"age": [20, 15, 200, None], "id": [1, 1, 2, 3], "plan": ["a", "b", "z", "a"],
                       "s": pd.to_datetime(["2020-01-01"] * 4), "e": pd.to_datetime(["2020-02-01", "2019-01-01", "2020-03-01", "2020-01-02"]),
                       "code": ["ABC", "AB", "XYZ", "QQQ"]})
    r = check_constraints(df, [{"type": "range", "column": "age", "min": 18, "max": 120}, {"type": "unique", "column": "id"},
                               {"type": "in_set", "column": "plan", "values": ["a", "b"]}, {"type": "not_null", "column": "age"},
                               {"type": "regex", "column": "code", "pattern": "^[A-Z]{3}$"}, {"type": "expr", "expr": "e >= s"}])
    counts = [x["violations"] for x in r["rules"]]
    assert counts == [2, 1, 1, 1, 1, 1]
    assert r["rows_violating"] == 3 and r["pass_pct"] == 25.0
    with pytest.raises(ValueError):
        check_constraints(df, [{"type": "bogus", "column": "age"}])


def test_edge_case_coverage(data):
    tr, _, synth = data
    good = edge_case_coverage(tr, synth)
    assert good["score"] > 60 and good["total"] > 5
    narrow = synth.copy()
    narrow["age"] = narrow["age"].clip(35, 45)
    narrow["plan"] = "basic"
    worse = edge_case_coverage(tr, narrow)
    assert worse["score"] < good["score"]
    assert {"column": "age", "case": "lower tail"} in worse["missing"]
    real = pd.DataFrame({"k": ["a"] * 96 + ["rare"] * 4, "v": [None] + [1.0] * 99})
    r = edge_case_coverage(real, pd.DataFrame({"k": ["a"] * 100, "v": [1.0] * 100}))
    assert {"column": "k", "case": "rare category 'rare'"} in r["missing"] and {"column": "v", "case": "nulls"} in r["missing"]


# ---------------------------------------------------------------- trust card
def test_trust_report_shape_weights_and_verdict(data):
    tr, ho, synth = data
    rep = build_trust_report(tr, synth, real_holdout=ho, rules=[{"type": "range", "column": "age", "min": 0, "max": 120}],
                             integrity={"total_violations": 0, "rows_checked": 3000, "by_kind": {}})
    assert [s["key"] for s in rep["sub_scores"]] == ["fidelity", "privacy", "validity"]
    assert sum(s["weight"] for s in rep["sub_scores"]) == pytest.approx(1)
    assert rep["trust_score"] == pytest.approx(sum(s["score"] * s["weight"] for s in rep["sub_scores"]))
    assert 0 <= rep["trust_score"] <= 100 and rep["label"] in rep["verdict"]
    assert rep["gates"] == []
    json.dumps(rep)
    validity = rep["sub_scores"][2]["components"]
    assert {c["key"] for c in validity} == {"constraints", "integrity", "coverage"}


def test_trust_gates_on_leak_and_integrity(data):
    tr, ho, _ = data
    leaked = tr.sample(300, random_state=0).reset_index(drop=True)
    rep = build_trust_report(tr, leaked, real_holdout=ho, integrity={"total_violations": 7, "rows_checked": 100, "by_kind": {}})
    assert any("exact copies" in g for g in rep["gates"]) and any("integrity" in g for g in rep["gates"])
    assert "Review before sharing" in rep["verdict"] and rep["label"] != "Strong"


def test_trust_with_locale_and_pdf(data):
    tr, ho, synth = data
    ldf = locale_frame("ur-PK", 200)
    rep = build_trust_report(ldf.iloc[:100], ldf.iloc[100:], locale_spec={"locale": "ur-PK", "columns": SPEC}, title="Bank customers (ur-PK)")
    assert any(c["key"] == "locale" and c["score"] == 100 for c in rep["sub_scores"][2]["components"])
    pdf = trust_report_pdf(rep)
    assert pdf.startswith(b"%PDF") and len(pdf) > 1500
    assert trust_report_pdf(build_trust_report(tr, synth, real_holdout=ho)).startswith(b"%PDF")
