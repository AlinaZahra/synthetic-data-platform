import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from sdp.datasets import make_customers
from sdp.evaluation import detect_task, evaluate_tstr
from sdp.tabular import TabularGenerator


@pytest.fixture(scope="module")
def split():
    real = make_customers(3000, seed=3)
    tr, te = train_test_split(real, test_size=0.3, random_state=0)
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
    synth = TabularGenerator().fit(tr).sample(len(tr), seed=0)
    return tr, te, synth


def test_detect_task():
    assert detect_task(pd.Series([0, 1, 0, 1])) == "classification"
    assert detect_task(pd.Series(["a", "b"])) == "classification"
    assert detect_task(pd.Series([True, False])) == "classification"
    assert detect_task(pd.Series(np.linspace(0, 1, 100))) == "regression"
    assert detect_task(pd.Series(np.arange(500))) == "regression"


def test_classification_report_shape_and_json(split):
    tr, te, synth = split
    rep = evaluate_tstr(tr, te, synth, "churned")
    assert rep["task"] == "classification"
    assert set(rep["models"]) == {"RandomForest", "LogisticRegression"}
    for m in rep["models"].values():
        assert set(m["baseline_real"]) == {"accuracy", "f1_macro", "auc"}
        assert set(m["gap_pct"]) == {"accuracy", "f1_macro", "auc"}
    json.dumps(rep)  # must be serialisable
    assert 0.5 < rep["models"]["LogisticRegression"]["baseline_real"]["auc"] <= 1


def test_synthetic_utility_is_close_to_baseline(split):
    tr, te, synth = split
    rep = evaluate_tstr(tr, te, synth, "churned")
    assert rep["summary"]["mean_gap_pct"] < 15  # good synthetic data loses little utility
    lr = rep["models"]["LogisticRegression"]
    assert lr["synthetic"]["auc"] > 0.55


def test_gap_math(split):
    tr, te, synth = split
    rep = evaluate_tstr(tr, te, synth, "churned")
    m = rep["models"]["RandomForest"]
    b, s = m["baseline_real"]["accuracy"], m["synthetic"]["accuracy"]
    assert m["gap_pct"]["accuracy"] == pytest.approx(100 * (b - s) / b)


def test_random_synthetic_is_much_worse(split):
    tr, te, _ = split
    rng = np.random.default_rng(0)
    junk = tr.copy()
    junk["churned"] = rng.permutation(junk["churned"].to_numpy())
    good = evaluate_tstr(tr, te, TabularGenerator().fit(tr).sample(len(tr), 0), "churned")
    bad = evaluate_tstr(tr, te, junk, "churned")
    assert bad["models"]["LogisticRegression"]["synthetic"]["auc"] < good["models"]["LogisticRegression"]["synthetic"]["auc"]


def test_regression_report(split):
    tr, te, synth = split
    rep = evaluate_tstr(tr, te, synth, "income")
    assert rep["task"] == "regression"
    assert set(rep["models"]) == {"RandomForest", "Ridge"}
    rf = rep["models"]["RandomForest"]
    assert set(rf["baseline_real"]) == {"r2", "rmse"}
    b, s = rf["baseline_real"]["rmse"], rf["synthetic"]["rmse"]
    assert rf["gap_pct"]["rmse"] == pytest.approx(100 * (s - b) / b)  # lower-is-better orientation
    json.dumps(rep)


def test_deterministic(split):
    tr, te, synth = split
    assert evaluate_tstr(tr, te, synth, "churned") == evaluate_tstr(tr, te, synth, "churned")


def test_single_class_synthetic_reports_error_not_crash(split):
    tr, te, synth = split
    one = synth.assign(churned=0)
    rep = evaluate_tstr(tr, te, one, "churned")
    assert "error" in rep["models"]["RandomForest"]["synthetic"]


def test_missing_target_raises(split):
    tr, te, synth = split
    with pytest.raises(ValueError):
        evaluate_tstr(tr, te, synth.drop(columns="churned"), "churned")
