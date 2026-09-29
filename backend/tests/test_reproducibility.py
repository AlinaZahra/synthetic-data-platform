import logging

import numpy as np
import pandas as pd
import pytest

from sdp.datasets import make_customers
from sdp.tabular import GenConfig, TabularGenerator

CFG = dict(rows=800, seed=42, null_rate={"income": 0.1, "region": 0.05},
           outlier_rate={"age": 0.02, "income": 0.03}, outlier_method="iqr")


@pytest.fixture(scope="module")
def gen():
    return TabularGenerator().fit(make_customers(1500, seed=2))


def test_same_seed_same_config_identical(gen):
    a = gen.generate(GenConfig(**CFG))
    b = gen.generate(GenConfig(**CFG))
    pd.testing.assert_frame_equal(a.data, b.data)
    assert a.log_dicts() == b.log_dicts()


def test_identical_across_freshly_fitted_generators():
    real = make_customers(1500, seed=2)
    a = TabularGenerator().fit(real).generate(GenConfig(**CFG)).data
    b = TabularGenerator().fit(real).generate(GenConfig(**CFG)).data
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_or_config_differs(gen):
    base = gen.generate(GenConfig(**CFG)).data
    assert not base.equals(gen.generate(GenConfig(**{**CFG, "seed": 43})).data)
    assert not base.equals(gen.generate(GenConfig(**{**CFG, "outlier_method": "zscore"})).data)


def test_none_seed_is_recorded_and_replayable(gen):
    r = gen.generate(GenConfig(**{**CFG, "seed": None}))
    replay = gen.generate(GenConfig(**{**CFG, "seed": r.seed}))
    pd.testing.assert_frame_equal(r.data, replay.data)


def test_null_counts_exact_and_logged(gen):
    r = gen.generate(GenConfig(**CFG))
    assert r.data["income"].isna().sum() == 80
    assert r.data["region"].isna().sum() == 40
    assert r.data["age"].isna().sum() == 0
    nulls = {x.column: x for x in r.log if x.action == "null"}
    assert nulls["income"].count == 80 and len(nulls["income"].row_indices) == 50


def test_nulls_do_not_change_clean_columns(gen):
    clean = gen.generate(GenConfig(rows=800, seed=42)).data
    dirty = gen.generate(GenConfig(**CFG)).data
    pd.testing.assert_series_equal(clean["tenure_months"], dirty["tenure_months"])


@pytest.mark.parametrize("method", ["iqr", "zscore", "scale"])
def test_outliers_are_extreme_and_logged(gen, method):
    cfg = GenConfig(rows=2000, seed=5, outlier_rate={"income": 0.02}, outlier_method=method)
    r = gen.generate(cfg)
    clean = gen.sample(2000, seed=5)["income"]
    rec = next(x for x in r.log if x.action == "outlier")
    assert rec.count == 40 and rec.method == method
    changed = np.flatnonzero(r.data["income"].to_numpy() != clean.to_numpy())
    assert len(changed) == 40
    if method == "iqr":
        q1, q3 = clean.quantile([0.25, 0.75])
        fence = 1.5 * (q3 - q1)
        vals = r.data["income"].iloc[changed]
        assert ((vals > q3 + fence) | (vals < q1 - fence)).all()


def test_int_column_stays_int_after_outliers_and_nullable_after_nulls(gen):
    r = gen.generate(GenConfig(rows=500, seed=1, outlier_rate={"age": 0.05}, null_rate={"tenure_months": 0.1}))
    assert pd.api.types.is_integer_dtype(r.data["age"])
    assert str(r.data["tenure_months"].dtype) == "Int64"


def test_logging_emitted(gen, caplog):
    with caplog.at_level(logging.INFO, logger="sdp.tabular"):
        gen.generate(GenConfig(**CFG))
    assert any("injected" in m for m in caplog.messages)


def test_bad_column_and_non_numeric_outlier(gen):
    with pytest.raises(ValueError, match="unknown columns"):
        gen.generate(GenConfig(rows=10, null_rate={"nope": 0.1}))
    with pytest.raises(ValueError, match="numeric"):
        gen.generate(GenConfig(rows=100, outlier_rate={"plan": 0.1}))
