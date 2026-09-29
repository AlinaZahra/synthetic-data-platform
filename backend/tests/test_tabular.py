import numpy as np
import pandas as pd
import pytest
from scipy import stats

from sdp.datasets import make_customers
from sdp.tabular import GenConfig, TabularGenerator
from sdp.tabular.quality import correlation_gap


@pytest.fixture(scope="module")
def real() -> pd.DataFrame:
    return make_customers(3000, seed=1)


@pytest.fixture(scope="module")
def synth(real) -> pd.DataFrame:
    return TabularGenerator().fit(real).sample(3000, seed=7)


def test_dtypes_and_columns_preserved(real, synth):
    assert list(synth.columns) == list(real.columns)
    for c in real.columns:
        assert synth[c].dtype == real[c].dtype, c


@pytest.mark.parametrize("col", ["age", "income", "tenure_months", "signup_date"])
def test_numeric_marginals_ks(real, synth, col):
    a, b = real[col], synth[col]
    if col == "signup_date":
        a, b = a.astype("int64"), b.astype("int64")
    res = stats.ks_2samp(a, b)
    assert res.pvalue > 0.01, (col, res)


@pytest.mark.parametrize("col", ["plan", "region", "churned", "is_active"])
def test_categorical_frequencies_chi_square(real, synth, col):
    freq = real[col].value_counts(normalize=True)
    obs = synth[col].value_counts().reindex(freq.index, fill_value=0)
    _, p = stats.chisquare(obs.to_numpy(), freq.to_numpy() * obs.sum())
    assert p > 0.01, (col, p)


def test_correlations_preserved(real, synth):
    assert correlation_gap(real, synth) < 0.06
    r = real[["age", "income"]].corr("spearman").iloc[0, 1]
    s = synth[["age", "income"]].corr("spearman").iloc[0, 1]
    assert abs(r - s) < 0.06 and r > 0.1


def test_values_stay_inside_observed_range(real, synth):
    assert synth["age"].between(real["age"].min(), real["age"].max()).all()
    assert set(synth["plan"]) <= set(real["plan"])


def test_seed_controls_sample(real):
    g = TabularGenerator().fit(real)
    pd.testing.assert_frame_equal(g.sample(200, seed=3), g.sample(200, seed=3))
    assert not g.sample(200, seed=3).equals(g.sample(200, seed=4))


def test_handles_nulls_and_category_dtype():
    df = pd.DataFrame({
        "x": [1.0, 2.0, np.nan, 4.0, 5.0] * 40,
        "c": pd.Categorical(["a", "b", "a", "c", "a"] * 40),
    })
    g = TabularGenerator().fit(df)
    out = g.sample(100, seed=0)
    assert isinstance(out["c"].dtype, pd.CategoricalDtype)
    assert g.null_rates() == {"x": pytest.approx(0.2)}


def test_from_schema_no_data():
    g = TabularGenerator.from_schema(
        {"age": {"type": "int", "min": 18, "max": 90, "mean": 40, "std": 12},
         "spend": {"type": "float", "min": 0, "max": 1000, "mean": 300, "std": 100},
         "tier": {"type": "category", "categories": {"a": 0.7, "b": 0.3}}},
        correlations={("age", "spend"): 0.7},
    )
    df = g.sample(3000, seed=0)
    assert df["age"].dtype == np.int64 and df["age"].between(18, 90).all()
    assert abs(df["tier"].eq("a").mean() - 0.7) < 0.03
    assert df[["age", "spend"]].corr().iloc[0, 1] > 0.55


def test_empty_frame_rejected():
    with pytest.raises(ValueError):
        TabularGenerator().fit(pd.DataFrame())
    with pytest.raises(ValueError):
        GenConfig(null_rate={"a": 1.5})
