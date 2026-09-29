import numpy as np
import pandas as pd

from sdp.common import is_identifier_like
from sdp.scoring import fidelity_score, privacy_score
from sdp.tabular import TabularGenerator


def people(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"name": [f"Customer {i:04d}" for i in range(1, n + 1)],
                         "code": [f"AB-{rng.integers(0, 10**7):07d}" for _ in range(n)],
                         "tier": rng.choice(["a", "b", "c"], n), "score": rng.normal(50, 10, n).round(1)})


def test_identifier_detection():
    df = people()
    assert is_identifier_like(df["name"]) and is_identifier_like(df["code"])
    assert not is_identifier_like(df["tier"]) and not is_identifier_like(df["score"])
    assert not is_identifier_like(pd.Series([f"x{i}" for i in range(10)]))   # too few distinct values to be an id column


def test_identifier_columns_get_new_values_in_the_real_format():
    real = people()
    synth = TabularGenerator().fit(real).sample(600, seed=1)
    assert not synth["name"].isin(real["name"]).mean() > 0.2          # real names are not resampled (chance overlap only)
    assert synth["name"].str.fullmatch(r"Customer \d{4}").all()        # ...but the format is kept
    assert synth["code"].str.match(r"^AB-\d{7}$").all()


def test_no_false_privacy_alarm_from_identifier_and_low_entropy_columns():
    real = people()
    synth = TabularGenerator().fit(real).sample(600, seed=2)
    p = privacy_score(real, synth)
    assert p["metrics"]["exact_matches"] == 0 and p["score"] > 80
    f = fidelity_score(real, synth)
    assert f["columns"]["name"]["kind"] == "identifier" and f["columns"]["name"]["score"] is None   # not scored, not penalised
    assert f["components"]["marginals"]["score"] > 90


def test_real_copies_of_unique_rows_are_still_caught():
    real = people()
    leaked = real.sample(200, random_state=0).reset_index(drop=True)
    p = privacy_score(real, leaked)
    assert p["metrics"]["exact_matches"] >= 190 and p["score"] < 60
