import itertools
import re

import numpy as np
import pandas as pd
import pytest

from sdp.locale import LocaleError, get_locale, registry
from sdp.locale.codemix import CodeMixError, english_token_ratio, generate_code_mixed, pairs, topics
from sdp.locale.coherent import COLUMNS, coherence_report, generate_people, parse_mix, resolve_locale
from sdp.locale.translit import generate_entities, matching_pairs, name_variants

ARABIC = re.compile(r"[؀-ۿ]")


# -------------------------------------------------------------------- M3
def test_mix_parsing_and_resolution():
    assert parse_mix("70% ur-PK, 30% en") == {"ur-PK": pytest.approx(0.7), "en-US": pytest.approx(0.3)}
    assert parse_mix("ur-PK:7, en-GB:3") == {"ur-PK": pytest.approx(0.7), "en-GB": pytest.approx(0.3)}
    assert parse_mix("50% pakistani and 50% french") == {"ur-PK": 0.5, "fr": 0.5}
    assert resolve_locale("en") == "en-US" and resolve_locale("Spanish") == "es" and resolve_locale("ar") == "ar"
    for bad in ("", "lots of everything", "0% en", "50% klingon"):
        with pytest.raises(LocaleError):
            parse_mix(bad)


def test_every_row_is_coherent_in_one_locale():
    df = generate_people(600, "70% ur-PK, 30% en", seed=1)
    assert list(df.columns) == COLUMNS and len(df) == 600
    rep = coherence_report(df)
    assert rep["coherent_pct"] == 100.0 and rep["failures"] == []
    assert all(v["failed"] == 0 for v in rep["by_check"].values())
    share = (df["locale"] == "ur-PK").mean()
    assert 0.65 < share < 0.75 and set(df["locale"]) == {"ur-PK", "en-US"}
    ur, en = df[df["locale"] == "ur-PK"], df[df["locale"] == "en-US"]
    assert ur["full_name"].map(lambda s: bool(ARABIC.search(s))).all() and not en["full_name"].map(lambda s: bool(ARABIC.search(s))).any()
    assert (ur["currency"] == "PKR").all() and (en["currency"] == "USD").all()
    assert ur["phone"].str.startswith("+92").all() and en["phone"].str.startswith("+1").all()
    assert ur["email"].str.endswith("@mail.pk.example").all() and en["email"].str.endswith("@mail.us.example").all()


def test_email_matches_the_persons_latin_name_and_is_ascii():
    df = generate_people(300, {"ar": 0.5, "es": 0.5}, seed=2)
    for _, r in df.head(100).iterrows():
        local = r["email"].split("@")[0]
        assert local.isascii() and re.fullmatch(r"[a-z0-9.]+", local)
        first, last = r["full_name_latin"].split()[0].lower(), r["full_name_latin"].split()[-1].lower()
        assert re.sub(r"[^a-z]", "", last) in local or first[:1] in local[:2] or re.sub(r"[^a-z]", "", first) in local


def test_coherence_checker_catches_mixed_up_rows():
    df = generate_people(50, {"ur-PK": 1.0}, seed=3)
    bad = df.copy()
    bad.loc[0, "phone"] = "+1 (212) 555-0100"                       # a US phone on a Pakistani record
    bad.loc[1, "currency"] = "EUR"
    bad.loc[2, "full_name"] = "John Smith"
    bad.loc[3, "email"] = "x@mail.us.example"
    bad.loc[4, "city"] = "Paris"
    rep = coherence_report(bad)
    assert {f["check"] for f in rep["failures"]} == {"phone", "currency", "name_script", "email_host", "city"}
    assert rep["coherent_pct"] == pytest.approx(100 * (1 - 5 / 50))


def test_people_deterministic_and_all_locales():
    a, b = generate_people(100, {c: 1 for c in registry().codes()}, seed=4), generate_people(100, {c: 1 for c in registry().codes()}, seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert coherence_report(a)["coherent_pct"] == 100.0 and a["locale"].nunique() == len(registry().codes())


# -------------------------------------------------------------------- M4
def test_the_headline_example_exists_as_variants():
    pack = get_locale("ur-PK")
    got = {t for _, _, t in name_variants(pack, "محمد", "علی", np.random.default_rng(0))}
    assert {"محمد علی", "Muhammad Ali", "Mohammad Ali", "Mohammed Ali", "Ali, Muhammad", "M. Ali", "MUHAMMAD ALI"} <= got


def test_entities_share_id_and_attributes_across_scripts():
    df = generate_entities(200, "ur-PK", seed=1, variants_per_entity=5)
    assert df["entity_id"].nunique() == 200 and df.groupby("entity_id").size().between(2, 5).all()
    for _, g in df.groupby("entity_id"):
        assert g["dob"].nunique() == g["national_id"].nunique() == g["phone"].nunique() == 1
        assert g["canonical"].sum() == 1 and g.iloc[0]["script"] == "native" and "latin" in set(g["script"])
        assert g["name"].nunique() == len(g)                      # no duplicate spellings within an entity
    assert set(df["script"]) == {"native", "latin"}
    assert df.loc[df["script"] == "native", "name"].map(lambda s: bool(ARABIC.search(s))).all()
    assert df.loc[df["script"] == "latin", "name"].map(lambda s: not ARABIC.search(s)).all()


@pytest.mark.parametrize("locale", ["ur-PK", "ar", "hi", "zh", "es", "fr", "en-GB"])
def test_every_locale_produces_multiple_variants(locale):
    df = generate_entities(60, locale, seed=2, variants_per_entity=6, typos=True)
    assert df.groupby("entity_id").size().min() >= 2 and df["style"].nunique() >= 4
    if get_locale(locale).script != "Latin":
        assert {"native", "latin"} <= set(df["script"])
    else:
        assert "latin_reordered" in set(df["style"]) and "latin_upper" in set(df["style"])


def test_diacritics_are_stripped_for_latin_locales():
    df = generate_entities(300, "es", seed=3, variants_per_entity=8)
    accented = df[df["name"].str.contains("[áéíóúñ]")]["entity_id"].unique()
    assert len(accented) > 0
    stripped = df[(df["style"] == "latin_no_diacritics") & df["entity_id"].isin(accented)]
    assert len(stripped) > 0 and not stripped["name"].str.contains("[áéíóúñÁÉÍÓÚÑ]").any()


def test_confusers_and_matching_pairs_ground_truth():
    df = generate_entities(100, "ar", seed=4, variants_per_entity=4, confusers=0.2)
    conf = df[df["confuser_of"].notna()]
    assert conf["entity_id"].nunique() == 20
    for eid, g in conf.groupby("entity_id"):                       # same NAME as its source, different person
        src = df[df["entity_id"] == g["confuser_of"].iloc[0]]
        assert set(g["name"]) & set(src["name"]) and g["dob"].iloc[0] != src["dob"].iloc[0] or g["national_id"].iloc[0] != src["national_id"].iloc[0]
    pairs_df = matching_pairs(df, seed=1)
    ent = dict(zip(df["record_id"], df["entity_id"]))
    for a, b, y in pairs_df.itertuples(index=False):
        assert (ent[a] == ent[b]) == bool(y)
    assert pairs_df["label"].sum() == sum(len(list(itertools.combinations(range(len(g)), 2))) for _, g in df.groupby("entity_id"))
    hard = pairs_df[pairs_df["label"] == 0].merge(df[["record_id", "name"]], left_on="record_a", right_on="record_id") \
        .merge(df[["record_id", "name"]], left_on="record_b", right_on="record_id", suffixes=("_a", "_b"))
    assert (hard["name_a"] == hard["name_b"]).sum() > 0             # hard negatives present


def test_translit_tables_cover_every_name_in_the_packs_and_are_deterministic():
    for code in ("ur-PK", "ar", "hi", "zh"):
        p = get_locale(code)
        toks = set(p.data["names"]["first_male"] + p.data["names"]["first_female"] + p.data["names"]["last"])
        assert toks <= set(p.data["translit"]), code
        assert all(v and v[0].isascii() for v in p.data["translit"].values())
    a, b = generate_entities(30, "hi", seed=9), generate_entities(30, "hi", seed=9)
    pd.testing.assert_frame_equal(a, b)


# -------------------------------------------------------------------- M5
def test_all_pairs_and_topics_available():
    assert set(pairs()) == {"roman_urdu", "hinglish", "spanglish"} and topics("hinglish") == ["banking", "shopping", "support"]


@pytest.mark.parametrize("pair", ["roman_urdu", "hinglish", "spanglish"])
@pytest.mark.parametrize("topic", ["banking", "shopping", "support"])
def test_level_controls_english_share(pair, topic):
    lo = generate_code_mixed(300, pair, topic, level=0.0, seed=1)
    mid = generate_code_mixed(300, pair, topic, level=0.5, seed=1)
    hi = generate_code_mixed(300, pair, topic, level=1.0, seed=1)
    assert lo["en_slots"].sum() == 0 and hi["realized_mix"].mean() > 0.95
    assert 0.3 < mid["realized_mix"].mean() < 0.7
    ratios = [df["tokens"].map(english_token_ratio).mean() for df in (lo, mid, hi)]
    assert ratios[0] < ratios[1] < ratios[2] and ratios[0] < 0.1


def test_text_is_well_formed_and_tagged():
    df = generate_code_mixed(50, "roman_urdu", "banking", level=0.6, seed=2)
    assert df["text"].map(lambda s: "{" not in s and "}" not in s and len(s) > 8).all()
    assert df["text"].nunique() > 10
    row = df.iloc[0]
    assert "".join(t for t, _ in row["tokens"]).replace(" ", "") == re.sub(r"\s", "", row["text"]).replace("’", "’")
    assert {lang for _, lang in row["tokens"]} <= {"base", "en", "punct"}
    hi = generate_code_mixed(20, "spanglish", "shopping", level=1.0, seed=3)
    assert any(lang == "en" for toks in hi["tokens"] for _, lang in toks) and hi["text"].str.contains("[¿áéíóú]").any()


def test_deterministic_and_validation():
    a, b = generate_code_mixed(40, "hinglish", "support", 0.4, seed=5), generate_code_mixed(40, "hinglish", "support", 0.4, seed=5)
    pd.testing.assert_frame_equal(a, b)
    assert not a["text"].equals(generate_code_mixed(40, "hinglish", "support", 0.4, seed=6)["text"])
    for args in (("klingon", "banking", .5), ("hinglish", "cooking", .5), ("hinglish", "banking", 1.5)):
        with pytest.raises(CodeMixError):
            generate_code_mixed(5, *args)
