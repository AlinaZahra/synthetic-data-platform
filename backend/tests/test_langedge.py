import unicodedata

import numpy as np
import pandas as pd
import pytest

from sdp.edgecases import ALL_PACKS, TAG, inject_edge_cases, visible
from sdp.locale import langedge as L
from sdp.tabular import GenConfig, TabularGenerator


def frame(n=200):
    return pd.DataFrame({"customer_name": [f"Person {i}" for i in range(n)], "city": ["Lahore"] * n, "id": range(n), "amount": np.linspace(1, 9, n)})


def test_all_six_packs_are_registered_with_edgecases():
    for p in L.PACK_NAMES:
        assert p in ALL_PACKS
    assert len(L.PACK_NAMES) == 6


@pytest.mark.parametrize("pack", L.PACK_NAMES)
def test_pack_injects_only_into_text_columns_and_reports_full_case_coverage(pack):
    r = inject_edge_cases(frame(400), {pack: 0.5}, seed=3, protect=["id"])
    info = r.report["packs"][pack]
    assert info["coverage_pct"] == 100.0 and set(info["columns"]) <= {"customer_name", "city"}
    tagged = r.data[r.data[TAG].notna()]
    assert len(tagged) == info["rows_affected"] > 0
    assert (r.data["id"] == range(400)).all() and (r.data["amount"] == frame(400)["amount"]).all()
    assert TAG not in visible(r.data).columns


def test_normalization_pack_makes_canonically_equal_but_bytewise_different_strings():
    rng = np.random.default_rng(1)
    nfc = L.make_value("unicode_normalization", "nfc_composed", np.random.default_rng(1))
    nfd = L.make_value("unicode_normalization", "nfd_decomposed", np.random.default_rng(1))
    half = L.make_value("unicode_normalization", "half_and_half", rng)
    assert nfc != nfd and unicodedata.normalize("NFC", nfd) == nfc
    assert L.normalize_key(nfc) == L.normalize_key(nfd) == L.normalize_key(nfd.upper()) or L.normalize_key(nfc) == L.normalize_key(nfd)
    assert L.describe(nfd)["is_nfc"] is False and L.describe(nfc)["is_nfc"] is True and L.describe(nfd)["has_combining"]
    assert half != unicodedata.normalize("NFC", half) or half != unicodedata.normalize("NFD", half)


def test_apostrophe_forms_are_all_present_and_unified_by_the_key():
    df = L.language_cases(["apostrophes"], per_case=6, seed=1)
    found = set(",".join(df["apostrophes"]).split(","))
    assert {"U+0027", "U+2019", "U+02BB", "U+0060"} <= found
    assert L.normalize_key("O\u2019Brien") == L.normalize_key("O'Brien")
    assert any("DROP TABLE" in v for v in df["value"])            # quote-escaping stress


def test_long_names_hit_exact_lengths():
    df = L.language_cases(["long_names"], per_case=1)
    assert sorted(df["length"]) == [80, 150, 255, 400]


def test_mixed_scripts_and_homoglyphs_are_flagged():
    df = L.language_cases(["mixed_scripts"], per_case=3, seed=2)
    assert df["mixed_script"].all()
    hom = df[df["case"] == "homoglyph"]
    assert not hom.empty and all(v.isascii() is False for v in hom["value"])
    assert any("CYRILLIC" in s for s in hom["scripts"])


def test_rtl_digit_cases_have_rtl_and_digits_and_marks():
    df = L.language_cases(["rtl_digits"], per_case=3, seed=1)
    assert df["has_digits"].all()
    assert df[df["case"] != "phone_arabic_indic"]["has_rtl"].all()     # digit-only phone numbers are bidi class AN, not R
    assert df[df["case"] == "bidi_marks"]["has_bidi_marks"].all()
    assert "U+" not in "".join(df[df["case"] == "rtl_with_latin_digits"]["apostrophes"])


def test_deterministic_and_unknown_pack():
    a = L.language_cases(seed=5)
    assert a.equals(L.language_cases(seed=5))
    with pytest.raises(ValueError):
        L.language_cases(["klingon"])


def test_works_through_gen_config_and_generator():
    real = frame(300)
    real["age"] = np.random.default_rng(0).integers(18, 80, 300)
    gen = TabularGenerator().fit(real[["age", "amount"]].assign(name=real["customer_name"]))
    res = gen.generate(GenConfig(rows=200, seed=1, edge_cases={"apostrophes": 0.1, "rtl_digits": 0.1}))
    tags = res.data[TAG].dropna()
    assert tags.str.contains("apostrophes").any() and tags.str.contains("rtl_digits").any()
