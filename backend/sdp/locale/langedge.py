"""M8. Language edge-case packs: the strings that break name/address handling in real systems.

Six packs, each with named cases: `unicode_diacritics`, `unicode_normalization` (NFC vs NFD of the same text), `apostrophes`
(straight/curly/modifier/backtick forms and quote-escaping stress), `long_names` (80-400 characters), `mixed_scripts` (incl. homoglyph
look-alikes) and `rtl_digits` (right-to-left text with embedded digits and bidi marks). All names are fictional composites.

They plug into `sdp.edgecases` (same rate/tag/coverage machinery: `GenConfig(edge_cases={"apostrophes": 0.05})`) and can be viewed
standalone with `language_cases()`. `describe(value)` reports the properties a consumer must handle (scripts, NFC/NFD, length, bidi).
"""

from __future__ import annotations

import unicodedata
from typing import Any, Callable

import numpy as np
import pandas as pd

LRM, RLM = "‎", "‏"
NFD = lambda s: unicodedata.normalize("NFD", s)  # noqa: E731

_LONG_PARTS = ["María", "de", "los", "Ángeles", "Fernández", "de", "la", "Vega", "y", "Ortiz", "de", "Zárate", "van", "der", "Berg", "Al-Hashimi", "bin", "Abdul", "Rahman"]

BANK: dict[str, dict[str, list[str]]] = {
    "unicode_diacritics": {
        "latin_accents": ["José Álvarez-Núñez", "Zoë Brontë", "François Lefèvre", "Ångström Öhman"],
        "polish_czech": ["Łukasz Wróblewski", "Dvořák Antonín", "Jiří Šebesta", "Żaneta Gąsiorowska"],
        "turkish_dotless_i": ["Şule Çelik", "İbrahim Işık", "Ayşe Yıldırım", "Gülşen Öztürk"],
        "vietnamese_stacked": ["Nguyễn Thị Ánh", "Trần Quốc Việt", "Phạm Thị Hồng Nhung", "Lê Hoàng Ánh Dương"],
        "nordic": ["Søren Ærøskøbing", "Björk Guðmundsdóttir", "Åsa Lindqvist", "Þórunn Sæmundsdóttir"],
    },
    "unicode_normalization": {   # value pairs are produced at apply time: same text in NFC, NFD or mixed
        "nfc_composed": ["José Núñez", "Zoë Müller", "Ångström Öhman", "Çağla Şahin"],
        "nfd_decomposed": ["José Núñez", "Zoë Müller", "Ångström Öhman", "Çağla Şahin"],
        "half_and_half": ["José Núñez", "Zoë Müller", "Ångström Öhman", "Çağla Şahin"],
        "hangul_jamo": ["김민수", "박지훈", "이서연", "최유진"],
    },
    "apostrophes": {
        "straight": ["Sean O'Brien", "Dina D'Angelo", "Mike O'Neil", "Sa'id Karim"],
        "curly_right": ["Sean O’Brien", "Dina D’Angelo", "N’Golo Kanté", "Mary O’Connor"],
        "modifier_letter": ["Kaʻana Kahale", "ʻOkalani Kealoha", "Saʻid Karim", "Hoʻoilina Kamai"],
        "backtick_acute": ["Ka`ana Kahale", "Anne D´Souza", "Ma‘ata Tui", "Jo’el Bern"],
        "quote_stress": ["Robert'); DROP TABLE students;--", "O'Malley's \"Pub\"", "Tom 'Tommy' Reyes", "Beth O''Hara"],
        "french_elision": ["L'Écuyer Beaulieu", "D'Aubigné Marie", "L’Hôpital Jean", "d'Artagnan Pierre"],
    },
    "mixed_scripts": {
        "latin_arabic": ["Ahmed أحمد Khan", "Sara ساره Malik", "Omar عمر Farooq"],
        "latin_han": ["山田 Taro", "Wang 伟 Wei", "Li Ming 李明", "陈 Chen Xiaoyu"],
        "latin_devanagari": ["Priya प्रिया Sharma", "Amit अमित Verma", "Neha नेहा Rao"],
        "latin_cyrillic": ["Иван Ivanov", "Olga Ольга Petrova", "Dmitri Дмитрий Sokolov"],
        "homoglyph": ["Аlex Morgan", "Joհn Smith", "Mаria Lopez", "Natаsha Roy"],   # Cyrillic A / Armenian h / Cyrillic a look like Latin
    },
    "rtl_digits": {
        "rtl_with_latin_digits": ["أحمد 123 خان", "شارع 15 رقم 7", "عمارة 42 طابق 3"],
        "rtl_with_native_digits": ["أحمد ١٢٣ خان", "شارع ١٥ رقم ٧", "گلی ۵ مکان ۱۲"],
        "ltr_run_in_rtl": ["مكتب Room 12B في الرياض", "شركة ACME-2024 المحدودة", "دفتر No. 7 لاہور"],
        "bidi_marks": [f"أحمد{LRM} (2024){RLM} خان", f"House 12{RLM}، گلی 5", f"شارع {LRM}45{RLM} جدة"],
        "phone_arabic_indic": ["٠٣٠٠١٢٣٤٥٦٧", "+٩٢ ٣٠٠ ١٢٣٤٥٦٧", "۰۳۰۰۱۲۳۴۵۶۷"],
    },
}
LONG_LENGTHS = {"long_80": 80, "long_150": 150, "long_255": 255, "long_400": 400}
PACK_NAMES = [*BANK, "long_names"]
CASES: dict[str, list[str]] = {**{p: sorted(c) for p, c in BANK.items()}, "long_names": sorted(LONG_LENGTHS)}


def _long_name(rng: np.random.Generator, length: int) -> str:
    out: list[str] = []
    while len(" ".join(out)) < length:
        out.append(_LONG_PARTS[int(rng.integers(len(_LONG_PARTS)))])
    s = " ".join(out)[:length].rstrip()
    return s if len(s) == length else s + "x" * (length - len(s))


def make_value(pack: str, case: str, rng: np.random.Generator) -> str:
    if pack == "long_names":
        return _long_name(rng, LONG_LENGTHS[case])
    v = BANK[pack][case][int(rng.integers(len(BANK[pack][case])))]
    if pack == "unicode_normalization":
        if case == "nfc_composed":
            return unicodedata.normalize("NFC", v)
        if case == "nfd_decomposed":
            return NFD(v)
        if case == "half_and_half":
            first, _, last = v.partition(" ")
            return unicodedata.normalize("NFC", first) + " " + NFD(last)
        return NFD(v)   # hangul_jamo: syllables decomposed into conjoining jamo
    return v


# --------------------------------------------------------------- inspection
def scripts_of(s: str) -> list[str]:
    out: set[str] = set()
    for ch in s:
        if ch.isalpha():
            out.add(unicodedata.name(ch, "UNKNOWN").split()[0])
    return sorted(out)


def describe(value: str) -> dict[str, Any]:
    """Properties a consumer of this string must cope with."""
    nfc = unicodedata.normalize("NFC", value)
    return {
        "length": len(value), "utf8_bytes": len(value.encode("utf-8")), "scripts": scripts_of(value),
        "is_nfc": value == nfc, "has_combining": any(unicodedata.combining(c) for c in value),
        "apostrophes": sorted({f"U+{ord(c):04X}" for c in value if c in "'’‘ʻ´`"}),
        "has_rtl": any(unicodedata.bidirectional(c) in ("R", "AL") for c in value),
        "has_digits": any(c.isdigit() for c in value), "has_bidi_marks": any(c in (LRM, RLM) for c in value),
        "mixed_script": len([s for s in scripts_of(value) if s != "UNKNOWN"]) > 1,
    }


def normalize_key(value: str) -> str:
    """The comparison key that makes NFC/NFD twins equal (NFC, casefold, trimmed, bidi marks removed, apostrophes unified)."""
    s = unicodedata.normalize("NFC", value).replace(LRM, "").replace(RLM, "")
    for a in "’‘ʻ´`":
        s = s.replace(a, "'")
    return s.strip().casefold()


def language_cases(packs: list[str] | None = None, per_case: int = 2, seed: int = 0) -> pd.DataFrame:
    """A table of edge-case strings with their properties, for demos and for testing your own pipeline."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in packs or PACK_NAMES:
        if p not in CASES:
            raise ValueError(f"unknown language pack {p!r}; available: {PACK_NAMES}")
        for case in CASES[p]:
            for _ in range(per_case):
                v = make_value(p, case, rng)
                rows.append({"pack": p, "case": case, "value": v, **{k: (",".join(x) if isinstance(x, list) else x) for k, x in describe(v).items()}})
    return pd.DataFrame(rows)


# ------------------------------------------------- edgecases.py integration
_NAMEY = ("name", "city", "address", "street", "title", "company", "customer", "patient", "contact", "description", "comment", "note")


def _is_str_series(s: pd.Series) -> bool:
    return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_bool_dtype(s)
                or isinstance(s.dtype, pd.CategoricalDtype)) and bool(s.dropna().map(lambda v: isinstance(v, str)).all())


def _columns(df: pd.DataFrame, protect: set[str]) -> list[str]:
    strs = [c for c in df.columns if c not in protect and not str(c).startswith("_") and _is_str_series(df[c])]
    named = [c for c in strs if any(h in str(c).lower() for h in _NAMEY)]
    return named or strs


def _make(pack: str) -> tuple[Callable[..., dict[str, list[str]]], Callable[..., str | None]]:
    def cases(df: pd.DataFrame, protect: set[str]) -> dict[str, list[str]]:
        cols = _columns(df, protect)
        return {c: cols for c in CASES[pack]} if cols else {}

    def apply(df: pd.DataFrame, case: str, col: str, i: int, rng: np.random.Generator) -> str | None:
        df.at[i, col] = make_value(pack, case, rng)
        return case
    return cases, apply


LANG_PACKS = {name: _make(name) for name in PACK_NAMES}
