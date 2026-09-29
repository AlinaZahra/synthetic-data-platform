import json
import unicodedata
from datetime import date
from decimal import Decimal

import numpy as np
import pytest

from sdp.locale import LocaleError, LocaleRegistry, get_locale, registry
from sdp.locale.checksums import validate_checksum

CODES = ["en-US", "en-GB", "ur-PK", "ar", "hi", "es", "fr", "zh"]


def test_all_required_locales_registered():
    assert set(CODES) <= set(registry().codes())


@pytest.mark.parametrize("vec,algo,ok", [
    ("79927398713", "luhn", True), ("79927398714", "luhn", False),
    ("2363", "verhoeff", True), ("2364", "verhoeff", False),
    ("12345678Z", "dni", True), ("12345678A", "dni", False),
    ("1 85 05 78 006 084 91", "insee", True), ("1 85 05 78 006 084 92", "insee", False),
    ("11010519491231002X", "iso7064_11_2", True), ("110105194912310021", "iso7064_11_2", False),
])
def test_checksum_known_vectors(vec, algo, ok):
    assert validate_checksum(algo, vec) is ok


@pytest.mark.parametrize("code", CODES)
def test_generated_values_validate_against_own_pack(code):
    p, rng = get_locale(code), np.random.default_rng(0)
    for _ in range(150):
        assert p.phone_valid(p.phone(rng)), p.phone(rng)
        nid = p.national_id(rng)
        ok, why = p.national_id_valid(nid)
        assert ok, (nid, why)
        assert p.script_ok(p.person_name(rng))
        assert get_locale("en-US").script_ok(p.person_name(rng, latin=True)) or p.script == "Latin"
        text, city = p.address(rng)
        assert city in text
        d = p.birth_date(rng)
        assert p.parse_date(p.format_date(d)) == d


def test_negative_cases():
    assert not get_locale("ur-PK").phone_valid("+92 4xx 1234567")
    assert not get_locale("hi").national_id_valid("2345 6789 0123")[0]  # bad Verhoeff digit (almost surely)
    assert get_locale("hi").national_id_valid("1234 5678 9012") == (False, "does not match Aadhaar format")
    assert not get_locale("ur-PK").script_ok("Ahmed Khan")
    assert not get_locale("en-US").script_ok("Ahmed علی")
    assert get_locale("zh").script_ok("王伟")


def test_number_and_currency_formats():
    v = Decimal("1234567.5")
    assert get_locale("en-US").format_currency(v) == "$1,234,567.50"
    assert get_locale("hi").format_currency(v) == "₹12,34,567.50"
    assert get_locale("es").format_currency(v) == "1.234.567,50 €"
    assert get_locale("fr").format_currency(Decimal("1234.56")) == "1 234,56 €"
    assert get_locale("ur-PK").format_currency(Decimal("99")) == "Rs 99.00"
    assert get_locale("en-GB").format_currency(Decimal("-5")) == "£-5.00"


@pytest.mark.parametrize("code", CODES)
@pytest.mark.parametrize("value", ["0", "7.05", "999.99", "1000", "12345.6", "9876543.21"])
def test_currency_roundtrip(code, value):
    p = get_locale(code)
    assert p.parse_currency(p.format_currency(Decimal(value))) == Decimal(value).quantize(Decimal("0.01"))


def test_currency_parse_rejects_wrong_locale_format():
    assert get_locale("en-US").parse_currency("1.234,56 €") is None
    assert get_locale("es").parse_currency("$1,234.56") is None


def test_tax_rules():
    us = get_locale("en-US")
    assert us.tax_rate("CA") == ("Sales Tax", Decimal("0.0725"))
    assert us.tax_rate("CA", "food")[1] == Decimal("0")
    assert us.tax_rate(None)[1] == Decimal("0.07")


def test_adding_a_locale_is_data_only(tmp_path):
    d = dict(get_locale("es").data)
    d.update(code="es-MX", country="MX", aliases=["mexican", "mexico"], phone={"pattern": "+52 55 #### ####", "regex": r"^\+52 55 \d{4} \d{4}$"})
    (tmp_path / "es-MX.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    reg = LocaleRegistry([tmp_path])
    p = reg.get("es-MX")
    assert p.phone_valid(p.phone(np.random.default_rng(1)))
    assert reg.find_by_alias("Mexican").code == "es-MX"


def test_bad_pack_rejected(tmp_path):
    (tmp_path / "x.json").write_text(json.dumps({"code": "xx"}), encoding="utf-8")
    with pytest.raises(LocaleError, match="missing keys"):
        LocaleRegistry([tmp_path])
    d = dict(get_locale("es").data)
    d["national_id"] = {**d["national_id"], "checksum": "nope"}
    (tmp_path / "x.json").write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(LocaleError, match="unknown checksum"):
        LocaleRegistry([tmp_path])
    with pytest.raises(LocaleError, match="unknown locale"):
        get_locale("xx-YY")
