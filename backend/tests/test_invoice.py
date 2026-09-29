import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

from sdp.documents import InvoiceSpec, LineSpec, generate_invoice, quantize, reconcile, render_invoice
from sdp.documents.render import count_pdf_pages

D = Decimal


def inv(lines, **kw):
    return generate_invoice(InvoiceSpec(lines=[LineSpec(**ln) for ln in lines], **kw))


# ---------------------------------------------------------------- rounding
def test_round_half_up_not_bankers():
    assert quantize(D("0.125"), 2) == D("0.13") and quantize(D("0.135"), 2) == D("0.14") and quantize(D("2.5"), 0) == D("3")
    i = inv([{"description": "x", "quantity": "1", "unit_price": "0.125"}])
    assert i["lines"][0]["line_total"] == "0.13"


def test_no_float_drift():
    i = inv([{"description": "cent", "quantity": "1", "unit_price": "0.10"}] * 500, locale="en-GB")
    assert i["subtotal"] == "50.00" and reconcile(i) == []
    assert 0.1 * 3 != 0.3 and quantize(D("0.1") * 3, 2) == D("0.30")  # the float trap this module avoids


def test_fractional_quantity_line_rounding():
    i = inv([{"description": "hours", "quantity": "2.5", "unit_price": "33.33"}])
    assert i["lines"][0]["line_total"] == "83.33"  # 83.325 -> half up


# --------------------------------------------------------------------- tax
def test_tax_is_computed_per_rate_group_not_per_line():
    lines = [{"description": "a", "quantity": "1", "unit_price": "3.35"}] * 3  # 10.05 total
    i = inv(lines, locale="en-US", region="CA")  # 7.25%
    assert i["tax_summary"] == [{"tax_label": "Sales Tax", "tax_rate": "0.0725", "taxable_amount": "10.05", "tax": "0.73"}]
    assert i["total"] == "10.78" and reconcile(i) == []  # per-line rounding would have given 0.72


def test_region_and_category_rules():
    line = {"description": "widget", "quantity": "1", "unit_price": "100", "category": "hardware"}
    assert inv([line], locale="en-US", region="CA")["tax_total"] == "7.25"
    assert inv([line], locale="en-US", region="TX")["tax_total"] == "6.25"
    assert inv([line], locale="en-US", region="OR")["tax_total"] == "0.00"
    assert inv([{**line, "category": "food"}], locale="en-US", region="CA")["tax_total"] == "0.00"
    assert inv([{**line, "category": "food"}], locale="fr")["tax_total"] == "5.50"
    assert inv([line], locale="ur-PK", region="SINDH")["tax_total"] == "13.00"
    mixed = inv([line, {**line, "category": "food"}], locale="es")
    assert [s["tax_rate"] for s in mixed["tax_summary"]] == ["0.10", "0.21"] and mixed["tax_total"] == "31.00"


def test_tax_inclusive_extracts_tax_exactly():
    i = inv([{"description": "x", "quantity": "1", "unit_price": "118.00", "category": "software"}], locale="hi", tax_inclusive=True)
    assert (i["subtotal"], i["tax_total"], i["total"]) == ("100.00", "18.00", "118.00") and reconcile(i) == []
    odd = inv([{"description": "x", "quantity": "3", "unit_price": "9.99"}] * 7, locale="en-GB", tax_inclusive=True)
    assert D(odd["subtotal"]) + D(odd["tax_total"]) == D(odd["total"]) and reconcile(odd) == []


def test_reconcile_detects_tampering():
    i = generate_invoice(InvoiceSpec(seed=5, locale="en-US", region="NY"))
    assert reconcile(i) == []
    bad = json.loads(json.dumps(i))
    bad["total"] = str(D(bad["total"]) + D("0.01"))
    assert any("total" in e for e in reconcile(bad))
    bad = json.loads(json.dumps(i))
    bad["lines"][0]["quantity"] = "99"
    assert any("line 0" in e for e in reconcile(bad))


# --------------------------------------------------------------- generation
@pytest.mark.parametrize("locale", ["en-US", "en-GB", "ur-PK", "ar", "hi", "es", "fr", "zh"])
def test_random_invoices_reconcile_exactly_in_every_locale(locale):
    for seed in range(25):
        i = generate_invoice(InvoiceSpec(seed=seed, locale=locale, n_lines=1 + seed % 12, tax_inclusive=bool(seed % 2)))
        assert reconcile(i) == [], (locale, seed)
        assert all(isinstance(v, str) for v in (i["subtotal"], i["tax_total"], i["total"]))
    json.dumps(i)


def test_deterministic_and_doc_id():
    a, b = InvoiceSpec(seed=3, locale="fr"), InvoiceSpec(seed=3, locale="fr")
    assert generate_invoice(a) == generate_invoice(b) and a.doc_id() == b.doc_id()
    assert InvoiceSpec(seed=4, locale="fr").doc_id() != a.doc_id()


def test_spec_validation():
    for bad in ({"locale": "xx"}, {"locale": "en-US", "region": "ZZ"}, {"n_lines": 0}, {"n_lines": 501}, {"unknown": 1},
                {"lines": [{"description": "x", "quantity": "-1", "unit_price": "1"}]},
                {"lines": [{"description": "x", "quantity": "1", "unit_price": "NaN"}]},
                {"lines": [{"description": "x", "quantity": "1", "unit_price": "1e12"}]},
                {"lines": [{"description": "", "quantity": "1", "unit_price": "1"}]}):
        with pytest.raises(ValidationError):
            InvoiceSpec(**bad)


# ----------------------------------------------------------------- render
def test_render_pdf_and_display_formats():
    i = generate_invoice(InvoiceSpec(seed=2, locale="hi", region="KA", n_lines=8))
    r = render_invoice(i)
    assert r.pdf.startswith(b"%PDF") and r.pdf.rstrip().endswith(b"%%EOF") and r.pages == count_pdf_pages(r.pdf) == 1
    assert r.unrenderable == []
    assert render_invoice(i).pdf == r.pdf  # byte-identical re-render


def test_render_is_paginated_for_many_lines():
    i = generate_invoice(InvoiceSpec(seed=1, n_lines=120))
    r = render_invoice(i)
    assert r.pages > 1 and count_pdf_pages(r.pdf) == r.pages
