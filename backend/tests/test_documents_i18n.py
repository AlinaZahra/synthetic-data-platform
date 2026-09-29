import io
import json
import re
from decimal import Decimal

import pandas as pd
import pytest
from pypdf import PdfReader

from sdp.documents import (DOC_TYPES, InvoiceSpec, ReceiptSpec, StatementSpec, build_layout, generate_invoice, generate_receipt,
                           generate_statement, reconcile, render_document)
from sdp.documents import fonts
from sdp.documents.pipeline import DocumentPipeline
from sdp.documents.render import MissingFontError
from sdp.documents.shaping import shape
from sdp.export import BOM, csv_bytes, json_bytes, zip_bytes
from sdp.locale import get_locale

ARABIC_FONT = fonts.find_font("arabic")
CJK_FONT = fonts.find_font("cjk")
needs_arabic = pytest.mark.skipif(ARABIC_FONT is None, reason="no Arabic-capable TrueType font installed")
needs_cjk = pytest.mark.skipif(CJK_FONT is None, reason="no CJK TrueType font installed")


def pdf_text(pdf: bytes) -> str:
    """All drawn text chunks (the visitor API is more faithful than plain extract_text for mixed-direction pages)."""
    out: list[str] = []
    for page in PdfReader(io.BytesIO(pdf)).pages:
        page.extract_text(visitor_text=lambda t, cm, tm, fd, fs: out.append(t.strip()) if t.strip() else None)
    return "\n".join(out)


def text_positions(pdf: bytes) -> list[tuple[str, float]]:
    out = []
    PdfReader(io.BytesIO(pdf)).pages[0].extract_text(visitor_text=lambda t, cm, tm, fd, fs: out.append((t.strip(), tm[4])) if t.strip() else None)
    return out


# -------------------------------------------------------------- shaping/bidi
def test_arabic_is_reshaped_and_reordered_but_numbers_stay_ltr():
    logical = "المجموع 1,234.50"
    visual = shape(logical, rtl=True)
    assert "1,234.50" in visual                                  # digit run keeps its internal order
    assert visual != logical and all(0xFE70 <= ord(c) <= 0xFEFF or c in " 1234567890,." for c in visual)  # presentation forms
    assert visual.index("1,234.50") < visual.index(visual.replace("1,234.50", "").strip()[0])  # number sits left of the Arabic word


def test_leading_number_in_latin_address_is_not_moved_in_rtl_documents():
    assert shape("57 Tahlia Street, Mecca 68516", rtl=True) == "57 Tahlia Street, Mecca 68516"


def test_latin_text_untouched_in_ltr_and_stable_in_rtl():
    assert shape("Total Rs 99.00", rtl=False) == "Total Rs 99.00"
    assert shape("Rs 1,234.00", rtl=True) == "Rs 1,234.00"      # a pure LTR run is not scrambled by an RTL paragraph
    assert shape("", rtl=True) == ""


def test_urdu_letters_shape():
    v = shape("انوائس", rtl=True)
    assert v != "انوائس" and len(v) >= 5


# ------------------------------------------------------ localized layouts
@pytest.mark.parametrize("code,title,tax", [("es", "FACTURA", "IVA"), ("fr", "FACTURE", "TVA"), ("en-GB", "INVOICE", "VAT"),
                                            ("ar", "فاتورة", "ضريبة القيمة المضافة"), ("ur-PK", "انوائس", "جی ایس ٹی"), ("zh", "发票", "增值税")])
def test_layout_uses_locale_labels_and_tax_names(code, title, tax):
    inv = generate_invoice(InvoiceSpec(seed=1, locale=code, native=True, n_lines=3))
    lay = build_layout(inv)
    assert lay.title == title and any(tax in t[0] for t in lay.totals)
    assert lay.direction == ("rtl" if code in ("ar", "ur-PK") else "ltr")


def test_native_digits_and_number_formats_in_layout():
    inv = generate_invoice(InvoiceSpec(seed=2, locale="ur-PK", native=True, native_digits=True, region="SINDH"))
    lay = build_layout(inv)
    total = lay.totals[-1][1]
    assert re.search("[۰-۹]", total) and not re.search("[0-9]", total)
    assert "٬" in build_layout(generate_invoice(InvoiceSpec(seed=3, locale="ar", native=True, native_digits=True,
                lines=[{"description": "x", "quantity": "1", "unit_price": "1234567"}])), None).totals[-1][1]  # Arabic thousands mark
    ascii_lay = build_layout(generate_invoice(InvoiceSpec(seed=2, locale="ur-PK", native=True)))
    assert re.search("[0-9]", ascii_lay.totals[-1][1])                 # ASCII digits unless asked otherwise
    hi = generate_invoice(InvoiceSpec(seed=1, locale="hi", lines=[{"description": "x", "quantity": "1", "unit_price": "1234567.5"}], native=True))
    assert build_layout(hi).totals[0][1].endswith("12,34,567.50") or "12,34,567.50" in build_layout(hi).totals[0][1]  # lakh grouping


def test_json_ground_truth_stays_ascii_decimal_even_with_native_digits():
    inv = generate_invoice(InvoiceSpec(seed=2, locale="ar", native=True, native_digits=True))
    assert re.fullmatch(r"\d+\.\d{2}", inv["total"]) and reconcile(inv) == []
    assert inv["presentation"] == {"native": True, "native_digits": True, "direction": "rtl", "script": "arabic", "notes": []}


def test_devanagari_falls_back_with_a_note_instead_of_rendering_wrongly():
    inv = generate_invoice(InvoiceSpec(seed=1, locale="hi", native=True))
    assert inv["presentation"]["native"] is False and inv["presentation"]["notes"]
    r = render_document(inv)
    assert r.unrenderable == [] and r.direction == "ltr"


# ------------------------------------------------------------- RTL PDFs
@needs_arabic
@pytest.mark.parametrize("code", ["ar", "ur-PK"])
def test_rtl_invoice_pdf_embeds_font_and_reads_right_to_left(code):
    inv = generate_invoice(InvoiceSpec(seed=4, locale=code, native=True, n_lines=6))
    r = render_document(inv)
    assert r.pdf.startswith(b"%PDF") and r.embedded_font and r.unrenderable == [] and r.direction == "rtl"
    assert b"FontFile2" in r.pdf and re.search(rb"/BaseFont\s*/[A-Z]{6}\+", r.pdf)  # subset TrueType embedded
    text = pdf_text(r.pdf)
    pack = get_locale(code)
    assert pack.format_number(Decimal(inv["total"])) in text            # Latin-digit amounts stay intact (mixed numbers)
    assert inv["invoice_number"] in text and inv["seller"]["tax_id"] in text
    pos = dict(text_positions(r.pdf))
    tax_x = next(x for t, x in text_positions(r.pdf) if inv["seller"]["tax_id"] in t)
    num_x = next(x for t, x in text_positions(r.pdf) if inv["invoice_number"] in t)
    assert tax_x > num_x                                                # seller block on the right, title/number block on the left
    ltr = generate_invoice(InvoiceSpec(seed=4, locale="en-US", n_lines=6))
    lt = text_positions(render_document(ltr).pdf)
    assert next(x for t, x in lt if ltr["seller"]["tax_id"] in t) < next(x for t, x in lt if ltr["invoice_number"] in t)


@needs_arabic
def test_urdu_native_digits_appear_in_pdf_text():
    inv = generate_invoice(InvoiceSpec(seed=5, locale="ur-PK", native=True, native_digits=True))
    text = pdf_text(render_document(inv).pdf)
    assert re.search("[۰-۹]", text)
    assert Decimal(inv["total"]) > 0 and get_locale("ur-PK").format_number(Decimal(inv["total"])) not in text  # ASCII form is not what was drawn


@needs_arabic
def test_mixed_script_and_mixed_numbers_render_without_missing_glyphs():
    inv = generate_invoice(InvoiceSpec(seed=6, locale="ar", native=True, customer_name="شركة Acme 2025 المحدودة",
                                       lines=[{"description": "خدمة Cloud 24/7", "quantity": "3", "unit_price": "199.99"}]))
    r = render_document(inv)
    assert r.unrenderable == []
    assert "2025" in pdf_text(r.pdf) and "199.99" in pdf_text(r.pdf).replace("٫", ".")


@needs_cjk
def test_chinese_invoice_pdf():
    inv = generate_invoice(InvoiceSpec(seed=1, locale="zh", native=True, n_lines=3))
    r = render_document(inv)
    assert r.embedded_font and r.unrenderable == [] and r.direction == "ltr"
    assert "发票" in pdf_text(r.pdf)


def test_missing_arabic_font_is_a_clean_error(monkeypatch):
    monkeypatch.setattr(fonts, "find_font", lambda script: None)
    monkeypatch.setattr("sdp.documents.render._TTF", {})
    inv = generate_invoice(InvoiceSpec(seed=1, locale="ar", native=True))
    with pytest.raises(MissingFontError, match="arabic"):
        render_document(inv)
    r = DocumentPipeline(sleep=lambda s: None).run([{"locale": "ar", "native": True}, {"locale": "en-US"}])
    assert (r.succeeded, r.failed) == (1, 1) and r.failures[0]["error_type"] == "MissingFontError"
    fb = DocumentPipeline(sleep=lambda s: None, fallback_font="Helvetica").run([{"locale": "ar", "native": True}])
    assert fb.failed == 1 and "cannot render" in fb.failures[0]["message"]  # Helvetica cannot draw Arabic: caught, not silently garbled


# ------------------------------------------------------- other doc types
@pytest.mark.parametrize("locale", ["en-US", "en-GB", "ur-PK", "ar", "hi", "es", "fr", "zh"])
def test_receipts_and_statements_reconcile_and_render_in_every_locale(locale):
    for seed in range(8):
        rc = generate_receipt(ReceiptSpec(seed=seed, locale=locale, payment_method="cash" if seed % 2 else "card", n_lines=1 + seed % 6))
        assert DOC_TYPES["receipt"].reconcile(rc) == [], (locale, seed)
        st = generate_statement(StatementSpec(seed=seed, locale=locale, n_transactions=5 + seed * 3))
        assert DOC_TYPES["statement"].reconcile(st) == [], (locale, seed)
    assert render_document(rc).unrenderable == [] and render_document(st).pages >= 1


def test_receipt_cash_change_and_statement_running_balance():
    rc = generate_receipt(ReceiptSpec(seed=3, locale="en-US", payment_method="cash"))
    assert Decimal(rc["payment"]["tendered"]) >= Decimal(rc["total"]) and \
        Decimal(rc["payment"]["change"]) == Decimal(rc["payment"]["tendered"]) - Decimal(rc["total"])
    card = generate_receipt(ReceiptSpec(seed=3, locale="en-US", payment_method="card"))
    assert card["payment"]["change"] == "0.00"
    st = generate_statement(StatementSpec(seed=1, locale="en-GB", n_transactions=40, opening_balance=Decimal("250.50")))
    assert st["transactions"][0]["balance"] != st["opening_balance"] and st["closing_balance"] == st["transactions"][-1]["balance"]
    bad = json.loads(json.dumps(st))
    bad["transactions"][10]["balance"] = "1.00"
    assert DOC_TYPES["statement"].reconcile(bad)
    bad = json.loads(json.dumps(rc))
    bad["payment"]["change"] = "99.99"
    assert DOC_TYPES["receipt"].reconcile(bad)


@needs_arabic
def test_statement_and_receipt_in_arabic_render_rtl():
    for doc in (generate_statement(StatementSpec(seed=2, locale="ar", native=True, n_transactions=60)),
                generate_receipt(ReceiptSpec(seed=2, locale="ur-PK", native=True))):
        r = render_document(doc)
        assert r.direction == "rtl" and r.unrenderable == [] and r.embedded_font
    assert render_document(generate_statement(StatementSpec(seed=2, locale="en-US", n_transactions=200))).pages > 1


def test_pipeline_handles_mixed_document_types():
    specs = [{"doc_type": "invoice", "seed": 1}, {"doc_type": "receipt", "seed": 1, "locale": "fr"},
             {"doc_type": "statement", "seed": 1, "locale": "es", "n_transactions": 30}, {"doc_type": "contract"},
             {"doc_type": "receipt", "payment_method": "bitcoin"}]
    r = DocumentPipeline(sleep=lambda s: None).run(specs)
    assert (r.succeeded, r.failed) == (3, 2) and {f["stage"] for f in r.failures} == {"validate"}
    assert "unknown doc_type" in r.failures[0]["message"]


def test_labels_cover_the_same_keys_in_every_locale():
    base = set(get_locale("en-US").data["labels"])
    for code in ("en-GB", "ur-PK", "ar", "hi", "es", "fr", "zh"):
        assert set(get_locale(code).data["labels"]) == base, code


# ------------------------------------------------------------------ export
def test_utf8_bom_option_round_trips_urdu():
    obj = {"name": "علی خان", "city": "لاہور"}
    plain, with_bom = json_bytes(obj), json_bytes(obj, bom=True)
    assert not plain.startswith(BOM) and with_bom.startswith(BOM) and with_bom[3:] == plain
    assert "علی خان".encode("utf-8") in plain                       # not \u-escaped
    assert json.loads(with_bom.decode("utf-8-sig")) == obj
    df = pd.DataFrame({"name": ["علی", "王伟", "José"], "n": [1, 2, 3]})
    assert csv_bytes(df, bom=True).startswith(BOM) and not csv_bytes(df).startswith(BOM)
    assert pd.read_csv(io.BytesIO(csv_bytes(df, bom=True)), encoding="utf-8-sig")["name"].tolist() == ["علی", "王伟", "José"]


def test_zip_export_applies_bom_to_every_csv():
    import zipfile
    z = zipfile.ZipFile(io.BytesIO(zip_bytes({"a": pd.DataFrame({"x": ["ع"]}), "b": pd.DataFrame({"y": [1]})}, bom=True, extra={"r.json": b"{}"})))
    assert sorted(z.namelist()) == ["a.csv", "b.csv", "r.json"]
    assert z.read("a.csv").startswith(BOM) and z.read("b.csv").startswith(BOM) and z.read("r.json") == b"{}"
