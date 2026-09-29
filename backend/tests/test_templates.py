import io
import json
import shutil
from decimal import Decimal

import pytest
from pydantic import ValidationError
from pypdf import PdfReader

from sdp.documents import DOC_TYPES, TEMPLATE_PACKS, register_template_pack, render_document, unregister_template_pack
from sdp.documents.pipeline import DocumentPipeline
from sdp.documents.templating import BUILTIN_DIR, Evaluator, TemplateError, load_pack, reconcile

LOCALES = ["en-US", "en-GB", "ur-PK", "ar", "hi", "es", "fr", "zh"]
TYPES = ["payslip", "retail_receipt", "purchase_order"]
D = Decimal


def make(t, **kw):
    dt = DOC_TYPES[t]
    return dt, dt.generate(dt.spec_cls(**kw))


def text_of(pdf: bytes) -> str:
    return "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf)).pages)


# ------------------------------------------------------------- the registry
def test_builtin_packs_are_registered_as_document_types():
    assert set(TYPES) <= set(TEMPLATE_PACKS) and set(TYPES) <= set(DOC_TYPES)
    assert {"invoice", "receipt", "statement"} <= set(DOC_TYPES)               # built-ins untouched
    assert all(DOC_TYPES[t].engine == "html" and DOC_TYPES[t].title for t in TYPES) and DOC_TYPES["invoice"].engine == "reportlab"


@pytest.mark.parametrize("t", TYPES)
@pytest.mark.parametrize("locale", LOCALES)
def test_every_type_reconciles_and_renders_in_every_locale(t, locale):
    for seed in range(4):
        dt, doc = make(t, seed=seed, locale=locale)
        assert dt.reconcile(doc) == [], (t, locale, seed)
        r = render_document(doc)
        assert r.pdf.startswith(b"%PDF") and r.pages >= 1 and r.unrenderable == [], (t, locale, seed, r.unrenderable)
    json.dumps(doc)


@pytest.mark.parametrize("t", TYPES)
def test_generation_and_pdf_are_deterministic(t):
    _, a = make(t, seed=7, locale="fr")
    _, b = make(t, seed=7, locale="fr")
    assert a == b and render_document(a).pdf == render_document(b).pdf
    assert make(t, seed=8, locale="fr")[1] != a


def test_pdf_contains_the_documents_own_values():
    _, ps = make("payslip", seed=2, locale="en-GB")
    txt = text_of(render_document(ps).pdf)
    assert ps["employee"]["name"] in txt and ps["pay_period"]["label"] in txt and "Gross pay" in txt
    assert f"{D(ps['net_pay']):,.2f}" in txt
    _, po = make("purchase_order", seed=2, locale="es")
    txt = text_of(render_document(po).pdf)
    assert po["po_number"] in txt and po["supplier"]["name"] in txt and "IVA" in txt


# ---------------------------------------------------------------- arithmetic
def test_payslip_arithmetic_is_exact_decimal():
    for seed in range(20):
        _, ps = make("payslip", seed=seed, locale="ur-PK")
        gross = sum(D(r["amount"]) for r in ps["earnings"])
        ded = sum(D(r["amount"]) for r in ps["deductions"])
        assert D(ps["gross"]) == gross and D(ps["total_deductions"]) == ded and D(ps["net_pay"]) == gross - ded > 0
        base = next(r for r in ps["earnings"] if r["description"] == "Basic salary")
        assert D(base["amount"]) == D(ps["base_salary"])
        ss = next(r for r in ps["deductions"] if r["description"].startswith("Social"))
        assert D(ss["amount"]) == (gross * D("0.06")).quantize(D("0.01"), rounding="ROUND_HALF_UP")


def test_progressive_income_tax_brackets():
    _, ps = make("payslip", seed=1, locale="en-US")
    ev = Evaluator(2, D("1000"), D("0.07"))
    assert ev.eval("bracket(gross, [[0, 0], [0.5, 0.05], [1.2, 0.15], [2.5, 0.25]])", {"gross": D("400")}) == D("0.00")
    assert ev.eval("bracket(gross, [[0, 0], [0.5, 0.05], [1.2, 0.15], [2.5, 0.25]])", {"gross": D("1000")}) == D("25.00")           # 500 x 5%
    assert ev.eval("bracket(gross, [[0, 0], [0.5, 0.05], [1.2, 0.15], [2.5, 0.25]])", {"gross": D("3000")}) == D("355.00")   # 700x5% + 1300x15% + 500x25%
    tax = next(r for r in ps["deductions"] if r["description"] == "Income tax")
    assert D(tax["amount"]) >= 0


def test_receipt_and_po_arithmetic_and_rounding():
    for seed in range(20):
        _, rc = make("retail_receipt", seed=seed, locale="en-US", region="CA", n_lines=5)
        assert len(rc["items"]) == 5
        sub = sum(D(i["line_total"]) for i in rc["items"])
        assert all(D(i["line_total"]) == (D(i["quantity"]) * D(i["unit_price"])).quantize(D("0.01"), rounding="ROUND_HALF_UP") for i in rc["items"])
        assert D(rc["subtotal"]) == sub and D(rc["tax"]) == (sub * D("0.0725")).quantize(D("0.01"), rounding="ROUND_HALF_UP")
        assert D(rc["total"]) == sub + D(rc["tax"]) and D(rc["change"]) == D(rc["tendered"]) - D(rc["total"]) >= 0
        if rc["payment"] == "card":
            assert rc["change"] == "0.00"
        _, po = make("purchase_order", seed=seed, locale="en-GB")
        assert po["delivery_date"] >= po["order_date"]
        taxable = D(po["subtotal"]) - D(po["discount"])
        assert D(po["taxable"]) == taxable and D(po["total"]) == taxable + D(po["tax"]) + D(po["shipping"])


def test_reconcile_catches_tampering():
    dt, ps = make("payslip", seed=3)
    bad = json.loads(json.dumps(ps))
    bad["net_pay"] = str(D(bad["net_pay"]) + D("0.01"))
    assert any("net_pay" in e or "net pay" in e for e in dt.reconcile(bad))
    bad = json.loads(json.dumps(ps))
    bad["deductions"][1]["amount"] = "1.00"                               # social security no longer 6% of gross
    assert any("Social security" in e for e in dt.reconcile(bad))
    dt, po = make("purchase_order", seed=3)
    bad = json.loads(json.dumps(po))
    bad["delivery_date"] = "2000-01-01"
    assert any("delivery" in e for e in dt.reconcile(bad))
    bad = json.loads(json.dumps(po))
    del bad["subtotal"]
    assert dt.reconcile(bad)


# --------------------------------------------------------------- specs/overrides
def test_params_and_overrides():
    _, rc = make("retail_receipt", seed=1, n_lines=12)
    assert len(rc["items"]) == 12
    _, ps = make("payslip", seed=1, overrides={"employee.name": "Test Person", "base_salary": "5000"})
    assert ps["employee"]["name"] == "Test Person" and ps["base_salary"] == "5000"
    assert next(r for r in ps["earnings"] if r["description"] == "Basic salary")["amount"] == "5000.00" or D(ps["gross"]) >= 5000
    assert DOC_TYPES["payslip"].reconcile(ps) == []
    spec = DOC_TYPES["payslip"].spec_cls
    for bad in ({"overrides": {"earnings": []}}, {"overrides": {"nope": 1}}, {"overrides": {"employee.name": "x" * 100}}, {"locale": "xx"}, {"unknown": 1}):
        with pytest.raises(ValidationError):
            spec(**bad)
    with pytest.raises(ValidationError):
        DOC_TYPES["retail_receipt"].spec_cls(n_lines=0)


def test_extreme_override_values_still_reconcile_or_fail_cleanly():
    _, ps = make("payslip", seed=1, overrides={"base_salary": "999999999999999"})
    assert DOC_TYPES["payslip"].reconcile(ps) == [] and D(ps["net_pay"]) > 0
    r = DocumentPipeline(sleep=lambda s: None).run([{"doc_type": "payslip", "overrides": {"base_salary": "999999999999999"}}, {"doc_type": "payslip", "seed": 1}])
    assert r.failed + r.succeeded == 2 and r.by_stage.get("unexpected", 0) == 0


# ------------------------------------------------------------------ security
def test_html_injection_is_escaped_and_templates_are_sandboxed():
    _, rc = make("retail_receipt", seed=1, overrides={"cashier": "<script>alert(1)</script><b>x</b>"})
    from sdp.documents.templating import render_html
    html = render_html(TEMPLATE_PACKS["retail_receipt"], rc)
    assert "<script>" not in html and "&lt;script&gt;" in html
    pack = TEMPLATE_PACKS["payslip"]
    from dataclasses import replace
    evil = replace(pack, html="{{ ''.__class__.__mro__[1].__subclasses__() }}")
    with pytest.raises(Exception):
        render_html(evil, make("payslip", seed=1)[1])


def test_expressions_are_a_safe_subset():
    ev = Evaluator(2, D("100"), D("0.1"))
    assert ev.eval("1 + 2 * 3", {}) == D(7) and ev.eval("pct(200, 0.075)", {}) == D("15.00") and ev.eval("round(1.005)", {}) == D("1.01")
    for bad in ("__import__('os')", "open('x')", "().__class__", "[x for x in y]", "lambda: 1", "a.b.c", "sum(1)", "rand(1, 2)"):
        with pytest.raises((TemplateError, TypeError, KeyError, AttributeError)):
            ev.eval(bad, {"a": {}, "y": []})


# ------------------------------------------------- adding a type = adding files
def test_a_new_document_type_needs_no_engine_change(tmp_path):
    """Drop a folder with type.json + template.html: generation, reconciliation, PDF, batch pipeline all work."""
    folder = tmp_path / "warranty"
    folder.mkdir()
    (folder / "type.json").write_text(json.dumps({
        "name": "warranty_card", "title": "Warranty card", "required": ["owner.name", "warranty_no", "price"],
        "build": [
            {"name": "owner", "kind": "person"},
            {"name": "warranty_no", "kind": "id", "prefix": "W-", "digits": 6},
            {"name": "purchased", "kind": "date", "start": "2024-01-01", "end": "2024-12-31"},
            {"name": "months", "kind": "choice", "type": "int", "values": ["12", "24", "36"]},
            {"name": "expires", "kind": "expr", "type": "date", "expr": "date_add(purchased, months * 30)"},
            {"name": "price", "kind": "money", "min": 50, "max": 900},
            {"name": "fee", "kind": "computed", "expr": "pct(price, 0.02)"}],
        "rules": [{"name": "expires after purchase", "expr": "expires > purchased"}, {"name": "fee is 2%", "expr": "fee == pct(price, 0.02)"}],
    }), encoding="utf-8")
    (folder / "template.html").write_text("<html><body><h1>Warranty {{ doc.warranty_no }}</h1><p>{{ doc.owner.name }}</p>"
                                          "<p>Price {{ doc.price | money }}, fee {{ doc.fee | money }}, valid to {{ doc.expires | date }}</p></body></html>", encoding="utf-8")
    pack = load_pack(folder)
    try:
        register_template_pack(pack)
        dt = DOC_TYPES["warranty_card"]
        doc = dt.generate(dt.spec_cls(seed=1, locale="fr"))
        assert dt.reconcile(doc) == [] and doc["expires"] > doc["purchased"]
        r = render_document(doc)
        assert doc["warranty_no"] in text_of(r.pdf)
        rep = DocumentPipeline(sleep=lambda s: None).run([{"doc_type": "warranty_card", "seed": i, "locale": "es"} for i in range(6)])
        assert (rep.succeeded, rep.failed) == (6, 0)
        bad = json.loads(json.dumps(doc))
        bad["fee"] = "9999.00"
        assert any("fee" in e for e in dt.reconcile(bad))
    finally:
        unregister_template_pack("warranty_card")
    assert "warranty_card" not in DOC_TYPES


def test_broken_packs_fail_at_load_time(tmp_path):
    def pack(meta, html="<html></html>"):
        d = tmp_path / f"p{len(list(tmp_path.iterdir()))}"
        d.mkdir()
        (d / "type.json").write_text(json.dumps(meta), encoding="utf-8")
        (d / "template.html").write_text(html, encoding="utf-8")
        return d
    ok = {"name": "x", "title": "X", "build": [{"name": "a", "kind": "int", "min": 1, "max": 2}]}
    load_pack(pack(ok))
    for meta, html in [({"name": "x"}, "<p/>"), ({**ok, "build": [{"name": "a", "kind": "wat"}]}, "<p/>"),
                       ({**ok, "build": [{"name": "a", "kind": "computed", "expr": "1 +"}]}, "<p/>"),
                       ({**ok, "build": [ok["build"][0], ok["build"][0]]}, "<p/>"), ({**ok, "rules": [{"expr": "(("}]}, "<p/>"), (ok, "{% for %}")]:
        with pytest.raises(TemplateError):
            load_pack(pack(meta, html))
    with pytest.raises(TemplateError):
        load_pack(tmp_path / "does-not-exist")
    clash = load_pack(pack({**ok, "name": "invoice"}))
    with pytest.raises(TemplateError, match="clashes"):
        register_template_pack(clash)


def test_pipeline_batches_mixed_builtin_and_template_types():
    specs = [{"doc_type": t, "seed": i, "locale": LOCALES[i % 8]} for i, t in enumerate(TYPES * 6)] + [{"doc_type": "invoice", "seed": 1}, {"doc_type": "payslip", "locale": "nope"}]
    r = DocumentPipeline(sleep=lambda s: None).run(specs)
    assert (r.succeeded, r.failed) == (19, 1) and r.failures[0]["stage"] == "validate"
