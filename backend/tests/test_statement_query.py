import csv
import io
import json
from datetime import date, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from sdp.documents import DOC_TYPES, StatementSpec, generate_statement, render_document
from sdp.documents.pipeline import DocumentPipeline
from sdp.documents.statement_query import QuerySyntaxError, apply_filter, parse_statement_query
from sdp.documents.types import statement_csv
from sdp.export import BOM

D = Decimal


@pytest.mark.parametrize("q,expect", [
    ("last 90 days, balance over $500", dict(days=90, min_balance=D("500.01"))),
    ("past 2 weeks and balance at least 1,000", dict(days=14, min_balance=D("1000"))),
    ("last 3 months, balance under Rs 2.5k", dict(days=90, max_balance=D("2499.99"))),
    ("balance between 100 and 900", dict(min_balance=D("100"), max_balance=D("900"))),
    ("debits only over 50", dict(kind="debit", min_amount=D("50.01"))),
    ("credits only", dict(kind="credit")),
    ("category groceries, last 30 days", dict(category="groceries", days=30)),
    ("fuel purchases under 40", dict(category="transport", max_amount=D("39.99"))),
    ("merchant restaurant", dict(merchant="restaurant")),
    ("last month", dict(days=30)),
])
def test_query_parsing(q, expect):
    f = parse_statement_query(q)
    for k, v in expect.items():
        assert getattr(f, k) == v, (q, k, getattr(f, k))
    assert f.unparsed == [], f.unparsed


def test_unparsed_words_and_unknown_category_are_reported():
    assert "banana" in parse_statement_query("last 30 days banana").unparsed
    with pytest.raises(QuerySyntaxError):
        parse_statement_query("category zzz")
    with pytest.raises(ValidationError, match="could not understand"):
        StatementSpec(query="give me pizza")


@pytest.fixture(scope="module")
def full():
    return generate_statement(StatementSpec(seed=3, n_transactions=300, days=365, opening_balance=D("800")))


def test_full_statement_reconciles_and_has_merchants_categories(full):
    assert DOC_TYPES["statement"].reconcile(full) == []
    assert {t["category"] for t in full["transactions"]} >= {"groceries", "income", "bills"}
    assert all((D(t["debit"]) == 0) != (D(t["credit"]) == 0) for t in full["transactions"])
    assert [t["seq"] for t in full["transactions"]] == list(range(300))


@pytest.mark.parametrize("q", ["last 90 days, balance over $500", "debits only over 50", "category groceries", "balance between 100 and 900",
                               "last 30 days credits only", "merchant salary, balance over 0", "last 7 days"])
def test_filtered_statements_match_the_query_and_still_reconcile(q):
    spec = StatementSpec(seed=3, n_transactions=300, days=365, opening_balance=D("800"), query=q)
    doc = generate_statement(spec)
    assert DOC_TYPES["statement"].reconcile(doc) == [], q
    f = parse_statement_query(q)
    assert doc["filter"]["matched"] == len(doc["transactions"]) <= doc["filter"]["of"] == 300
    end = date.fromisoformat(doc["period_end"])
    for t in doc["transactions"]:
        if f.days:
            assert t["date"] >= (end - timedelta(days=f.days)).isoformat()
        if f.min_balance is not None:
            assert D(t["balance"]) >= f.min_balance
        if f.max_balance is not None:
            assert D(t["balance"]) <= f.max_balance
        if f.min_amount is not None:
            assert max(D(t["debit"]), D(t["credit"])) >= f.min_amount
        if f.kind == "debit":
            assert D(t["debit"]) > 0
        if f.kind == "credit":
            assert D(t["credit"]) > 0
        if f.category:
            assert t["category"] == f.category


def test_filter_actually_filters_and_empty_result_is_valid(full):
    kept = apply_filter(full, parse_statement_query("last 30 days, balance over 500"))
    assert 0 < len(kept["transactions"]) < len(full["transactions"])
    empty = apply_filter(full, parse_statement_query("balance over 999999999"))
    assert empty["transactions"] == [] and DOC_TYPES["statement"].reconcile(empty) == [] and empty["filter"]["matched"] == 0


def test_tampering_a_filtered_statement_is_detected(full):
    doc = apply_filter(full, parse_statement_query("category groceries"))
    bad = json.loads(json.dumps(doc))
    consecutive = next(i for i in range(1, len(bad["transactions"])) if bad["transactions"][i]["seq"] == bad["transactions"][i - 1]["seq"] + 1) \
        if any(bad["transactions"][i]["seq"] == bad["transactions"][i - 1]["seq"] + 1 for i in range(1, len(bad["transactions"]))) else None
    bad["transactions"][0]["balance"] = "1.00"
    assert DOC_TYPES["statement"].reconcile(bad)
    if consecutive:
        bad = json.loads(json.dumps(doc))
        bad["transactions"][consecutive]["balance"] = "1.00"
        assert DOC_TYPES["statement"].reconcile(bad)


def test_exports_csv_json_pdf(full):
    doc = apply_filter(full, parse_statement_query("last 60 days"))
    rows = list(csv.DictReader(io.StringIO(statement_csv(doc).decode("utf-8"))))
    assert len(rows) == len(doc["transactions"]) and set(rows[0]) == {"date", "description", "category", "debit", "credit", "balance"}
    assert D(rows[-1]["balance"]) == D(doc["closing_balance"]) and statement_csv(doc, bom=True).startswith(BOM)
    r = render_document(doc)
    assert r.pdf.startswith(b"%PDF") and r.unrenderable == []
    from pypdf import PdfReader
    text = " ".join(p.extract_text() for p in PdfReader(io.BytesIO(r.pdf)).pages)
    assert "last 60 days" in text
    json.dumps(doc)


def test_pipeline_runs_statement_queries_and_localises():
    specs = [{"doc_type": "statement", "seed": 1, "n_transactions": 80, "days": 200, "query": "last 60 days, balance over 100", "locale": "en-GB"},
             {"doc_type": "statement", "seed": 2, "n_transactions": 50, "query": "nonsense words"}]
    r = DocumentPipeline(sleep=lambda s: None).run(specs)
    assert (r.succeeded, r.failed) == (1, 1) and r.failures[0]["stage"] == "validate"
    doc = generate_statement(StatementSpec(seed=1, locale="ur-PK", n_transactions=40, query="debits only"))
    assert doc["currency"] == "PKR" and DOC_TYPES["statement"].reconcile(doc) == []
