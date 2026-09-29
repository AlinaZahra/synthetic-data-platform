"""Document types: invoice, receipt, bank statement. Each = spec model + generator + independent reconciler.

All amounts are Decimal, serialised as strings. Adding a type = a spec, a generator, a reconciler and a layout builder.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from sdp.documents.invoice import InvoiceSpec, _s, generate_invoice, quantize, reconcile
from sdp.documents.layout import LAYOUTS
from sdp.documents.statement_query import MERCHANTS, QuerySyntaxError, apply_filter, parse_statement_query
from sdp.locale import LocaleError, get_locale


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = 0
    locale: str = "en-US"
    region: str | None = None
    font: str = "Helvetica"
    native: bool = False
    native_digits: bool = False

    @field_validator("locale")
    @classmethod
    def _known_locale(cls, v: str) -> str:
        try:
            return get_locale(v).code
        except LocaleError as e:
            raise ValueError(str(e)) from e

    def doc_id(self) -> str:
        return hashlib.sha256((type(self).__name__ + self.model_dump_json()).encode()).hexdigest()[:16]


# ----------------------------------------------------------------- receipt
class ReceiptSpec(_Base):
    n_lines: int = Field(4, ge=1, le=60)
    payment_method: str = Field("card", pattern="^(cash|card)$")


def generate_receipt(spec: ReceiptSpec) -> dict[str, Any]:
    pack = get_locale(spec.locale)
    dp = pack.data["currency"]["decimals"]
    inv = generate_invoice(InvoiceSpec(seed=spec.seed, locale=spec.locale, region=spec.region, n_lines=spec.n_lines,
                                       tax_inclusive=True, native=spec.native, native_digits=spec.native_digits, font=spec.font))
    total = Decimal(inv["total"])
    if spec.payment_method == "cash":
        step = Decimal(5) if total < 100 else Decimal(50)
        tendered = quantize((total / step).to_integral_value(rounding="ROUND_CEILING") * step, dp)
    else:
        tendered = total
    return {
        "doc_type": "receipt", "receipt_number": "RCP-" + spec.doc_id()[:10].upper(), "issue_date": inv["issue_date"],
        "locale": inv["locale"], "region": inv["region"], "currency": inv["currency"], "tax_inclusive": True,
        "merchant": inv["seller"], "lines": inv["lines"], "tax_summary": inv["tax_summary"], "subtotal": inv["subtotal"],
        "tax_total": inv["tax_total"], "total": inv["total"],
        "payment": {"method": spec.payment_method, "tendered": _s(tendered), "change": _s(tendered - total)},
        "presentation": inv["presentation"],
    }


def reconcile_receipt(doc: dict[str, Any]) -> list[str]:
    errs = reconcile({**doc, "doc_type": "invoice"})
    p = doc["payment"]
    if Decimal(p["tendered"]) - Decimal(doc["total"]) != Decimal(p["change"]) or Decimal(p["change"]) < 0:
        errs.append("change != tendered - total")
    return errs


# --------------------------------------------------------------- statement
class StatementSpec(_Base):
    n_transactions: int = Field(20, ge=1, le=5000)
    days: int = Field(30, ge=1, le=730)
    opening_balance: Decimal = Field(Decimal("1000.00"), ge=0, le=Decimal("1e12"), decimal_places=2)
    query: str | None = Field(None, max_length=300)   # e.g. "last 90 days, balance over 500"

    @field_validator("query")
    @classmethod
    def _query_parses(cls, v: str | None) -> str | None:
        if v:
            try:
                f = parse_statement_query(v)
            except QuerySyntaxError as e:
                raise ValueError(str(e)) from e
            if f.unparsed:
                raise ValueError(f"could not understand: {' '.join(f.unparsed)}")
        return v


def generate_statement(spec: StatementSpec) -> dict[str, Any]:
    """Merchants, debits/credits and a running balance that always reconciles; optionally filtered by a query."""
    pack = get_locale(spec.locale)
    dp = pack.data["currency"]["decimals"]
    rng = np.random.default_rng(spec.seed)
    factor = Decimal(pack.data["scale"]["price_factor"])
    start = date(2025, 1, 1) + timedelta(days=int(rng.integers(0, 300)))
    days = sorted(int(x) for x in rng.integers(0, spec.days, spec.n_transactions))
    bal = spec.opening_balance
    txs, debits, credits = [], Decimal(0), Decimal(0)
    zero = quantize(Decimal(0), dp)
    names = list(MERCHANTS)
    for i, d in enumerate(days):
        desc = names[int(rng.integers(0, len(names)))]
        cat, is_credit = MERCHANTS[desc]
        boost = {"Salary": 6, "Transfer received": 4}.get(desc, 1)  # keeps balances hovering at a realistic level
        amt = quantize(Decimal(str(round(float(rng.lognormal(3.4, 0.9)), 2))) * factor * boost, dp)
        deb, cred = (zero, amt) if is_credit else (amt, zero)
        bal = bal + cred - deb
        debits, credits = debits + deb, credits + cred
        txs.append({"seq": i, "date": (start + timedelta(days=d)).isoformat(), "description": desc, "category": cat,
                    "debit": _s(deb), "credit": _s(cred), "balance": _s(bal)})
    a1, _ = pack.address(rng, latin=not spec.native)
    a2, _ = pack.address(rng, latin=not spec.native)
    presentation = generate_invoice(InvoiceSpec(seed=spec.seed, locale=spec.locale, n_lines=1, native=spec.native,
                                                native_digits=spec.native_digits, font=spec.font))["presentation"]
    native = presentation["native"]
    doc = {
        "doc_type": "statement", "statement_number": "STM-" + spec.doc_id()[:10].upper(), "locale": pack.code, "currency": pack.data["currency"]["code"],
        "period_start": start.isoformat(), "period_end": (start + timedelta(days=spec.days)).isoformat(),
        "bank": {"name": f"{pack.person_name(rng, latin=True).split()[-1]} Bank", "address": a1},
        "holder": {"name": pack.person_name(rng, latin=not native), "address": a2},
        "account_number": f"****{int(rng.integers(1000, 9999))}",
        "opening_balance": _s(spec.opening_balance.quantize(zero)), "transactions": txs,
        "total_debits": _s(debits), "total_credits": _s(credits), "closing_balance": _s(bal), "presentation": presentation,
    }
    if spec.query:
        doc = apply_filter(doc, parse_statement_query(spec.query))
    return doc


def reconcile_statement(doc: dict[str, Any]) -> list[str]:
    """Every shown row must follow from the previous *original* row (seq is consecutive), the first shown row from the
    opening balance, and the totals/closing balance from the shown rows. A filtered statement reconciles the same way."""
    D = Decimal
    errs, deb, cred = [], D(0), D(0)
    prev_bal, prev_seq = D(doc["opening_balance"]), None
    for i, t in enumerate(doc["transactions"]):
        seq = t.get("seq", i)
        if prev_seq is None or seq == prev_seq + 1:
            if D(t["balance"]) != prev_bal + D(t["credit"]) - D(t["debit"]):
                errs.append(f"row {i}: balance {t['balance']} != {prev_bal + D(t['credit']) - D(t['debit'])}")
        deb, cred = deb + D(t["debit"]), cred + D(t["credit"])
        prev_bal, prev_seq = D(t["balance"]), seq
    if D(doc["total_debits"]) != deb or D(doc["total_credits"]) != cred:
        errs.append("debit/credit totals do not match transactions")
    if doc["transactions"] and D(doc["closing_balance"]) != prev_bal:
        errs.append(f"closing balance {doc['closing_balance']} != {prev_bal}")
    return errs


def statement_csv(doc: dict[str, Any], bom: bool = False) -> bytes:
    """Machine-readable transactions: ASCII decimals, ISO dates, UTF-8 (BOM optional)."""
    import pandas as pd
    from sdp.export import csv_bytes
    cols = ["date", "description", "category", "debit", "credit", "balance"]
    return csv_bytes(pd.DataFrame([{c: t[c] for c in cols} for t in doc["transactions"]], columns=cols), bom)


# ---------------------------------------------------------------- registry
@dataclass(frozen=True)
class DocType:
    name: str
    spec_cls: type[BaseModel]
    generate: Callable[[Any], dict[str, Any]]
    reconcile: Callable[[dict[str, Any]], list[str]]
    required: tuple[str, ...]
    render: Callable[..., Any] | None = None     # template types render themselves (HTML -> PDF); built-ins use the layout engine
    html: Callable[[dict[str, Any]], str] | None = None
    title: str = ""
    engine: str = "reportlab"


DOC_TYPES: dict[str, DocType] = {
    "invoice": DocType("invoice", InvoiceSpec, generate_invoice, reconcile,
                       ("invoice_number", "issue_date", "due_date", "currency", "total", "subtotal", "tax_total")),
    "receipt": DocType("receipt", ReceiptSpec, generate_receipt, reconcile_receipt,
                       ("receipt_number", "issue_date", "currency", "total", "payment")),
    "statement": DocType("statement", StatementSpec, generate_statement, reconcile_statement,
                         ("statement_number", "period_start", "currency", "closing_balance", "opening_balance")),
}
assert set(DOC_TYPES) == set(LAYOUTS)


# ------------------------------------------------------------ template packs
# Every folder in documents/templates/ (and $SDP_TEMPLATE_DIR) becomes a document type with no engine change.
from sdp.documents.templating import TemplateError, TemplatePack, discover, load_pack, make_doc_type  # noqa: E402

TEMPLATE_PACKS: dict[str, TemplatePack] = {}


def register_template_pack(pack: TemplatePack) -> DocType:
    if pack.name in DOC_TYPES and pack.name not in TEMPLATE_PACKS:
        raise TemplateError(f"template type {pack.name!r} clashes with a built-in document type")
    spec, gen, rec, render, html, required = make_doc_type(pack)
    DOC_TYPES[pack.name] = DocType(pack.name, spec, gen, rec, required, render=render, html=html, title=pack.title, engine="html")
    TEMPLATE_PACKS[pack.name] = pack
    return DOC_TYPES[pack.name]


def unregister_template_pack(name: str) -> None:
    TEMPLATE_PACKS.pop(name, None)
    DOC_TYPES.pop(name, None)


for _pack in discover().values():
    register_template_pack(_pack)
