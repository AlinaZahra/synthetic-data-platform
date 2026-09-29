"""Document layouts: one generic structure (Layout) that any document type fills in, plus locale-aware formatting.

The same Layout drives both the PDF renderer and the HTML preview in the UI, so what you preview is what you print.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Callable

from sdp.locale import get_locale

EN_LABELS = {
    "invoice_title": "INVOICE", "receipt_title": "RECEIPT", "statement_title": "ACCOUNT STATEMENT", "bill_to": "Bill to",
    "issued": "Issued", "due": "Due", "page": "Page", "description": "Description", "qty": "Qty", "unit_price": "Unit price",
    "amount": "Amount", "subtotal": "Subtotal", "total": "Total", "tax": "Tax", "prices_include_tax": "Prices include tax.",
    "currency": "Currency", "seller_tax_id": "Tax ID", "payment_method": "Payment method", "date": "Date", "balance": "Balance",
    "debit": "Debit", "credit": "Credit", "opening_balance": "Opening balance", "closing_balance": "Closing balance",
    "account": "Account", "thank_you": "Thank you", "tendered": "Tendered", "change": "Change", "period": "Period",
    "cash": "Cash", "card": "Card", "filter": "Filter",
}


@dataclass
class Col:
    label: str
    width: float           # fraction of table width (sums to 1)
    align: str = "start"   # start | end


@dataclass
class Layout:
    doc_type: str
    direction: str                     # ltr | rtl
    title: str
    number: str
    header_start: list[tuple[str, bool]]   # (text, bold), on the reading-start side
    header_end: list[str]                  # on the reading-end side
    block_label: str
    block_lines: list[str]
    columns: list[Col]
    rows: list[list[str]]
    totals: list[tuple[str, str, bool]]    # (label, formatted amount, emphasised)
    notes: list[str] = field(default_factory=list)
    page_label: str = "Page"
    script: str = "latin"
    keys: dict[str, Any] = field(default_factory=dict)    # semantic field keys (see _attach_keys)
    raw: dict[str, Any] = field(default_factory=dict)     # key -> ground-truth value from the document

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_type": self.doc_type, "direction": self.direction, "title": self.title, "number": self.number,
            "header_start": [{"text": t, "bold": b} for t, b in self.header_start], "header_end": self.header_end,
            "block_label": self.block_label, "block_lines": self.block_lines,
            "columns": [{"label": c.label, "width": c.width, "align": c.align} for c in self.columns],
            "rows": self.rows, "totals": [{"label": a, "value": b, "strong": s} for a, b, s in self.totals],
            "notes": self.notes, "script": self.script,
        }


class Fmt:
    """Formats labels, numbers, money and dates for one document, given the font's capabilities."""

    def __init__(self, locale: str, native: bool = False, native_digits: bool = False,
                 supports: Callable[[str], bool] | None = None) -> None:
        self.pack = get_locale(locale)
        pdf = self.pack.data.get("pdf", {})
        self.native = native and not pdf.get("complex_shaping", False)
        self.direction = pdf.get("direction", "ltr") if self.native else "ltr"
        self.script = pdf.get("font_script", "latin") if self.native else "latin"
        self.digits = self.pack.data.get("native_digits") if native_digits and self.native else None
        self.supports = supports or (lambda text: True)
        self.labels = {**EN_LABELS, **(self.pack.data.get("labels", {}) if self.native else {})}

    def label(self, key: str) -> str:
        return self.labels.get(key, EN_LABELS.get(key, key))

    def tax_label(self, label: str) -> str:
        if self.native and self.pack.data["tax"].get("label_native"):
            return self.pack.data["tax"]["label_native"]
        return label

    def localize(self, text: str) -> str:
        if not self.digits:
            return text
        table = {ord(str(i)): d for i, d in enumerate(self.digits["digits"])}
        table[ord(".")] = self.digits["decimal"]
        table[ord(",")] = self.digits["thousands"]
        return text.translate(table)

    def num(self, value: str | Decimal, decimals: int | None = None) -> str:
        return self.localize(self.pack.format_number(Decimal(value), decimals))

    def money(self, value: str | Decimal) -> str:
        text =self.pack.format_currency(Decimal(value)).replace(" ", " ")
        if not self.supports(text):  # e.g. INR sign in a WinAnsi font: fall back to the ISO code
            text = f"{self.pack.data['currency']['code']} {self.pack.format_number(Decimal(value)).replace(chr(0x202f), chr(0xa0))}"
        return self.localize(text)

    def date(self, iso: str) -> str:
        return self.localize(self.pack.format_date(date.fromisoformat(iso)))

    def qty(self, s: str) -> str:
        d = Decimal(s)
        return self.localize(format(d.normalize(), "f") if d == d.to_integral() else format(d, "f"))

    def pct(self, rate: str) -> str:
        return self.localize(format((Decimal(rate) * 100).normalize(), "f") + "%")


# ------------------------------------------------------------ builders
def invoice_layout(inv: dict[str, Any], f: Fmt) -> Layout:
    rows = [[ln["description"], f.qty(ln["quantity"]), f.money(ln["unit_price"]), f.money(ln["line_total"])] for ln in inv["lines"]]
    totals = [(f.label("subtotal"), f.money(inv["subtotal"]), False)]
    totals += [(f"{f.tax_label(s['tax_label'])} {f.pct(s['tax_rate'])}", f.money(s["tax"]), False) for s in inv["tax_summary"]]
    totals.append((f.label("total"), f.money(inv["total"]), True))
    return Layout(
        "invoice", f.direction, f.label("invoice_title"), inv["invoice_number"],
        [(inv["seller"]["name"], True), (inv["seller"]["address"], False), (f"{f.label('seller_tax_id')}: {inv['seller']['tax_id']}", False)],
        [f"{f.label('issued')} {f.date(inv['issue_date'])}", f"{f.label('due')} {f.date(inv['due_date'])}", f"{f.label('currency')}: {inv['currency']}"],
        f.label("bill_to"), [inv["customer"]["name"], inv["customer"]["address"]],
        [Col(f.label("description"), 0.46), Col(f.label("qty"), 0.12, "end"), Col(f.label("unit_price"), 0.21, "end"), Col(f.label("amount"), 0.21, "end")],
        rows, totals, [f.label("prices_include_tax")] if inv["tax_inclusive"] else [], f.label("page"), f.script)


def receipt_layout(doc: dict[str, Any], f: Fmt) -> Layout:
    rows = [[ln["description"], f.qty(ln["quantity"]), f.money(ln["line_total"])] for ln in doc["lines"]]
    totals = [(f.label("subtotal"), f.money(doc["subtotal"]), False)]
    totals += [(f"{f.tax_label(s['tax_label'])} {f.pct(s['tax_rate'])}", f.money(s["tax"]), False) for s in doc["tax_summary"]]
    totals += [(f.label("total"), f.money(doc["total"]), True), (f.label("tendered"), f.money(doc["payment"]["tendered"]), False),
               (f.label("change"), f.money(doc["payment"]["change"]), False)]
    method = f.label(doc["payment"]["method"])
    return Layout(
        "receipt", f.direction, f.label("receipt_title"), doc["receipt_number"],
        [(doc["merchant"]["name"], True), (doc["merchant"]["address"], False), (f"{f.label('seller_tax_id')}: {doc['merchant']['tax_id']}", False)],
        [f.date(doc["issue_date"]), f"{f.label('payment_method')}: {method}"],
        "", [], [Col(f.label("description"), 0.62), Col(f.label("qty"), 0.12, "end"), Col(f.label("amount"), 0.26, "end")],
        rows, totals, [f.label("prices_include_tax"), f.label("thank_you")], f.label("page"), f.script)


def statement_layout(doc: dict[str, Any], f: Fmt) -> Layout:
    rows = [[f.date(t["date"]), t["description"], f.money(t["debit"]) if t["debit"] != "0.00" else "",
             f.money(t["credit"]) if t["credit"] != "0.00" else "", f.money(t["balance"])] for t in doc["transactions"]]
    totals = [(f.label("opening_balance"), f.money(doc["opening_balance"]), False),
              (f.label("debit"), f.money(doc["total_debits"]), False), (f.label("credit"), f.money(doc["total_credits"]), False),
              (f.label("closing_balance"), f.money(doc["closing_balance"]), True)]
    return Layout(
        "statement", f.direction, f.label("statement_title"), doc["statement_number"],
        [(doc["bank"]["name"], True), (doc["bank"]["address"], False)],
        [f"{f.label('period')}: {f.date(doc['period_start'])} - {f.date(doc['period_end'])}", f"{f.label('currency')}: {doc['currency']}"],
        f.label("account"), [doc["holder"]["name"], f"{doc['holder']['address']}", f"{f.label('account')}: {f.localize(doc['account_number'])}"],
        [Col(f.label("date"), 0.16), Col(f.label("description"), 0.30), Col(f.label("debit"), 0.18, "end"), Col(f.label("credit"), 0.18, "end"),
         Col(f.label("balance"), 0.18, "end")],
        rows, totals, ([f"{f.label('filter')}: {doc['filter']['query']} ({doc['filter']['matched']}/{doc['filter']['of']})"] if doc.get("filter") else []),
        f.label("page"), f.script)


LAYOUTS: dict[str, Callable[[dict[str, Any], Fmt], Layout]] = {
    "invoice": invoice_layout, "receipt": receipt_layout, "statement": statement_layout,
}


# ------------------------------------------------------- field keys / ground truth
def _attach_keys(lay: Layout, doc: dict[str, Any]) -> Layout:
    """Semantic key for every drawn string + the raw (unformatted) ground-truth value where one exists.

    keys: header_start / header_end / block / columns / totals -> lists of keys; raw: key -> value straight from the document.
    Table cells are keyed `row{r}.{column_key}`; a total's label is `{key}.label`.
    """
    t = lay.doc_type
    raw: dict[str, Any] = {}
    if t == "invoice":
        keys = {"header_start": ["seller_name", "seller_address", "seller_tax_id"], "header_end": ["issue_date", "due_date", "currency"],
                "number": "invoice_number", "block": ["customer_name", "customer_address"],
                "columns": ["description", "quantity", "unit_price", "line_total"]}
        raw.update(seller_name=doc["seller"]["name"], seller_address=doc["seller"]["address"], seller_tax_id=doc["seller"]["tax_id"],
                   issue_date=doc["issue_date"], due_date=doc["due_date"], currency=doc["currency"], invoice_number=doc["invoice_number"],
                   customer_name=doc["customer"]["name"], customer_address=doc["customer"]["address"], subtotal=doc["subtotal"], total=doc["total"])
        rows = doc["lines"]
        tot = ["subtotal"] + [f"tax_{i}" for i in range(len(doc["tax_summary"]))] + ["total"]
        for i, s in enumerate(doc["tax_summary"]):
            raw[f"tax_{i}"] = s["tax"]
    elif t == "receipt":
        keys = {"header_start": ["merchant_name", "merchant_address", "merchant_tax_id"], "header_end": ["issue_date", "payment_method"],
                "number": "receipt_number", "block": [], "columns": ["description", "quantity", "line_total"]}
        raw.update(merchant_name=doc["merchant"]["name"], merchant_address=doc["merchant"]["address"], merchant_tax_id=doc["merchant"]["tax_id"],
                   issue_date=doc["issue_date"], payment_method=doc["payment"]["method"], receipt_number=doc["receipt_number"],
                   subtotal=doc["subtotal"], total=doc["total"], tendered=doc["payment"]["tendered"], change=doc["payment"]["change"])
        rows = doc["lines"]
        tot = ["subtotal"] + [f"tax_{i}" for i in range(len(doc["tax_summary"]))] + ["total", "tendered", "change"]
        for i, s in enumerate(doc["tax_summary"]):
            raw[f"tax_{i}"] = s["tax"]
    else:
        keys = {"header_start": ["bank_name", "bank_address"], "header_end": ["period", "currency"], "number": "statement_number",
                "block": ["holder_name", "holder_address", "account_number"], "columns": ["date", "description", "debit", "credit", "balance"]}
        raw.update(bank_name=doc["bank"]["name"], bank_address=doc["bank"]["address"], currency=doc["currency"], statement_number=doc["statement_number"],
                   holder_name=doc["holder"]["name"], holder_address=doc["holder"]["address"], account_number=doc["account_number"],
                   opening_balance=doc["opening_balance"], total_debits=doc["total_debits"], total_credits=doc["total_credits"],
                   closing_balance=doc["closing_balance"])
        rows = [{**x, "debit": "" if x["debit"] == "0.00" else x["debit"], "credit": "" if x["credit"] == "0.00" else x["credit"]} for x in doc["transactions"]]
        tot = ["opening_balance", "total_debits", "total_credits", "closing_balance"]
    for r, row in enumerate(rows):
        for ck in keys["columns"]:
            if ck in row:
                raw[f"row{r}.{ck}"] = row[ck]
    lay.keys = {**keys, "totals": tot}
    lay.raw = raw
    return lay


def _with_keys(fn):
    def wrapper(doc: dict[str, Any], f: Fmt) -> Layout:
        return _attach_keys(fn(doc, f), doc)
    return wrapper


LAYOUTS = {k: _with_keys(v) for k, v in LAYOUTS.items()}
