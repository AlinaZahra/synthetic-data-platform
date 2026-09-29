"""D1. Invoice generator. All money is `decimal.Decimal` (never float); totals reconcile exactly.

Rounding rule (documented and tested): every monetary amount is rounded HALF_UP to the currency's decimals.
  tax-exclusive prices: line = round(qty * unit_price); tax is computed ONCE PER TAX RATE on the sum of
                        that rate's lines (not per line), total = subtotal + sum(tax)
  tax-inclusive prices: line = round(qty * unit_price) is gross; tax = round(G - G/(1+r)) per rate group,
                        net = G - tax, so subtotal + tax == total exactly
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal, localcontext
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sdp.locale import LocaleError, get_locale

CATALOG = json.loads((Path(__file__).parent / "catalog.json").read_text(encoding="utf-8"))
MAX_UNIT_PRICE = Decimal("1e10")
MAX_QTY = Decimal("1e9")


def _s(d: Decimal) -> str:
    return format(d, "f")


def quantize(value: Decimal, decimals: int) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = 60
        return value.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)


class LineSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = Field(min_length=1, max_length=200)
    quantity: Decimal = Field(gt=0, le=MAX_QTY, decimal_places=3)
    unit_price: Decimal = Field(ge=0, le=MAX_UNIT_PRICE, decimal_places=4)
    category: str = "general"


class InvoiceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = 0
    locale: str = "en-US"
    region: str | None = None
    lines: list[LineSpec] | None = None
    n_lines: int = Field(5, ge=1, le=500)
    customer_name: str | None = Field(None, max_length=120)
    font: str = "Helvetica"
    tax_inclusive: bool = False
    native: bool = False          # localized labels, native-script names/addresses, layout direction (RTL for Arabic/Urdu)
    native_digits: bool = False   # Arabic-Indic / Devanagari digits in the rendered document (JSON stays ASCII)

    @field_validator("locale")
    @classmethod
    def _known_locale(cls, v: str) -> str:
        try:
            return get_locale(v).code
        except LocaleError as e:
            raise ValueError(str(e)) from e

    @model_validator(mode="after")
    def _known_region(self) -> "InvoiceSpec":
        regions = get_locale(self.locale).data["tax"]["regions"]
        if self.region is not None and self.region not in regions:
            raise ValueError(f"region {self.region!r} not in {self.locale} tax regions {sorted(regions)}")
        if self.lines is not None and not 1 <= len(self.lines) <= 500:
            raise ValueError("lines must contain 1..500 items")
        return self

    def doc_id(self) -> str:
        """Stable id: same spec -> same id (used for idempotent jobs)."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]


def _random_lines(spec: InvoiceSpec, rng: np.random.Generator, factor: Decimal, dp: int) -> list[LineSpec]:
    out = []
    for _ in range(spec.n_lines):
        item = CATALOG[int(rng.integers(0, len(CATALOG)))]
        base = Decimal(str(round(float(rng.uniform(item["min"], item["max"])), 2)))
        price = quantize(base * factor, dp)
        qty = Decimal(int(rng.integers(1, 11))) if item["step"] == "1" else Decimal(int(rng.integers(2, 41))) / 2
        out.append(LineSpec(description=item["description"], quantity=qty, unit_price=price, category=item["category"]))
    return out


def generate_invoice(spec: InvoiceSpec) -> dict[str, Any]:
    """Structured ground truth (JSON-serialisable; Decimals as strings)."""
    pack = get_locale(spec.locale)
    dp = pack.data["currency"]["decimals"]
    rng = np.random.default_rng(spec.seed)
    factor = Decimal(pack.data["scale"]["price_factor"])
    pdf = pack.data.get("pdf", {})
    complex_script = bool(pdf.get("complex_shaping"))
    native = spec.native and not complex_script  # Devanagari needs conjunct shaping, which the PDF library lacks
    notes = ["complex-script shaping unavailable: rendered with Latin names and English labels"] if spec.native and complex_script else []
    lines = spec.lines if spec.lines is not None else _random_lines(spec, rng, factor, dp)

    rows, groups = [], {}
    for ln in lines:
        label, rate = pack.tax_rate(spec.region, ln.category)
        amount = quantize(ln.quantity * ln.unit_price, dp)
        rows.append({"description": ln.description, "category": ln.category, "quantity": _s(ln.quantity),
                     "unit_price": _s(ln.unit_price), "line_total": _s(amount), "tax_label": label, "tax_rate": _s(rate)})
        g = groups.setdefault((label, rate), Decimal(0))
        groups[(label, rate)] = g + amount

    summary, subtotal, tax_total = [], Decimal(0), Decimal(0)
    for (label, rate), amount in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if spec.tax_inclusive:
            tax = quantize(amount - amount / (1 + rate), dp) if rate else Decimal(0).quantize(Decimal(1).scaleb(-dp))
            taxable = amount - tax
        else:
            taxable = amount
            tax = quantize(amount * rate, dp)
        subtotal += taxable
        tax_total += tax
        summary.append({"tax_label": label, "tax_rate": _s(rate), "taxable_amount": _s(taxable), "tax": _s(tax)})
    total = subtotal + tax_total

    issue = date(2025, 1, 1) + timedelta(days=int(rng.integers(0, 365)))
    a_seller, _ = pack.address(rng, latin=not native)
    a_cust, _ = pack.address(rng, latin=not native)
    seller_last = pack.person_name(rng, latin=not native).split()[-1]
    return {
        "doc_type": "invoice",
        "invoice_number": "INV-" + spec.doc_id()[:10].upper(),
        "issue_date": issue.isoformat(), "due_date": (issue + timedelta(days=30)).isoformat(),
        "locale": pack.code, "region": spec.region, "currency": pack.data["currency"]["code"],
        "tax_inclusive": spec.tax_inclusive,
        "seller": {"name": f"{seller_last} Trading Co.", "address": a_seller,
                   "tax_id": f"TAX-{int(rng.integers(10**7, 10**8))}"},
        "customer": {"name": spec.customer_name or pack.person_name(rng, latin=not native), "address": a_cust},
        "presentation": {"native": native, "native_digits": bool(spec.native_digits and native), "direction": pdf.get("direction", "ltr") if native else "ltr",
                         "script": pdf.get("font_script", "latin") if native else "latin", "notes": notes},
        "lines": rows, "tax_summary": summary,
        "subtotal": _s(subtotal), "tax_total": _s(tax_total), "total": _s(total),
        "rounding": "HALF_UP; tax computed once per rate group",
    }


def reconcile(inv: dict[str, Any]) -> list[str]:
    """Independent re-computation from the JSON; returns a list of discrepancies (empty = exact)."""
    pack = get_locale(inv["locale"])
    dp = pack.data["currency"]["decimals"]
    errs: list[str] = []
    D = Decimal
    groups: dict[tuple[str, Decimal], Decimal] = {}
    for i, ln in enumerate(inv["lines"]):
        expect = quantize(D(ln["quantity"]) * D(ln["unit_price"]), dp)
        if D(ln["line_total"]) != expect:
            errs.append(f"line {i}: line_total {ln['line_total']} != {expect}")
        key = (ln["tax_label"], D(ln["tax_rate"]))
        groups[key] = groups.get(key, D(0)) + D(ln["line_total"])
    subtotal, tax_total = D(0), D(0)
    by_key = {(s["tax_label"], D(s["tax_rate"])): s for s in inv["tax_summary"]}
    if set(by_key) != set(groups):
        errs.append("tax_summary groups do not match line tax rates")
    for key, amount in groups.items():
        s = by_key.get(key)
        if s is None:
            continue
        rate = key[1]
        if inv["tax_inclusive"]:
            tax = quantize(amount - amount / (1 + rate), dp) if rate else D(0)
            taxable = amount - tax
        else:
            tax, taxable = quantize(amount * rate, dp), amount
        if D(s["tax"]) != tax:
            errs.append(f"tax {key}: {s['tax']} != {tax}")
        if D(s["taxable_amount"]) != taxable:
            errs.append(f"taxable {key}: {s['taxable_amount']} != {taxable}")
        subtotal += taxable
        tax_total += tax
    if D(inv["subtotal"]) != subtotal:
        errs.append(f"subtotal {inv['subtotal']} != {subtotal}")
    if D(inv["tax_total"]) != tax_total:
        errs.append(f"tax_total {inv['tax_total']} != {tax_total}")
    if D(inv["total"]) != subtotal + tax_total:
        errs.append(f"total {inv['total']} != {subtotal + tax_total}")
    if inv["tax_inclusive"] and D(inv["total"]) != sum((D(x["line_total"]) for x in inv["lines"]), D(0)):
        errs.append("inclusive total must equal the sum of line totals")
    return errs
