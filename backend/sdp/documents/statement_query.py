"""D2. Query-style statements: parse "last 90 days, balance over $500" into filter parameters and apply them.

Supported phrases (combine freely, separated by commas / "and"):
  time      last|past N days|weeks|months|years        (relative to the statement's end date)
  balance   balance over|above|more than|at least X ; balance under|below|less than|at most X ; balance between X and Y
  amount    amount|payments|purchases|debits|credits over|under X   (size of a single transaction)
  type      debits only | credits only | deposits | withdrawals | spending | income
  where     category groceries|dining|...   or   at|from|merchant <name>
Amounts accept $, £, €, Rs, ₹, ¥, thousands separators and k/m suffixes ("1.5k"). Anything not understood is reported in
`unparsed` rather than silently ignored.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

MERCHANTS = {  # merchant -> (category, credit?)
    "Grocery market": ("groceries", False), "Fuel station": ("transport", False), "Online store": ("shopping", False),
    "Utility bill": ("bills", False), "Restaurant": ("dining", False), "Pharmacy": ("health", False),
    "Transfer received": ("income", True), "Salary": ("income", True), "ATM withdrawal": ("cash", False),
    "Subscription": ("bills", False), "Refund": ("income", True), "Insurance premium": ("bills", False),
}
CATEGORIES = sorted({c for c, _ in MERCHANTS.values()})
CATEGORY_ALIASES = {"grocery": "groceries", "food": "dining", "restaurants": "dining", "fuel": "transport", "gas": "transport",
                    "utilities": "bills", "medical": "health", "atm": "cash", "salary": "income", "deposits": "income"}

_NUM = r"(?:[$£€₹¥]|rs\.?|pkr|usd|sar|inr|eur|gbp|cny)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k|m)?"
_OVER = r"(?:over|above|more than|greater than|exceeding|at least|>=?|≥)"
_UNDER = r"(?:under|below|less than|at most|no more than|<=?|≤)"


class QuerySyntaxError(ValueError):
    pass


@dataclass
class StatementFilter:
    days: int | None = None
    min_balance: Decimal | None = None
    max_balance: Decimal | None = None
    min_amount: Decimal | None = None
    max_amount: Decimal | None = None
    kind: str | None = None          # debit | credit
    category: str | None = None
    merchant: str | None = None
    raw: str = ""
    unparsed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Decimal) else v) for k, v in d.items()}


def _amount(num: str, suffix: str | None) -> Decimal:
    v = Decimal(num.replace(",", ""))
    return v * {"k": 1000, "m": 1_000_000}.get((suffix or "").lower(), 1)


def _strict_gt(text: str) -> bool:
    return bool(re.search(r">(?!=)|over|above|more than|greater than|exceeding", text)) and not re.search(r"at least|>=|≥", text)


def parse_statement_query(query: str) -> StatementFilter:
    q = query.strip()
    f = StatementFilter(raw=q)
    rest = q.lower()

    def take(pattern: str) -> re.Match[str] | None:
        nonlocal rest
        m = re.search(pattern, rest)
        if m:
            rest = rest[:m.start()] + " " + rest[m.end():]
        return m

    m = take(r"(?:last|past|previous)\s+(\d+)\s*(day|week|month|year)s?")
    if m:
        f.days = int(m.group(1)) * {"day": 1, "week": 7, "month": 30, "year": 365}[m.group(2)]
    elif (m := take(r"(?:last|this)\s+(week|month|year)")):
        f.days = {"week": 7, "month": 30, "year": 365}[m.group(1)]

    m = take(rf"balance\s+between\s+{_NUM}\s+and\s+{_NUM}")
    if m:
        f.min_balance, f.max_balance = _amount(m.group(1), m.group(2)), _amount(m.group(3), m.group(4))
    else:
        m = take(rf"balance\s+({_OVER})\s*{_NUM}")
        if m:
            f.min_balance = _amount(m.group(2), m.group(3))
            if _strict_gt(m.group(1)):
                f.min_balance += Decimal("0.01")
        m = take(rf"balance\s+({_UNDER})\s*{_NUM}")
        if m:
            f.max_balance = _amount(m.group(2), m.group(3))
            if not re.search(r"at most|no more than|<=|≤", m.group(1)):
                f.max_balance -= Decimal("0.01")

    m = take(rf"(?:amounts?|transactions?|payments?|purchases?|debits?|credits?|withdrawals?|deposits?)\s+(?:only\s+)?({_OVER})\s*{_NUM}")
    if m:
        f.min_amount = _amount(m.group(2), m.group(3))
        if _strict_gt(m.group(1)):
            f.min_amount += Decimal("0.01")
        word = m.group(0)
        f.kind = f.kind or ("debit" if re.match(r"debit|payment|purchase|withdrawal", word) else "credit" if re.match(r"credit|deposit", word) else None)
    m = take(rf"(?:amounts?|transactions?|payments?|purchases?|debits?|credits?|withdrawals?|deposits?)\s+(?:only\s+)?({_UNDER})\s*{_NUM}")
    if m:
        f.max_amount = _amount(m.group(2), m.group(3))
        if not re.search(r"at most|no more than|<=|≤", m.group(1)):
            f.max_amount -= Decimal("0.01")

    if take(r"(?:debits?|withdrawals?|spending|expenses?|purchases?|payments?)\s+only|only\s+(?:debits?|withdrawals?|spending|expenses?)|\bspending\b|\bexpenses?\b"):
        f.kind = "debit"
    if take(r"(?:credits?|deposits?|income)\s+only|only\s+(?:credits?|deposits?|income)"):
        f.kind = "credit"

    m = take(r"(?:merchant|at|from)\s+([a-z][a-z ]{2,30}?)(?=,|$|\s+and\b|\s+with\b)")  # explicit merchant wins over category words
    if m:
        f.merchant = m.group(1).strip()
    m = take(r"category\s+([a-z]+)")
    if m:
        f.category = CATEGORY_ALIASES.get(m.group(1), m.group(1))
    elif not f.merchant:
        for word in re.findall(r"[a-z]+", rest):
            if word in CATEGORIES or word in CATEGORY_ALIASES:
                f.category = CATEGORY_ALIASES.get(word, word)
                rest = re.sub(rf"\b{word}\b", " ", rest, count=1)
                break
    if f.category and f.category not in CATEGORIES:
        raise QuerySyntaxError(f"unknown category {f.category!r}; known: {', '.join(CATEGORIES)}")

    leftovers = [w for w in re.findall(r"[a-z0-9$£€₹¥.]+", re.sub(r"\b(and|with|the|a|of|only|show|me|my|all|statement|transactions?|where|for|in|over|last)\b", " ", rest)) if w.strip(".")]
    f.unparsed = leftovers
    return f


def apply_filter(doc: dict[str, Any], f: StatementFilter) -> dict[str, Any]:
    """Return a new statement document containing only matching transactions; still reconciles exactly."""
    D = Decimal
    txs = doc["transactions"]
    end = date.fromisoformat(doc["period_end"])
    cutoff = (end - timedelta(days=f.days)).isoformat() if f.days else None
    keep = []
    for t in txs:
        amt = max(D(t["debit"]), D(t["credit"]))
        if cutoff and t["date"] < cutoff:
            continue
        if f.min_balance is not None and D(t["balance"]) < f.min_balance:
            continue
        if f.max_balance is not None and D(t["balance"]) > f.max_balance:
            continue
        if f.min_amount is not None and amt < f.min_amount:
            continue
        if f.max_amount is not None and amt > f.max_amount:
            continue
        if f.kind == "debit" and D(t["debit"]) == 0:
            continue
        if f.kind == "credit" and D(t["credit"]) == 0:
            continue
        if f.category and t.get("category") != f.category:
            continue
        if f.merchant and f.merchant.lower() not in t["description"].lower():
            continue
        keep.append(t)
    out = dict(doc)
    out["transactions"] = keep
    if keep:
        first = keep[0]
        out["opening_balance"] = format(D(first["balance"]) - D(first["credit"]) + D(first["debit"]), "f")
        out["closing_balance"] = keep[-1]["balance"]
        out["period_start"] = max(doc["period_start"], cutoff) if cutoff else doc["period_start"]
    else:
        out["closing_balance"] = out["opening_balance"] = doc["closing_balance"]
    out["total_debits"] = format(sum((D(t["debit"]) for t in keep), D(0)), "f")
    out["total_credits"] = format(sum((D(t["credit"]) for t in keep), D(0)), "f")
    out["filter"] = {"query": f.raw, "params": f.to_dict(), "matched": len(keep), "of": len(txs)}
    return out
