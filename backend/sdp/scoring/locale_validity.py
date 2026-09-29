"""Q3. Locale validity: phone, ID + checksum, name script, currency/date format, Unicode normalisation."""

from __future__ import annotations

import unicodedata
from decimal import Decimal
from typing import Any

import pandas as pd

from sdp.locale import get_locale

CHECKS = ("phone", "national_id", "name_script", "currency", "date", "unicode_nfc")


def locale_validity(df: pd.DataFrame, locale: str | None = None, columns: dict[str, Any] | None = None,
                    locale_column: str | None = None, max_failures: int = 200) -> dict[str, Any]:
    """columns: {"phone": col, "national_id": col, "name": [cols], "currency": [cols], "date": [cols], "text": [cols]}
    (`text` defaults to every string column). Row locale = df[locale_column] if given, else `locale`."""
    if not locale and not locale_column:
        raise ValueError("give `locale` or `locale_column`")
    columns = columns or {}
    listy = lambda v: [v] if isinstance(v, str) else list(v or [])  # noqa: E731
    phone, nid = listy(columns.get("phone")), listy(columns.get("national_id"))
    names, money, dates = listy(columns.get("name")), listy(columns.get("currency")), listy(columns.get("date"))
    text = listy(columns.get("text")) or [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) in ("str", "string")]
    used = set(phone + nid + names + money + dates + text)
    missing = used - set(df.columns)
    if missing:
        raise ValueError(f"columns not in DataFrame: {sorted(missing)}")

    stats = {k: {"checked": 0, "failed": 0} for k in CHECKS}
    failures: list[dict[str, Any]] = []
    bad_rows: set[int] = set()

    def fail(row: int, col: str, check: str, value: Any, reason: str) -> None:
        stats[check]["failed"] += 1
        bad_rows.add(row)
        if len(failures) < max_failures:
            failures.append({"row": row, "column": col, "check": check, "value": str(value), "reason": reason})

    loc_vals = df[locale_column].tolist() if locale_column else None
    for i in range(len(df)):
        pack = get_locale(loc_vals[i] if loc_vals else locale)  # type: ignore[arg-type]
        row = df.iloc[i]

        def cells(cs: list[str]):
            for c in cs:
                v = row[c]
                if isinstance(v, str) and v != "":
                    yield c, v

        for c, v in cells(phone):
            stats["phone"]["checked"] += 1
            if not pack.phone_valid(v):
                fail(i, c, "phone", v, f"not a valid {pack.data['country']} phone number")
        for c, v in cells(nid):
            stats["national_id"]["checked"] += 1
            ok, why = pack.national_id_valid(v)
            if not ok:
                fail(i, c, "national_id", v, why)
        for c, v in cells(names):
            stats["name_script"]["checked"] += 1
            if not pack.script_ok(v):
                fail(i, c, "name_script", v, f"letters outside the {pack.script} script")
        for c, v in cells(money):
            stats["currency"]["checked"] += 1
            if pack.parse_currency(v) is None:
                fail(i, c, "currency", v, f"not formatted like {pack.format_currency(Decimal('1234.5'))}")
        for c, v in cells(dates):
            stats["date"]["checked"] += 1
            if pack.parse_date(v) is None:
                fail(i, c, "date", v, f"expected format {pack.data['date']['format']}")
        for c, v in cells(text):
            stats["unicode_nfc"]["checked"] += 1
            if not unicodedata.is_normalized("NFC", v):
                fail(i, c, "unicode_nfc", v, "not NFC-normalised")

    by_check = {k: {**v, "pct_valid": (100.0 * (1 - v["failed"] / v["checked"]) if v["checked"] else None)}
                for k, v in stats.items() if v["checked"]}
    n = len(df)
    return {
        "valid_pct": 100.0 * (1 - len(bad_rows) / n) if n else 100.0,
        "n_rows": n, "n_invalid_rows": len(bad_rows), "by_check": by_check,
        "failures": failures, "failures_truncated": sum(v["failed"] for v in stats.values()) > len(failures),
    }
