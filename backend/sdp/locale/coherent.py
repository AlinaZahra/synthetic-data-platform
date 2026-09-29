"""M3. Culturally coherent records: every row is generated from ONE locale context, so name, script, city, address,
phone, national ID, email host and currency all agree. `mix` sets the share of rows per locale (e.g. 70% ur-PK, 30% en).

Coherence is checkable: `coherence_report(df)` re-validates each row against its own locale pack.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from sdp.locale.registry import LocaleError, LocalePack, get_locale, registry

COLUMNS = ["locale", "gender", "first_name", "last_name", "full_name", "full_name_latin", "city", "address", "phone", "email",
           "national_id", "date_of_birth", "currency", "monthly_income"]


def strip_diacritics(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def resolve_locale(token: str) -> str:
    """'ur-PK' -> 'ur-PK'; 'en' -> 'en-US'; 'pakistani' -> 'ur-PK'; 'Spanish' -> 'es'."""
    reg = registry()
    try:
        return reg.get(token).code
    except LocaleError:
        pass
    p = reg.find_by_alias(token)
    if p:
        return p.code
    hits = [c for c in reg.codes() if reg.get(c).data["language"] == token.lower()]
    if hits:
        return sorted(hits, key=lambda c: (c != f"{token.lower()}-US", c))[0]
    raise LocaleError(f"unknown locale {token!r}; available: {reg.codes()}")


def parse_mix(text: str) -> dict[str, float]:
    """'70% ur-PK, 30% en' | 'ur-PK:0.7,en:0.3' | 'ur-PK=70 en=30' -> {'ur-PK': 0.7, 'en-US': 0.3} (normalised)."""
    mix: dict[str, float] = {}
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%?\s*(?:of\s+)?([A-Za-z][A-Za-z-]*)|([A-Za-z][A-Za-z-]*)\s*[:=]\s*(\d+(?:\.\d+)?)\s*%?", text):
        w, loc = (float(m.group(1)), m.group(2)) if m.group(1) else (float(m.group(4)), m.group(3))
        code = resolve_locale(loc)
        mix[code] = mix.get(code, 0.0) + w
    if not mix:
        raise LocaleError(f"could not read a locale mix from {text!r}; try '70% ur-PK, 30% en'")
    return normalise_mix(mix)


def normalise_mix(mix: dict[str, float]) -> dict[str, float]:
    resolved: dict[str, float] = {}
    for k, v in mix.items():
        if v < 0:
            raise LocaleError("mix weights must be non-negative")
        c = resolve_locale(k)
        resolved[c] = resolved.get(c, 0.0) + float(v)
    total = sum(resolved.values())
    if total <= 0:
        raise LocaleError("mix weights must sum to more than zero")
    return {k: v / total for k, v in resolved.items()}


def native_name_parts(pack: LocalePack, rng: np.random.Generator) -> dict[str, str]:
    """A person in one locale: native-script parts + a canonical Latin rendering derived from the SAME tokens."""
    n = pack.data["names"]
    male = rng.random() < 0.5
    first = n["first_male" if male else "first_female"][int(rng.integers(0, len(n["first_male" if male else "first_female"])))]
    last = n["last"][int(rng.integers(0, len(n["last"])))]
    tr = pack.data.get("translit")
    lat = (lambda t: tr[t][0]) if tr else (lambda t: t)
    joiner = n.get("joiner", " ")
    full = f"{last}{joiner}{first}" if n.get("order") == "family_given" else f"{first}{joiner}{last}"
    latin_full = f"{lat(last)} {lat(first)}" if n.get("order") == "family_given" else f"{lat(first)} {lat(last)}"
    return {"gender": "male" if male else "female", "first": first, "last": last, "full": full,
            "latin_first": lat(first), "latin_last": lat(last), "latin_full": latin_full}


def email_for(pack: LocalePack, latin_first: str, latin_last: str, rng: np.random.Generator) -> str:
    def clean(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", strip_diacritics(s).lower())
    style = int(rng.integers(0, 3))
    a, b = clean(latin_first), clean(latin_last)
    local = [f"{a}.{b}", f"{a[:1]}{b}", f"{a}{b}{int(rng.integers(1, 99))}"][style]
    return f"{local}@{pack.data['email_host']}"


def generate_people(n: int, mix: dict[str, float] | str | None = None, seed: int = 0) -> pd.DataFrame:
    """n coherent records. mix: {'ur-PK': .7, 'en': .3} or '70% ur-PK, 30% en'; default en-US."""
    m = parse_mix(mix) if isinstance(mix, str) else normalise_mix(mix or {"en-US": 1.0})
    rng = np.random.default_rng(seed)
    codes = list(m)
    locs = rng.choice(codes, size=n, p=[m[c] for c in codes])
    rows: list[dict[str, Any]] = []
    for code in locs:
        p = get_locale(str(code))
        nm = native_name_parts(p, rng)
        addr, city = p.address(rng)
        income = Decimal(str(round(float(rng.lognormal(0, 0.5)), 4))) * Decimal(p.data["scale"]["income_median"])
        rows.append({
            "locale": p.code, "gender": nm["gender"], "first_name": nm["first"], "last_name": nm["last"], "full_name": nm["full"],
            "full_name_latin": nm["latin_full"], "city": city, "address": addr, "phone": p.phone(rng),
            "email": email_for(p, nm["latin_first"], nm["latin_last"], rng), "national_id": p.national_id(rng),
            "date_of_birth": p.format_date(p.birth_date(rng)), "currency": p.data["currency"]["code"],
            "monthly_income": float(income.quantize(Decimal("0.01"))),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def coherence_report(df: pd.DataFrame, max_failures: int = 50) -> dict[str, Any]:
    """Re-check every row against ITS OWN locale: phone, ID (+checksum), name script, city, email host, currency, DOB format."""
    checks = ["phone", "national_id", "name_script", "city", "email_host", "currency", "date_of_birth"]
    stats = {c: {"checked": 0, "failed": 0} for c in checks}
    failures: list[dict[str, Any]] = []
    bad_rows: set[int] = set()
    for i, r in enumerate(df.to_dict("records")):
        p = get_locale(r["locale"])
        results = {
            "phone": p.phone_valid(r["phone"]),
            "national_id": p.national_id_valid(r["national_id"])[0],
            "name_script": p.script_ok(r["full_name"]),
            "city": r["city"] in p.data["address"]["cities"] and r["city"] in r["address"],
            "email_host": r["email"].endswith("@" + p.data["email_host"]),
            "currency": r["currency"] == p.data["currency"]["code"],
            "date_of_birth": p.parse_date(r["date_of_birth"]) is not None,
        }
        for c, ok in results.items():
            stats[c]["checked"] += 1
            if not ok:
                stats[c]["failed"] += 1
                bad_rows.add(i)
                if len(failures) < max_failures:
                    failures.append({"row": i, "locale": r["locale"], "check": c, "value": str(r.get(c if c in r else "full_name"))})
    mix = df["locale"].value_counts(normalize=True).to_dict() if len(df) else {}
    return {"coherent_pct": 100.0 * (1 - len(bad_rows) / len(df)) if len(df) else 100.0, "rows": len(df), "by_check": stats,
            "failures": failures, "mix": {k: float(v) for k, v in mix.items()}}
