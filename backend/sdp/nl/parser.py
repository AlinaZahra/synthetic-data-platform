"""A3. Natural language -> validated DatasetConfig (deterministic rule-based parser; no network, no LLM).

"5,000 Pakistani bank customers, 3% fraud, 12 months of history" ->
  rows=5000, locale=ur-PK, domain=bank_customers, flag is_fraud rate 0.03, history_months=12
The parser NEVER generates; it only proposes a config with an explanation of where each value came from,
so the UI can show it for confirmation. Domains are JSON files in `domains/` (add one = add a file).
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sdp.locale import LocaleError, get_locale, registry

DOMAIN_DIR = Path(__file__).parent / "domains"
MAX_ROWS = 1_000_000
UNITS = {"k": 1_000, "thousand": 1_000, "m": 1_000_000, "million": 1_000_000}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "twelve": 12, "twenty": 20, "fifty": 50, "hundred": 100}


@lru_cache(maxsize=1)
def domains() -> dict[str, dict[str, Any]]:
    out = {}
    for f in sorted(DOMAIN_DIR.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        out[d["name"]] = d
    return out


class FlagSpec(BaseModel):
    name: str
    rate: float = Field(ge=0.0, le=1.0)


class DatasetConfig(BaseModel):
    """The confirmed contract between the chat box and the generator."""

    model_config = ConfigDict(extra="forbid")

    domain: str
    locale: str
    rows: int = Field(ge=1, le=MAX_ROWS)
    seed: int = 0
    locale_mix: dict[str, float] | None = None   # e.g. {'ur-PK': 0.7, 'en-US': 0.3}: each row is generated from ONE of them
    flag: FlagSpec | None = None
    history_months: int | None = Field(None, ge=1, le=120)
    # echoed from the domain so the user can see the schema, distributions and rules being applied
    schema_columns: list[dict[str, Any]] = Field(default_factory=list)
    rules: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("locale")
    @classmethod
    def _locale_exists(cls, v: str) -> str:
        try:
            return get_locale(v).code
        except LocaleError as e:
            raise ValueError(str(e)) from e

    @field_validator("locale_mix")
    @classmethod
    def _mix_valid(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        if v is None:
            return v
        from sdp.locale.coherent import normalise_mix
        return normalise_mix(v)

    @model_validator(mode="after")
    def _domain_consistent(self) -> "DatasetConfig":
        dom = domains().get(self.domain)
        if dom is None:
            raise ValueError(f"unknown domain {self.domain!r}; available: {sorted(domains())}")
        if self.flag and self.flag.name != dom["flag"]["name"]:
            raise ValueError(f"domain {self.domain} supports flag {dom['flag']['name']!r}, not {self.flag.name!r}")
        if self.history_months and "history" not in dom:
            raise ValueError(f"domain {self.domain} has no history table")
        return self

    def config_hash(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]


class ParseResult(BaseModel):
    ok: bool
    config: DatasetConfig | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    explanation: list[dict[str, str]] = Field(default_factory=list)  # {field, value, source}
    confirmation_required: bool = True
    config_hash: str | None = None
    issues: list[dict[str, Any]] = Field(default_factory=list)  # stable codes for errors/warnings: {level, code, params}, used for localised answers


# ---------------------------------------------------------------- extractors
def _rows(text: str) -> tuple[int | None, str | None]:
    for m in re.finditer(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(k|m|thousand|million)?\b(?!\s*(?:%|percent|months?|years?|mo\b))", text):
        raw, unit = m.group(1), m.group(2)
        try:
            val = float(raw.replace(",", ""))
        except ValueError:
            continue
        if unit:
            val *= UNITS[unit]
        return int(round(val)), m.group(0).strip()
    m = re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|twenty|fifty|hundred)\s+(thousand|million)\b", text)
    if m:
        return NUMBER_WORDS[m.group(1)] * UNITS[m.group(2)], m.group(0)
    return None, None


def _rate(text: str, keywords: list[str]) -> tuple[float | None, str | None]:
    kw = "|".join(re.escape(k) for k in keywords)
    for pat in (rf"(\d+(?:\.\d+)?)\s*(?:%|percent|per\s*cent)\s*(?:of\s+\w+\s+)?(?:{kw})", rf"(?:{kw})\w*[^\d%]{{0,25}}?(\d+(?:\.\d+)?)\s*(?:%|percent)"):
        m = re.search(pat, text)
        if m:
            return float(m.group(1)) / 100.0, m.group(0)
    return None, None


def _months(text: str) -> tuple[int | None, str | None]:
    m = re.search(r"(\d+|a|one|two|three)\s*[- ]?(month|months|year|years|yr|yrs)\b(?:\s+of)?\s*(?:history|data|transactions?|activity|records)", text)
    if not m:
        m = re.search(r"(?:history|over|for)\s+(?:of\s+|the\s+last\s+)?(\d+)\s*[- ]?(month|months|year|years)", text)
    if not m:
        return None, None
    n = m.group(1)
    n = int(n) if n.isdigit() else {"a": 1, "one": 1, "two": 2, "three": 3}[n]
    return (n * 12 if m.group(2).startswith(("year", "yr")) else n), m.group(0)


def _locale(text: str) -> tuple[list[str], str | None]:
    reg = registry()
    found: list[tuple[int, str, str]] = []
    for code in reg.codes():
        for alias in [code.lower(), *(a.lower() for a in reg.get(code).data.get("aliases", []))]:
            for m in re.finditer(rf"(?<![\w-]){re.escape(alias)}(?![\w-])", text):
                found.append((m.start(), code, alias))
    found.sort()
    codes = list(dict.fromkeys(c for _, c, _ in found))
    return codes, (found[0][2] if found else None)


def _domain(text: str) -> tuple[str | None, str | None]:
    best: tuple[int, str, str] | None = None
    for name, d in domains().items():
        for kw in d["keywords"]:
            if kw in text and (best is None or len(kw) > best[0]):
                best = (len(kw), name, kw)
    return (best[1], best[2]) if best else (None, None)


def _mix(text: str) -> tuple[dict[str, float] | None, str | None]:
    """'70% ur-PK, 30% en' -> {'ur-PK': .7, 'en-US': .3}. A percentage counts only if the word after it is a locale."""
    from sdp.locale.coherent import resolve_locale
    found: dict[str, float] = {}
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)?([a-z][a-z-]*)", text):
        try:
            code = resolve_locale(m.group(2))
        except LocaleError:
            continue
        found[code] = found.get(code, 0.0) + float(m.group(1))
    if len(found) < 2:
        return None, None
    total = sum(found.values())
    return {k: v / total for k, v in found.items()}, ", ".join(f"{v:g}% {k}" for k, v in found.items())


def parse_request(text: str, seed: int = 0) -> ParseResult:
    t = text.lower().strip()
    errors: list[str] = []
    warnings: list[str] = []
    issues: list[dict[str, Any]] = []
    expl: list[dict[str, str]] = []

    def err(code: str, text: str, **params: Any) -> None:
        errors.append(text)
        issues.append({"level": "error", "code": code, "params": params})

    def warn(code: str, text: str, **params: Any) -> None:
        warnings.append(text)
        issues.append({"level": "warning", "code": code, "params": params})

    if not t:
        return ParseResult(ok=False, errors=["Describe the dataset you want, e.g. '5,000 Pakistani bank customers, 3% fraud'."],
                           issues=[{"level": "error", "code": "empty", "params": {}}])

    domain, dsrc = _domain(t)
    if domain is None:
        err("no_domain", f"I couldn't tell what kind of records you want. Supported: {', '.join(sorted(domains()))}.", supported=sorted(domains()))
    rows, rsrc = _rows(t)
    if rows is None:
        err("no_rows", "I couldn't find how many rows you want (e.g. '5,000' or '10k').")
    elif not 1 <= rows <= MAX_ROWS:
        err("rows_range", f"Row count {rows:,} is outside 1..{MAX_ROWS:,}.", rows=rows, max=MAX_ROWS)

    mix, msrc = _mix(t)
    codes, lsrc = _locale(t)
    if mix:
        codes, lsrc = list(mix), msrc
    if not codes:
        locale = "en-US"
        warn("default_locale", "No country or language mentioned; assuming en-US.")
    else:
        locale = codes[0]
        if len(codes) > 1:
            warn("multi_locale", f"Several locales mentioned ({', '.join(codes)}); using {locale}. Generate them separately for mixed sets.", locales=codes, used=locale)

    dom = domains().get(domain) if domain else None
    flag = None
    months = None
    if dom:
        rate, fsrc = _rate(t, dom["flag"]["keywords"])
        if rate is not None:
            if rate > 0.5:
                warn("high_flag", f"A {rate:.0%} {dom['flag']['name']} rate is unusually high.", rate=rate, flag=dom["flag"]["name"])
            flag = {"name": dom["flag"]["name"], "rate": rate}
            expl.append({"field": "flag", "value": f"{dom['flag']['name']} = {rate:.1%}", "source": fsrc or ""})
        elif re.search(r"\d\s*(%|percent)", t):
            warn("pct_ignored", "A percentage was given but it isn't attached to a known flag; ignored.")
        months, msrc = _months(t)
        if months is not None:
            expl.append({"field": "history_months", "value": str(months), "source": msrc or ""})
        elif "history" in t:
            warn("history_default", "History was mentioned without a length; using the domain default.")
            months = dom["history"]["default_months"]
        if months is not None and months > 120:
            err("history_long", "History longer than 120 months isn't supported.")
    if errors:
        return ParseResult(ok=False, errors=errors, warnings=warnings, issues=issues)

    assert dom and rows
    expl = [{"field": "domain", "value": domain, "source": dsrc or ""}, {"field": "rows", "value": f"{rows:,}", "source": rsrc or ""},
            {"field": "locale", "value": locale, "source": lsrc or "default"}] + expl
    try:
        cfg = DatasetConfig(domain=domain, locale=locale, rows=rows, seed=seed, flag=flag, history_months=months, locale_mix=mix,
                            schema_columns=dom["columns"], rules=dom["rules"])
    except ValueError as e:
        return ParseResult(ok=False, errors=[str(e)], warnings=warnings, issues=[*issues, {"level": "error", "code": "invalid", "params": {"detail": str(e)}}])
    return ParseResult(ok=True, config=cfg, warnings=warnings, explanation=expl, config_hash=cfg.config_hash(), issues=issues)
