"""A2. Realistic content synthesis: names, addresses and free text (reviews, descriptions, support tickets).

Batched LLM calls (one request per `batch_size` rows, each row carrying its own context such as rating or city), an on-disk cache
so repeats are free and reproducible, per-item validation (count, script for the locale, no e-mail/phone leakage, no duplicates),
and a deterministic local fallback used per item whenever the API is unavailable or an answer is rejected.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from sdp.ai import llm
from sdp.locale import get_locale
from sdp.locale.coherent import resolve_locale

KINDS = ("name", "address", "review", "description", "ticket")
MAX_LEN = {"name": 80, "address": 160, "review": 600, "description": 600, "ticket": 800}
_PII = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|\+?\d[\d\s().-]{8,}\d")
LANG_NAMES = {"en-US": "English", "en-GB": "British English", "ur-PK": "Urdu (Arabic script)", "ar": "Arabic", "hi": "Hindi (Devanagari)",
              "es": "Spanish", "fr": "French", "zh": "Simplified Chinese"}


class ContentError(ValueError):
    pass


@dataclass
class SynthResult:
    values: list[str]
    stats: dict[str, int] = field(default_factory=lambda: {"rows": 0, "llm": 0, "cache_batches": 0, "llm_batches": 0, "fallback": 0, "rejected": 0})
    warnings: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)   # per row: llm | fallback


@lru_cache(maxsize=1)
def _fallback_data() -> dict[str, Any]:
    return json.loads((Path(__file__).parent / "data" / "fallback_text.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ fallback
def _sentiment(ctx: dict[str, Any]) -> str:
    r = ctx.get("rating")
    try:
        r = float(r)
    except (TypeError, ValueError):
        return str(ctx.get("sentiment", "neutral")) if str(ctx.get("sentiment", "")) in ("positive", "neutral", "negative") else "neutral"
    return "positive" if r >= 4 else "negative" if r <= 2 else "neutral"


def fallback_text(kind: str, locale: str, ctx: dict[str, Any], rng: np.random.Generator) -> str:
    lang = locale.split("-")[0] if locale.split("-")[0] in _fallback_data() else "en"
    d = _fallback_data()[lang]
    product = str(ctx.get("product") or ctx.get("item") or d["default_product"])
    city = str(ctx.get("city") or d["default_city"])

    def pick(xs: list[str]) -> str:
        return xs[int(rng.integers(len(xs)))].format(product=product, city=city)

    if kind == "review":
        b = d["review"][_sentiment(ctx)]
    elif kind == "description":
        b = d["description"]
    else:
        pr = str(ctx.get("priority", "medium")).lower()
        b = {"open": d["ticket"]["open"][pr if pr in d["ticket"]["open"] else "medium"], "detail": d["ticket"]["detail"], "close": d["ticket"]["close"]}
    parts = [pick(b["open"])]
    if rng.random() < 0.85:
        parts.append(pick(b["detail"]))
    if rng.random() < 0.7:
        parts.append(pick(b["close"]))
    return " ".join(parts)


def fallback_one(kind: str, locale: str, ctx: dict[str, Any], rng: np.random.Generator) -> str:
    pack = get_locale(locale)
    if kind == "name":
        return pack.person_name(rng)
    if kind == "address":
        a, city = pack.address(rng)
        return f"{a}, {city}"
    return fallback_text(kind, locale, ctx, rng)


# ------------------------------------------------------------------ validation
def _valid(kind: str, text: Any, locale: str, seen: set[str]) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip()
    if not t or len(t) > MAX_LEN[kind]:
        return False
    if kind in ("name", "address", "review", "description", "ticket") and _PII.search(t) and kind != "address":
        return False
    if kind in ("name", "address") and not get_locale(locale).script_ok(t):
        return False
    if kind == "name" and t in seen:
        return False
    if "```" in t or t.startswith(("{", "[")):
        return False
    return True


# ------------------------------------------------------------------- prompts
def _system(kind: str, locale: str) -> str:
    lang = LANG_NAMES.get(locale, locale)
    what = {"name": "realistic but fictional full person names typical of the locale",
            "address": "realistic but fictional street addresses (street, area, city) typical of the locale; they must not be real households",
            "review": "short customer product reviews (1-3 sentences) whose tone matches each row's rating/sentiment",
            "description": "short product descriptions (1-3 sentences) fitting each row's product/category",
            "ticket": "customer-support ticket texts (1-3 sentences) matching each row's category, product and priority"}[kind]
    return (f"You write synthetic test data. Produce {what}. Language/script: {lang}. Never include real people, real businesses, e-mail addresses "
            f"or phone numbers. Reply with ONE JSON array of strings, exactly one string per input row, in order, no prose.")


def _prompt(contexts: list[dict[str, Any]]) -> str:
    return json.dumps({"rows": [{k: v for k, v in c.items() if isinstance(v, (str, int, float, bool)) or v is None} for c in contexts]}, ensure_ascii=False, default=str)


def synthesize(kind: str, n: int, locale: str = "en-US", contexts: list[dict[str, Any]] | None = None, seed: int = 0,
               client: llm.LLMClient | None | bool = True, batch_size: int = 20, use_cache: bool = True) -> SynthResult:
    """Generate `n` values of `kind`. `contexts[i]` (optional) describes row i (city, rating, product, priority...) and steers the text."""
    if kind not in KINDS:
        raise ContentError(f"unknown kind {kind!r}; available: {list(KINDS)}")
    if n < 0 or n > 100_000:
        raise ContentError("n must be between 0 and 100,000")
    locale = resolve_locale(locale) if locale else "en-US"
    ctxs = list(contexts or [])
    if ctxs and len(ctxs) != n:
        raise ContentError(f"contexts has {len(ctxs)} entries for {n} rows")
    ctxs = ctxs or [{} for _ in range(n)]
    c = llm.get_client() if client is True else (client or None)
    res = SynthResult(values=[], stats={"rows": n, "llm": 0, "cache_batches": 0, "llm_batches": 0, "fallback": 0, "rejected": 0})
    seen: set[str] = set()
    down = False
    for b0 in range(0, n, batch_size):
        batch = ctxs[b0:b0 + batch_size]
        answers: list[Any] | None = None
        if c is not None and not down:
            prompt = _prompt(batch) + f"\nvariant:{seed}:{b0 // batch_size}"   # seed/batch in the prompt: same seed -> same cache hit
            try:
                text, cached = llm.cached_complete(c, _system(kind, locale), prompt, max_tokens=min(4096, 120 * len(batch) + 200), use_cache=use_cache)
                parsed = llm.extract_json(text)
                if isinstance(parsed, dict):
                    parsed = next((v for v in parsed.values() if isinstance(v, list)), None)
                if isinstance(parsed, list):
                    answers = parsed
                    res.stats["cache_batches" if cached else "llm_batches"] += 1
                else:
                    res.warnings.append(f"batch {b0 // batch_size}: reply was not a JSON array; used local generator.")
            except llm.LLMError as e:
                res.warnings.append(f"LLM unavailable ({e}); using local generators for the remaining rows.")
                down = True
        if answers is not None and len(answers) != len(batch):
            res.warnings.append(f"batch {b0 // batch_size}: expected {len(batch)} items, got {len(answers)}; unmatched rows use the local generator.")
        for i, ctx in enumerate(batch):
            a = answers[i] if answers is not None and i < len(answers) else None
            if a is not None and _valid(kind, a, locale, seen):
                v = a.strip()
                res.sources.append("llm")
                res.stats["llm"] += 1
            else:
                if a is not None:
                    res.stats["rejected"] += 1
                rng = np.random.default_rng([seed, b0 + i, KINDS.index(kind)])   # per-row stream: fallback is stable if the LLM path changes
                v = fallback_one(kind, locale, ctx, rng)
                res.sources.append("fallback")
                res.stats["fallback"] += 1
            seen.add(v)
            res.values.append(v)
    if kind not in ("name", "address") and locale.split("-")[0] not in _fallback_data() and res.stats["fallback"]:
        res.warnings.append(f"no offline free-text templates for {locale}; fallback text is English.")
    return res
