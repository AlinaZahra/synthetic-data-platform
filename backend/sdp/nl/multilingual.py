"""M7. Multilingual prompts: accept a request in en / es / fr / ur / ar / hi / zh and answer in the same language.

The parser itself stays English and deterministic. This layer (1) detects the language (script, then vocabulary for Latin scripts),
(2) normalises digits/separators/letters and rewrites known phrases into canonical English tokens using `lexicon.json`
(data, not code: a new language is a new entry), (3) parses, and (4) renders the confirmation, warnings and errors from
`messages.json` using the parser's stable issue codes. Anything it cannot map is reported, never guessed.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

from sdp.nl.parser import ParseResult, domains, parse_request

DATA = Path(__file__).parent
SUPPORTED = ["en", "es", "fr", "ur", "ar", "hi", "zh"]
LATIN = {"en", "es", "fr"}
_URDU_LETTERS = set("ٹڈڑںےگچپھژ")
_EN_WORDS = {"customers", "customer", "bank", "fraud", "months", "month", "history", "of", "with", "and", "rows", "generate", "the", "users", "shoppers", "year", "years"}
_ARABIC_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ی": "ي", "ې": "ي", "ک": "ك", "ہ": "ه", "ھ": "ه", "ة": "ه", "ؤ": "و", "ئ": "ي", "ۀ": "ه", "ے": "ي", "ں": "ن"})


def strip_marks(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def norm(text: str, lang: str) -> str:
    """Canonical comparison form: NFKC, ASCII digits, unified separators/percent, lowercase; accents (Latin) / harakat + letter variants (Arabic) folded."""
    t = unicodedata.normalize("NFKC", text)
    t = "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in t)
    t = t.replace("٬", ",").replace("٫", ".").replace("،", ",").replace("，", ",").replace("٪", "%").replace("’", "'").replace("‘", "'")
    t = t.lower()
    if lang in LATIN:
        t = strip_marks(t)
    elif lang in ("ar", "ur"):
        t = "".join(c for c in unicodedata.normalize("NFD", t) if not unicodedata.combining(c)).replace("ـ", "").translate(_ARABIC_MAP)
        t = unicodedata.normalize("NFC", t)
    elif lang == "hi":
        t = unicodedata.normalize("NFC", t)
    return re.sub(r"\s+", " ", t).strip()


@lru_cache(maxsize=1)
def lexicon() -> dict[str, dict[str, Any]]:
    raw = json.loads((DATA / "lexicon.json").read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for lang, d in raw.items():
        if lang.startswith("_"):
            continue
        phrases = sorted(((norm(k, lang), v) for k, v in d.get("phrases", {}).items()), key=lambda kv: -len(kv[0]))
        mult = sorted(((norm(k, lang), v) for k, v in d.get("multipliers", {}).items()), key=lambda kv: -len(kv[0]))
        out[lang] = {"phrases": phrases, "stop": {norm(w, lang) for w in d.get("stop", [])}, "mult": mult, "pre": [(re.compile(a), b) for a, b in d.get("pre_regex", [])]}
    return out


@lru_cache(maxsize=1)
def messages() -> dict[str, dict[str, str]]:
    raw = json.loads((DATA / "messages.json").read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


# -------------------------------------------------------------- detection
def _script_counts(text: str) -> dict[str, int]:
    c = {"han": 0, "deva": 0, "arab": 0, "latin": 0}
    for ch in text:
        if not ch.isalpha():
            continue
        n = unicodedata.name(ch, "")
        key = "han" if n.startswith("CJK") else "deva" if n.startswith("DEVANAGARI") else "arab" if n.startswith("ARABIC") else "latin" if n.startswith("LATIN") else None
        if key:
            c[key] += 1
    return c


def _hits(t: str, lang: str) -> int:
    lex = lexicon()[lang]
    score = 0
    for k, _ in lex["phrases"]:
        if k and re.search(rf"(?<![a-z]){re.escape(k)}(?![a-z])" if lang in LATIN else re.escape(k), t):
            score += 2
    score += sum(1 for w in re.findall(r"[^\W\d_]+", t) if w in lex["stop"])
    return score


def detect_language(text: str) -> str:
    c = _script_counts(text)
    top = max(c, key=lambda k: c[k])
    if c[top] == 0:
        return "en"
    if top == "han":
        return "zh"
    if top == "deva":
        return "hi"
    if top == "arab":
        if any(ch in _URDU_LETTERS for ch in text):
            return "ur"
        return "ur" if _hits(norm(text, "ur"), "ur") > _hits(norm(text, "ar"), "ar") else "ar"
    t = norm(text, "es")
    scores = {"es": _hits(t, "es") + (3 if re.search(r"[¿¡ñ]", text.lower()) else 0), "fr": _hits(norm(text, "fr"), "fr") + (1 if re.search(r"[çœ]", text.lower()) else 0),
              "en": sum(1 for w in re.findall(r"[a-z]+", t) if w in _EN_WORDS)}
    best = max(scores, key=lambda k: (scores[k], k == "en"))
    return best if scores[best] > 0 else "en"


# -------------------------------------------------------------- translation
def _latin_numbers(t: str, lang: str) -> str:
    """es/fr: '5.000' / '5 000' are thousands, '3,5' is a decimal."""
    if lang == "fr":
        t = re.sub(r"(?<=\d)[   ](?=\d{3}(?!\d))", "", t)
    t = re.sub(r"(?<=\d)\.(?=\d{3}(?!\d))", "", t)
    return re.sub(r"(?<=\d),(?=\d{1,2}(?!\d))", ".", t)


def to_canonical(text: str, lang: str) -> tuple[str, list[str]]:
    """Rewrite a request into the English the parser understands. Returns (text, notes about what was mapped)."""
    if lang not in lexicon():
        raise ValueError(f"unsupported language {lang!r}; supported: {SUPPORTED}")
    if lang == "en":
        return text, []
    lex = lexicon()[lang]
    t = norm(text, lang)
    if lang in ("es", "fr"):
        t = _latin_numbers(t, lang)
    for pat, rep in lex["pre"]:
        t = pat.sub(rep, t)
    latin = lang in LATIN
    for word, mult in lex["mult"]:
        tail = r"(?![^\W\d_])" if latin else ""
        t = re.sub(rf"(\d+(?:\.\d+)?)\s*{re.escape(word)}{tail}", lambda m: f" {int(round(float(m.group(1)) * mult))} ", t)
    notes: list[str] = []
    for key, val in lex["phrases"]:
        pat = rf"(?<![^\W\d_]){re.escape(key)}(?![^\W\d_])" if latin else re.escape(key)
        new = re.sub(pat, f" {val} ", t)
        if new != t:
            notes.append(f"{key} -> {val}")
            t = new
    if lex["stop"]:
        if latin or lang in ("ar", "ur", "hi"):
            toks = re.split(r"([\s,.;:!?()\[\]'\"«»،؛]+)", t)
            t = "".join("" if w in lex["stop"] else w for w in toks)
        else:                                            # no word spacing (zh): drop the words wherever they occur, longest first
            for w in sorted(lex["stop"], key=len, reverse=True):
                t = t.replace(w, " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t, notes


# ------------------------------------------------------------------ answers
def msg(lang: str, key: str, **params: Any) -> str:
    cat = messages().get(lang) or messages()["en"]
    text = cat.get(key) or messages()["en"][key]
    return text.format(**params) if params else text


def _fmt_issue(lang: str, issue: dict[str, Any]) -> str:
    code, p = issue["code"], dict(issue["params"])
    key = ("err_" if issue["level"] == "error" else "warn_") + code
    if code == "no_domain":
        p["supported"] = ", ".join(msg(lang, f"domain_{d}") for d in sorted(domains()))
    if code == "rows_range":
        p["rows"], p["max"] = f"{p['rows']:,}", f"{p['max']:,}"
    if code == "multi_locale":
        p["locales"] = ", ".join(msg(lang, f"loc_{c}") for c in p["locales"])
        p["used"] = msg(lang, f"loc_{p['used']}")
    if code == "high_flag":
        p["rate"] = f"{p['rate']:.0%}"
        p["flag"] = msg(lang, f"flag_{p['flag']}")
    return msg(lang, key, **p)


def localize(result: ParseResult, lang: str) -> dict[str, Any]:
    """The reply in the user's language: headline, summary lines (when parsed), errors and warnings."""
    errs = [_fmt_issue(lang, i) for i in result.issues if i["level"] == "error"]
    warns = [_fmt_issue(lang, i) for i in result.issues if i["level"] == "warning"]
    lines: list[str] = []
    if result.ok and result.config:
        c = result.config
        lines = [f"{msg(lang, 'rows')}: {c.rows:,}", f"{msg(lang, 'domain')}: {msg(lang, f'domain_{c.domain}')}",
                 f"{msg(lang, 'locale')}: " + (", ".join(f"{msg(lang, f'loc_{k}')} {v:.0%}" for k, v in c.locale_mix.items()) if c.locale_mix else msg(lang, f"loc_{c.locale}"))]
        if c.flag:
            lines.append(f"{msg(lang, 'flag')} ({msg(lang, f'flag_{c.flag.name}')}): {c.flag.rate:.1%}")
        if c.history_months:
            lines.append(f"{msg(lang, 'history')}: {c.history_months}")
    return {"lang": lang, "rtl": lang in ("ar", "ur"), "headline": msg(lang, "understood") if result.ok else (errs[0] if errs else ""),
            "lines": lines, "errors": errs, "warnings": warns}


def parse_multilingual(text: str, seed: int = 0, language: str | None = None) -> dict[str, Any]:
    """Detect (or use `language`), translate to canonical English, parse, and answer in the same language."""
    lang = language or detect_language(text)
    if lang not in SUPPORTED:
        raise ValueError(f"unsupported language {lang!r}; supported: {SUPPORTED}")
    canonical, notes = to_canonical(text, lang)
    result = parse_request(canonical, seed)
    return {"language": lang, "detected": language is None, "canonical_text": canonical, "mapped": notes,
            "result": json.loads(result.model_dump_json()), "reply": localize(result, lang), "parse": result}
