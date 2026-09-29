"""M5. Code-mixed text: Roman Urdu-English, Hinglish and Spanglish, with a configurable mixing level and topic.

Method (matrix-language frame): each sentence template keeps the base language's grammar and function words; every
content-word slot is filled with the English form with probability `level` (0 = pure base language, 1 = every slot English),
otherwise with the base-language form. Fillers ("yaar", "actually") are added at the same rates. Because slots are
inserted where the base grammar allows a noun/verb, the result resembles real code-switching. Token-level language tags are
returned as ground truth for language-ID / NER training.

Language pairs and topics are JSON files in `codemix_data/`; adding one is data only. Templates are hand-written and small:
have a fluent speaker review them before using the output as evidence about how people actually write.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "codemix_data"
SLOT = re.compile(r"\{(\w+)\}")
TOKEN = re.compile(r"[\w'’áéíóúñüÁÉÍÓÚÑ]+|[^\w\s]", re.UNICODE)


class CodeMixError(ValueError):
    pass


@lru_cache(maxsize=1)
def pairs() -> dict[str, dict[str, Any]]:
    out = {}
    for f in sorted(DATA_DIR.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        out[d["name"]] = d
    return out


def topics(pair: str) -> list[str]:
    return sorted(_pair(pair)["topics"])


def _pair(name: str) -> dict[str, Any]:
    if name not in pairs():
        raise CodeMixError(f"unknown language pair {name!r}; available: {sorted(pairs())}")
    return pairs()[name]


def _tag(text: str, en_words: set[str], base_words: set[str]) -> list[tuple[str, str]]:
    """Language tag per token. Words present only in the English lexicon side are 'en'; punctuation is 'punct'."""
    out = []
    for t in TOKEN.findall(text):
        if not re.match(r"\w", t):
            out.append((t, "punct"))
        elif t.lower() in en_words and t.lower() not in base_words:
            out.append((t, "en"))
        else:
            out.append((t, "base"))
    return out


def generate_code_mixed(n: int, pair: str = "roman_urdu", topic: str = "banking", level: float = 0.5, seed: int = 0) -> pd.DataFrame:
    """n sentences. level in [0, 1]. Columns: text, pair, topic, level, slots, en_slots, realized_mix, tokens (list of (token, lang))."""
    if not 0.0 <= level <= 1.0:
        raise CodeMixError("level must be within [0, 1]")
    p = _pair(pair)
    if topic not in p["topics"]:
        raise CodeMixError(f"unknown topic {topic!r} for {pair}; available: {sorted(p['topics'])}")
    t = p["topics"][topic]
    rng = np.random.default_rng(seed)
    en_words = {w.lower() for v in t["lexicon"].values() for w in TOKEN.findall(v["e"])} | {w.lower() for w in p["fillers"]["en"]}
    base_words = {w.lower() for v in t["lexicon"].values() for w in TOKEN.findall(v["b"])} | {w.lower() for w in p["fillers"]["base"]}
    # function words are always base; a word that is spelled the same in both languages ("card", "app") counts as base
    rows = []
    for _ in range(n):
        tpl = t["templates"][int(rng.integers(0, len(t["templates"])))]
        slots = SLOT.findall(tpl)
        en_slots = 0
        swappable = sum(1 for k in slots if t["lexicon"][k]["e"].lower() != t["lexicon"][k]["b"].lower())  # same spelling in both = not a switch

        def fill(m: re.Match[str]) -> str:
            nonlocal en_slots
            entry = t["lexicon"][m.group(1)]
            use_en = rng.random() < level
            en_slots += use_en and entry["e"].lower() != entry["b"].lower()
            return entry["e"] if use_en else entry["b"]
        text = SLOT.sub(fill, tpl)
        if rng.random() < 0.25 * (1 - level) and p["fillers"]["base"]:
            text = f"{p['fillers']['base'][int(rng.integers(0, len(p['fillers']['base'])))].capitalize()}, {text[0].lower()}{text[1:]}"
        elif rng.random() < 0.4 * level and p["fillers"]["en"]:
            text = f"{p['fillers']['en'][int(rng.integers(0, len(p['fillers']['en'])))].capitalize()}, {text[0].lower()}{text[1:]}"
        rows.append({"text": text, "pair": pair, "topic": topic, "level": level, "slots": len(slots), "en_slots": int(en_slots),
                     "realized_mix": en_slots / swappable if swappable else float("nan"), "tokens": _tag(text, en_words, base_words)})
    return pd.DataFrame(rows)


def english_token_ratio(tokens: list[tuple[str, str]]) -> float:
    words = [lang for _, lang in tokens if lang != "punct"]
    return sum(1 for w in words if w == "en") / len(words) if words else 0.0
