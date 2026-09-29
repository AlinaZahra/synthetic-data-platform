"""M4. Transliteration variants: the same entity in several scripts and spellings, with a shared entity ID.

  محمد علی  /  Muhammad Ali  /  Mohammad Ali  /  Mohammed Ali  /  Ali, Muhammad  /  M. Ali  /  MUHAMMAD ALI

Built from the locale pack's `translit` table (native token -> Latin spellings, first = canonical), so adding a language
is data. Latin-script locales (es, fr, en-*) get diacritic-stripped, reordered, initial and upper-case variants.
Attributes that identify the person (DOB, national ID, phone) are shared by all variants of an entity, so a matcher can
be scored on name evidence alone or on name + attributes. `confusers` adds DIFFERENT people who carry an identical name.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from sdp.locale.coherent import native_name_parts, strip_diacritics
from sdp.locale.registry import get_locale

STYLES = ["native_canonical", "latin_canonical", "latin_spelling", "native_reordered", "latin_reordered", "latin_initial",
          "latin_upper", "latin_no_diacritics", "latin_typo"]


def _variants(pack, tokens_native: list[str]) -> list[list[str]]:
    tr = pack.data.get("translit")
    return [(tr[t] if tr and t in tr else [t]) for t in tokens_native]


def name_variants(pack, first: str, last: str, rng: np.random.Generator, max_variants: int = 12, typos: bool = False) -> list[tuple[str, str, str]]:
    """[(style, script, text)] for one person; the first item is always the native canonical form."""
    latin_script = pack.script == "Latin"
    n = pack.data["names"]
    family_first = n.get("order") == "family_given"
    joiner = n.get("joiner", " ")
    native = f"{last}{joiner}{first}" if family_first else f"{first}{joiner}{last}"
    (f_vars,), (l_vars,) = _variants(pack, [first]), _variants(pack, [last])
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    def add(style: str, script: str, text: str) -> None:
        if text not in seen:
            seen.add(text)
            out.append((style, script, text))

    add("native_canonical", "native" if not latin_script else "latin", native)
    def order(f: str, l: str, sep: str = " ") -> str:
        return f"{l}{sep}{f}" if family_first else f"{f}{sep}{l}"
    add("latin_canonical", "latin", order(f_vars[0], l_vars[0]))
    # spelling variants: vary the given name first (the usual source of mismatches), then the family name, then both
    combos = [(f, l_vars[0]) for f in f_vars[1:]] + [(f_vars[0], l) for l in l_vars[1:]] +         [(f, l) for f in f_vars[1:] for l in l_vars[1:]]
    for f, l in itertools.islice(combos, 5):
        add("latin_spelling", "latin", order(f, l))
    if not latin_script:
        add("native_reordered", "native", f"{first}{joiner}{last}" if family_first else f"{last}{joiner}{first}")
    add("latin_reordered", "latin", f"{l_vars[0]}, {f_vars[0]}")
    add("latin_initial", "latin", f"{f_vars[0][0]}. {l_vars[0]}")
    add("latin_upper", "latin", order(f_vars[0], l_vars[0]).upper())
    stripped = strip_diacritics(order(f_vars[0], l_vars[0]))
    if latin_script and stripped != order(f_vars[0], l_vars[0]):
        add("latin_no_diacritics", "latin", stripped)
    if typos:
        s = order(f_vars[0], l_vars[0])
        k = int(rng.integers(1, max(2, len(s) - 1)))
        add("latin_typo", "latin", s[:k] + s[k + 1:] if len(s) > 4 else s + s[-1])
    return out[:max_variants]


def generate_entities(n_entities: int, locale: str = "ur-PK", seed: int = 0, variants_per_entity: int = 5,
                      confusers: float = 0.0, typos: bool = False) -> pd.DataFrame:
    """Rows = name records. Columns: entity_id, record_id, name, script, style, canonical, dob, national_id, phone, confuser_of.

    Records with the same entity_id are the same person. confuser_of (if set) names the entity whose NAME this different
    person duplicates (a hard negative for matchers).
    """
    pack = get_locale(locale)
    rng = np.random.default_rng(seed)
    people: list[dict[str, Any]] = []
    for e in range(n_entities):
        nm = native_name_parts(pack, rng)
        people.append({"entity_id": f"E{e + 1:06d}", "first": nm["first"], "last": nm["last"], "dob": pack.birth_date(rng).isoformat(),
                       "national_id": pack.national_id(rng), "phone": pack.phone(rng), "confuser_of": None})
    for c in range(int(round(confusers * n_entities))):
        src = people[int(rng.integers(0, n_entities))]
        people.append({"entity_id": f"E{n_entities + c + 1:06d}", "first": src["first"], "last": src["last"],
                       "dob": pack.birth_date(rng).isoformat(), "national_id": pack.national_id(rng), "phone": pack.phone(rng),
                       "confuser_of": src["entity_id"]})
    rows = []
    for p in people:
        vs = name_variants(pack, p["first"], p["last"], rng, typos=typos)
        keep = [vs[0], vs[1]] if len(vs) > 1 else vs[:1]           # native + canonical Latin always present
        rest = vs[2:]
        if rest and variants_per_entity > 2:
            idx = sorted(rng.choice(len(rest), size=min(len(rest), variants_per_entity - 2), replace=False))
            keep += [rest[i] for i in idx]
        for j, (style, script, text) in enumerate(keep):
            rows.append({"entity_id": p["entity_id"], "record_id": f"{p['entity_id']}-{j + 1}", "name": text, "script": script, "style": style,
                         "canonical": j == 0, "dob": p["dob"], "national_id": p["national_id"], "phone": p["phone"],
                         "confuser_of": p["confuser_of"]})
    return pd.DataFrame(rows)


def matching_pairs(df: pd.DataFrame, n_negative: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Labelled record pairs for evaluating a matcher: every within-entity pair is a match (label 1);
    non-matches (label 0) are sampled across entities, preferring hard negatives (confusers with the same name)."""
    rng = np.random.default_rng(seed)
    pos = [(a, b, 1) for _, g in df.groupby("entity_id") for a, b in itertools.combinations(g["record_id"], 2)]
    neg: list[tuple[str, str, int]] = []
    conf = df[df["confuser_of"].notna()]
    for _, r in conf.iterrows():
        twin = df[(df["entity_id"] == r["confuser_of"]) & (df["name"] == r["name"])]
        neg += [(r["record_id"], t, 0) for t in twin["record_id"]]
    target = n_negative if n_negative is not None else len(pos)
    ids = df["record_id"].to_numpy()
    ent = dict(zip(df["record_id"], df["entity_id"]))
    while len(neg) < target and len(ids) > 1:
        a, b = rng.choice(ids, 2, replace=False)
        if ent[a] != ent[b]:
            neg.append((a, b, 0))
    return pd.DataFrame(pos + neg[:target] if n_negative is not None else pos + neg, columns=["record_a", "record_b", "label"])
