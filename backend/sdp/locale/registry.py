"""Locale registry. Every locale is a JSON file in `packs/` (or in $SDP_LOCALE_DIR); adding one needs no code.

Pattern mini-language used by phone/ID/postal patterns:
  #        random digit                 [abc] [1-5]  random char from the class
  {check}  check digit(s) computed with the pack's `checksum` algorithm over everything before it
  {date:%Y%m%d}  random adult birth date in that strptime format
  anything else is literal
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, localcontext
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from sdp.locale.checksums import ALGORITHMS, compute_check, validate_checksum

PACK_DIR = Path(__file__).parent / "packs"
REQUIRED = ("code", "language", "country", "script", "names", "address", "phone", "national_id",
            "currency", "number", "date", "tax", "scale")
SCRIPT_MARKERS = {"Latin": ("LATIN",), "Arabic": ("ARABIC",), "Devanagari": ("DEVANAGARI",), "Han": ("CJK",)}


class LocaleError(ValueError):
    pass


def _expand_class(spec: str) -> str:
    out, i = [], 0
    while i < len(spec):
        if i + 2 < len(spec) and spec[i + 1] == "-":
            out.append("".join(chr(c) for c in range(ord(spec[i]), ord(spec[i + 2]) + 1)))
            i += 3
        else:
            out.append(spec[i])
            i += 1
    return "".join(out)


def expand_pattern(pattern: str, rng: np.random.Generator, checksum: str | None = None) -> str:
    out: list[str] = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "#":
            out.append(str(int(rng.integers(0, 10))))
        elif ch == "[":
            j = pattern.index("]", i)
            chars = _expand_class(pattern[i + 1:j])
            out.append(chars[int(rng.integers(0, len(chars)))])
            i = j
        elif ch == "{":
            j = pattern.index("}", i)
            token = pattern[i + 1:j]
            if token == "check":
                if not checksum:
                    raise LocaleError("pattern has {check} but pack declares no checksum")
                out.append(compute_check(checksum, "".join(out)))
            elif token.startswith("date:"):
                start, end = date(1950, 1, 1), date(2005, 12, 31)
                d = start + timedelta(days=int(rng.integers(0, (end - start).days + 1)))
                out.append(d.strftime(token[5:]))
            else:
                raise LocaleError(f"unknown pattern token {{{token}}}")
            i = j
        else:
            out.append(ch)
        i += 1
    return "".join(out)


@dataclass(frozen=True)
class LocalePack:
    data: dict[str, Any]

    # ---- identity
    @property
    def code(self) -> str:
        return self.data["code"]

    @property
    def script(self) -> str:
        return self.data["script"]

    def _pick(self, rng: np.random.Generator, items: list[str]) -> str:
        return items[int(rng.integers(0, len(items)))]

    def person_name(self, rng: np.random.Generator, latin: bool = False) -> str:
        n = self.data["names_latin"] if latin and "names_latin" in self.data else self.data["names"]
        first = self._pick(rng, n["first_male"] if rng.random() < 0.5 else n["first_female"])
        last = self._pick(rng, n["last"])
        joiner = n.get("joiner", " ")
        return f"{last}{joiner}{first}" if n.get("order") == "family_given" else f"{first}{joiner}{last}"

    def phone(self, rng: np.random.Generator) -> str:
        return expand_pattern(self.data["phone"]["pattern"], rng)

    def national_id(self, rng: np.random.Generator) -> str:
        nid = self.data["national_id"]
        return expand_pattern(nid["pattern"], rng, nid.get("checksum"))

    def address(self, rng: np.random.Generator, latin: bool = False) -> tuple[str, str]:
        a = self.data["address_latin"] if latin and "address_latin" in self.data else self.data["address"]
        city = self._pick(rng, a["cities"])
        text = a["format"].format(number=int(rng.integers(1, 200)), street=self._pick(rng, a["streets"]),
                                  city=city, postal=expand_pattern(a["postal_pattern"], rng))
        return text, city

    def birth_date(self, rng: np.random.Generator) -> date:
        return date(1950, 1, 1) + timedelta(days=int(rng.integers(0, 20000)))

    # ---- formatting / parsing
    def format_date(self, d: date | datetime) -> str:
        return d.strftime(self.data["date"]["format"])

    def parse_date(self, s: str) -> date | None:
        try:
            return datetime.strptime(s, self.data["date"]["format"]).date()
        except (ValueError, TypeError):
            return None

    def format_number(self, value: Decimal, decimals: int | None = None) -> str:
        num = self.data["number"]
        dec = self.data["currency"]["decimals"] if decimals is None else decimals
        q = Decimal(1).scaleb(-dec)
        with localcontext() as ctx:
            ctx.prec = 100  # amounts can legitimately exceed the default 28 digits
            v = value.quantize(q)
        neg, v = v < 0, abs(v)
        whole, _, frac = f"{v:f}".partition(".")
        if num.get("grouping") == "indian" and len(whole) > 3:
            head, tail = whole[:-3], whole[-3:]
            groups = []
            while len(head) > 2:
                groups.insert(0, head[-2:])
                head = head[:-2]
            whole = num["thousands"].join(([head] if head else []) + groups + [tail])
        else:
            groups = []
            while len(whole) > 3:
                groups.insert(0, whole[-3:])
                whole = whole[:-3]
            whole = num["thousands"].join([whole] + groups)
        s = whole + (num["decimal"] + frac if dec else "")
        return "-" + s if neg else s

    def format_currency(self, value: Decimal) -> str:
        c = self.data["currency"]
        sp = " " if c.get("space") else ""
        n = self.format_number(value)
        return f"{c['symbol']}{sp}{n}" if c["position"] == "before" else f"{n}{sp}{c['symbol']}"

    def currency_regex(self) -> re.Pattern[str]:
        c, num = self.data["currency"], self.data["number"]
        sp = " " if c.get("space") else ""
        body = rf"-?\d{{1,3}}(?:{re.escape(num['thousands'])}\d{{2,3}})*(?:{re.escape(num['decimal'])}\d{{{c['decimals']}}})?" \
            if c["decimals"] else r"-?\d{1,3}(?:%s\d{2,3})*" % re.escape(num["thousands"])
        sym = re.escape(c["symbol"])
        return re.compile(rf"^{sym}{sp}{body}$" if c["position"] == "before" else rf"^{body}{sp}{sym}$")

    def parse_currency(self, s: str) -> Decimal | None:
        if not isinstance(s, str) or not self.currency_regex().match(s):
            return None
        num, c = self.data["number"], self.data["currency"]
        t = s.replace(c["symbol"], "").replace(" ", "").replace(num["thousands"], "")
        t = t.replace(num["decimal"], ".")
        try:
            return Decimal(t)
        except ArithmeticError:
            return None

    # ---- validation helpers
    def phone_valid(self, s: str) -> bool:
        return bool(re.match(self.data["phone"]["regex"], s or ""))

    def national_id_valid(self, s: str) -> tuple[bool, str]:
        nid = self.data["national_id"]
        if not re.match(nid["regex"], s or ""):
            return False, f"does not match {nid['name']} format"
        if nid.get("checksum") and not validate_checksum(nid["checksum"], s):
            return False, f"{nid['name']} check digit invalid"
        return True, ""

    def script_ok(self, text: str) -> bool:
        """Every letter belongs to the locale's script (digits, spaces, punctuation ignored)."""
        markers = SCRIPT_MARKERS.get(self.script, (self.script.upper(),))
        for ch in text:
            if ch.isalpha() and not any(unicodedata.name(ch, "").startswith(m) for m in markers):
                return False
        return True

    def tax_rate(self, region: str | None, category: str | None = None) -> tuple[str, Decimal]:
        t = self.data["tax"]
        if category and category in t.get("category_rates", {}):
            return t["label"], Decimal(t["category_rates"][category])
        rate = t["regions"].get(region, t["default_rate"]) if region else t["default_rate"]
        return t["label"], Decimal(rate)


def _validate_pack(d: dict[str, Any], source: str) -> None:
    missing = [k for k in REQUIRED if k not in d]
    if missing:
        raise LocaleError(f"{source}: missing keys {missing}")
    nid = d["national_id"]
    for key in ("name", "pattern", "regex"):
        if key not in nid:
            raise LocaleError(f"{source}: national_id.{key} missing")
    if nid.get("checksum") and nid["checksum"] not in ALGORITHMS:
        raise LocaleError(f"{source}: unknown checksum {nid['checksum']!r}; known: {sorted(ALGORITHMS)}")
    for rx in (d["phone"]["regex"], nid["regex"]):
        re.compile(rx)
    if d["script"] not in SCRIPT_MARKERS:
        raise LocaleError(f"{source}: script must be one of {sorted(SCRIPT_MARKERS)}")
    for r in d["tax"]["regions"].values():
        Decimal(r)


class LocaleRegistry:
    def __init__(self, dirs: list[Path] | None = None) -> None:
        self.packs: dict[str, LocalePack] = {}
        extra = os.environ.get("SDP_LOCALE_DIR")
        for d in dirs or [PACK_DIR, *([Path(extra)] if extra else [])]:
            for f in sorted(Path(d).glob("*.json")):
                data = json.loads(f.read_text(encoding="utf-8"))
                _validate_pack(data, f.name)
                self.packs[data["code"]] = LocalePack(data)

    def get(self, code: str) -> LocalePack:
        if code in self.packs:
            return self.packs[code]
        lc = {k.lower(): k for k in self.packs}
        if code.lower() in lc:
            return self.packs[lc[code.lower()]]
        raise LocaleError(f"unknown locale {code!r}; available: {sorted(self.packs)}")

    def codes(self) -> list[str]:
        return sorted(self.packs)

    def find_by_alias(self, word: str) -> LocalePack | None:
        w = word.lower()
        for p in self.packs.values():
            if w in [a.lower() for a in p.data.get("aliases", [])]:
                return p
        return None


@lru_cache(maxsize=1)
def registry() -> LocaleRegistry:
    return LocaleRegistry()


def get_locale(code: str) -> LocalePack:
    return registry().get(code)
