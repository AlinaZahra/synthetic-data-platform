"""Right-to-left text preparation for PDF drawing.

reportlab draws glyphs left to right in the order it is given, and does no Arabic shaping. So for Arabic-script text we
  1. reshape letters into their contextual presentation forms (initial/medial/final/isolated), then
  2. run the Unicode bidirectional algorithm to get *visual* order, keeping numbers and Latin runs left-to-right.
"""

from __future__ import annotations

import re

import arabic_reshaper
from bidi import get_display

_ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_reshaper = arabic_reshaper.ArabicReshaper(configuration={"delete_harakat": False, "support_ligatures": True})


def has_arabic(text: str) -> bool:
    return bool(_ARABIC.search(text))


def shape(text: str, rtl: bool) -> str:
    """Visual-order string ready for left-to-right glyph drawing. Pure-LTR text in an LTR document is returned unchanged."""
    if not text:
        return text
    if not has_arabic(text):
        # pure LTR (Latin, digits, CJK) never needs reordering, even inside an RTL document: running it through the
        # bidi algorithm would move a leading number ("57 High Street") to the wrong end of the line
        return text
    return get_display(_reshaper.reshape(text), base_dir="R" if rtl else "L")
