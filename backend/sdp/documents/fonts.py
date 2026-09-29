"""Font discovery and registration for non-Latin PDFs.

Noto is preferred (Noto Naskh/Sans Arabic, Noto Sans Devanagari, Noto Sans), then common system fonts.
Search order: $SDP_FONT_DIR, backend/sdp/fonts/ (drop Noto TTFs here; see scripts/fetch_fonts.py), OS font folders.

Limits (reportlab): only TrueType outlines can be embedded, so the CFF-based Noto Sans CJK is skipped; for Chinese we use
a TrueType CJK face (e.g. WenQuanYi, Microsoft YaHei). Complex shaping (Devanagari conjuncts) is not available.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFError, TTFont

BUNDLED = Path(__file__).resolve().parent.parent / "fonts"

# (regular, bold) candidates in preference order; matching is case-insensitive on file name
CANDIDATES: dict[str, list[tuple[str, str | None]]] = {
    "latin": [("NotoSans-Regular.ttf", "NotoSans-Bold.ttf"), ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
              ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"), ("arial.ttf", "arialbd.ttf")],
    "arabic": [("NotoNaskhArabic-Regular.ttf", "NotoNaskhArabic-Bold.ttf"), ("NotoSansArabic-Regular.ttf", "NotoSansArabic-Bold.ttf"),
               ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"), ("arial.ttf", "arialbd.ttf"), ("tahoma.ttf", "tahomabd.ttf")],
    "devanagari": [("NotoSansDevanagari-Regular.ttf", "NotoSansDevanagari-Bold.ttf"), ("Nirmala.ttc", "NirmalaB.ttc")],
    "cjk": [("NotoSansSC-Regular.ttf", "NotoSansSC-Bold.ttf"), ("wqy-microhei.ttc", None), ("wqy-zenhei.ttc", None),
            ("msyh.ttc", "msyhbd.ttc"), ("simsun.ttc", None), ("DroidSansFallbackFull.ttf", None)],
}


def search_dirs() -> list[Path]:
    dirs = []
    if os.environ.get("SDP_FONT_DIR"):
        dirs.append(Path(os.environ["SDP_FONT_DIR"]))
    dirs.append(BUNDLED)
    if sys.platform.startswith("win"):
        dirs.append(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts")
    dirs += [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path("/Library/Fonts"), Path("/System/Library/Fonts"),
             Path.home() / ".fonts", Path.home() / "Library/Fonts"]
    return [d for d in dirs if d.exists()]


@lru_cache(maxsize=1)
def _index() -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for d in search_dirs():
        for p in d.rglob("*"):
            if p.suffix.lower() in (".ttf", ".ttc") and p.name.lower() not in idx:
                idx[p.name.lower()] = p
    return idx


@dataclass(frozen=True)
class FoundFont:
    script: str
    regular_path: Path
    bold_path: Path | None
    is_noto: bool


def find_font(script: str) -> FoundFont | None:
    idx = _index()
    for reg, bold in CANDIDATES.get(script, []):
        p = idx.get(reg.lower())
        if p:
            return FoundFont(script, p, idx.get(bold.lower()) if bold else None, reg.lower().startswith("noto"))
    return None


def ensure_font(script: str) -> tuple[str, str] | None:
    """Register (once) and return (regular, bold) reportlab font names for a script, or None if nothing usable is installed."""
    from sdp.documents import render  # local import: render depends on this module's callers

    name = f"sdp-{script}"
    if name in render._TTF:
        return name, render._TTF[name]
    f = find_font(script)
    if f is None:
        return None
    try:
        render.register_font(name, str(f.regular_path), str(f.bold_path) if f.bold_path else None)
    except render.MissingFontError:
        return None
    return name, render._TTF[name]


def describe() -> dict[str, dict]:
    """What is installed, for the API/UI and docs: script -> file and whether it is Noto."""
    out = {}
    for script in CANDIDATES:
        f = find_font(script)
        out[script] = {"file": f.regular_path.name if f else None, "noto": bool(f and f.is_noto)}
    return out
