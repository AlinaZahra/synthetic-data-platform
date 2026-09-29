"""PDF rendering (reportlab) for every document type, with RTL mirroring, Arabic shaping and embedded Unicode fonts.

Fonts: the three built-ins (WinAnsi), or TrueType fonts registered with `register_font`. For native-script documents the
renderer auto-selects an installed font for the script (Noto preferred, see documents/fonts.py) and *embeds a subset* of it.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFError, TTFont
from reportlab.pdfgen import canvas

from sdp.documents.layout import LAYOUTS, Fmt, Layout
from sdp.documents.shaping import shape

BUILTIN_FONTS = {"Helvetica": "Helvetica-Bold", "Times-Roman": "Times-Bold", "Courier": "Courier-Bold"}
_TTF: dict[str, str] = {}  # regular name -> bold name (same face if no bold supplied)


class MissingFontError(RuntimeError):
    """The requested font is not built in / registered / installed (not retryable)."""


class RenderOverflowError(RuntimeError):
    """A value could not be fitted into its column even at the minimum font size."""


def register_font(name: str, ttf_path: str, bold_ttf_path: str | None = None) -> None:
    try:
        pdfmetrics.registerFont(TTFont(name, ttf_path))
        bold = name
        if bold_ttf_path:
            bold = name + "-Bold"
            pdfmetrics.registerFont(TTFont(bold, bold_ttf_path))
    except (TTFError, OSError) as e:
        raise MissingFontError(f"cannot load font file {ttf_path!r}: {e}") from e
    _TTF[name] = bold


def resolve_font(name: str) -> tuple[str, str]:
    if name in BUILTIN_FONTS:
        return name, BUILTIN_FONTS[name]
    if name in _TTF:
        return name, _TTF[name]
    raise MissingFontError(f"font {name!r} is not available; built-ins: {sorted(BUILTIN_FONTS)}, registered: {sorted(_TTF)}")


def font_supports(font: str, text: str) -> bool:
    if font in pdfmetrics.standardFonts:  # WinAnsi (cp1252) encoded
        try:
            text.encode("cp1252")
            return True
        except UnicodeEncodeError:
            return False
    face = pdfmetrics.getFont(font).face
    return all(ord(ch) in face.charToGlyph or ch.isspace() for ch in text)


def choose_font(presentation: dict[str, Any], font: str, strict: bool = False) -> tuple[str, str]:
    """Native-script documents with the default font get an installed Unicode font for their script.
    strict=True uses exactly the font asked for (used for an explicit fallback font)."""
    if presentation.get("native") and font in ("Helvetica", "auto") and not strict:
        from sdp.documents.fonts import ensure_font
        script = presentation.get("script", "latin")
        got = ensure_font(script)
        if got is None:
            raise MissingFontError(f"no TrueType font installed for the {script} script; install Noto (fonts-noto-core) or "
                                   f"set SDP_FONT_DIR / place TTFs in backend/sdp/fonts")
        return got
    if font == "auto":
        font = "Helvetica"
    return resolve_font(font)


@dataclass
class RenderResult:
    pdf: bytes
    pages: int
    fonts_used: list[str]
    unrenderable: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    embedded_font: bool = False
    direction: str = "ltr"
    boxes: list[dict[str, Any]] = field(default_factory=list)   # every drawn field: key, text, raw value, page, bbox (points, top-left origin)
    page_size: tuple[float, float] = (595.28, 841.89)


def build_layout(doc: dict[str, Any], font: str | None = None) -> Layout:
    """The layout a document renders with (also used by the UI preview; `font=None` skips font-capability fallbacks)."""
    pres = doc.get("presentation") or {}
    supports = (lambda t: font_supports(font, t)) if font else None
    f = Fmt(doc["locale"], pres.get("native", False), pres.get("native_digits", False), supports)
    return LAYOUTS[doc.get("doc_type", "invoice")](doc, f)


def render_document(doc: dict[str, Any], font: str = "Helvetica", strict: bool = False) -> RenderResult:
    from sdp.documents.types import DOC_TYPES
    dt = DOC_TYPES.get(doc.get("doc_type", "invoice"))
    if dt is not None and dt.render is not None:  # template document types render themselves (HTML -> PDF)
        return dt.render(doc, font, strict)
    pres = doc.get("presentation") or {}
    regular, bold = choose_font(pres, font, strict)
    layout = build_layout(doc, regular)
    rtl = layout.direction == "rtl"
    native = bool(pres.get("native", False))
    fmt = Fmt(doc["locale"], native, pres.get("native_digits", False))

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4, invariant=1)  # invariant=1: byte-identical output for identical input
    W, H = A4
    m = 42.0
    CW = W - 2 * m
    unrenderable: set[str] = set()
    warnings: list[str] = []
    boxes: list[dict[str, Any]] = []
    page = 1

    def X(lx: float) -> float:
        """Logical distance from the reading-start edge -> page x (mirrored for RTL)."""
        return W - m - lx if rtl else m + lx

    def txt(lx: float, y: float, s: str, f: str = regular, size: float = 9, align: str = "start", maxw: float | None = None, key: str | None = None) -> None:
        orig = s
        s = shape(s, rtl) if native else s
        if not font_supports(f, s):
            unrenderable.update(ch for ch in s if not font_supports(f, ch))
            s = "".join(ch if font_supports(f, ch) else "?" for ch in s)
        if maxw is not None:
            while size > 6 and pdfmetrics.stringWidth(s, f, size) > maxw:
                size -= 0.5
            if pdfmetrics.stringWidth(s, f, size) > maxw:
                if align == "end":
                    raise RenderOverflowError(f"value {s!r} does not fit its column")
                while len(s) > 1 and pdfmetrics.stringWidth(s + "...", f, size) > maxw:
                    s = s[:-1]
                s += "..."
                warnings.append("text truncated")
        c.setFont(f, size)
        right_side = (align == "start") == rtl  # start-aligned text hugs the right edge in RTL
        (c.drawRightString if right_side else c.drawString)(X(lx), y, s)
        if key:
            w, x = pdfmetrics.stringWidth(s, f, size), X(lx)
            x0, x1 = (x - w, x) if right_side else (x, x + w)
            boxes.append({"key": key, "text": orig.replace("\u00a0", " "), "value": layout.raw.get(key), "page": page,
                          "bbox": [x0, H - y - size * 0.78, x1, H - y + size * 0.22], "size": size})

    def header(page_no: int) -> float:
        y = H - 50
        ks = layout.keys
        for i, (t, b) in enumerate(layout.header_start):
            txt(0, y - 15 * i, t, bold if b else regular, 14 if (b and i == 0) else 8, maxw=CW * 0.55, key=(ks.get("header_start") or [None] * 9)[i])
        txt(CW, y, layout.title, bold, 14, "end", key="title")
        end_keys = (ks.get("header_end") or []) + [ks.get("number")]
        for i, t in enumerate(layout.header_end + [layout.number]):
            txt(CW, y - 15 - 12 * i, t, size=8 if i < len(layout.header_end) else 9, align="end", maxw=CW * 0.5, key=end_keys[i] if i < len(end_keys) else None)
        y2 = H - 122
        if layout.block_label:
            txt(0, y2, layout.block_label, bold, 9, key="block_label")
            for i, t in enumerate(layout.block_lines):
                txt(0, y2 - 13 * (i + 1), t, size=9 if i == 0 else 8, maxw=CW * 0.6, key=(ks.get("block") or [None] * 9)[i])
        txt(CW, H - 150, f"{layout.page_label} {fmt.localize(str(page_no))}", size=8, align="end", key="page_number")
        return H - 175

    edges = [0.0]
    for col in layout.columns:
        edges.append(edges[-1] + col.width * CW)

    def table_head(y: float) -> float:
        c.setStrokeColorRGB(0.6, 0.6, 0.6)
        c.line(m, y + 12, W - m, y + 12)
        for i, col in enumerate(layout.columns):
            ck = (layout.keys.get("columns") or [None] * 9)[i]
            txt(edges[i] if col.align == "start" else edges[i + 1], y, col.label, bold, align=col.align, maxw=edges[i + 1] - edges[i] - 4, key=f"header.{ck}" if ck else None)
        c.line(m, y - 5, W - m, y - 5)
        return y - 20

    y = table_head(header(page))
    ckeys = layout.keys.get("columns") or [None] * 9
    for r, row in enumerate(layout.rows):
        if y < 150:
            c.showPage()
            page += 1
            y = table_head(header(page))
        for i, (col, cell) in enumerate(zip(layout.columns, row)):
            if cell:
                txt(edges[i] if col.align == "start" else edges[i + 1], y, cell, align=col.align, maxw=edges[i + 1] - edges[i] - 6, key=f"row{r}.{ckeys[i]}" if ckeys[i] else None)
        y -= 15
    if y < 60 + 16 * (len(layout.totals) + len(layout.notes)):
        c.showPage()
        page += 1
        y = header(page)
    y -= 10
    for ti, (label, value, strong) in enumerate(layout.totals):
        tk = (layout.keys.get("totals") or [None] * 20)[ti]
        txt(CW - 130, y, label, bold if strong else regular, 11 if strong else 9, "end", maxw=CW - 140, key=f"{tk}.label" if tk else None)
        txt(CW, y, value, bold if strong else regular, 11 if strong else 9, "end", maxw=125, key=tk)
        y -= 16 if strong else 14
    for ni, n in enumerate(layout.notes):
        y -= 4
        txt(0, y, n, size=8, key=f"note{ni}")
        y -= 11
    c.save()
    pdf = buf.getvalue()
    return RenderResult(pdf=pdf, pages=page, fonts_used=sorted({regular, bold}), unrenderable=sorted(unrenderable),
                        warnings=sorted(set(warnings)), embedded_font=b"/FontFile2" in pdf, direction=layout.direction, boxes=boxes, page_size=(W, H))


def render_invoice(inv: dict[str, Any], font: str = "Helvetica") -> RenderResult:
    return render_document(inv, font)


def count_pdf_pages(pdf: bytes) -> int:
    """Independent page count straight from the PDF bytes."""
    return len(re.findall(rb"/Type\s*/Page(?![s\w])", pdf))
