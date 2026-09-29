"""Charts for the scorecard PDF, drawn from the small JSON the Visualize endpoint returns (never from rows). Native reportlab drawing, no image files.

Two-colour palette: REAL (slate blue) and SYNTHETIC (teal). Every chart is followed by its plain-English caption.
"""

from __future__ import annotations

import math
from typing import Any

from reportlab.pdfgen import canvas

REAL, SYN = (0.42, 0.50, 0.66), (0.05, 0.58, 0.53)
INK, MUTED, GRID = (0.11, 0.10, 0.09), (0.47, 0.44, 0.42), (0.88, 0.87, 0.86)
ORDER = ("distribution", "categorical", "correlation", "tstr", "cardinality", "privacy_distance", "locale_validity", "batch_summary")
TITLES = {"distribution": "Distribution", "categorical": "Category mix", "correlation": "Correlations", "tstr": "Usefulness for machine learning (TSTR)",
          "cardinality": "Rows per parent", "privacy_distance": "Distance to the closest real record", "locale_validity": "Valid for the country",
          "batch_summary": "Document batch"}


def _fold(s: Any) -> str:
    from sdp.scoring.pdf import ascii_fold
    return ascii_fold(s)


def _set(c: canvas.Canvas, color: tuple[float, float, float]) -> None:
    c.setFillColorRGB(*color)
    c.setStrokeColorRGB(*color)


def _text(c: canvas.Canvas, x: float, y: float, s: str, size: float = 8, color=MUTED, font: str = "Helvetica", align: str = "left") -> None:
    c.setFillColorRGB(*color)
    c.setFont(font, size)
    (c.drawRightString if align == "right" else c.drawCentredString if align == "center" else c.drawString)(x, y, _fold(s))


def _wrap(c: canvas.Canvas, text: str, width: float, size: float = 8) -> list[str]:
    out, cur = [], ""
    for w in _fold(text).split():
        t = f"{cur} {w}".strip()
        if c.stringWidth(t, "Helvetica", size) <= width:
            cur = t
        else:
            out.append(cur)
            cur = w
    return out + ([cur] if cur else [])


def _legend(c: canvas.Canvas, x: float, y: float, a: str = "Real", b: str = "Synthetic") -> None:
    for i, (lab, col) in enumerate(((a, REAL), (b, SYN))):
        _set(c, col)
        c.rect(x + i * 70, y, 8, 8, stroke=0, fill=1)
        _text(c, x + i * 70 + 12, y + 1, lab, 8)


# ------------------------------------------------------------------ charts
def gauge(c: canvas.Canvas, x: float, y: float, w: float, score: float, label: str) -> float:
    """Half-circle gauge: grey track, teal arc up to the score. Returns the height used."""
    r = min(w / 2 - 10, 70)
    cx, cy = x + w / 2, y - r - 6
    c.setLineWidth(9)
    c.setLineCap(1)
    for frac, col in ((1.0, GRID), (max(0.0, min(1.0, score / 100)), SYN if score >= 70 else (0.75, 0.25, 0.25))):
        if frac > 0:
            c.setStrokeColorRGB(*col)
            _arc(c, cx, cy, r, frac)
    c.setLineWidth(1)
    _text(c, cx, cy + 4, f"{score:.0f}", 26, INK, "Helvetica-Bold", "center")
    _text(c, cx, cy - 12, f"/ 100   {label}", 9, MUTED, "Helvetica", "center")
    return r + 30


def _arc(c: canvas.Canvas, cx: float, cy: float, r: float, frac: float) -> None:
    p = c.beginPath()
    steps = max(2, int(60 * frac))
    for i in range(steps + 1):
        a = math.pi - math.pi * frac * i / steps          # 180deg (left) sweeping clockwise over the top
        px, py = cx + r * math.cos(a), cy + r * math.sin(a)
        p.moveTo(px, py) if i == 0 else p.lineTo(px, py)
    c.drawPath(p, stroke=1, fill=0)


def _bars(c: canvas.Canvas, x: float, y: float, w: float, h: float, labels: list[str], series: list[tuple[list[float | None], tuple[float, float, float]]],
          vmax: float | None = None, pct: bool = False) -> None:
    """Grouped vertical bars in a w x h box whose top-left is (x, y)."""
    vals = [v for s, _ in series for v in s if v is not None]
    vmax = vmax or (max(vals) if vals else 1.0) or 1.0
    base = y - h + 14
    c.setStrokeColorRGB(*GRID)
    c.line(x, base, x + w, base)
    n, k = len(labels), len(series)
    gw = w / max(n, 1)
    bw = min(gw * 0.8 / max(k, 1), 18)
    for i in range(n):
        for j, (s, col) in enumerate(series):
            v = s[i]
            if v is None:
                continue
            bh = (h - 26) * (v / vmax)
            _set(c, col)
            c.rect(x + i * gw + (gw - bw * k) / 2 + j * bw, base, bw, max(bh, 0.4), stroke=0, fill=1)
    step = max(1, n // 8)
    for i in range(0, n, step):
        _text(c, x + i * gw + gw / 2, base - 9, labels[i][:10], 6, MUTED, align="center")
    _text(c, x, y - 4, f"{vmax * 100:.0f}%" if pct else f"{vmax:.2g}", 6)


def _heat(c: canvas.Canvas, x: float, y: float, size: float, m: list[list[float]], hi: tuple[float, float, float], lo: tuple[float, float, float] | None, title: str) -> None:
    n = len(m)
    cell = size / max(n, 1)
    for i in range(n):
        for j in range(n):
            v = m[i][j]
            tint = lo if (lo is not None and v < 0) else hi          # white -> teal for positive, white -> slate for negative
            col = tuple(1 + (tint[t] - 1) * min(1.0, abs(v)) for t in range(3))
            c.setFillColorRGB(*col)
            c.rect(x + j * cell, y - (i + 1) * cell, cell, cell, stroke=0, fill=1)
    _text(c, x + size / 2, y + 4, title, 8, INK, "Helvetica-Bold", "center")


def draw(c: canvas.Canvas, p: dict[str, Any], x: float, y: float, w: float) -> float:
    """Draw one chart (title, chart, caption) with its top-left at (x, y). Returns the y just below it."""
    typ = p["type"]
    _text(c, x, y, TITLES.get(typ, typ) + (f": {p['column']}" if p.get("column") else ""), 10, INK, "Helvetica-Bold")
    y -= 8
    h = 110
    if typ in ("distribution", "privacy_distance", "cardinality"):
        b = p["bins"]
        has_real = any(r.get("real") is not None for r in b)
        rl = "Real (unseen)" if typ == "privacy_distance" else "Real"
        if has_real:
            _legend(c, x + w - 150, y - 4, rl, "Synthetic")
        _bars(c, x, y - 10, w, h, [str(r["label"]) for r in b],
              ([([r["real"] for r in b], REAL)] if has_real else []) + [([r["synthetic"] for r in b], SYN)], pct=True)
        y -= h + 10
    elif typ == "categorical":
        cats = p["categories"]
        has_real = any(r.get("real") is not None for r in cats)
        if has_real:
            _legend(c, x + w - 150, y - 4)
        _bars(c, x, y - 10, w, h, [r["label"] for r in cats], ([([r["real"] for r in cats], REAL)] if has_real else []) + [([r["synthetic"] for r in cats], SYN)], pct=True)
        y -= h + 10
    elif typ == "correlation":
        n = len(p["columns"])
        size = min((w - 30) / 3, 120)
        top = y - 14
        _heat(c, x, top, size, p["real_matrix"], SYN, REAL, "Real")
        _heat(c, x + size + 15, top, size, p["synthetic_matrix"], SYN, REAL, "Synthetic")
        _heat(c, x + 2 * (size + 15), top, size, p["diff_matrix"], (0.85, 0.55, 0.20), (0.85, 0.55, 0.20), "Difference")
        _text(c, x, top - size - 10, f"{n} numeric columns: " + ", ".join(p["columns"][:8]) + ("..." if n > 8 else ""), 6)
        y = top - size - 16
    elif typ == "tstr":
        ms = p["models"]
        _legend(c, x + w - 150, y - 4, "Trained on real", "Trained on synthetic")
        _bars(c, x, y - 10, w, h, [m["model"] for m in ms], [([m["real"] for m in ms], REAL), ([m["synthetic"] for m in ms], SYN)], vmax=1.0)
        y -= h + 10
    elif typ == "locale_validity":
        ch = p["checks"]
        yy = y - 12
        for r in ch:
            _text(c, x, yy, r["label"], 8, INK)
            c.setFillColorRGB(*GRID)
            c.rect(x + 150, yy - 1, w - 210, 6, stroke=0, fill=1)
            _set(c, SYN if r["pass_pct"] >= 95 else (0.8, 0.4, 0.3))
            c.rect(x + 150, yy - 1, (w - 210) * r["pass_pct"] / 100, 6, stroke=0, fill=1)
            _text(c, x + w, yy, f"{r['pass_pct']:.0f}%", 8, INK, align="right")
            yy -= 14
        y = yy
    elif typ == "batch_summary":
        tot = max(p["total"], 1)
        bw = w - 20
        xs = x
        for n, col in ((p["succeeded"] + p["skipped"], SYN), (p["failed"], (0.78, 0.36, 0.30))):
            _set(c, col)
            c.rect(xs, y - 22, bw * n / tot, 12, stroke=0, fill=1)
            xs += bw * n / tot
        _text(c, x, y - 36, f"{p['succeeded']} succeeded   {p['failed']} failed   reconciliation {p['reconciliation_rate_pct']:.1f}%   success {p['success_rate_pct']:.1f}%", 8, INK)
        y -= 46
    for line in _wrap(c, p.get("caption", ""), w, 7.5):
        y -= 10
        _text(c, x, y, line, 7.5, MUTED)
    return y - 22


def draw_pages(c: canvas.Canvas, visuals: dict[str, Any], report: dict[str, Any], page_size: tuple[float, float]) -> None:
    """One or more pages of charts: the gauge first, then every supplied view in a fixed order."""
    W, H = page_size
    m = 50
    c.showPage()
    y = H - 60
    _text(c, m, y, "Charts", 14, INK, "Helvetica-Bold")
    _legend(c, W - m - 150, y)
    y -= 20
    used = gauge(c, m, y, W - 2 * m, float(report["trust_score"]), str(report["label"]))
    y -= used + 10
    for typ in ORDER:
        p = visuals.get(typ)
        if not isinstance(p, dict) or p.get("type") != typ:
            continue
        need = 190 if typ != "locale_validity" else 40 + 14 * len(p.get("checks", []))
        if y - need < 50:
            c.showPage()
            y = H - 60
        y = draw(c, p, m, y, W - 2 * m)
