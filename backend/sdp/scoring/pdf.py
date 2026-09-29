"""Trust Score PDF export (reportlab, built-in Helvetica; text is ASCII-folded so it can't fail on glyphs)."""

from __future__ import annotations

import io
import unicodedata
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

INK, MUTED, RULE, ACCENT = (0.11, 0.10, 0.09), (0.47, 0.44, 0.42), (0.9, 0.89, 0.89), (0.18, 0.36, 0.92)


def ascii_fold(s: Any) -> str:
    s = str(s).replace("—", "-").replace("–", "-")
    return unicodedata.normalize("NFKD", s).encode("ascii", "replace").decode("ascii")


def _wrap(c: canvas.Canvas, text: str, font: str, size: int, width: float) -> list[str]:
    lines, cur = [], ""
    for w in ascii_fold(text).split():
        trial = f"{cur} {w}".strip()
        if c.stringWidth(trial, font, size) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    return lines + ([cur] if cur else [])


def trust_report_pdf(report: dict[str, Any], visuals: dict[str, Any] | None = None) -> bytes:
    """`visuals` (optional): Visualize payloads keyed by type; when given, a Charts section (gauge + each chart with its caption) is appended."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    m = 50
    y = H - 60

    def text(x: float, y: float, s: str, font="Helvetica", size=10, color=INK) -> None:
        c.setFillColorRGB(*color)
        c.setFont(font, size)
        c.drawString(x, y, ascii_fold(s))

    text(m, y, report["title"], "Helvetica-Bold", 16)
    text(m, y - 16, f"Trust Score report - {report['generated_at']}", size=9, color=MUTED)
    y -= 80
    text(m, y, f"{report['trust_score']:.0f}", "Helvetica-Bold", 54, ACCENT)
    text(m + 100, y + 22, f"/ 100   {report['label']}", "Helvetica-Bold", 14)
    y -= 26
    for line in _wrap(c, report["verdict"], "Helvetica", 10, W - 2 * m):
        text(m, y, line)
        y -= 14
    y -= 14
    for s in report["sub_scores"]:
        c.setStrokeColorRGB(*RULE)
        c.line(m, y + 10, W - m, y + 10)
        text(m, y - 6, s["label"], "Helvetica-Bold", 12)
        text(W - m - 150, y - 6, f"weight {s['weight'] * 100:.0f}%", size=9, color=MUTED)
        text(W - m - 50, y - 6, f"{s['score']:.0f}", "Helvetica-Bold", 12)
        y -= 24
        for comp in s["components"]:
            text(m + 12, y, comp["label"], size=9)
            text(m + 170, y, comp["summary"][:80], size=8, color=MUTED)
            text(W - m - 50, y, f"{comp['score']:.0f}", size=9)
            y -= 13
        y -= 12
    if report["gates"]:
        text(m, y, "Review before sharing", "Helvetica-Bold", 11)
        y -= 14
        for g in report["gates"]:
            text(m + 8, y, "- " + g, size=9)
            y -= 13

    cols = report["details"].get("fidelity_columns", {})
    if cols:
        c.showPage()
        y = H - 60
        text(m, y, "Per-column fidelity (lowest first)", "Helvetica-Bold", 13)
        y -= 24
        for name, v in sorted(((k, x) for k, x in cols.items() if x["score"] is not None), key=lambda kv: kv[1]["score"])[:35]:
            text(m, y, name, size=9)
            text(m + 200, y, f"{v['test']} = {v.get('statistic', float('nan')):.3f}", size=9, color=MUTED)
            text(W - m - 50, y, f"{v['score']:.0f}", size=9)
            y -= 14
            if y < 60:
                break
    if visuals:
        from sdp.scoring.pdf_charts import draw_pages
        draw_pages(c, visuals, report, A4)
    c.save()
    return buf.getvalue()
