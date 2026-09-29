"""D4. Scan-style realism: turn a rendered page into something that looks like a scanned/photographed paper document,
and export labelled ground truth (field values + bounding boxes) for OCR / document-AI training.

Pipeline per page (seeded; every step is configurable and can be switched off):
  1 rasterise the PDF at `dpi`            2 paper tint + uneven lighting        3 handwriting (signature, note, chosen fields)
  4 stamp overlays                        5 rotation + horizontal skew           6 gaussian blur
  7 sensor noise                          8 low resolution                       9 JPEG artifacts
Bounding boxes come from the renderer (exact text extents, `key` = semantic field). Geometric steps transform the boxes with the
same affine matrix as the image; resolution changes rescale them; stamps and handwriting add their own labelled boxes.

Labels: {"pages": [{"image", "width", "height", "augmentations", "fields": [{"key", "text", "value", "bbox", "polygon",
"kind": text|handwritten|stamp, "handwritten"}]}]} in output-pixel coordinates (origin top-left). `text` is what is printed
(logical order, also for RTL); `value` is the unformatted value from the document JSON when one exists.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pydantic import BaseModel, ConfigDict, Field

from sdp.documents.render import RenderResult, render_document

STAMP_TEXTS = ["PAID", "RECEIVED", "APPROVED", "COPY", "VERIFIED", "PROCESSED"]
NOTES = ["checked ok", "paid cash", "recd - thanks", "see email", "urgent", "filed"]
LATIN_NAMES = ["A. Malik", "S. Ahmed", "R. Sharma", "M. Garcia", "L. Chen", "J. Smith"]


class ScanConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = 0
    dpi: int = Field(150, ge=72, le=300)
    rotate_deg: float = Field(1.0, ge=0, le=10, description="max |rotation|; the actual angle is drawn uniformly from [-max, max]")
    skew: float = Field(0.01, ge=0, le=0.2, description="max horizontal shear")
    blur_radius: float = Field(0.6, ge=0, le=6)
    noise_sigma: float = Field(6.0, ge=0, le=60)
    jpeg_quality: int = Field(60, ge=5, le=100)
    low_res_scale: float = Field(1.0, ge=0.2, le=1.0, description="1.0 = keep resolution; 0.5 = half the pixels per side")
    stamps: int = Field(1, ge=0, le=6)
    stamp_text: str | None = Field(None, max_length=20)
    handwriting: bool = True
    handwrite_fields: list[str] = Field(default_factory=list, description="field keys re-drawn by 'hand' (e.g. total, customer_name)")
    paper_tint: bool = True
    output: Literal["png", "jpeg"] = "png"


PRESETS: dict[str, dict[str, Any]] = {
    "clean": dict(rotate_deg=0, skew=0, blur_radius=0, noise_sigma=0, jpeg_quality=100, low_res_scale=1.0, stamps=0, handwriting=False, paper_tint=False),
    "office_scan": dict(rotate_deg=0.8, skew=0.005, blur_radius=0.5, noise_sigma=5, jpeg_quality=70, stamps=1, handwriting=True),
    "photocopy": dict(rotate_deg=1.5, skew=0.01, blur_radius=0.9, noise_sigma=14, jpeg_quality=45, stamps=1, handwriting=True),
    "bad_fax": dict(rotate_deg=2.5, skew=0.03, blur_radius=1.3, noise_sigma=24, jpeg_quality=25, low_res_scale=0.55, stamps=2, handwriting=True),
    "phone_photo": dict(rotate_deg=4, skew=0.06, blur_radius=0.8, noise_sigma=10, jpeg_quality=55, low_res_scale=0.8, stamps=1, handwriting=True),
}


def preset(name: str, **overrides: Any) -> ScanConfig:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    return ScanConfig(**{**PRESETS[name], **overrides})


@dataclass
class ScanResult:
    images: list[bytes]
    labels: dict[str, Any]
    mime: str

    def to_zip(self) -> bytes:
        buf = io.BytesIO()
        ext = "png" if self.mime == "image/png" else "jpg"
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for i, b in enumerate(self.images):
                z.writestr(f"page-{i + 1}.{ext}", b)
            z.writestr("labels.json", json.dumps(self.labels, indent=2, ensure_ascii=False))
            lines = [json.dumps({"image": p["image"], "key": f["key"], "text": f["text"], "bbox": f["bbox"], "kind": f["kind"]}, ensure_ascii=False)
                     for p in self.labels["pages"] for f in p["fields"]]
            z.writestr("labels.jsonl", "\n".join(lines))
        return buf.getvalue()


# ------------------------------------------------------------------ helpers
def _poly(bbox: list[float]) -> list[list[float]]:
    x0, y0, x1, y1 = bbox
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _bbox_of(poly: list[list[float]]) -> list[float]:
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=max(6, size))


def _ascii(s: str) -> str:
    return s if s.isascii() else ""


def _handwrite(img: Image.Image, text: str, xy: tuple[float, float], height: float, rng: np.random.Generator,
               ink: tuple[int, int, int]) -> list[float]:
    """Draw `text` as jittery pen strokes (per-glyph rotation, baseline wobble, spacing, ink opacity). Returns its bbox."""
    scale = 3
    font = _font(int(height * scale * 0.95))
    x = xy[0]
    x0, y0, x1, y1 = x, xy[1], x, xy[1] + height
    for ch in text:
        if ch == " ":
            x += height * 0.35
            continue
        glyph = Image.new("RGBA", (int(height * scale * 1.4), int(height * scale * 1.6)), (0, 0, 0, 0))
        ImageDraw.Draw(glyph).text((height * scale * 0.2, height * scale * 0.1), ch, font=font,
                                   fill=(*ink, int(rng.integers(190, 256))))
        glyph = glyph.rotate(float(rng.uniform(-11, 11)), resample=Image.BICUBIC, expand=True)
        glyph = glyph.resize((max(1, glyph.width // scale), max(1, glyph.height // scale)), Image.LANCZOS)
        gy = xy[1] + float(rng.uniform(-0.12, 0.12)) * height
        img.paste(glyph, (int(x), int(gy)), glyph)
        bb = glyph.getbbox()
        if bb:
            x1 = max(x1, x + bb[2])
            y0, y1 = min(y0, gy + bb[1]), max(y1, gy + bb[3])
        x += max(height * 0.2, font.getlength(ch) / scale * float(rng.uniform(0.9, 1.05)))
    return [x0, y0, min(x1, x + 2), y1]


def _stamp(size: tuple[int, int], text: str, colour: tuple[int, int, int], rng: np.random.Generator, dpi: int) -> Image.Image:
    d = int(dpi * 1.15)
    layer = Image.new("RGBA", (d * 2, d), (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    a = int(rng.integers(120, 175))
    lw = max(2, dpi // 40)
    dr.rounded_rectangle((lw, lw, d * 2 - lw, d - lw), radius=d // 6, outline=(*colour, a), width=lw)
    dr.rounded_rectangle((lw * 3, lw * 3, d * 2 - lw * 3, d - lw * 3), radius=d // 8, outline=(*colour, a), width=max(1, lw // 2))
    f = _font(int(d * 0.36))
    tw = dr.textlength(text, font=f)
    dr.text(((d * 2 - tw) / 2, d * 0.28), text, font=f, fill=(*colour, a))
    layer = layer.rotate(float(rng.uniform(-22, 22)), resample=Image.BICUBIC, expand=True)
    # speckle the ink so it does not look digitally perfect
    alpha = np.asarray(layer.getchannel("A"), dtype=float) * (1 - 0.35 * (rng.random(layer.size[::-1]) < 0.08))
    layer.putalpha(Image.fromarray(alpha.astype(np.uint8)))
    return layer


def _paper(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    arr = np.asarray(img, dtype=float)
    tint = np.array([0.985, 0.965, 0.92])
    grid = rng.normal(0, 1, (6, 6))
    shade = np.asarray(Image.fromarray(((grid - grid.min()) / (np.ptp(grid) + 1e-9) * 255).astype(np.uint8)).resize(img.size, Image.BICUBIC), dtype=float) / 255
    arr = arr * tint * (0.94 + 0.06 * shade[..., None])
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ------------------------------------------------------------------ pipeline
def scan_render(rendered: RenderResult, cfg: ScanConfig, doc_info: dict[str, Any] | None = None) -> ScanResult:
    """Augment an already-rendered document (use scan_document for the common one-call path)."""
    pdf = pdfium.PdfDocument(rendered.pdf)
    ppt = cfg.dpi / 72.0
    ext, mime = ("png", "image/png") if cfg.output == "png" else ("jpg", "image/jpeg")
    images: list[bytes] = []
    pages_out: list[dict[str, Any]] = []
    for pi in range(len(pdf)):
        rng = np.random.default_rng([cfg.seed, pi])
        img = pdf[pi].render(scale=ppt).to_pil().convert("RGB")
        W, H = img.size
        fields: list[dict[str, Any]] = []
        for b in rendered.boxes:
            if b["page"] == pi + 1:
                bb = [v * ppt for v in b["bbox"]]
                fields.append({"key": b["key"], "text": b["text"], "value": b["value"], "bbox": bb, "polygon": _poly(bb), "kind": "text", "handwritten": False})
        applied: dict[str, Any] = {"dpi": cfg.dpi}
        if cfg.paper_tint:
            img = _paper(img, rng)
            applied["paper_tint"] = True
        paper = tuple(int(v) for v in np.asarray(img)[: max(2, H // 50), : max(2, W // 50)].reshape(-1, 3).mean(axis=0))

        # handwriting: re-drawn fields, then signature + note near the bottom of the last page
        ink = (18, 34, 120) if rng.random() < 0.7 else (30, 30, 30)
        for f in fields:
            if f["key"] in cfg.handwrite_fields and _ascii(f["text"]) and f["kind"] == "text":
                x0, y0, x1, y1 = f["bbox"]
                ImageDraw.Draw(img).rectangle((x0 - 2, y0 - 2, x1 + 2, y1 + 2), fill=paper)
                bb = _handwrite(img, f["text"], (x0, y0 - 1), (y1 - y0) * 1.15, rng, ink)
                f.update(bbox=bb, polygon=_poly(bb), kind="handwritten", handwritten=True)
        if cfg.handwriting and pi == len(pdf) - 1:
            h = 0.028 * H
            sy = H * 0.905
            dr = ImageDraw.Draw(img)
            dr.line((W * 0.08, sy + h * 1.15, W * 0.36, sy + h * 1.15), fill=(90, 90, 90), width=max(1, cfg.dpi // 100))
            dr.text((W * 0.08, sy + h * 1.25), "Signature", font=_font(int(h * 0.5)), fill=(110, 110, 110))
            name = LATIN_NAMES[int(rng.integers(0, len(LATIN_NAMES)))]
            bb = _handwrite(img, name, (W * 0.1, sy), h, rng, ink)
            fields.append({"key": "handwritten_signature", "text": name, "value": name, "bbox": bb, "polygon": _poly(bb), "kind": "handwritten", "handwritten": True})
            note = NOTES[int(rng.integers(0, len(NOTES)))]
            bb = _handwrite(img, note, (W * 0.42, sy + h * 0.1), h * 0.8, rng, ink)
            fields.append({"key": "handwritten_note", "text": note, "value": note, "bbox": bb, "polygon": _poly(bb), "kind": "handwritten", "handwritten": True})
            applied["handwriting"] = ["handwritten_signature", "handwritten_note"]

        # stamps
        stamped = []
        for _ in range(cfg.stamps):
            text = cfg.stamp_text or STAMP_TEXTS[int(rng.integers(0, len(STAMP_TEXTS)))]
            colour = (196, 28, 36) if rng.random() < 0.65 else (28, 62, 168)
            layer = _stamp(img.size, text, colour, rng, cfg.dpi)
            px = int(rng.uniform(0.12, 0.88) * W - layer.width / 2)
            py = int(rng.uniform(0.22, 0.86) * H - layer.height / 2)
            base = img.convert("RGBA")
            base.alpha_composite(layer, (max(0, px), max(0, py)))
            img = base.convert("RGB")
            bb = layer.getbbox() or (0, 0, 1, 1)
            box = [max(0, px) + bb[0], max(0, py) + bb[1], max(0, px) + bb[2], max(0, py) + bb[3]]
            fields.append({"key": "stamp", "text": text, "value": text, "bbox": box, "polygon": _poly(box), "kind": "stamp", "handwritten": False})
            stamped.append(text)
        if stamped:
            applied["stamps"] = stamped

        # rotation + horizontal shear, applied to image and labels with the same matrix
        theta = float(rng.uniform(-cfg.rotate_deg, cfg.rotate_deg)) if cfg.rotate_deg else 0.0
        shear = float(rng.uniform(-cfg.skew, cfg.skew)) if cfg.skew else 0.0
        if theta or shear:
            c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
            cx, cy = W / 2, H / 2
            T = lambda dx, dy: np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])  # noqa: E731
            F = T(cx, cy) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]) @ np.array([[1, shear, 0], [0, 1, 0], [0, 0, 1.0]]) @ T(-cx, -cy)
            inv = np.linalg.inv(F)
            img = img.transform(img.size, Image.AFFINE, tuple(inv[:2].reshape(-1)), resample=Image.BICUBIC, fillcolor=paper)
            for f in fields:
                pts = np.array([[x, y, 1.0] for x, y in f["polygon"]]) @ F.T
                f["polygon"] = pts[:, :2].tolist()
                f["bbox"] = _bbox_of(f["polygon"])
            applied.update(rotation_deg=round(theta, 4), shear=round(shear, 5))

        if cfg.blur_radius > 0:
            img = img.filter(ImageFilter.GaussianBlur(cfg.blur_radius * cfg.dpi / 150))
            applied["blur_radius"] = cfg.blur_radius
        if cfg.noise_sigma > 0:
            arr = np.asarray(img, dtype=float)
            grey = rng.normal(0, cfg.noise_sigma, arr.shape[:2])[..., None]
            arr = arr + 0.75 * grey + 0.25 * rng.normal(0, cfg.noise_sigma, arr.shape)
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            applied["noise_sigma"] = cfg.noise_sigma
        if cfg.low_res_scale < 1.0:
            new = (max(1, int(round(img.width * cfg.low_res_scale))), max(1, int(round(img.height * cfg.low_res_scale))))
            k = (new[0] / img.width, new[1] / img.height)
            img = img.resize(new, Image.LANCZOS)
            for f in fields:
                f["polygon"] = [[x * k[0], y * k[1]] for x, y in f["polygon"]]
                f["bbox"] = _bbox_of(f["polygon"])
            applied["low_res_scale"] = cfg.low_res_scale
        buf = io.BytesIO()
        if cfg.jpeg_quality < 100 or cfg.output == "jpeg":
            img.save(buf, "JPEG", quality=cfg.jpeg_quality)
            applied["jpeg_quality"] = cfg.jpeg_quality
            if cfg.output == "png":
                img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
                buf = io.BytesIO()
        if cfg.output == "png":
            img.save(buf, "PNG")
        images.append(buf.getvalue())

        w, h = img.size
        for f in fields:  # clip to the image, round for readable JSON
            f["polygon"] = [[round(x, 1), round(y, 1)] for x, y in f["polygon"]]
            f["bbox"] = [round(max(0, f["bbox"][0]), 1), round(max(0, f["bbox"][1]), 1), round(min(w, f["bbox"][2]), 1), round(min(h, f["bbox"][3]), 1)]
        pages_out.append({"page": pi + 1, "image": f"page-{pi + 1}.{ext}", "width": w, "height": h, "augmentations": applied, "fields": fields})
    labels = {"schema": "sdp-scan-labels/1", "config": cfg.model_dump(), "coordinate_system": "pixels, origin top-left", "pages": pages_out, **(doc_info or {})}
    return ScanResult(images, labels, mime)


def scan_document(doc: dict[str, Any], cfg: ScanConfig | None = None, font: str = "Helvetica") -> ScanResult:
    cfg = cfg or ScanConfig()
    rendered = render_document(doc, font)
    info = {"doc_type": doc.get("doc_type", "invoice"), "locale": doc["locale"],
            "document_id": doc.get("invoice_number") or doc.get("receipt_number") or doc.get("statement_number")}
    return scan_render(rendered, cfg, info)
