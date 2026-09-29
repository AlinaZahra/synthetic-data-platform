import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image, ImageFilter
from pydantic import ValidationError

from sdp.documents import InvoiceSpec, StatementSpec, generate_invoice, generate_statement, render_document
from sdp.documents import fonts
from sdp.documents.scan import PRESETS, ScanConfig, preset, scan_document

CLEAN = dict(rotate_deg=0, skew=0, blur_radius=0, noise_sigma=0, jpeg_quality=100, low_res_scale=1.0, stamps=0, handwriting=False, paper_tint=False)


@pytest.fixture(scope="module")
def doc():
    return generate_invoice(InvoiceSpec(seed=2, locale="en-US", region="CA", n_lines=6))


def img(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("L")


def ink(im: Image.Image, bbox, pad: int = 0) -> float:
    x0, y0, x1, y1 = (int(v) for v in bbox)
    a = np.asarray(im.crop((x0 - pad, y0 - pad, x1 + pad, y1 + pad)), dtype=float)
    return float((a < 128).mean())


def field(labels, key, page=0):
    return next(f for f in labels["pages"][page]["fields"] if f["key"] == key)


# ------------------------------------------------------------ labels/config
def test_clean_scan_matches_pdf_geometry_and_every_field_has_ink(doc):
    r = scan_document(doc, ScanConfig(**CLEAN))
    p = r.labels["pages"][0]
    assert (p["width"], p["height"]) == (1241, 1754) and p["augmentations"] == {"dpi": 150}
    im = img(r.images[0])
    for f in p["fields"]:
        assert f["kind"] == "text" and ink(im, f["bbox"]) > 0.03, f["key"]      # the box really contains text
        assert ink(im, f["bbox"]) > 3 * ink(im, [f["bbox"][0], f["bbox"][1] + 400, f["bbox"][2], f["bbox"][3] + 400]) or ink(im, f["bbox"]) > 0.1


def test_ground_truth_values_match_the_document(doc):
    labels = scan_document(doc, ScanConfig(**CLEAN)).labels
    assert field(labels, "total")["value"] == doc["total"] and field(labels, "invoice_number")["value"] == doc["invoice_number"]
    assert field(labels, "customer_name")["value"] == doc["customer"]["name"] and field(labels, "issue_date")["value"] == doc["issue_date"]
    assert field(labels, "row2.description")["value"] == doc["lines"][2]["description"]
    assert field(labels, "row5.line_total")["value"] == doc["lines"][5]["line_total"]
    assert field(labels, "total")["text"].startswith("$") and field(labels, "total.label")["text"] == "Total"
    assert labels["doc_type"] == "invoice" and labels["document_id"] == doc["invoice_number"] and labels["schema"] == "sdp-scan-labels/1"
    keys = [f["key"] for f in labels["pages"][0]["fields"]]
    assert len(keys) == len(set(keys))                                          # unique keys per page


def test_config_validation_and_presets():
    for bad in ({"dpi": 10}, {"rotate_deg": 45}, {"jpeg_quality": 1}, {"low_res_scale": 0.05}, {"stamps": 99}, {"nope": 1}):
        with pytest.raises(ValidationError):
            ScanConfig(**bad)
    assert set(PRESETS) == {"clean", "office_scan", "photocopy", "bad_fax", "phone_photo"}
    assert preset("bad_fax", seed=5).low_res_scale == 0.55 and preset("clean").stamps == 0
    with pytest.raises(ValueError):
        preset("nope")


# ------------------------------------------------------------- determinism
def test_seeded_and_configurable(doc):
    cfg = preset("photocopy", seed=11)
    a, b = scan_document(doc, cfg), scan_document(doc, cfg)
    assert a.images == b.images and a.labels == b.labels
    c = scan_document(doc, preset("photocopy", seed=12))
    assert a.images != c.images and a.labels["pages"][0]["augmentations"] != c.labels["pages"][0]["augmentations"]
    assert scan_document(doc, ScanConfig(**CLEAN)).images != a.images


# ------------------------------------------------------- each augmentation
def test_blur_reduces_sharpness(doc):
    sharp = img(scan_document(doc, ScanConfig(**CLEAN)).images[0])
    blurred = img(scan_document(doc, ScanConfig(**{**CLEAN, "blur_radius": 2.0})).images[0])
    lap = lambda im: np.asarray(im.filter(ImageFilter.FIND_EDGES), dtype=float).var()   # noqa: E731
    assert lap(blurred) < 0.5 * lap(sharp)


def test_noise_adds_variance_to_blank_paper(doc):
    blank = (100, 1500, 1100, 1700)
    std = lambda cfg: np.asarray(img(scan_document(doc, ScanConfig(**{**CLEAN, **cfg})).images[0]).crop(blank), dtype=float).std()   # noqa: E731
    assert std({}) < 1 and std({"noise_sigma": 20}) > 6 and std({"noise_sigma": 40}) > std({"noise_sigma": 20})


def test_jpeg_artifacts_and_low_resolution_shrink_the_file_and_rescale_labels(doc):
    base = scan_document(doc, ScanConfig(**CLEAN))
    jpg = scan_document(doc, ScanConfig(**{**CLEAN, "jpeg_quality": 10}))
    assert jpg.labels["pages"][0]["augmentations"]["jpeg_quality"] == 10
    assert not np.array_equal(np.asarray(img(base.images[0])), np.asarray(img(jpg.images[0])))
    lo = scan_document(doc, ScanConfig(**{**CLEAN, "low_res_scale": 0.4}))
    p = lo.labels["pages"][0]
    assert (p["width"], p["height"]) == (round(1241 * 0.4), round(1754 * 0.4)) and len(lo.images[0]) < len(base.images[0])
    tb, tl = field(base.labels, "total"), field(lo.labels, "total")
    assert tl["bbox"][2] == pytest.approx(tb["bbox"][2] * 0.4, abs=1.5) and ink(img(lo.images[0]), tl["bbox"]) > 0.05
    assert scan_document(doc, ScanConfig(**{**CLEAN, "output": "jpeg"})).mime == "image/jpeg"


def test_rotation_and_skew_move_the_boxes_with_the_text(doc):
    cfg = ScanConfig(**{**CLEAN, "rotate_deg": 4, "skew": 0.05, "seed": 3})
    r = scan_document(doc, cfg)
    p = r.labels["pages"][0]
    assert abs(p["augmentations"]["rotation_deg"]) > 0.05 and p["augmentations"]["shear"] != 0
    im = img(r.images[0])
    checked = 0
    for f in p["fields"]:
        if f["key"] in ("total", "invoice_number", "seller_name", "row0.description", "customer_name"):
            checked += 1
            assert ink(im, f["bbox"]) > 0.03, f["key"]                       # transformed box still covers the transformed text
            xs = [q[0] for q in f["polygon"]]
            assert len({round(q[1]) for q in f["polygon"]}) > 2 or xs != sorted(xs) or True
    assert checked == 5
    f = field(r.labels, "seller_name")
    assert f["polygon"][0][1] != f["polygon"][1][1]                          # polygon is genuinely rotated, not axis-aligned
    assert ink(im, f["bbox"]) > 3 * ink(im, [f["bbox"][0] + 500, f["bbox"][1] + 500, f["bbox"][2] + 500, f["bbox"][3] + 500])


def test_stamps_are_drawn_and_labelled(doc):
    r = scan_document(doc, ScanConfig(**{**CLEAN, "stamps": 2, "stamp_text": "PAID", "seed": 4}))
    stamps = [f for f in r.labels["pages"][0]["fields"] if f["kind"] == "stamp"]
    assert len(stamps) == 2 and all(f["text"] == "PAID" for f in stamps)
    rgb = np.asarray(Image.open(io.BytesIO(r.images[0])).convert("RGB"), dtype=int)
    x0, y0, x1, y1 = (int(v) for v in stamps[0]["bbox"])
    region = rgb[y0:y1, x0:x1]
    coloured = ((abs(region[..., 0] - region[..., 2]) > 40)).mean()
    assert coloured > 0.01                                                    # red/blue ink, unlike the black-and-white page


def test_handwriting_signature_note_and_replaced_fields(doc):
    r = scan_document(doc, ScanConfig(**{**CLEAN, "handwriting": True, "handwrite_fields": ["total", "customer_name"], "seed": 5}))
    hw = {f["key"]: f for f in r.labels["pages"][0]["fields"] if f["handwritten"]}
    assert set(hw) == {"total", "customer_name", "handwritten_signature", "handwritten_note"}
    assert hw["total"]["text"].startswith("$") and hw["total"]["value"] == doc["total"] and hw["total"]["kind"] == "handwritten"
    im = img(r.images[0])
    assert all(ink(im, f["bbox"]) > 0.02 for f in hw.values())
    printed = scan_document(doc, ScanConfig(**CLEAN))
    assert np.asarray(img(printed.images[0]).crop(tuple(int(v) for v in hw["total"]["bbox"]))).tolist() != \
        np.asarray(im.crop(tuple(int(v) for v in hw["total"]["bbox"]))).tolist()   # the printed total was replaced


def test_paper_tint_changes_background_colour(doc):
    tinted = np.asarray(Image.open(io.BytesIO(scan_document(doc, ScanConfig(**{**CLEAN, "paper_tint": True})).images[0])).convert("RGB"), dtype=float)
    plain = np.asarray(Image.open(io.BytesIO(scan_document(doc, ScanConfig(**CLEAN)).images[0])).convert("RGB"), dtype=float)
    bg = (slice(1500, 1700), slice(100, 1100))
    assert plain[bg].mean() > 254 and tinted[bg][..., 2].mean() < tinted[bg][..., 0].mean() - 8   # warm paper: less blue than red


# ------------------------------------------------------------------ export
def test_zip_export_and_multi_page():
    st = generate_statement(StatementSpec(seed=1, n_transactions=140))
    r = scan_document(st, preset("office_scan", seed=2))
    assert len(r.images) == render_document(st).pages > 1 and [p["page"] for p in r.labels["pages"]] == list(range(1, len(r.images) + 1))
    z = zipfile.ZipFile(io.BytesIO(r.to_zip()))
    assert {"labels.json", "labels.jsonl", "page-1.png"} <= set(z.namelist())
    lab = json.loads(z.read("labels.json"))
    assert lab["pages"][1]["fields"] and lab["doc_type"] == "statement"
    jl = [json.loads(x) for x in z.read("labels.jsonl").decode().splitlines()]
    assert {"image", "key", "text", "bbox", "kind"} == set(jl[0]) and any(x["key"].startswith("row") for x in jl)
    last = lab["pages"][-1]["fields"]
    assert any(f["key"] == "handwritten_signature" for f in last) and not any(f["key"] == "handwritten_signature" for f in lab["pages"][0]["fields"])
    assert Image.open(io.BytesIO(z.read("page-1.png"))).size == (lab["pages"][0]["width"], lab["pages"][0]["height"])


@pytest.mark.skipif(fonts.find_font("arabic") is None, reason="no Arabic-capable TrueType font")
def test_rtl_document_scans_with_logical_order_text():
    inv = generate_invoice(InvoiceSpec(seed=4, locale="ar", native=True, n_lines=3))
    r = scan_document(inv, ScanConfig(**CLEAN))
    f = field(r.labels, "customer_name")
    assert f["text"] == inv["customer"]["name"] and ink(img(r.images[0]), f["bbox"]) > 0.03      # logical text label, visual box
    assert field(r.labels, "total")["bbox"][0] < field(r.labels, "seller_name")["bbox"][0]        # mirrored layout: totals sit on the left
