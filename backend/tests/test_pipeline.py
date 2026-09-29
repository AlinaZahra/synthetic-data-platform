import io
import json
import logging
import time

import pytest

from sdp.documents import register_font
from sdp.documents.invoice import LineSpec, generate_invoice
from sdp.documents.pipeline import (DocumentPipeline, JsonLogFormatter, RetryPolicy, TransientError, job_id_for,
                                    structured_logging)
from sdp.documents.render import MissingFontError, RenderOverflowError, render_invoice


def spec(i: int, **kw):
    return {"seed": i, "locale": ["en-US", "en-GB", "ur-PK", "ar", "hi", "es", "fr", "zh"][i % 8], "n_lines": 1 + i % 9, **kw}


def pipe(tmp_path=None, **kw):
    return DocumentPipeline(out_dir=tmp_path, sleep=lambda s: None, **kw)


# ------------------------------------------------------------- happy path
def test_single_doc_end_to_end(tmp_path):
    p = pipe(tmp_path)
    r = p.run([spec(1)])
    assert (r.succeeded, r.failed, r.skipped) == (1, 0, 0)
    doc = next(tmp_path.glob("*.json"))
    assert not doc.name.endswith(".report.json")
    data = json.loads(doc.read_text())
    assert doc.with_suffix(".pdf").read_bytes().startswith(b"%PDF") and data["doc_id"] == doc.stem
    assert (tmp_path / f"{r.job_id}.report.json").exists()
    assert not list(tmp_path.glob(".*.tmp"))  # atomic writes leave no temp files


def test_idempotent_rerun_and_job_id(tmp_path):
    specs = [spec(i) for i in range(20)]
    a = pipe(tmp_path).run(specs)
    b = pipe(tmp_path).run(specs)
    assert a.job_id == b.job_id == job_id_for(specs) and a.succeeded == 20
    assert (b.succeeded, b.skipped) == (0, 20)
    files = {f.name: f.stat().st_mtime_ns for f in tmp_path.glob("*.pdf")}
    pipe(tmp_path).run(specs)
    assert {f.name: f.stat().st_mtime_ns for f in tmp_path.glob("*.pdf")} == files  # untouched
    assert pipe(tmp_path).run(specs, force=True).succeeded == 20
    assert job_id_for(specs[:5]) != a.job_id


def test_duplicates_in_a_batch_are_skipped(tmp_path):
    r = pipe(tmp_path).run([spec(3), spec(3), {**spec(3)}])
    assert (r.succeeded, r.skipped) == (1, 2)
    equivalent = [{"seed": 0}, {"seed": 0, "locale": "en-US", "n_lines": 5}]  # same after defaults -> same doc
    assert pipe().run(equivalent).skipped == 1


# ---------------------------------------------------------- error isolation
MALFORMED = [None, "not a dict", 42, [], {"locale": "xx"}, {"n_lines": 0}, {"n_lines": "many"}, {"bogus": 1},
             {"lines": [{"description": "x", "quantity": "-1", "unit_price": "1"}]},
             {"lines": [{"description": "x", "quantity": "abc", "unit_price": "1"}]},
             {"locale": "en-US", "region": "ZZ"}, {"seed": "x"}]


def test_malformed_specs_never_stop_the_batch():
    good = [spec(i) for i in range(10)]
    mixed = [x for pair in zip(good, MALFORMED[:10]) for x in pair] + MALFORMED[10:] + good[:0]
    r = pipe().run(mixed)
    assert r.succeeded == 10 and r.failed == len(MALFORMED) and r.total == len(mixed)
    assert r.by_stage == {"validate": len(MALFORMED)} and r.by_error == {"ValueError": len(MALFORMED)}
    assert all(f["message"] and f["attempts"] == 1 for f in r.failures)  # validation errors are not retried
    assert sorted(f["index"] for f in r.failures) == sorted(i for i, x in enumerate(mixed) if x in MALFORMED)
    json.loads(r.to_json())


def test_unexpected_exception_is_isolated():
    def boom(s):
        if s.seed == 2:
            raise ZeroDivisionError("oops")
        return generate_invoice(s)
    r = pipe(generate_fn=boom).run([spec(i) for i in range(5)])
    assert (r.succeeded, r.failed) == (4, 1)
    assert r.failures[0]["stage"] == "generate" and r.failures[0]["error_type"] == "ZeroDivisionError"


# ------------------------------------------------------------------ retries
def test_transient_failures_are_retried_with_backoff():
    calls, sleeps = {"n": 0}, []

    def flaky(s):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise TransientError("rate limited")
        return generate_invoice(s)
    p = DocumentPipeline(sleep=sleeps.append, generate_fn=flaky, retry=RetryPolicy(retries=3, base_delay=0.1, jitter=0))
    r = p.run([spec(1)])
    assert r.succeeded == 1 and calls["n"] == 3
    assert sleeps == pytest.approx([0.1, 0.2])  # exponential backoff


def test_retries_exhausted_reports_attempts():
    p = DocumentPipeline(sleep=lambda s: None, generate_fn=lambda s: (_ for _ in ()).throw(TransientError("down")),
                         retry=RetryPolicy(retries=2))
    r = p.run([spec(1)])
    f = r.failures[0]
    assert (r.failed, f["stage"], f["error_type"], f["attempts"]) == (1, "generate", "TransientError", 3)


def test_oserror_on_export_is_retried_then_recovers(tmp_path, monkeypatch):
    p = pipe(tmp_path)
    real, n = p._write_atomic, {"c": 0}

    def flaky_write(path, payload):
        n["c"] += 1
        if n["c"] == 1:
            raise OSError("disk hiccup")
        real(path, payload)
    monkeypatch.setattr(p, "_write_atomic", flaky_write)
    assert p.run([spec(1)]).succeeded == 1


# ------------------------------------------------------------- fonts / limits
def test_missing_font_fails_that_doc_only_or_falls_back():
    specs = [spec(0), spec(1, font="NoSuchFont"), spec(2, font="Times-Roman")]
    r = pipe().run(specs)
    assert (r.succeeded, r.failed) == (2, 1)
    assert r.failures[0]["stage"] == "render" and r.failures[0]["error_type"] == "MissingFontError"
    r2 = pipe(fallback_font="Helvetica").run(specs)
    assert r2.failed == 0 and r2.warnings == 1


def test_register_font_with_bad_path():
    with pytest.raises(MissingFontError):
        register_font("Broken", "does/not/exist.ttf")


def test_non_encodable_text_is_caught_in_post_validation():
    def with_cjk(s):
        d = generate_invoice(s)
        d["customer"]["name"] = "王伟"
        return d
    r = pipe(generate_fn=with_cjk).run([spec(1)])
    assert r.failures[0]["stage"] == "post_validate" and "font cannot render" in r.failures[0]["message"]


def test_totals_tampering_is_caught_in_post_validation():
    def tampered(s):
        d = generate_invoice(s)
        d["total"] = "1.00"
        return d
    f = pipe(generate_fn=tampered).run([spec(1)]).failures[0]
    assert f["stage"] == "post_validate" and "totals" in f["message"]


def test_page_limit_enforced():
    r = pipe(max_pages=2).run([spec(1, n_lines=200), spec(2, n_lines=3)])
    assert (r.succeeded, r.failed) == (1, 1) and "page count" in r.failures[0]["message"]


# ------------------------------------------------------------ extreme values
def line(**kw):
    return {"description": "x", "quantity": "1", "unit_price": "1", **kw}


def test_extreme_values_either_reconcile_or_fail_cleanly():
    extremes = [
        {"lines": [line(quantity="999999999.999", unit_price="9999999999.9999")] * 500, "locale": "hi"},   # ~5e21 totals
        {"lines": [line(quantity="0.001", unit_price="0.0001")], "locale": "en-US"},                        # rounds to 0.00
        {"lines": [line(unit_price="0")] * 3, "locale": "fr"},                                               # free
        {"lines": [line(description="é" * 200)], "locale": "es"},                                            # max-length text
        {"lines": [line(quantity="1", unit_price="9999999999.9999")], "locale": "zh", "tax_inclusive": True},
        {"n_lines": 500, "locale": "en-GB"},
        spec(1),
    ]
    r = pipe(max_pages=20).run(extremes)
    assert r.succeeded + r.failed == len(extremes) and r.by_stage.get("unexpected", 0) == 0
    for f in r.failures:
        assert f["stage"] in {"render", "post_validate"}
    assert r.succeeded >= 4
    _, art = pipe().process_one({"lines": [line(quantity="0.001", unit_price="0.0001")]})
    assert art.data["total"] == "0.00" and art.data["lines"][0]["line_total"] == "0.00"


def test_column_overflow_raises_render_overflow():
    d = generate_invoice(__import__("sdp.documents", fromlist=["InvoiceSpec"]).InvoiceSpec(
        lines=[LineSpec(description="x", quantity="999999999", unit_price="9999999999")] * 500, locale="en-US"))
    d["total"] = "9" * 60 + ".00"
    with pytest.raises(RenderOverflowError):
        render_invoice(d)


# ------------------------------------------------------------------ scale
def test_thousand_plus_documents_with_faults_injected(tmp_path):
    specs = [spec(i) for i in range(1200)]
    for i in range(0, 1200, 37):
        specs[i] = MALFORMED[i % len(MALFORMED)]
    bad = sum(1 for i in range(0, 1200, 37))
    t0 = time.perf_counter()
    r = DocumentPipeline(out_dir=tmp_path, sleep=lambda s: None, workers=4, batch_size=100).run(specs)
    elapsed = time.perf_counter() - t0
    assert r.total == 1200 and r.failed == bad and r.succeeded + r.skipped == 1200 - bad
    assert len(list(tmp_path.glob("*.pdf"))) == r.succeeded
    assert elapsed < 90, elapsed
    again = DocumentPipeline(out_dir=tmp_path, sleep=lambda s: None).run(specs)
    assert again.succeeded == 0 and again.skipped == r.succeeded + r.skipped and again.failed == bad


# --------------------------------------------------------------------- logs
def test_structured_json_logs():
    buf = io.StringIO()
    structured_logging(logging.INFO, buf)
    pipe().run([spec(1), {"locale": "xx"}])
    lines = [json.loads(x) for x in buf.getvalue().splitlines()]
    msgs = [x["msg"] for x in lines]
    assert "batch started" in msgs and "batch finished" in msgs and "document failed" in msgs
    failed = next(x for x in lines if x["msg"] == "document failed")
    assert failed["stage"] == "validate" and failed["error"] == "ValueError" and "job_id" in failed
    logging.getLogger("sdp.pipeline").handlers.clear()
    assert isinstance(JsonLogFormatter(), logging.Formatter)
