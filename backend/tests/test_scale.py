import gzip
import hashlib
import io
import json
import sqlite3

import pandas as pd
import pytest

from sdp.datasets import make_customers
from sdp.scale import ScaleError, chunk_seed, generate_large, plan
from sdp.tabular import TabularGenerator


@pytest.fixture(scope="module")
def gen():
    real = make_customers(800, seed=0)
    real.insert(0, "customer_id", range(1, 801))
    return TabularGenerator().fit(real)


def test_plan_and_chunk_seeds_are_stable_and_distinct():
    assert plan(250, 100) == [100, 100, 50] and plan(100, 100) == [100]
    assert chunk_seed(1, 0) == chunk_seed(1, 0) and len({chunk_seed(1, i) for i in range(50)}) == 50 and chunk_seed(1, 0) != chunk_seed(2, 0)
    with pytest.raises(ScaleError):
        plan(0, 100)
    with pytest.raises(ScaleError):
        plan(10, 5)


def test_output_is_identical_for_any_worker_count_and_matches_row_count(gen, tmp_path):
    reps = [generate_large(gen, 2500, tmp_path / f"w{w}.csv", seed=7, chunk_rows=400, workers=w) for w in (1, 3, 6)]
    assert len({r.sha256 for r in reps}) == 1
    assert all(r.rows == 2500 and r.chunks == 7 for r in reps)
    df = pd.read_csv(tmp_path / "w1.csv")
    assert len(df) == 2500 and (tmp_path / "w1.csv").stat().st_size == reps[0].bytes
    assert hashlib.sha256((tmp_path / "w3.csv").read_bytes()).hexdigest() == reps[1].sha256
    assert generate_large(gen, 2500, tmp_path / "other.csv", seed=8, chunk_rows=400).sha256 != reps[0].sha256


def test_keys_stay_unique_across_chunks_and_chunks_differ(gen, tmp_path):
    generate_large(gen, 1500, tmp_path / "a.csv", seed=1, chunk_rows=300)
    df = pd.read_csv(tmp_path / "a.csv")
    key = "customer_id" if "customer_id" in df.columns else df.columns[0]
    assert df[key].is_unique and df[key].tolist() == list(range(1, 1501))
    assert not df.iloc[:300].drop(columns=[key]).reset_index(drop=True).equals(df.iloc[300:600].drop(columns=[key]).reset_index(drop=True))
    explicit = generate_large(gen, 600, tmp_path / "b.csv", seed=1, chunk_rows=200, sequence_columns=[])
    assert explicit.rows == 600


def test_progress_reports_are_monotonic_and_throughput_is_reported(gen, tmp_path):
    seen = []
    r = generate_large(gen, 2000, tmp_path / "p.csv", seed=1, chunk_rows=500, on_progress=lambda d, t: seen.append((d, t)))
    assert [d for d, _ in seen] == [500, 1000, 1500, 2000] and all(t == 2000 for _, t in seen)
    assert r.rows_per_second > 0 and r.mb_per_second > 0 and r.seconds > 0 and r.max_rows_in_memory == 8 * 500
    assert len(r.per_chunk_seconds) == 4
    assert r.peak_rss_mb is None or r.peak_rss_mb > 0


def test_memory_is_bounded_by_the_window_not_the_total(gen, tmp_path):
    import tracemalloc
    tracemalloc.start()
    generate_large(gen, 20_000, tmp_path / "m.csv", seed=1, chunk_rows=1000, workers=2)
    _, peak_small = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    generate_large(gen, 80_000, tmp_path / "m2.csv", seed=1, chunk_rows=1000, workers=2)
    _, peak_big = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak_big < peak_small * 1.6                # 4x the rows, roughly the same peak: it does not grow with the total


def test_cancel_leaves_a_valid_partial_file(gen, tmp_path):
    n = {"c": 0}

    def cancel():
        n["c"] += 1
        return n["c"] > 6

    r = generate_large(gen, 50_000, tmp_path / "c.jsonl", seed=1, fmt="jsonl", chunk_rows=500, workers=2, should_cancel=cancel)
    assert r.cancelled and 0 < r.rows < 50_000
    lines = (tmp_path / "c.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == r.rows and all(json.loads(x) for x in lines)


def test_formats_json_sql_and_gzip(gen, tmp_path):
    r = generate_large(gen, 1200, tmp_path / "s.json", seed=2, fmt="json", chunk_rows=500)
    assert len(json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))) == 1200 == r.rows
    r = generate_large(gen, 1200, tmp_path / "s.sql", seed=2, fmt="sql", chunk_rows=500, export_options={"dialect": "sqlite"})
    con = sqlite3.connect(":memory:")
    con.executescript((tmp_path / "s.sql").read_text(encoding="utf-8"))
    assert con.execute("SELECT COUNT(*) FROM synthetic").fetchone()[0] == 1200
    r = generate_large(gen, 1200, tmp_path / "z.csv", seed=2, chunk_rows=500, compress=True)
    assert r.file == "z.csv.gz" and len(pd.read_csv(io.BytesIO(gzip.decompress((tmp_path / "z.csv.gz").read_bytes())))) == 1200
    assert generate_large(gen, 1200, tmp_path / "z2.csv", seed=2, chunk_rows=500, compress=True).sha256 == r.sha256     # gzip is reproducible
    with pytest.raises(ScaleError):
        generate_large(gen, 100, tmp_path / "x.pdf", fmt="pdf")


def test_null_and_outlier_injection_apply_per_chunk(gen, tmp_path):
    generate_large(gen, 3000, tmp_path / "n.csv", seed=3, chunk_rows=1000, null_rate={"income": 0.1}, outlier_rate={"age": 0.05})
    df = pd.read_csv(tmp_path / "n.csv")
    assert 0.07 < df["income"].isna().mean() < 0.13
    assert "_edge_case" not in df.columns


def test_process_executor_gives_the_same_bytes_as_threads(gen, tmp_path):
    a = generate_large(gen, 1200, tmp_path / "t.csv", seed=5, chunk_rows=300, workers=2, executor="thread")
    b = generate_large(gen, 1200, tmp_path / "p.csv", seed=5, chunk_rows=300, workers=2, executor="process")
    assert a.sha256 == b.sha256 and a.executor == "thread" and b.executor == "process"
    assert generate_large(gen, 500, tmp_path / "x.csv", seed=5, chunk_rows=250).executor == "thread"      # auto: small jobs use threads
    with pytest.raises(ScaleError):
        generate_large(gen, 500, tmp_path / "y.csv", executor="gpu")
