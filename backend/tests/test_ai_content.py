import json

import pytest

from sdp.ai import llm
from sdp.ai.content import ContentError, synthesize


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("SDP_DATA_DIR", str(tmp_path))
    llm.set_client(None)
    yield
    llm.set_client(False)


class Echo:
    """Fake model: answers with one string per input row, built from the row context."""
    name = "echo"

    def __init__(self, fn=None):
        self.calls, self.fn = 0, fn

    def complete(self, system, prompt, max_tokens=1024):
        self.calls += 1
        rows = json.loads(prompt.split("\nvariant")[0])["rows"]
        return json.dumps(self.fn(rows) if self.fn else [f"Review for {r.get('product', 'x')} rated {r.get('rating')} #{i}" for i, r in enumerate(rows)])


def test_offline_fallback_is_deterministic_and_context_aware():
    ctx = [{"rating": 5, "product": "kettle"}, {"rating": 1, "product": "kettle"}] * 10
    a = synthesize("review", 20, contexts=ctx, seed=3, client=None)
    b = synthesize("review", 20, contexts=ctx, seed=3, client=None)
    c = synthesize("review", 20, contexts=ctx, seed=4, client=None)
    assert a.values == b.values and a.values != c.values
    assert a.stats["fallback"] == 20 and set(a.sources) == {"fallback"}
    assert all("kettle" in v or len(v) > 10 for v in a.values)
    pos = " ".join(v for v, x in zip(a.values, ctx) if x["rating"] == 5).lower()
    neg = " ".join(v for v, x in zip(a.values, ctx) if x["rating"] == 1).lower()
    assert any(w in pos for w in ("happy", "great", "pleased", "recommended", "exceeded", "five"))
    assert any(w in neg for w in ("disappointed", "not", "poor", "refund", "stopped", "late"))


def test_names_and_addresses_follow_the_locale_script():
    from sdp.locale import get_locale
    n = synthesize("name", 15, locale="ur-PK", seed=1, client=None).values
    assert all(get_locale("ur-PK").script_ok(x) for x in n) and len(set(n)) > 5
    a = synthesize("address", 5, locale="fr", seed=1, client=None).values
    assert all(x and "," in x for x in a)


def test_llm_batches_cache_and_context_are_used():
    fake = Echo()
    ctx = [{"rating": i % 5 + 1, "product": "lamp"} for i in range(45)]
    r = synthesize("review", 45, contexts=ctx, client=fake, batch_size=20)
    assert fake.calls == 3 and r.stats["llm"] == 45 and r.stats["fallback"] == 0
    assert r.values[0].startswith("Review for lamp rated 1")
    r2 = synthesize("review", 45, contexts=ctx, client=fake, batch_size=20)
    assert fake.calls == 3 and r2.stats["cache_batches"] == 3 and r2.values == r.values


def test_bad_items_fall_back_individually():
    def bad(rows):
        out = [f"fine text {i}" for i in range(len(rows))]
        out[0] = "call me on +92 300 1234567"        # phone number leakage
        out[1] = "```json"                            # junk
        out[2] = 42                                   # not a string
        return out

    r = synthesize("review", 6, client=Echo(bad), batch_size=6)
    assert r.sources[:3] == ["fallback"] * 3 and r.sources[3:] == ["llm"] * 3 and r.stats["rejected"] == 3


def test_wrong_length_reply_uses_fallback_for_missing_rows():
    r = synthesize("review", 5, client=Echo(lambda rows: ["only one"]), batch_size=5)
    assert r.sources == ["llm"] + ["fallback"] * 4 and any("expected 5" in w for w in r.warnings)


def test_wrong_script_names_are_rejected_for_the_locale():
    r = synthesize("name", 4, locale="ur-PK", client=Echo(lambda rows: ["John Smith"] * len(rows)), batch_size=4)
    assert r.stats["llm"] == 0 and r.stats["fallback"] == 4


def test_api_down_falls_back_once_and_reports():
    class Down:
        name = "down"
        calls = 0

        def complete(self, *a, **k):
            Down.calls += 1
            raise llm.LLMError("boom")

    r = synthesize("ticket", 50, contexts=[{"priority": "high"}] * 50, client=Down(), batch_size=10)
    assert Down.calls == 1                                       # stops hammering a dead API
    assert r.stats["fallback"] == 50 and any("unavailable" in w for w in r.warnings)


def test_validation_errors_and_unsupported_locale_note():
    with pytest.raises(ContentError):
        synthesize("poem", 3, client=None)
    with pytest.raises(ContentError):
        synthesize("review", 3, contexts=[{}], client=None)
    r = synthesize("review", 3, locale="ar", client=None)
    assert any("English" in w for w in r.warnings)
    assert synthesize("review", 0, client=None).values == []
