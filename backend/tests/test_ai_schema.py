import json

import numpy as np
import pandas as pd
import pytest

from sdp.ai import llm
from sdp.ai.schema_infer import SchemaError, apply_edits, infer_schema, to_rules


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("SDP_DATA_DIR", str(tmp_path))
    llm.set_client(None)
    yield
    llm.set_client(False)


def sample(n=30):
    rng = np.random.default_rng(1)
    start = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 200, n), unit="D")
    return pd.DataFrame({
        "cust_id": np.arange(1, n + 1),
        "full_name": [f"Person {i} Example" for i in range(n)],
        "contact": [f"user{i}@example.com" for i in range(n)],
        "mobile": [f"+92 300 {1000000 + i}" for i in range(n)],
        "signup": start.strftime("%d/%m/%Y"),
        "last_seen": (start + pd.to_timedelta(rng.integers(1, 50, n), unit="D")).strftime("%d/%m/%Y"),
        "balance": [f"${x:,.2f}" for x in rng.uniform(10, 9000, n)],
        "tier": rng.choice(["gold", "silver", "basic"], n),
        "age": rng.integers(18, 80, n),
        "note": ["customer asked about the new savings plan today"] * n,
    })


class Fake:
    name = "fake"

    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def complete(self, system, prompt, max_tokens=1024):
        self.calls += 1
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply)


def test_heuristics_find_types_formats_and_constraints_offline():
    p = infer_schema(sample(), client=None)
    t = {c.name: c for c in p.tables["table"]}
    assert t["cust_id"].semantic_type == "identifier" and t["cust_id"].unique
    assert t["contact"].semantic_type == "email" and t["mobile"].semantic_type == "phone"
    assert t["signup"].semantic_type == "date" and t["signup"].date_format == "%d/%m/%Y"
    assert t["balance"].semantic_type == "currency_amount" and t["balance"].currency == "$"
    assert t["tier"].semantic_type == "category" and set(t["tier"].allowed_values) == {"gold", "silver", "basic"}
    assert t["age"].semantic_type == "integer" and t["full_name"].semantic_type == "person_name"
    assert t["note"].semantic_type == "free_text"
    kinds = {(c.type, tuple(c.columns)) for c in p.constraints}
    assert ("unique", ("cust_id",)) in kinds and ("order", ("signup", "last_seen")) in kinds and ("range", ("age",)) in kinds
    assert all(c.holds_on_sample for c in p.constraints) and p.used_llm is False and p.proposal_hash


def test_too_few_rows_is_a_clear_error():
    with pytest.raises(SchemaError, match="at least"):
        infer_schema(sample(3), client=None)


def test_llm_refines_only_what_the_data_supports_and_reports_rejections():
    fake = Fake({
        "columns": [
            {"name": "note", "semantic_type": "free_text"},
            {"name": "full_name", "semantic_type": "email"},           # contradicted by the values
            {"name": "ghost", "semantic_type": "email"},               # not a column
            {"name": "age", "semantic_type": "wizard"},                # not in the enum
            {"name": "signup", "semantic_type": "date", "date_format": "%Y-%m-%d"},   # format does not parse
        ],
        "constraints": [{"type": "range", "table": "table", "columns": ["age"], "params": {"min": 0, "max": 10}, "description": "young"}],
    })
    p = infer_schema(sample(), client=fake)
    t = {c.name: c for c in p.tables["table"]}
    assert p.used_llm and fake.calls == 1
    assert t["full_name"].semantic_type == "person_name"     # LLM claim rejected
    assert t["signup"].date_format == "%d/%m/%Y"
    joined = " ".join(p.warnings)
    assert "ghost" in joined and "wizard" in joined and "does not parse" in joined and "violated" in joined
    bad = [c for c in p.constraints if c.source == "llm"]
    assert bad and not bad[0].holds_on_sample
    assert "age <= 10" not in json.dumps(to_rules(p))          # a violated constraint is never turned into a rule


def test_llm_can_upgrade_a_low_confidence_column_and_is_cached():
    df = sample()
    df["mystery"] = [f"{i % 7}-{(i * 3) % 10}" for i in range(len(df))]
    fake = Fake({"columns": [{"name": "mystery", "semantic_type": "category", "reason": "codes"}]})
    p = infer_schema(df, client=fake)
    assert p.column("table", "mystery").semantic_type == "category" and p.column("table", "mystery").source == "llm"
    infer_schema(df, client=fake)
    assert fake.calls == 1                                     # second call served from the cache


def test_llm_failure_or_garbage_falls_back_to_heuristics():
    class Down:
        name = "down"

        def complete(self, *a, **k):
            raise llm.LLMError("offline")

    p = infer_schema(sample(), client=Down())
    assert not p.used_llm and any("unavailable" in w for w in p.warnings)
    p = infer_schema(sample(), client=Fake("I cannot help with that"))
    assert not p.used_llm and p.tables["table"]


def test_json_reply_in_a_code_fence_is_accepted():
    p = infer_schema(sample(), client=Fake("Sure!\n```json\n{\"columns\": []}\n```"))
    assert p.used_llm


def test_multi_table_relationships_and_edits():
    cust = pd.DataFrame({"customer_id": range(1, 21), "name": [f"n{i} x" for i in range(20)]})
    orders = pd.DataFrame({"order_id": range(100, 130), "customer_id": [1 + i % 20 for i in range(30)], "total": np.linspace(5, 90, 30)})
    p = infer_schema({"customers": cust, "orders": orders}, client=None)
    assert any(r.child_table == "orders" and r.parent_table == "customers" for r in p.relationships)
    p2 = apply_edits(p, [{"action": "set_type", "table": "orders", "column": "total", "value": "currency_amount"},
                         {"action": "remove_relationship", "value": 0}, {"action": "confirm"}])
    assert p2.confirmed and p2.column("orders", "total").source == "user" and len(p2.relationships) == len(p.relationships) - 1
    assert p2.proposal_hash != p.proposal_hash
    with pytest.raises(SchemaError):
        apply_edits(p, [{"action": "set_type", "table": "orders", "column": "nope", "value": "email"}])
    with pytest.raises(SchemaError):
        apply_edits(p, [{"action": "set_type", "table": "orders", "column": "total", "value": "wizard"}])
    p3 = apply_edits(p, [{"action": "add_constraint", "value": {"type": "range", "table": "orders", "columns": ["total"], "params": {"min": 1}}}])
    assert any(c.source == "user" for c in p3.constraints)


def test_self_reference_and_rules_export():
    df = pd.DataFrame({"emp_id": range(1, 21), "manager_id": [None] + [1 + i // 3 for i in range(19)], "salary": np.linspace(100, 900, 20)})
    p = infer_schema(df, client=None)
    assert any(r.child_column == "manager_id" and r.parent_column == "emp_id" for r in p.relationships)
    assert "salary >= 0" in to_rules(p)["table"]


def test_anthropic_client_never_leaks_key_and_wraps_errors(monkeypatch):
    c = llm.AnthropicClient("sk-secret", url="http://127.0.0.1:1/none", timeout=1)
    with pytest.raises(llm.LLMError) as e:
        c.complete("s", "p")
    assert "sk-secret" not in str(e.value)
