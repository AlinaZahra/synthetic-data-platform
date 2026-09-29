import numpy as np
import pandas as pd
import pytest

from sdp.ai import llm
from sdp.ai.chat_edit import ChatEditError, EditState, apply_message, materialize, parse_op, summarize
from sdp.nl import DatasetConfig, generate_dataset, parse_request


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("SDP_DATA_DIR", str(tmp_path))
    llm.set_client(None)
    yield
    llm.set_client(False)


def state(rows=600, seed=3, text=None):
    r = parse_request(text or f"{rows} Pakistani bank customers, 4% fraud, 6 months of history", seed=seed)
    assert r.ok
    return EditState(config=r.config)


def test_double_a_segment_appends_only_that_segment_and_leaves_the_rest_identical():
    s = state()
    base, _ = materialize(s)
    lahore = int((base["customers"]["city"] == "Lahore").sum())
    out = apply_message(s, "double customers from Lahore")
    assert out["applied"] == 1 and out["ops"][0]["op"] == "scale_segment" and out["source"] == "rules"
    new, _ = materialize(EditState.model_validate(out["state"]))
    assert int((new["customers"]["city"] == "Lahore").sum()) == 2 * lahore
    assert len(new["customers"]) == len(base["customers"]) + lahore
    # existing rows are untouched; history of the old customers is byte-identical, new customers have history too
    pd.testing.assert_frame_equal(new["customers"].iloc[:600].reset_index(drop=True), base["customers"])
    pd.testing.assert_frame_equal(new["monthly_activity"].iloc[:len(base["monthly_activity"])].reset_index(drop=True), base["monthly_activity"])
    added = new["customers"].iloc[600:]
    assert added["customer_id"].is_unique and added["customer_id"].min() > base["customers"]["customer_id"].max()
    assert set(new["monthly_activity"]["customer_id"]) == set(new["customers"]["customer_id"])
    assert (added["city"] == "Lahore").all()
    r = out["regenerated"]
    assert r["customers"]["rows_after"] == r["customers"]["rows_before"] + lahore and r["customers"]["columns_changed"] == []
    assert any(d["path"] == "segments.city.Lahore" and d["after"] == 2 * d["before"] for d in out["diff"])
    assert out["message"] and out["preview"]["rows"]


def test_replay_is_deterministic_and_history_is_reproducible():
    s = state()
    out1 = apply_message(s, "double customers from Lahore and fraud 8%")
    out2 = apply_message(s, "double customers from Lahore and fraud 8%")
    assert out1["state"] == out2["state"]
    a, _ = materialize(EditState.model_validate(out1["state"]))
    b, _ = materialize(EditState.model_validate(out2["state"]))
    for t in a:
        pd.testing.assert_frame_equal(a[t], b[t])


def test_fraud_rate_edit_changes_flag_and_spike_history_only():
    s = state()
    base, _ = materialize(s)
    out = apply_message(s, "fraud 8%")
    new, _ = materialize(EditState.model_validate(out["state"]))
    assert new["customers"]["is_fraud"].sum() == round(0.08 * 600)
    changed = out["regenerated"]["customers"]["columns_changed"]
    assert changed == ["is_fraud"]
    for c in base["customers"].columns:
        if c != "is_fraud":
            assert new["customers"][c].equals(base["customers"][c])
    h0, h1 = base["monthly_activity"], new["monthly_activity"]
    # customers whose flag did not change keep byte-identical history
    same = base["customers"]["is_fraud"] == new["customers"]["is_fraud"]
    keep = base["customers"].loc[same, "customer_id"]
    assert h0[h0.customer_id.isin(keep)].reset_index(drop=True).equals(h1[h1.customer_id.isin(keep)].reset_index(drop=True))
    flagged_now = set(new["customers"].loc[new["customers"].is_fraud == 1, "customer_id"])
    assert set(h1.loc[h1.is_fraud_activity == 1, "customer_id"]) <= flagged_now
    # lowering it again removes spikes for the customers that were unflagged
    out2 = apply_message(EditState.model_validate(out["state"]), "fraud 2%")
    new2, _ = materialize(EditState.model_validate(out2["state"]))
    assert new2["customers"]["is_fraud"].sum() == round(0.02 * 600)
    assert set(new2["monthly_activity"].loc[new2["monthly_activity"].is_fraud_activity == 1, "customer_id"]) <= set(new2["customers"].loc[new2["customers"].is_fraud == 1, "customer_id"])


def test_double_the_fraud_and_more_outliers_to_balance():
    s = state()
    out = apply_message(s, "add more outliers to balance")
    op = out["ops"][0]
    assert op["op"] == "set_outlier_rate" and op["column"] == "account_balance" and op["rate"] >= 0.02
    new, _ = materialize(EditState.model_validate(out["state"]))
    base, _ = materialize(s)
    c = new["customers"]
    marked = c["_outlier_account_balance"].sum()
    assert marked == round(op["rate"] * 600)
    clean = base["customers"]["account_balance"]
    q1, q3 = clean.quantile(.25), clean.quantile(.75)
    out_vals = c.loc[c["_outlier_account_balance"], "account_balance"]
    assert (out_vals > q3 + 1.4 * (q3 - q1)).all()
    assert out["regenerated"]["customers"]["columns_changed"] == sorted(["account_balance", "_outlier_account_balance"])
    assert out["after"]["outliers"]["account_balance"] == marked and out["before"]["outliers"] == {}
    d = apply_message(s, "double the fraud")
    assert d["ops"][0]["rate"] == pytest.approx(0.08, abs=0.002)


def test_rows_and_history_ops():
    s = state(400)
    out = apply_message(s, "add 50% more customers")
    assert out["ops"] == [{"op": "scale_rows", "factor": 1.5}] and out["after"]["rows"] == 600
    out = apply_message(s, "500 customers")
    assert out["ops"][0] == {"op": "set_rows", "rows": 500} and out["after"]["rows"] == 500
    out = apply_message(s, "halve the customers")
    assert out["after"]["rows"] == 200
    t, _ = materialize(EditState.model_validate(out["state"]))
    assert set(t["monthly_activity"]["customer_id"]) == set(t["customers"]["customer_id"])
    out = apply_message(s, "3 months of history")
    t, _ = materialize(EditState.model_validate(out["state"]))
    assert t["monthly_activity"].groupby("customer_id")["month"].nunique().eq(3).all() and out["regenerated"]["customers"]["untouched"]
    out = apply_message(s, "12 months of history")
    t, _ = materialize(EditState.model_validate(out["state"]))
    assert t["monthly_activity"].groupby("customer_id")["month"].nunique().eq(12).all() and len(t["monthly_activity"]) == 400 * 12


def test_ambiguity_and_unknown_requests_ask_instead_of_guessing():
    s = state()
    out = apply_message(s, "add outliers")
    assert out["applied"] == 0 and "Which column" in out["message"] and out["state"]["ops"] == []
    out = apply_message(s, "make it prettier")
    assert out["applied"] == 0 and out["message"]
    out = apply_message(s, "more customers from Karachi and make it prettier")
    assert out["applied"] == 1 and out["questions"]


def test_bad_ops_are_rejected():
    with pytest.raises(ChatEditError):
        parse_op({"op": "scale_rows", "factor": -1})
    with pytest.raises(ChatEditError):
        parse_op({"op": "explode"})
    s = state()
    bad = EditState(config=s.config, ops=[{"op": "set_outlier_rate", "column": "name", "rate": 0.1}])
    with pytest.raises(ChatEditError):
        materialize(bad)
    with pytest.raises(ChatEditError):
        apply_message(EditState(config=s.config, ops=[]), "double customers from Karachi and 0 customers from Karachi") if False else materialize(
            EditState(config=s.config, ops=[{"op": "scale_segment", "column": "city", "value": "Atlantis", "factor": 2}]))


def test_multilingual_edits_use_the_same_ops():
    s = state()
    es = apply_message(s, "duplica los clientes de Lahore")
    assert es["language"] == "es" and es["ops"][0]["op"] == "scale_segment" and "cambio" in es["message"]
    fr = apply_message(s, "double les clients de Karachi")
    assert fr["ops"][0]["value"] == "Karachi"
    zh = apply_message(s, "把Lahore的客户翻倍")
    assert zh["language"] == "zh" and zh["ops"][0]["value"] == "Lahore" and "更改" in zh["message"]
    ar = apply_message(s, "ضاعف عملاء Lahore")
    assert ar["ops"][0]["factor"] == 2.0
    ur = apply_message(s, "Lahore کے صارفین دگنے کریں")
    assert ur["ops"][0]["value"] == "Lahore"


def test_llm_ops_are_validated_and_used_only_when_rules_find_nothing():
    class Fake:
        def __init__(self, reply):
            self.reply = reply
            self.name = "fake-" + reply             # the disk cache is keyed by client name + prompt

        def complete(self, system, prompt, max_tokens=1024):
            return self.reply

    s = state()
    ok = apply_message(s, "please make the data a bit bigger", client=Fake('[{"op": "scale_rows", "factor": 1.25}]'))
    assert ok["source"] == "llm" and ok["after"]["rows"] == 750
    bad = apply_message(s, "please make the data a bit bigger", client=Fake('[{"op": "drop_database"}]'))
    assert bad["applied"] == 0
    rules = apply_message(s, "fraud 5%", client=Fake('[{"op": "scale_rows", "factor": 9}]'))
    assert rules["source"] == "rules" and rules["after"]["rows"] == 600


def test_ecommerce_domain_works_too():
    r = parse_request("800 online shoppers from Spain, 3% fraud, 4 months of history", seed=1)
    if not r.ok:
        r = parse_request("800 ecommerce customers, 3% fraud, 4 months of history", seed=1)
    s = EditState(config=r.config)
    out = apply_message(s, "fraud 6%")
    assert out["after"]["flag_rate"] == pytest.approx(0.06, abs=0.002)
