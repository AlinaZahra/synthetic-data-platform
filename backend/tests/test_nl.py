import pandas as pd
import pytest
from pydantic import ValidationError

from sdp.nl import DatasetConfig, domains, generate_dataset, parse_request

PROMPT = "5,000 Pakistani bank customers, 3% fraud, 12 months of history"


def test_headline_example_parses_to_expected_config():
    r = parse_request(PROMPT)
    assert r.ok and r.confirmation_required and r.errors == []
    c = r.config
    assert (c.rows, c.locale, c.domain, c.history_months) == (5000, "ur-PK", "bank_customers", 12)
    assert c.flag.name == "is_fraud" and c.flag.rate == pytest.approx(0.03)
    assert {e["field"] for e in r.explanation} == {"domain", "rows", "locale", "flag", "history_months"}
    assert [e["source"] for e in r.explanation if e["field"] == "rows"] == ["5,000"]
    assert c.schema_columns and c.rules and r.config_hash == c.config_hash()


@pytest.mark.parametrize("text,rows,locale,rate,months", [
    ("10k Indian bank customers with 1.5% fraud and 2 years of history", 10000, "hi", 0.015, 24),
    ("0.9 million french bank clients", 900_000, "fr", None, None),
    ("generate 300 bank customers from Saudi Arabia, fraud rate of 5 percent", 300, "ar", 0.05, None),
    ("1000 e-commerce customers in spain, 0.5% chargebacks, 6 months history", 1000, "es", 0.005, 6),
    ("500 bank customers", 500, "en-US", None, None),
    ("2.5 thousand british bank account holders 12-month history", 2500, "en-GB", None, 12),
])
def test_variants(text, rows, locale, rate, months):
    r = parse_request(text)
    assert r.ok, r.errors
    assert (r.config.rows, r.config.locale, r.config.history_months) == (rows, locale, months)
    assert (r.config.flag.rate if r.config.flag else None) == (pytest.approx(rate) if rate else None)


def test_defaults_and_warnings_are_explicit():
    r = parse_request("500 bank customers")
    assert any("assuming en-US" in w for w in r.warnings)
    r = parse_request("100 pakistani and indian bank customers")
    assert r.ok and any("Several locales" in w for w in r.warnings) and r.config.locale in ("ur-PK", "hi")
    r = parse_request("100 bank customers, 5% churn")
    assert r.ok and r.config.flag is None and any("ignored" in w for w in r.warnings)
    assert any("unusually high" in w for w in parse_request("100 bank customers 80% fraud").warnings)


@pytest.mark.parametrize("text", ["", "   ", "hello there", "bank customers", "5000 unicorns", "0 bank customers",
                                  "5000000000 bank customers", "100 bank customers, 500 months of history"])
def test_unparseable_or_invalid_requests_return_errors_not_exceptions(text):
    r = parse_request(text)
    assert not r.ok and r.config is None and r.errors


def test_config_validation():
    base = dict(domain="bank_customers", locale="ur-PK", rows=10)
    assert DatasetConfig(**base).locale == "ur-PK"
    for bad in (dict(rows=0), dict(rows=2_000_000), dict(locale="xx"), dict(domain="nope"),
                dict(flag={"name": "is_fraud", "rate": 1.5}), dict(flag={"name": "other", "rate": 0.1}),
                dict(history_months=0), dict(extra=1)):
        with pytest.raises(ValidationError):
            DatasetConfig(**{**base, **bad})


def test_domains_are_data_files():
    assert {"bank_customers", "ecommerce_customers"} <= set(domains())


# -------------------------------------------------------------- generation
@pytest.fixture(scope="module")
def ds():
    cfg = parse_request(PROMPT).config
    return generate_dataset(cfg.model_copy(update={"rows": 5000}))


def test_shapes_and_exact_fraud_rate(ds):
    c, h = ds.tables["customers"], ds.tables["monthly_activity"]
    assert len(c) == 5000 and len(h) == 5000 * 12
    assert c["is_fraud"].sum() == 150 and c["is_fraud"].mean() == pytest.approx(0.03)
    assert h["month"].nunique() == 12
    assert set(c.columns) >= {"customer_id", "name", "phone", "national_id", "address", "city", "date_of_birth", "monthly_income", "is_fraud"}


def test_locale_validity_constraints_and_integrity_all_clean(ds):
    v = ds.validate()
    assert v["locale"]["valid_pct"] == 100.0 and v["constraints"]["pass_pct"] == 100.0
    assert v["integrity"]["ok"] and v["integrity"]["total_violations"] == 0
    assert v["flag"]["achieved"] == pytest.approx(0.03) and v["flag"]["requested"] == 0.03
    assert all("؀" <= ch <= "ۿ" or ch == " " for ch in ds.tables["customers"]["name"].iloc[0])  # Urdu script
    assert ds.tables["customers"]["phone"].str.startswith("+92 3").all()


def test_fraud_customers_have_anomalous_history(ds):
    c, h = ds.tables["customers"], ds.tables["monthly_activity"]
    fraud_ids = set(c.loc[c["is_fraud"] == 1, "customer_id"])
    h = h.assign(f=h["customer_id"].isin(fraud_ids))
    assert h["is_fraud_activity"].sum() > 0 and set(h.loc[h["is_fraud_activity"] == 1, "customer_id"]) <= fraud_ids
    spike, normal = h[h["is_fraud_activity"] == 1], h[h["is_fraud_activity"] == 0]
    assert spike["txn_amount"].mean() > 3 * normal["txn_amount"].mean()


def test_reproducible_and_seed_sensitive():
    cfg = DatasetConfig(domain="bank_customers", locale="hi", rows=200, flag={"name": "is_fraud", "rate": 0.05}, history_months=3, seed=7)
    a, b = generate_dataset(cfg), generate_dataset(cfg)
    for k in a.tables:
        pd.testing.assert_frame_equal(a.tables[k], b.tables[k])
    c = generate_dataset(cfg.model_copy(update={"seed": 8}))
    assert not a.tables["customers"]["name"].equals(c.tables["customers"]["name"])


@pytest.mark.parametrize("locale", ["en-US", "en-GB", "ur-PK", "ar", "hi", "es", "fr", "zh"])
def test_every_locale_generates_valid_data(locale):
    cfg = DatasetConfig(domain="bank_customers", locale=locale, rows=300, flag={"name": "is_fraud", "rate": 0.1}, history_months=2)
    v = generate_dataset(cfg).validate()
    assert v["locale"]["valid_pct"] == 100.0 and v["integrity"]["ok"] and v["flag"]["count"] == 30


def test_ecommerce_domain_without_flag_or_history():
    ds = generate_dataset(DatasetConfig(domain="ecommerce_customers", locale="fr", rows=100))
    assert list(ds.tables) == ["customers"] and "is_fraud" not in ds.tables["customers"]
    assert ds.validate()["locale"]["valid_pct"] == 100.0
