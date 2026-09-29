import pytest

from sdp.nl.multilingual import SUPPORTED, detect_language, lexicon, messages, norm, parse_multilingual, to_canonical

CASES = [
    ("en", "5,000 Pakistani bank customers, 3% fraud, 12 months of history", 5000, "ur-PK", "bank_customers", 0.03, 12),
    ("es", "Quiero 5.000 clientes bancarios pakistaníes, 3% de fraude y 12 meses de historial", 5000, "ur-PK", "bank_customers", 0.03, 12),
    ("es", "2 mil clientes de comercio electrónico españoles con 5% de fraude", 2000, "es", "ecommerce_customers", 0.05, None),
    ("fr", "Je veux 5 000 clients bancaires français, 3 % de fraude, 6 mois d'historique", 5000, "fr", "bank_customers", 0.03, 6),
    ("fr", "10 mille clients e-commerce britanniques avec 2,5 % de fraude", 10000, "en-GB", "ecommerce_customers", 0.025, None),
    ("ur", "۵۰۰۰ پاکستانی بینک صارفین، ۳ فیصد فراڈ، ۱۲ ماہ کی ہسٹری", 5000, "ur-PK", "bank_customers", 0.03, 12),
    ("ur", "5 ہزار بھارتی بینک کے صارفین 2 فیصد فراڈ", 5000, "hi", "bank_customers", 0.02, None),
    ("ar", "٥٠٠٠ عملاء البنك سعوديين، ٣٪ احتيال، ١٢ أشهر من السجل", 5000, "ar", "bank_customers", 0.03, 12),
    ("hi", "5000 भारतीय बैंक ग्राहक, 3% धोखाधड़ी, 12 महीनों का इतिहास", 5000, "hi", "bank_customers", 0.03, 12),
    ("hi", "२ लाख पाकिस्तानी बैंक ग्राहक", 200000, "ur-PK", "bank_customers", None, None),
    ("zh", "生成5000名中国银行客户，百分之3欺诈，12个月的历史", 5000, "zh", "bank_customers", 0.03, 12),
    ("zh", "2万美国电商客户，5%欺诈", 20000, "en-US", "ecommerce_customers", 0.05, None),
]


@pytest.mark.parametrize("lang,text,rows,locale,domain,rate,months", CASES)
def test_request_in_each_language_parses_to_the_same_config(lang, text, rows, locale, domain, rate, months):
    out = parse_multilingual(text)
    assert out["language"] == lang, out["canonical_text"]
    r = out["parse"]
    assert r.ok, (out["canonical_text"], r.errors)
    c = r.config
    assert (c.rows, c.locale, c.domain) == (rows, locale, domain)
    assert (c.flag.rate if c.flag else None) == pytest.approx(rate) if rate is not None else c.flag is None
    assert c.history_months == months
    rep = out["reply"]
    assert rep["lang"] == lang and rep["lines"] and not rep["errors"]


def test_reply_is_in_the_users_language_and_rtl_flagged():
    ar = parse_multilingual("٥٠٠٠ عملاء البنك سعوديين، ٣٪ احتيال")["reply"]
    assert ar["rtl"] and any("الصفوف" in x for x in ar["lines"]) and "5,000" in ar["lines"][0]
    es = parse_multilingual("5.000 clientes bancarios pakistaníes")["reply"]
    assert es["lines"][0].startswith("Filas") and "Pakistán" in es["lines"][2] and not es["rtl"], es
    zh = parse_multilingual("5000名银行客户")["reply"]
    assert zh["lines"][0].startswith("行数")


def test_errors_and_warnings_are_localised():
    es = parse_multilingual("dame 500 cosas raras", language="es")["reply"]
    assert es["errors"] and "No pude" in es["errors"][0] and "clientes bancarios" in es["errors"][0]
    fr = parse_multilingual("clients bancaires", language="fr")["reply"]
    assert "lignes" in fr["errors"][0]
    ur = parse_multilingual("۲۰۰ پاکستانی بینک صارفین 60 فیصد فراڈ")["reply"]
    assert any("غیر معمولی" in w for w in ur["warnings"])
    zh = parse_multilingual("100名银行客户")["reply"]
    assert any("默认" in w for w in zh["warnings"])


def test_explicit_language_overrides_detection_and_unknown_is_rejected():
    out = parse_multilingual("5000 bank customers", language="es")
    assert out["language"] == "es" and out["detected"] is False and out["reply"]["lines"][0].startswith("Filas")
    with pytest.raises(ValueError):
        parse_multilingual("x", language="xx")


def test_detection_of_close_pairs():
    assert detect_language("عملاء البنك السعوديين") == "ar"
    assert detect_language("بینک صارفین پاکستانی") == "ur"
    assert detect_language("clients bancaires français") == "fr"
    assert detect_language("clientes bancarios españoles") == "es"
    assert detect_language("just some english words about customers") == "en"
    assert detect_language("12345") == "en"


def test_normalisation_and_number_formats():
    assert norm("٣٫٥٪ ۱۲", "ur") == "3.5% 12"
    assert to_canonical("1,5 millones de clientes bancarios", "es")[0].startswith("1500000")
    assert to_canonical("5 000 clients bancaires", "fr")[0].startswith("5000")
    assert to_canonical("百分之3欺诈", "zh")[0].startswith("3 %")


def test_catalog_is_complete_for_every_language():
    en = set(messages()["en"])
    for lang in SUPPORTED:
        assert set(messages()[lang]) == en, lang
        assert lang in lexicon()


def test_english_path_is_unchanged():
    out = parse_multilingual("1,000 Pakistani bank customers, 3% fraud, 6 months of history")
    assert out["language"] == "en" and out["parse"].config.rows == 1000 and out["canonical_text"].startswith("1,000")
