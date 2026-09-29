# Synthetic Data Platform

Tabular, relational and document engines, locale packs, an NL-to-dataset assistant, a Trust Score, and a React workspace. All demo data is generated in code; no real personal data.

```
backend/sdp/
  common.py               seeding helpers, dtype families
  datasets.py             fictional demo data (customers table; customers/products/orders/order_items)
  tabular/                GenConfig, TabularGenerator (Gaussian copula / CTGAN), null+outlier injection, fidelity metrics
  evaluation/tstr.py      evaluate_tstr(...) -> scorecard JSON
  relational/             schema graph, PK/FK inference, DDL parser, cardinality, generator, integrity checker
  locale/                 registry + packs/*.json (en-US, en-GB, ur-PK, ar, hi, es, fr, zh); a new locale is a data file
  scoring/                fidelity, privacy, locale validity, constraints/coverage, Trust Score (+PDF)
  documents/              Decimal invoices, PDF render, robust batch pipeline
  nl/                     'text -> validated config' parser + builder, domains/*.json
  api/main.py             FastAPI
frontend/                 Vite + React + TypeScript workspace
```

## Run

```bash
docker compose up --build        # single container -> http://localhost:8000 (drop extra locale JSON into ./locales)
```

Or locally:

```bash
cd backend && pip install -r requirements.txt && python -m pytest          # ~240 tests
cd backend && python -m uvicorn sdp.api.main:app --port 8000
cd frontend && npm install && npm run dev                                   # http://localhost:5173
```

## Python quick start

```python
from sdp.datasets import make_customers, make_shop
from sdp.tabular import TabularGenerator, GenConfig
from sdp.evaluation import evaluate_tstr
from sdp.relational import infer_graph, RelationalGenerator, CardinalityConfig

real = make_customers(3000)
gen = TabularGenerator().fit(real)                       # or TabularGenerator("ctgan") after `pip install ctgan`
out = gen.generate(GenConfig(rows=1000, seed=42, null_rate={"income": .05},
                             outlier_rate={"age": .01}, outlier_method="iqr"))
out.data, out.log_dicts()                                # identical for the same seed + config

tables = make_shop(300)
graph = infer_graph(tables); print(graph.to_json())      # editable: add_foreign_key / remove_foreign_key / ...
res = RelationalGenerator(graph, {"orders(customer_id)->customers": CardinalityConfig(kind="zipf", a=1.8)}
                          ).fit(tables).generate(scale=2, seed=1)
res.integrity.ok, res.scorecard()
```

### Rules, cross-table consistency and relation metrics

```python
from sdp.datasets import make_shop_full, SHOP_FULL_RULES
from sdp.relational.metrics import relation_metrics

real = make_shop_full(400)
graph = infer_graph(real)
gen = RelationalGenerator(graph).fit(real)
res = gen.generate(seed=1, rules=SHOP_FULL_RULES)          # DSL or plain language, across tables
res.rules["reconciliation_pass_rate"]                       # 100.0 (naive generation: ~50%)
relation_metrics(real, res.tables, graph)["score"]          # 0-100: cardinality, join size, cross-table effects, FK coverage, JOIN queries

from sdp.rules import sample_with_rules, compare_naive_vs_enforced
sample_with_rules(TabularGenerator().fit(make_customers()), 1000, seed=1,
                  rules_text=["age must be at least 25", "income <= 60000"], mode="hybrid")   # repair | reject | hybrid
```

Rule examples: `discount <= 0.3`, `delivery_date > order_date`, `orders.total = SUM(order_items.quantity * order_items.unit_price)`,
`orders.status = 'cancelled' => COUNT(shipments) = 0`, or plain English such as *"age must be at least 25"*.

### REST API, CLI and reproducibility

```bash
export SDP_API_KEYS=my-key     # if unset, an ephemeral key is printed in the server log
python -m uvicorn sdp.api.main:app --port 8000        # OpenAPI docs at /docs, "Authorize" with X-API-Key

curl -H "X-API-Key: my-key" -X POST localhost:8000/generate -H "Content-Type: application/json" \
  -d '{"kind":"tabular","params":{"rows":500,"seed":1,"rules":["age must be at least 25"]},"wait":true}'
curl -H "X-API-Key: my-key" localhost:8000/jobs/<id>           # status, request, outputs, lineage
curl -H "X-API-Key: my-key" localhost:8000/score/<id>          # Trust Score (or validation score for described datasets)

# CLI mirror for CI: exit code 2 = quality gate failed
python -m sdp.cli generate --kind tabular --params '{"rows": 500, "seed": 1}' --wait --min-score 80 --out out/
python -m sdp.cli rerun <id>            # replays the manifest; exit 2 if the output hash differs
python -m sdp.cli --url http://host:8000 --api-key my-key score <id>
```

Long jobs run in the background: `GET /jobs/{id}/progress` (done/total, percent, phase, failures, ETA), `POST /jobs/{id}/cancel` (keeps what finished), `GET /jobs/{id}/download.zip`. A 20,000-document batch does not block the UI or the API; the **Documents → Bulk** tab and **History** show live progress.

```bash
curl -H "X-API-Key: my-key" -X POST localhost:8000/generate -H "Content-Type: application/json" \
  -d '{"kind":"document","params":{"doc_type":"payslip","count":5000,"spec":{"locale":"en-GB"}}}'
```

### Adding a document type (no code)

Create `backend/sdp/documents/templates/<name>/type.json` and `template.html` (or point `SDP_TEMPLATE_DIR` at your own folder). `type.json` lists the fields to generate, computed values and reconciliation rules; `template.html` is a Jinja2 template converted to PDF. Payslip, retail receipt and purchase order ship this way. See `sdp/documents/templating.py` for the field kinds and the expression language.

Each job stores a manifest (request, seed, schema, dataset hash, model/code fingerprint, output hashes, scores, lineage). The **History** page lists versions and reruns from a manifest.

### Schema understanding, content, chat editing, multilingual (optional Claude, always with offline fallbacks)

```python
from sdp.ai.schema_infer import infer_schema, apply_edits
p = infer_schema(df_of_20_to_50_rows)                  # semantic types, date formats, relationships, constraints; client=None = offline only
p = apply_edits(p, [{"action": "set_type", "table": "table", "column": "tier", "value": "category"}, {"action": "confirm"}])

from sdp.ai.content import synthesize                  # names, addresses, reviews, descriptions, tickets; batched + cached; local fallback
synthesize("review", 20, locale="es", contexts=[{"rating": 2, "product": "lamp"}] * 20, seed=1)

from sdp.ai.chat_edit import EditState, apply_message  # "double customers from Lahore", "fraud 8%", "add more outliers to balance"
apply_message(EditState(config=cfg), "double customers from Lahore")   # ops, before/after, config diff, what was regenerated

from sdp.nl.multilingual import parse_multilingual     # es fr ur ar hi zh en: request and answer in the same language
parse_multilingual("Quiero 3.000 clientes bancarios pakistaníes, 3% de fraude")["reply"]
```
Set `ANTHROPIC_API_KEY` to let Claude refine schema guesses and text (`use_llm=true`). Its answers are validated against your data and cached under
`SDP_DATA_DIR/llm_cache`; without a key everything still works. Only column names and up to 50 sample rows are sent.

### Export, databases, contracts, large runs

```bash
curl -X POST localhost:8000/api/export -H "Content-Type: application/json" -o shop.sql \
  -d '{"source":{"kind":"demo","name":"shop_full","rows":300},"format":"sql","options":{"dialect":"postgres"}}'   # csv json jsonl sql pdf zip
curl -H "X-API-Key: my-key" -X POST localhost:8000/connect/load -H "Content-Type: application/json" \
  -d '{"url":"postgresql://user:pw@localhost/db","source":{"kind":"pack","name":"banking"},"dry_run":true}'     # rolls back; dry_run=false commits
curl -X POST localhost:8000/api/contracts/run -H "Content-Type: application/json" -d '{"pack":"healthcare","rows":300}'   # banking | ecommerce | healthcare
curl -H "X-API-Key: my-key" -X POST localhost:8000/generate -H "Content-Type: application/json" \
  -d '{"kind":"large","params":{"rows":5000000,"format":"csv","chunk_rows":100000,"workers":4}}'                    # progress, cancel, throughput in the manifest
```
Database loads only reach `localhost` unless `SDP_CONNECTOR_ALLOW_HOSTS` lists more hosts; the UI route needs `SDP_CONNECTORS_UI=1`. Drivers are optional:
`pip install "psycopg[binary]"`, `pymysql`, `pymongo`. Language edge-case packs: `GenConfig(edge_cases={"apostrophes": 0.05, "rtl_digits": 0.05})`
(`unicode_diacritics`, `unicode_normalization`, `apostrophes`, `long_names`, `mixed_scripts`, `rtl_digits`).

### Visualize

Every workspace (Tabular, Relational, Documents, Assistant) has a **Visualize** tab: real-vs-synthetic distributions and category mix for a chosen column, correlation
heatmaps (real, synthetic, difference), a TSTR bar chart, orders-per-customer histogram, an ER diagram with an integrity counter, a distance-to-closest-record privacy
chart, locale validity, a document batch summary and a Trust Score gauge. Each chart has a plain-English caption and **Save PNG**; "PDF with charts" embeds them in the scorecard.

```bash
curl "localhost:8000/api/visualize/customers?type=distribution&column=income&bins=20"     # UI route, no key
curl -H "X-API-Key: my-key" "localhost:8000/visualize/shop_full?type=cardinality"          # same, API key
# type = distribution | categorical | correlation | tstr | cardinality | privacy_distance | locale_validity | batch_summary
python scripts/visualize_demo.py out/       # prints every summary, writes visualize_demo.json + a scorecard PDF with charts
```
Responses are small summaries (bins, shares, matrices, counts), never rows. `dataset_id` is a built-in name or a finished saved run id (`GET /api/visualize` lists them).

### Scan realism, edge cases, locale tools

- `sdp.documents.scan`: rotation, skew, blur, noise, JPEG artifacts, stamps, handwriting, low resolution (seeded; presets `office_scan`, `photocopy`, `bad_fax`, `phone_photo`) with per-field ground-truth boxes for OCR training.
- `sdp.edgecases`: packs `boundary_values`, `duplicates`, `typos`, `negative_balances`, `leap_year_dates`, `timezone_shifts`; each takes a rate, tags rows in a hidden `_edge_case` column and reports coverage (`GenConfig(edge_cases={"typos": 0.02})`).
- `sdp.documents.statement_query`: `"last 90 days, balance over $500"` becomes filter parameters; statements always reconcile.
- `sdp.locale.coherent` / `translit` / `codemix`: per-row locale context with a mix (`"70% ur-PK, 30% en"`), the same entity in several scripts and spellings with a shared `entity_id`, and Roman Urdu / Hinglish / Spanglish text with a mixing level.

### Documents and locales

Invoices, receipts and bank statements in en-US, en-GB, ur-PK, ar, hi, es, fr, zh: localized labels, number/date formats, tax names,
native digits, and right-to-left layout for Arabic and Urdu. PDFs embed a Noto (or system TrueType) font; JSON/CSV exports take a UTF-8 BOM option.
Devanagari is not rendered natively (no complex-script shaping); it falls back to English labels with a note. Optional: `python scripts/fetch_fonts.py`
(dry run by default) to fetch Noto locally; the Docker image installs `fonts-noto-core` and `fonts-wqy-microhei`.

## Known limits

- Gaussian copula captures monotone dependence; categorical-to-numeric effects across tables are preserved via target-encoded conditioning, not a deep model.
- CTGAN path is implemented but untested here (needs `ctgan`/`torch`; conditional sampling is copula-only).
- Rule repair handles the shapes documented in `rules/enforce.py`; other rule shapes are verified and reported, not repaired.
- Devanagari PDFs and Noto Sans CJK embedding are not supported by the PDF library (see `CLAUDE.md`).
- The Dockerfile has not been built on the dev machine (Docker not installed there).
- PostgreSQL, MySQL and MongoDB connectors are tested against fakes, not live servers; SQLite is tested for real.
- Multilingual lexicon, messages and fallback text (es, fr) are hand-written and need a fluent-speaker review; free-text fallbacks exist for en/es/fr only.
- Chat editing understands a fixed set of edits (rows, segments, fraud rate, outliers, history months) and asks when unsure; Claude can propose ops for other phrasings.
