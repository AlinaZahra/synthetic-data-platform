# Synthetic Data Platform: project context

Hackathon project: three engines (tabular, relational, document) on one schema-aware pipeline, an NL layer, a rule engine,
a unified Trust Score, and a small React workspace. **All data is fictional. Never add real personal data.**

## Layout
```
backend/sdp/
  common.py            seeding (resolve_seed, spawn_rngs), dtype_family, is_identifier_like
  datasets.py          demo data: make_customers, make_shop, make_shop_full (+ SHOP_FULL_RULES)
  tabular/             GenConfig, TabularGenerator (Gaussian copula; CTGAN optional; sample_conditional), injection, quality/overlay
  evaluation/tstr.py   evaluate_tstr -> scorecard JSON
  relational/          schema graph, inference, ddl, cardinality, generator, integrity, metrics (R5)
  rules/               dsl (parser + plain-language), engine (compile/evaluate, cross-table), enforce (repair), tabular (A4 modes)
  locale/              registry + checksums; packs/*.json = one file per locale (DATA, not code)
  scoring/             fidelity, privacy, locale_validity, constraints(+edge coverage), trust (single + relational), pdf
  documents/           invoice/receipt/statement (Decimal), layout (labels/formats/direction), render (reportlab), shaping (RTL),
                       fonts (Noto discovery), pipeline (retries/isolation/idempotency)
  nl/                  parser (text -> DatasetConfig), builder, domains/*.json
  edgecases.py         T6 scenario packs (+ hidden `_edge_case` tag column, coverage report)
  locale/coherent.py   M3 per-row locale context + mix; translit.py (M4 name variants, shared entity_id); codemix.py (+ codemix_data/*.json, M5)
  documents/statement_query.py (D2 query parsing/filtering), scan.py (D4 scan realism + labels)
  documents/templating.py + templates/<type>/{type.json,template.html}   D3: data-driven document types (schema + Jinja2 + rules -> HTML -> PDF)
  lineage.py + service.py + cli.py   P4/P2: manifests, versions, rerun; job runner; CLI mirroring the REST API
  export.py            UTF-8 exports with optional BOM (csv/json/zip); superseded for new work by exporters/
  ai/                  llm.py (optional Claude client, disk cache, extract_json), schema_infer.py (A1), content.py (A2), chat_edit.py (A6)
  nl/multilingual.py   M7: detect language, lexicon.json -> canonical English -> parser; messages.json = localized answers
  locale/langedge.py   M8 language edge-case packs (registered into edgecases.PACKS)
  exporters/           P1: pluggable exporters (csv, json, jsonl, sql, pdf, zip) + chunk-level encode protocol used by scale.py
  connectors/          P3: PostgreSQL/MySQL/MongoDB/SQLite loaders (dry run, transactions, allowlisted hosts)
  contracts/           P5: starter packs (packs/*.json = generator + expectations), GE-style expectations engine, healthcare generator
  scale.py             P6: chunked, parallel, streaming generation (job kind "large")
  api/main.py + v2.py + public.py   FastAPI; public.py = API-key endpoints (/generate, /jobs, /score) + /api/history for the UI;
                       serves frontend/dist when SDP_STATIC_DIR is set (Docker)
backend/tests/         pytest, ~610 tests
frontend/src/          Vite + React + TS: Guided (4-step), Assistant (+chat edit, multilingual), Tabular, Relational, Documents (+Scan, Bulk), Schema & text,
                       Export & load, Contracts, Locale lab, Trust Score, History. polish.css + ui.tsx (Term tooltips, Advanced, keyboard tabs, dark mode)
scripts/fetch_fonts.py optional Noto download (dry run unless --yes)
Dockerfile, docker-compose.yml   single container (UI built in stage 1, fonts via apt, served by FastAPI)
```

## Commands
```bash
cd backend && python -m pytest -q              # all tests (~3 min)
cd backend && python -m uvicorn sdp.api.main:app --port 8000
cd frontend && npm run dev                     # :5173, proxies /api to :8000
cd frontend && npm run build                   # tsc --noEmit + vite build (type-check gate)
docker compose up --build                      # http://localhost:8000
```

## Conventions and invariants (do not break)
- **Determinism**: same seed + config => identical output. Sampling and injection use independent RNG streams
  (`SeedSequence.spawn`). Any new generator must take a seed and never touch global RNG state.
- **Money is `Decimal`, never float.** Round HALF_UP; tax once per rate group; `reconcile()` recomputes totals from the JSON.
- **Locales are data.** `backend/sdp/locale/packs/<code>.json` (or `$SDP_LOCALE_DIR`). Pack keys incl. `labels`, `pdf`
  (direction, font_script, complex_shaping), `native_digits`, `tax.label_native`. Checksums are named algorithms in
  `locale/checksums.py`. A new document label needs an entry in every pack (a test enforces identical label keys).
- **Rules** (`rules/`): safe parser + vectorised 3-valued evaluator (no `eval`; NULL is unknown, not a violation). A rule is
  evaluated on its *owner* table (child-most table mentioned); ancestors are joined via FKs, aggregates group children by FK.
  `enforce()` repairs (derive / bound / conditional / unique) but NEVER edits key columns. Derived values are snapped exactly.
  Plain-language rules are translated to DSL and echoed back for confirmation, never guessed.
- **Relational generation** conditions child attributes on ancestor attributes (Gaussian conditional sampling; categorical
  ancestors are target-encoded by their effect on the child's numeric columns) and stratifies children-per-parent counts by the
  most explanatory ancestor attribute. Disable with `condition_on_parents=False` to see the difference (R5 score drops).
- **Privacy scoring**: exact copies are judged against the real data's own duplicate rate (chance), identifier-like text columns
  (`is_identifier_like`) are generated fresh in the real format rather than resampled, and the no-holdout baseline is a half-split
  (a leave-one-out baseline is biased low). Do not "simplify" these back.
- **PDFs**: built-in fonts are WinAnsi only. Native-script documents auto-select and embed an installed TrueType font
  (Noto preferred). Arabic/Urdu: reshape + bidi (`shaping.py`), mirrored layout. Pure-LTR strings are NOT passed through bidi.
  Devanagari (complex shaping) is deliberately unsupported: falls back to Latin/English with a note in `presentation.notes`.
  Noto Sans CJK is CFF-based and cannot be embedded by reportlab; Chinese uses a TrueType CJK face (msyh / WenQuanYi).
- **Pipeline**: per-document isolation, retries only for TransientError/OSError/TimeoutError, idempotent ids, atomic writes.
- **Trust Score** = Fidelity 0.40 + Privacy 0.30 + Validity & coverage 0.30; gates (exact copies, integrity violations, rules
  still broken) cap the verdict. Relational Fidelity = 0.65 per-table + 0.35 relation-preservation (R5).
- **Document templates**: a type is a folder (built-ins in `documents/templates/`, extras via `$SDP_TEMPLATE_DIR`) with `type.json` (ordered
  `build` of fields/groups/computed, `rules`) + `template.html`. Adding one must never need engine changes: `register_template_pack` builds the
  spec class, generator, reconciler and renderer from data. Expressions are a safe `ast` subset on Decimal (no eval); `computed` values and
  `recompute` row fields are re-verified by `reconcile()`; `rand()` is generation-only. Jinja is sandboxed + autoescaped; use `doc['items']`, not
  `doc.items`. `reportlab.rl_config.invariant = 1` is set in templating.py so PDFs are byte-stable (lineage hashes depend on it). Template docs are
  Latin-script only (xhtml2pdf has no bidi/shaping); the templated receipt is `retail_receipt` because `receipt` is a built-in name.
- **Job queue (D6)**: `service.submit` returns immediately; jobs run on a 2-thread pool, document batches use the pipeline with 4 workers in chunks
  of 50. Progress is persisted (throttled) in the manifest; cancel is cooperative (checked between chunks/phases), keeps finished output, and a
  cancelled job has `output_hash=None` and no score. `download.zip` builds from the job dir (private `input*` files excluded). On startup
  `service.recover` marks jobs that were running when the process died as failed. Document job outputs are `index.json` + `report.json` + `docs/*`;
  the report is stored without timings so the output hash stays reproducible.
- **LLM layer (A1/A2/A6)**: the platform must work with no API key and no network. Every LLM answer is validated against the data (types must
  parse the sample, relationships must overlap, constraints are re-checked and flagged if the sample violates them, ops go through the same
  pydantic validation as rule-based ops) and cached on disk (`SDP_DATA_DIR/llm_cache`, keyed by client name + prompt). Never send more than
  column names and <= 50 sample rows; UI/API default `use_llm=false`. Tests inject fakes with `llm.set_client(...)` (name must be unique per reply,
  the cache key includes it).
- **Chat editing (A6)**: state = (base DatasetConfig, ops history); every op has seed `cfg.seed*100003 + 7919*(i+1)`, so replay is exact and the
  server is stateless. Ops append/drop only the affected rows and their history; unaffected columns/tables stay byte-identical (tests assert it).
  Outlier edits keep hidden `_outlier_<col>` markers. Ambiguity becomes a question, never a guess.
- **Multilingual (M7)**: parser stays English. `parse_request` emits stable issue codes (`ParseResult.issues`) that `messages.json` localizes. Lexicon
  and messages are hand-written data and need fluent-speaker review. Add a language = add entries to both JSON files (a test enforces key parity).
- **Exporters (P1)**: `export_bytes(tables, fmt, schema, **opts)`. Hidden `_` columns never leave. Row-oriented exporters also implement
  `stream_begin/encode_chunk/stream_end` so scale.py serialises in workers. SQL rows are re-batched across chunks so output does not depend on chunk size.
- **Connectors (P3)**: host allowlist (`SDP_CONNECTOR_ALLOW_HOSTS`, default localhost), sqlite files only inside the data dir, passwords masked, dry run by
  default. UI route is off unless `SDP_CONNECTORS_UI=1`; `POST /connect/load` needs the API key. MySQL cannot roll back DDL (dry run is plan-only,
  failures drop what the run created); Mongo without a replica set is plan-only / compensating deletes. PostgreSQL/MySQL/Mongo paths are tested with
  fakes only (no servers on the dev machine); SQLite is tested for real.
- **Contracts (P5)**: expectations use GE names/kwargs (+3 custom types, marked `meta.custom` on export). A broken expectation is reported, not raised.
- **Large runs (P6)**: chunk seed = f(seed, chunk index), so the file is identical for any worker count and executor; chunks are serialised in the worker
  and written in order; memory is window x chunk_rows. Integer `*_id` columns are renumbered globally (the generator would otherwise repeat ids per chunk).
  `executor=auto` uses processes from 1M rows (~2.4x faster at 4M rows on the dev machine: 152k rows/s vs 63k with threads). Downloads stream from disk.
- **Visualize** (`sdp/visualize.py`, `api/visualize.py`, `frontend/src/viz/`): `GET /api/visualize/{id}?type=` (UI) and `GET /visualize/{id}` (API key) return
  only summaries (histogram bins, category shares, correlation matrices, orphan counts, pass rates), never rows: identifier-like columns are refused, categories
  with < 5 real rows merge into "other", bins/categories/matrix/rows are capped (tests assert < 30 KB per payload). `id` = built-in (customers, students, employees,
  shop_full, bank_customers, ecommerce_customers, documents-demo) or a finished saved run. Types: distribution, categorical, correlation, tstr, cardinality (+ER graph
  + integrity), privacy_distance, locale_validity, batch_summary. Defaults pick the first table with something to chart. Every payload carries a plain-English `caption`.
  The scorecard PDF embeds charts drawn natively with reportlab from those payloads (`scoring/pdf_charts.py`, `POST /api/trust/pdf {report, visuals}`). Frontend:
  Recharts + React Flow + html-to-image (PNG export), lazy-loaded (`LazyVisualize`) so the main bundle stays small; palette = `--real` slate, `--syn` teal (`viz.css`).
  Demo: `python scripts/visualize_demo.py`.
- **History autosave** (`api/autosave.py`): every explicit generate/export/download (nl generate, export tabular/relational/nl, single document PDF,
  contract run, pack export) queues a background job with the same request, so History shows it and it can be rerun. Live previews and scores are
  never recorded. Off with `SDP_AUTOSAVE=0` (tests/conftest.py turns it off by default; test_autosave.py turns it on). Autosave failures never fail the
  user's request. Job kind `pack` = starter pack + contract report. Known gap: a relational download with an edited graph/cardinality is recorded
  without them (RelationalParams has no graph field), so its rerun may differ.
- **Lineage**: a rerun of an unchanged request must reproduce the same `output_hash`; the manifest records `reproduced` and `code_changed`
  (code fingerprint = hash of all sdp .py/.json). Outputs are written as separate files (no zips, whose timestamps break hashing).
  Inline input data lives in the job dir (`input*.csv`), manifests keep only references.
- **Public API auth**: `/generate`, `/jobs*`, `/score/*` need `X-API-Key` (or Bearer) from `SDP_API_KEYS`; if unset an ephemeral key is logged at
  startup. `/api/*` (UI, incl. `/api/history`) is deliberately unauthenticated: same-origin UI only, put it behind your own boundary.
- **Scan labels**: bounding boxes come from the renderer (`RenderResult.boxes`, keyed via `Layout.keys/raw`); geometric augmentations transform
  boxes with the same affine matrix as the pixels. New drawn text must pass `key=` to `txt()` or it will be unlabeled.
- **Edge cases** tag rows in `_edge_case` (hidden: `visible()` strips `_`-prefixed columns for previews/exports). Coverage counts only cases that
  are applicable to the dataset's columns. Adding `edge_cases` to GenConfig must not change the clean sampling stream (spawn_rngs(seed, 3)).
- **Statements**: every row keeps a `seq`; filtered statements reconcile because the opening balance is recomputed from the first shown row.
- Code-mix templates and transliteration tables are hand-written data: a fluent-speaker review is still needed.
- NL parsing is deterministic (no LLM); `/api/nl/generate` returns 409 unless `confirmed=true`.

## Environment notes
- Python 3.13/3.14, **pandas 3.x** (string dtype `str`; nullable int FKs arrive as float). Windows dev machine.
- Git Bash heredocs containing apostrophes/non-ASCII break: create such files with the editor tools instead.
- Docker is not installed on the dev machine, so the Dockerfile is unbuilt locally.
- Fonts on the dev machine: Arial (Arabic), Nirmala (Devanagari, unused), msyh (CJK). Noto not installed; tests that need a font skip if none.

## Known limits / next steps
- Gaussian copula: monotone dependence only; categorical->numeric effects rely on the target-encoding trick above.
- CTGAN path is written but untested (needs `pip install ctgan`; conditional sampling is copula-only).
- Rule repair covers the shapes listed in `rules/enforce.py`; composite check rules are verified but not repaired.
- Rules are enforced before null/outlier injection on purpose; injected defects can break them.
- Privacy metrics are distance-based heuristics, not differential privacy.
- Trust Score needs real reference data; NL-generated datasets get a validation summary instead.
- Frontend: guided flow supports CSV upload (client-side parsing, <=8 MB each); ER layout is layered, not force-directed.
