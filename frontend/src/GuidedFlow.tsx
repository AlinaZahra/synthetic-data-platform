import { useMemo, useState, type ChangeEvent } from "react";
import { post, postBlob, saveBlob, type NLResult, type ParseResponse, type RelationalFullResponse, type Row, type TabularResponse,
  type TabularRulesReport, type Overlay, type RulesReport, type TrustReport } from "./api";
import { ChartGrid } from "./Charts";
import { parseCsv } from "./csv";
import { ERDiagram } from "./ERDiagram";
import { PreviewFrame, Skeleton } from "./Preview";
import { RulesEditor } from "./RulesEditor";
import { QualityCard, TrustCard } from "./TrustCard";
import { DataTable, Field, Tabs, pct, useLive } from "./ui";

type Kind = "describe" | "tabular-demo" | "relational-demo" | "upload";
const STEPS = [
  { title: "Upload or describe", help: "Start from a sentence, a sample dataset, or your own CSV files." },
  { title: "Configure", help: "Choose how much data to make and any rules it must obey." },
  { title: "Preview and score", help: "The preview updates as you change settings. Score it when it looks right." },
  { title: "Export", help: "Download the data and the trust report." },
] as const;
const PRIMARY = ["Continue", "Generate preview", "Looks good, continue", "Download data"] as const;

const SHOP_RULES = [
  "orders.total = SUM(order_items.quantity * order_items.unit_price)", "order_items.unit_price = products.price",
  "shipments.shipped_date >= orders.order_date", "shipments.delivered_date >= shipments.shipped_date", "orders.discount <= 0.3",
  "orders.status = 'cancelled' => COUNT(shipments) = 0", "orders.status IN ('shipped', 'delivered') => COUNT(shipments) >= 1",
  "shipments.delivered_date IS NOT NULL => orders.status = 'delivered'"];
const EXAMPLES = ["5,000 Pakistani bank customers, 3% fraud, 12 months of history", "800 French e-commerce customers, 0.5% chargebacks"];

interface Upload { name: string; rows: Row[]; columns: string[]; truncated: boolean }

export function GuidedFlow() {
  const [step, setStep] = useState(0);
  const [kind, setKind] = useState<Kind>("tabular-demo");
  const [text, setText] = useState(EXAMPLES[0]);
  const [parsed, setParsed] = useState<ParseResponse | null>(null);
  const [parseBusy, setParseBusy] = useState(false);
  const [uploads, setUploads] = useState<Upload[]>([]);
  const [uploadErr, setUploadErr] = useState<string | null>(null);
  const [rows, setRows] = useState(500);
  const [seed, setSeed] = useState(0);
  const [scale, setScale] = useState(1);
  const [ruleText, setRuleText] = useState("");
  const [rulesOk, setRulesOk] = useState<string[]>([]);
  const [ruleMode, setRuleMode] = useState<"hybrid" | "repair" | "reject">("hybrid");
  const [enforce, setEnforce] = useState(true);
  const [bom, setBom] = useState(false);
  const [tab, setTab] = useState("Rows");
  const [trust, setTrust] = useState<TrustReport | null>(null);
  const [showQuality, setShowQuality] = useState(false);   // described datasets: the score is revealed by the button, like Compute Trust Score elsewhere
  const [trustBusy, setTrustBusy] = useState(false);
  const [trustErr, setTrustErr] = useState<string | null>(null);
  const [exportErr, setExportErr] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const [table, setTable] = useState<string | null>(null);

  /** Back to a clean slate: step 1, sample table, default rows/seed/rules, no upload, no description, no scores. History and downloads are untouched. */
  const startOver = () => {
    setStep(0); setKind("tabular-demo"); setText(EXAMPLES[0]); setParsed(null); setParseBusy(false); setUploads([]); setUploadErr(null);
    setRows(500); setSeed(0); setScale(1); setRuleText(""); setRulesOk([]); setRuleMode("hybrid"); setEnforce(true); setBom(false);
    setTab("Rows"); setTrust(null); setShowQuality(false); setTrustBusy(false); setTrustErr(null); setExportErr(null); setSaved(null); setTable(null);
  };

  // What "kind" means for the API
  const mode: "nl" | "tabular" | "relational" = kind === "describe" ? "nl" : kind === "tabular-demo" ? "tabular" : kind === "relational-demo" ? "relational"
    : uploads.length > 1 ? "relational" : "tabular";
  const tablesBody = useMemo(() => (kind === "upload" && uploads.length > 1 ? Object.fromEntries(uploads.map((u) => [u.name.replace(/\.csv$/i, ""), u.rows])) : undefined), [kind, uploads]);
  const config = parsed?.config ? { ...parsed.config, rows, seed } : null;

  const body = useMemo(() => {
    if (mode === "tabular") {
      return { dataset: kind === "upload" ? "inline" : "demo", data: kind === "upload" ? uploads[0]?.rows : undefined, config: { rows, seed }, rules: rulesOk, rule_mode: ruleMode };
    }
    if (mode === "relational") {
      return { dataset: kind === "upload" ? "inline" : "shop_full", tables: tablesBody, seed, scale, rules: rulesOk, enforce };
    }
    return { config, confirmed: true };
  }, [mode, kind, uploads, rows, seed, scale, rulesOk, ruleMode, enforce, tablesBody, config]);

  const live = step === 2;
  const gen = useLive(async (signal) => {
    if (!live) return null;
    if (mode === "tabular") return post<TabularResponse & { rules: TabularRulesReport | null; overlay: Overlay }>("/api/tabular/generate", body, signal);
    if (mode === "relational") return post<RelationalFullResponse>("/api/relational/generate", body, signal);
    return config ? post<NLResult>("/api/nl/generate", body, signal) : null;
  }, [live, JSON.stringify(body)], 500);
  const [retry, setRetry] = useState(0);
  void retry;

  const canContinue = [
    kind === "describe" ? !!parsed?.ok : kind === "upload" ? uploads.length > 0 : true,
    true, true, true][step];

  // ---- step 1 actions
  const doParse = async () => {
    setParseBusy(true);
    try { const p = await post<ParseResponse>("/api/nl/parse", { text, seed }); setParsed(p); if (p.ok && p.config) setRows(Number((p.config as { rows: number }).rows)); }
    finally { setParseBusy(false); }
  };
  const onFiles = async (e: ChangeEvent<HTMLInputElement>) => {
    setUploadErr(null);
    const list = Array.from(e.target.files ?? []);
    const out: Upload[] = [];
    for (const f of list) {
      if (f.size > 8_000_000) { setUploadErr(`${f.name} is larger than 8 MB.`); return; }
      const p = parseCsv(await f.text());
      if (!p.rows.length) { setUploadErr(`${f.name} has no rows.`); return; }
      out.push({ name: f.name, rows: p.rows, columns: p.columns, truncated: p.truncated });
    }
    setUploads(out);
    if (out.length === 1) setRows(Math.min(2000, Math.max(200, out[0].rows.length)));
  };
  const pickKind = (k: Kind) => {
    setKind(k); setTrust(null); setShowQuality(false); setTab("Rows"); setTable(null);
    if (k === "relational-demo" && !ruleText) setRuleText(SHOP_RULES.join("\n"));
    if (k !== "relational-demo" && ruleText === SHOP_RULES.join("\n")) setRuleText("");
  };

  // ---- step 3 actions
  const runTrust = async () => {
    setTrustBusy(true); setTrustErr(null);
    try {
      setTrust(mode === "relational"
        ? await post<TrustReport>("/api/relational/trust", { ...body })
        : await post<TrustReport>("/api/trust/report", { dataset: kind === "upload" ? "inline" : "demo", data: kind === "upload" ? uploads[0]?.rows : undefined, rows: Math.max(100, rows), seed, rules: rulesOk }));
    } catch (e) { setTrustErr((e as Error).message); } finally { setTrustBusy(false); }
  };
  const doExport = async () => {
    setExportErr(null);
    try {
      if (mode === "tabular") saveBlob(await postBlob("/api/export/tabular", { ...body, bom }), "synthetic.csv");
      else if (mode === "relational") saveBlob(await postBlob("/api/export/relational", { ...body, bom }), "synthetic-tables.zip");
      else saveBlob(await postBlob("/api/export/nl", { config, bom }), "synthetic-dataset.zip");
    } catch (e) { setExportErr((e as Error).message); }
  };

  const saveHistory = async () => {
    setExportErr(null); setSaved(null);
    try {
      const b = body as Record<string, unknown>;
      const payload = mode === "tabular" ? { kind: "tabular", params: { dataset: b.dataset, data: b.data, rows, seed, rules: rulesOk, rule_mode: ruleMode } }
        : mode === "relational" ? { kind: "relational", params: { dataset: b.dataset, tables: b.tables, seed, scale, rules: rulesOk, enforce } }
        : { kind: "nl", params: { config, seed } };
      const r = await post<{ id: string }>("/api/history/generate", { ...payload, wait: false });
      setSaved(r.id);
    } catch (e) { setExportErr((e as Error).message); }
  };
  const next = () => { if (step === 3) void doExport(); else setStep(step + 1); };
  const data = gen.data as (TabularResponse & { rules: TabularRulesReport | null; overlay: Overlay }) | RelationalFullResponse | NLResult | null;

  return (
    <>
      <main className="center">
        <h1>Guided</h1>
        <ol className="stepper" aria-label="Progress">
          {STEPS.map((s, i) => (
            <li key={s.title} aria-current={i === step ? "step" : undefined} className={i < step ? "done" : ""}>
              <button onClick={() => i < step && setStep(i)} disabled={i >= step} aria-label={`Step ${i + 1}: ${s.title}`}><span className="num">{i < step ? "✓" : i + 1}</span>{s.title}</button>
            </li>
          ))}
        </ol>
        <p className="lede">{STEPS[step].help}</p>

        {step === 0 && (
          <section>
            <div className="choices" role="radiogroup" aria-label="Data source">
              {([["describe", "Describe it", "“5,000 Pakistani bank customers, 3% fraud”"], ["tabular-demo", "Sample table", "2,000 fictional customers"],
                ["relational-demo", "Sample shop", "5 linked tables with rules"], ["upload", "Upload CSV", "One file, or several linked files"]] as const).map(([k, t, d]) => (
                <button key={k} role="radio" aria-checked={kind === k} className={`choice${kind === k ? " on" : ""}`} onClick={() => pickKind(k)}><b>{t}</b><span>{d}</span></button>
              ))}
            </div>
            {kind === "describe" && (
              <div style={{ marginTop: 20 }}>
                <div className="chat">
                  <textarea aria-label="Describe your dataset" rows={2} value={text} onChange={(e) => { setText(e.target.value); setParsed(null); }}
                    onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void doParse(); } }} />
                  <button className="btn primary" onClick={doParse} disabled={parseBusy || !text.trim()}>{parseBusy ? "Reading…" : "Parse"}</button>
                </div>
                <div className="chips">{EXAMPLES.map((x) => <button key={x} className="chip" onClick={() => { setText(x); setParsed(null); }}>{x}</button>)}</div>
                {parsed && !parsed.ok && <div className="state error"><b>I need a bit more</b><ul>{parsed.errors.map((e) => <li key={e}>{e}</li>)}</ul></div>}
                {parsed?.ok && (
                  <div className="card" style={{ marginTop: 16 }}>
                    <b>Here’s what I understood</b>
                    <table><tbody>{parsed.explanation.map((e) => <tr key={e.field}><td className="muted">{e.field}</td><td><b>{e.value}</b></td><td className="muted">from “{e.source}”</td></tr>)}</tbody></table>
                    {parsed.warnings.map((w) => <p key={w} className="muted">⚠ {w}</p>)}
                    <p className="muted">Press Continue to confirm this and set it up.</p>
                  </div>
                )}
              </div>
            )}
            {kind === "upload" && (
              <div style={{ marginTop: 20 }}>
                <label className="dropzone">
                  <input type="file" accept=".csv,text/csv" multiple onChange={onFiles} aria-label="Choose CSV files" />
                  <span>{uploads.length ? "Choose different files" : "Choose one or more CSV files"}</span>
                </label>
                {uploadErr && <div className="err" role="alert">{uploadErr}</div>}
                {uploads.map((u) => <div key={u.name} className="muted">{u.name}: {u.rows.length.toLocaleString()} rows, {u.columns.length} columns{u.truncated ? " (first 50,000 used)" : ""}</div>)}
                {uploads.length > 1 && <p className="muted">Several files: keys and relationships are inferred automatically. Name key columns like <code>customer_id</code> for best results.</p>}
              </div>
            )}
          </section>
        )}

        {step === 1 && (
          <section className="config-step">
            {mode === "nl" && (<div className="row2"><Field label="Rows"><input type="number" min={1} max={200000} value={rows} onChange={(e) => setRows(Math.max(1, +e.target.value || 1))} /></Field>
              <Field label="Seed"><input type="number" value={seed} onChange={(e) => setSeed(+e.target.value || 0)} /></Field></div>)}
            {mode === "tabular" && (
              <>
                <div className="row2"><Field label="Rows to generate"><input type="number" min={1} max={20000} value={rows} onChange={(e) => setRows(Math.max(1, +e.target.value || 1))} /></Field>
                  <Field label="Seed (same seed, same data)"><input type="number" value={seed} onChange={(e) => setSeed(+e.target.value || 0)} /></Field></div>
                <h3>Rules the data must obey</h3>
                <RulesEditor value={ruleText} onChange={setRuleText} dataset={kind === "upload" ? "inline" : "customers"} tables={undefined} onValid={setRulesOk}
                  examples={kind === "upload" ? [] : ["age must be at least 25", "income <= 60000", "plan must be one of basic, pro"]} />
                <Field label="How to enforce"><select value={ruleMode} onChange={(e) => setRuleMode(e.target.value as typeof ruleMode)}>
                  <option value="hybrid">Repair, then re-draw what is left (recommended)</option><option value="repair">Repair only (keeps every row)</option><option value="reject">Reject and re-draw (exact, slower)</option></select></Field>
              </>
            )}
            {mode === "relational" && (
              <>
                <div className="row2"><Field label="Size (× real rows)"><input type="number" min={0.1} max={20} step={0.5} value={scale} onChange={(e) => setScale(Math.max(0.1, +e.target.value || 1))} /></Field>
                  <Field label="Seed"><input type="number" value={seed} onChange={(e) => setSeed(+e.target.value || 0)} /></Field></div>
                <h3>Rules across tables</h3>
                <RulesEditor value={ruleText} onChange={setRuleText} dataset={kind === "upload" ? "inline" : "shop_full"} tables={tablesBody} onValid={setRulesOk} />
                <label className="field"><span>Enforce the rules (turn off to see what naive generation would produce)</span><input type="checkbox" checked={enforce} onChange={(e) => setEnforce(e.target.checked)} /></label>
              </>
            )}
          </section>
        )}

        {step === 2 && (
          <section>
            <PreviewFrame loading={gen.busy} error={gen.error} hasData={!!data} onRetry={() => setRetry((r) => r + 1)}
              emptyTitle="Ready when you are" emptyText="Adjust settings in step 2 and the preview appears here." skeleton={<Skeleton rows={8} cols={6} />}>
              {data && mode === "tabular" && <TabularPreview d={data as TabularResponse & { rules: TabularRulesReport | null; overlay: Overlay }} tab={tab} setTab={setTab} />}
              {data && mode === "relational" && <RelationalPreview d={data as RelationalFullResponse} tab={tab} setTab={setTab} table={table} setTable={setTable} />}
              {data && mode === "nl" && <NLPreview d={data as NLResult} table={table} setTable={setTable} />}
            </PreviewFrame>
            {mode === "nl" && (
              <div style={{ marginTop: 24 }}>
                {!showQuality && <button className="btn" onClick={() => setShowQuality(true)} disabled={!(data as NLResult | null)?.quality}>Compute Trust Score</button>}
                {showQuality && (data as NLResult | null)?.quality && <QualityCard quality={(data as NLResult).quality!} />}
              </div>
            )}
            {mode !== "nl" && (
              <div style={{ marginTop: 24 }}>
                {!trust && <button className="btn" onClick={runTrust} disabled={trustBusy || !data}>{trustBusy ? "Scoring… (runs privacy attacks)" : "Compute Trust Score"}</button>}
                {trustBusy && <Skeleton rows={4} cols={3} />}
                {trustErr && <div className="err" role="alert">{trustErr}</div>}
                {trust && <TrustCard report={trust} />}
              </div>
            )}
          </section>
        )}

        {step === 3 && (
          <section>
            <div className="card">
              <p>{mode === "tabular" ? "One CSV file with your synthetic rows." : "A ZIP with one CSV per table, the relationship graph and the scorecard."}</p>
              <label className="field"><span>UTF-8 with byte-order mark (helps Excel open Urdu, Arabic and Chinese text)</span><input type="checkbox" checked={bom} onChange={(e) => setBom(e.target.checked)} /></label>
              {exportErr && <div className="err" role="alert">{exportErr}</div>}
              <p className="muted">Same seed and settings always give the same file. Seed: {seed}.</p>
              <button className="btn" onClick={saveHistory}>Save to history</button>
              {saved && <p className="muted" role="status">Saved as job <code>{saved}</code>. Open <b>History</b> to rerun it from its manifest.</p>}
            </div>
            {trust && <div style={{ marginTop: 16 }}><TrustCard report={trust} /></div>}
          </section>
        )}

        <div className="stepfoot">
          <button className="btn" onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0}>Back</button>
          <button className="btn primary" onClick={next} disabled={!canContinue || (step === 2 && !data)}>{PRIMARY[step]}</button>
        </div>
      </main>
      <aside className="config">
        <h2>Your choices</h2>
        <dl className="summary">
          <dt>Source</dt><dd>{kind === "describe" ? (parsed?.ok ? "Described in words" : "Describe it") : kind === "tabular-demo" ? "Sample table" : kind === "relational-demo" ? "Sample shop (5 tables)" : uploads.length ? uploads.map((u) => u.name).join(", ") : "Upload"}</dd>
          <dt>{mode === "relational" ? "Size" : "Rows"}</dt><dd>{mode === "relational" ? `${scale}× real` : rows.toLocaleString()}</dd>
          <dt>Seed</dt><dd>{seed}</dd>
          <dt>Rules</dt><dd>{mode === "nl" ? "from the description" : `${rulesOk.length} active`}</dd>
        </dl>
        <button className="btn" onClick={startOver}>Start over</button>
      </aside>
    </>
  );
}

// ------------------------------------------------------------------ previews
function RulesTable({ rules, naive }: { rules: RulesReport | TabularRulesReport; naive?: boolean }) {
  return (
    <>
      <div className="cards">
        <div className="card"><div className="big">{pct(rules.pass_rate_before, 0)}</div><div className="label">rows passing, naive generation</div></div>
        <div className="card"><div className="big" style={{ color: rules.violations_after ? "var(--bad)" : "var(--ok)" }}>{pct(rules.reconciliation_pass_rate, 0)}</div><div className="label">rows passing after enforcement ({rules.violations_after} violations)</div></div>
        {!naive && "derived_pass_rate" in rules && rules.rules.some((r) => r.kind === "derive") && <div className="card"><div className="big">{pct(rules.derived_pass_rate, 0)}</div><div className="label">derived fields reconcile (was {pct(rules.derived_pass_rate_before, 0)})</div></div>}
      </div>
      <div className="tablewrap"><table><thead><tr><th>rule</th><th>violations before</th><th>after</th><th>how it was fixed</th></tr></thead>
        <tbody>{rules.rules.map((r) => <tr key={r.name}><td><code>{r.dsl}</code></td><td>{r.violations_before}</td><td>{r.violations_after}</td><td>{r.unrepairable ?? (r.strategy || (r.violations_before ? "—" : "already held"))}</td></tr>)}</tbody></table></div>
    </>
  );
}

function TabularPreview({ d, tab, setTab }: { d: TabularResponse & { rules: TabularRulesReport | null; overlay: Overlay }; tab: string; setTab: (t: string) => void }) {
  const tabs = ["Rows", "Distributions", ...(d.rules ? ["Rules"] : [])] as const;
  return (
    <>
      <Tabs tabs={tabs} value={(tabs as readonly string[]).includes(tab) ? tab as typeof tabs[number] : "Rows"} onChange={setTab} />
      {tab === "Rows" || !(tabs as readonly string[]).includes(tab) ? (<><p className="muted">{d.n_rows.toLocaleString()} rows generated · first {d.preview.length} shown</p><DataTable columns={d.columns} rows={d.preview} /></>) : null}
      {tab === "Distributions" && <ChartGrid overlay={d.overlay} />}
      {tab === "Rules" && d.rules && <RulesTable rules={d.rules} />}
    </>
  );
}

function RelationalPreview({ d, tab, setTab, table, setTable }: { d: RelationalFullResponse; tab: string; setTab: (t: string) => void; table: string | null; setTable: (t: string) => void }) {
  const tabs = ["Tables", "Relations", ...(d.scorecard.rules ? ["Rules"] : [])] as const;
  const active = (tabs as readonly string[]).includes(tab) ? tab : "Tables";
  const t = table && d.preview[table] ? table : d.order[0];
  const sc = d.scorecard, rm = d.relation_metrics;
  return (
    <>
      <div className="cards">
        <div className="card"><div className="big" style={{ color: sc.integrity.ok ? "var(--ok)" : "var(--bad)" }}>{sc.integrity.total_violations}</div><div className="label">integrity violations</div></div>
        {rm && <div className="card"><div className="big">{rm.score.toFixed(0)}</div><div className="label">relation-preservation score</div></div>}
        {sc.rules && <div className="card"><div className="big">{pct(sc.rules.reconciliation_pass_rate, 0)}</div><div className="label">cross-table rules hold</div></div>}
      </div>
      <Tabs tabs={tabs} value={active as typeof tabs[number]} onChange={setTab} />
      {active === "Tables" && (<><div className="tabs">{d.order.map((n) => <button key={n} aria-selected={n === t} onClick={() => setTable(n)}>{n} ({sc.row_counts[n]})</button>)}</div>
        <DataTable columns={d.columns[t]} rows={d.preview[t]} /></>)}
      {active === "Relations" && (
        <>
          <ERDiagram graph={d.graph} rowCounts={sc.row_counts} />
          {rm && (<><h3>How well the relationships survived</h3>
            {Object.entries(rm.components).filter(([, c]) => c.score !== null).map(([k, c]) => (
              <div className="metric" key={k}><span>{k.replace("_", " ")}</span><span className="bar wide"><i style={{ width: `${c.score}%` }} /></span><b>{c.score!.toFixed(0)}</b><span className="muted">{c.summary}</span></div>
            ))}</>)}
        </>
      )}
      {active === "Rules" && sc.rules && <RulesTable rules={sc.rules} naive={false} />}
    </>
  );
}

function NLPreview({ d, table, setTable }: { d: NLResult; table: string | null; setTable: (t: string) => void }) {
  const names = Object.keys(d.preview);
  const t = table && d.preview[table] ? table : names[0];
  const v = d.validation;
  return (
    <>
      <div className="cards">
        <div className="card"><div className="big">{v.locale.valid_pct.toFixed(0)}%</div><div className="label">records locale-valid</div></div>
        <div className="card"><div className="big" style={{ color: v.integrity.total_violations ? "var(--bad)" : "var(--ok)" }}>{v.integrity.total_violations}</div><div className="label">integrity violations</div></div>
        <div className="card"><div className="big">{v.constraints.pass_pct.toFixed(0)}%</div><div className="label">rows passing rules</div></div>
        {v.flag && <div className="card"><div className="big">{(v.flag.achieved * 100).toFixed(1)}%</div><div className="label">{v.flag.name} (asked {(v.flag.requested * 100).toFixed(1)}%)</div></div>}
      </div>
      <div className="tabs">{names.map((n) => <button key={n} aria-selected={n === t} onClick={() => setTable(n)}>{n} ({v.rows[n].toLocaleString()})</button>)}</div>
      <DataTable columns={d.columns[t]} rows={d.preview[t]} />
    </>
  );
}
