import { useMemo, useState } from "react";
import { get, post, postBlob, saveBlob, type Row, type Overlay, type TabularResponse, type TabularRulesReport, type TstrResponse } from "./api";
import { ChartGrid } from "./Charts";
import { parseCsv } from "./csv";
import { VisualizePanel } from "./viz/LazyVisualize";
import { PreviewFrame, Skeleton } from "./Preview";
import { RulesEditor } from "./RulesEditor";
import { DataTable, Field, Panel, Tabs, pct, useLive } from "./ui";

interface Demo { columns: string[]; dtypes: Record<string, string>; target: string }
interface EdgeReport { coverage_pct: number | null; rows_tagged: number; cases_possible: number; cases_exercised: number; not_applicable: string[];
  packs: Record<string, { rate: number; rows_affected: number; rows_added?: number; coverage_pct: number | null; cases_possible: string[]; cases_exercised: string[]; columns: string[] }> }
type Gen = TabularResponse & { rules: TabularRulesReport | null; overlay: Overlay; edge_cases: EdgeReport | null; edge_tags: (string | null)[] | null };
const EDGE_PACKS: [string, string][] = [["boundary_values", "Boundary values (0, max int, empty string)"], ["duplicates", "Duplicates (exact and near)"], ["typos", "Typos"],
  ["negative_balances", "Negative balances"], ["leap_year_dates", "Leap-year and year-end dates"], ["timezone_shifts", "Timezone shifts and DST"]];
const TABS = ["Preview", "Visualize", "Distributions", "Rules", "Edge cases", "Quality", "Injection log", "Utility (TSTR)"] as const;
const isNumeric = (dt: string) => /int|float/.test(dt);

export function useDemo(sample = "customers") {
  return useLive(() => get<Demo>(`/api/demo/tabular?sample=${encodeURIComponent(sample)}`), [sample], 0);
}

interface Upload { name: string; rows: Row[]; columns: string[]; dtypes: Record<string, string>; truncated: boolean }
const MAX_UPLOAD_ROWS = 20_000;
const RULE_EXAMPLES: Record<string, string[]> = {
  customers: ["age must be at least 25", "income <= 60000", "plan must be one of basic, pro"],
  students: ["prior_gpa >= 2", "attendance_pct >= 50", "study_hours_week <= 40"],
  employees: ["salary must be at least 40000", "performance_rating >= 2", "age must be at least 21"],
};

/** Column types for an uploaded table, in the same words the server uses (int64, float64, bool, datetime64, object). */
function guessDtypes(columns: string[], rows: Row[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const c of columns) {
    const vals = rows.map((r) => r[c]).filter((v) => v !== null && v !== undefined && v !== "");
    out[c] = !vals.length ? "object" : vals.every((v) => typeof v === "boolean") ? "bool"
      : vals.every((v) => typeof v === "number") ? (vals.every((v) => Number.isInteger(v)) ? "int64" : "float64")
        : vals.every((v) => typeof v === "string" && /^\d{4}-\d\d-\d\d/.test(v)) ? "datetime64[ns]" : "object";
  }
  return out;
}

export function TabularWorkspace() {
  const [sample, setSample] = useState("customers");           // a built-in table, or "upload"
  const [upload, setUpload] = useState<Upload | null>(null);
  const [uploadErr, setUploadErr] = useState<string | null>(null);
  const [targetPick, setTargetPick] = useState<string | null>(null);
  const samples = useLive(() => get<{ name: string; title: string; target: string }[]>("/api/demo/tabular/samples"), [], 0).data;
  const builtin = useDemo(sample === "upload" ? "customers" : sample).data;
  const demo: Demo | undefined = sample === "upload" && upload ? { columns: upload.columns, dtypes: upload.dtypes, target: upload.columns[upload.columns.length - 1] } : builtin ?? undefined;
  // what tells the server which table to use (built-in by name, or the uploaded rows)
  const dataBody = useMemo(() => (sample === "upload" && upload ? { dataset: "inline", data: upload.rows } : { dataset: "demo", sample: sample === "upload" ? "customers" : sample }),
    [sample, upload]);
  const dataKey = sample === "upload" ? `upload:${upload?.name}:${upload?.rows.length}` : sample;
  const target = targetPick && demo?.columns.includes(targetPick) ? targetPick : demo?.target;
  const [tab, setTab] = useState<(typeof TABS)[number]>("Preview");
  const [rows, setRows] = useState(500);
  const [seed, setSeed] = useState(0);
  const [method, setMethod] = useState<"gaussian_copula" | "ctgan">("gaussian_copula");
  const [outlierMethod, setOutlierMethod] = useState<"iqr" | "zscore" | "scale">("iqr");
  const [nulls, setNulls] = useState<Record<string, number>>({});
  const [outliers, setOutliers] = useState<Record<string, number>>({});
  const [tstr, setTstr] = useState<TstrResponse | null>(null);
  const [tstrBusy, setTstrBusy] = useState(false);
  const [tstrErr, setTstrErr] = useState<string | null>(null);
  const [ruleText, setRuleText] = useState("");
  const [edge, setEdge] = useState<Record<string, number>>({});
  const [rulesOk, setRulesOk] = useState<string[]>([]);
  const [ruleMode, setRuleMode] = useState<"hybrid" | "repair" | "reject">("hybrid");
  const [retry, setRetry] = useState(0);

  const config = useMemo(() => {
    const clean = (m: Record<string, number>) => Object.fromEntries(Object.entries(m).filter(([, v]) => v > 0));
    return { rows, seed, null_rate: clean(nulls), outlier_rate: clean(outliers), outlier_method: outlierMethod, edge_cases: clean(edge) };
  }, [rows, seed, nulls, outliers, outlierMethod, edge]);

  const ready = sample !== "upload" || !!upload;
  const { data, error, busy } = useLive(
    (signal) => (ready ? post<Gen>("/api/tabular/generate", { ...dataBody, method, config, rules: rulesOk, rule_mode: ruleMode }, signal) : Promise.resolve(null)),
    [JSON.stringify(config), method, JSON.stringify(rulesOk), ruleMode, retry, dataKey],
  );

  /** Switching table: the columns differ, so per-column settings and results from the old table are cleared. */
  const changeSample = (s: string) => {
    setSample(s); setNulls({}); setOutliers({}); setEdge({}); setRuleText(""); setRulesOk([]); setTstr(null); setTstrErr(null); setTargetPick(null); setTab("Preview");
  };
  const onFile = async (f: File | undefined) => {
    if (!f) return;
    setUploadErr(null);
    try {
      if (f.size > 8_000_000) throw new Error("That file is over 8 MB. Use a smaller sample of it.");
      const { rows: r, columns, truncated } = parseCsv(await f.text(), MAX_UPLOAD_ROWS);
      if (r.length < 30 || columns.length < 2) throw new Error("Need a CSV with a header row, at least 2 columns and at least 30 rows.");
      setUpload({ name: f.name, rows: r, columns, dtypes: guessDtypes(columns, r), truncated });
      changeSample("upload");
    } catch (e) { setUploadErr((e as Error).message); }
  };

  const marks = useMemo(() => {
    const hit = new Set<string>();
    data?.injection_log.filter((r) => r.action === "outlier").forEach((r) => r.row_indices.forEach((i) => hit.add(`${r.column}:${i}`)));
    return (col: string, i: number) => (hit.has(`${col}:${i}`) ? ("outlier" as const) : undefined);
  }, [data]);

  const runTstr = async () => {
    setTstrBusy(true); setTstrErr(null);
    try { setTstr(await post<TstrResponse>("/api/tabular/tstr", { ...dataBody, method, target, config: { rows, seed } })); }
    catch (e) { setTstrErr((e as Error).message); }
    finally { setTstrBusy(false); }
  };

  const [dlBusy, setDlBusy] = useState(false);
  const [saved, setSaved] = useState(false);
  const download = async () => {
    setDlBusy(true); setSaved(false);
    try { saveBlob(await postBlob("/api/export/tabular", { ...dataBody, method, config, rules: rulesOk, rule_mode: ruleMode }), "synthetic.csv"); setSaved(true); }
    catch (e) { setTstrErr((e as Error).message); }
    finally { setDlBusy(false); }
  };

  const cols = demo?.columns ?? [];
  return (
    <>
      <main className="center">
        <h1>Tabular <span className={`status ${busy ? "" : "live"}`}>{busy ? "updating…" : `live · seed ${data?.seed ?? seed}`}</span></h1>
        <p className="lede">{sample === "upload" && upload ? <>Synthetic version of <b>{upload.name}</b> ({upload.rows.length.toLocaleString()} rows{upload.truncated ? `, first ${MAX_UPLOAD_ROWS.toLocaleString()} used` : ""}).</>
          : <>Synthetic version of the <b>{samples?.find((s) => s.name === sample)?.title ?? "sample"}</b> table.</>} Change anything on the right; the preview regenerates.</p>
        {sample === "upload" && !upload && <div className="state"><b>Choose a CSV file</b><p>Use “Data” on the right to pick a sample table or upload your own.</p></div>}
        <Tabs tabs={TABS} value={tab} onChange={setTab} />
        {!data && <PreviewFrame loading={busy} error={error} hasData={false} onRetry={() => setRetry((r) => r + 1)} skeleton={<Skeleton rows={8} cols={7} />}><></></PreviewFrame>}
        {data && error && <div className="err" role="alert">{error} <button className="btn" onClick={() => setRetry((r) => r + 1)}>Retry</button></div>}
        {data && tab === "Distributions" && <ChartGrid overlay={data.overlay} />}
        {data && tab === "Rules" && (data.rules ? (
          <>
            <div className="cards">
              <div className="card"><div className="big">{Math.round(data.rules.pass_rate_before)}%</div><div className="label">rows passing, naive sampling</div></div>
              <div className="card"><div className="big" style={{ color: data.rules.violations_after ? "var(--bad)" : "var(--ok)" }}>{Math.round(data.rules.reconciliation_pass_rate)}%</div><div className="label">rows passing, enforced ({data.rules.mode})</div></div>
              <div className="card"><div className="big">{data.rules.rows_dropped}</div><div className="label">rows re-drawn</div></div>
            </div>
            <div className="tablewrap"><table><thead><tr><th>rule</th><th>naive violations</th><th>enforced</th><th>how</th></tr></thead>
              <tbody>{data.rules.rules.map((r) => <tr key={r.name}><td><code>{r.dsl}</code></td><td>{data.rules!.naive_violations[r.name] ?? r.violations_before}</td><td>{r.violations_after}</td><td>{r.strategy || "already held"}</td></tr>)}</tbody></table></div>
          </>
        ) : <div className="state"><b>No rules yet</b><p>Add rules in the panel on the right, in plain English or DSL.</p></div>)}
        {data && tab === "Preview" && (
          <>
            <p className="lede">{data.n_rows.toLocaleString()} rows · first {data.preview.length} shown · highlighted cells are injected outliers, <i>null</i> cells are injected nulls.</p>
            <DataTable columns={data.columns} rows={data.preview} marks={marks} rowMark={(i) => data.edge_tags?.[i] ?? undefined} />
          </>
        )}
        {data && tab === "Edge cases" && (data.edge_cases ? (
          <>
            <div className="cards">
              <div className="card"><div className="big">{data.edge_cases.coverage_pct == null ? "–" : `${Math.round(data.edge_cases.coverage_pct)}%`}</div><div className="label">edge-case coverage ({data.edge_cases.cases_exercised} of {data.edge_cases.cases_possible} applicable cases)</div></div>
              <div className="card"><div className="big">{data.edge_cases.rows_tagged}</div><div className="label">rows tagged in the hidden _edge_case column</div></div>
            </div>
            <div className="tablewrap"><table><thead><tr><th>pack</th><th>rate</th><th>rows</th><th>coverage</th><th>cases exercised</th></tr></thead>
              <tbody>{Object.entries(data.edge_cases.packs).map(([k, v]) => (
                <tr key={k}><td>{k}</td><td>{pct(v.rate * 100)}</td><td>{v.rows_affected}{v.rows_added ? ` (+${v.rows_added} added)` : ""}</td>
                  <td>{v.coverage_pct == null ? <span className="muted">not applicable to these columns</span> : pct(v.coverage_pct, 0)}</td>
                  <td className="muted">{v.cases_exercised.join(", ") || "–"}</td></tr>))}</tbody></table></div>
            <p className="muted">Highlighted rows in the Preview tab carry a tag (hover to read it). Exports leave the tag column out unless you ask for it.</p>
          </>
        ) : <div className="state"><b>No scenario packs selected</b><p>Set a rate for a pack in the panel on the right.</p></div>)}
        {data && tab === "Quality" && <Quality f={data.fidelity} />}
        {data && tab === "Injection log" && (
          data.injection_log.length === 0 ? <div className="empty">No nulls or outliers requested.</div> : (
            <div className="tablewrap"><table>
              <thead><tr><th>column</th><th>action</th><th>method</th><th>rate</th><th>rows affected</th><th>first rows</th></tr></thead>
              <tbody>{data.injection_log.map((r, i) => (
                <tr key={i}><td>{r.column}</td><td>{r.action}</td><td>{r.method ?? "–"}</td><td>{pct(r.requested_rate * 100)}</td><td>{r.count}</td><td>{r.row_indices.slice(0, 8).join(", ")}…</td></tr>
              ))}</tbody>
            </table></div>
          )
        )}
        {tab === "Visualize" && <VisualizePanel datasetId={sample === "upload" ? null : sample} kind="tabular"
          emptyNote="Charts use the built-in tables and saved runs. Set “Data” on the right to a built-in table to see them for your table's sample." />}
        {tab === "Utility (TSTR)" && (
          <>
            <p className="lede">Train on synthetic, test on real, versus the real-trained baseline. Target: <b>{target}</b>. Positive gap = synthetic is worse.</p>
            {demo && <label className="field" style={{ maxWidth: 320 }}><span>Column to predict</span>
              <select value={target} onChange={(e) => { setTargetPick(e.target.value); setTstr(null); }}>{demo.columns.map((c) => <option key={c}>{c}</option>)}</select></label>}
            <button className="btn primary" onClick={runTstr} disabled={tstrBusy}>{tstrBusy ? "Training models…" : "Run evaluation"}</button>
            {tstrErr && <div className="err">{tstrErr}</div>}
            {tstr && <Tstr r={tstr} />}
          </>
        )}
      </main>
      <aside className="config">
        <h2>Settings</h2>
        <Field label="Data" hint="Pick a built-in fictional table, or upload your own CSV.">
          <select value={sample} onChange={(e) => (e.target.value === "upload" && !upload ? document.getElementById("tab-upload")?.click() : changeSample(e.target.value))}>
            {(samples ?? [{ name: "customers", title: "Customers", target: "" }]).map((s) => <option key={s.name} value={s.name}>{s.title}</option>)}
            <option value="upload">{upload ? `Your file: ${upload.name}` : "Upload your own CSV…"}</option>
          </select>
        </Field>
        <input id="tab-upload" type="file" accept=".csv,text/csv" className="sr-only" tabIndex={-1} onChange={(e) => { void onFile(e.target.files?.[0]); e.target.value = ""; }} />
        <button className="btn" style={{ width: "100%", marginBottom: 12 }} onClick={() => document.getElementById("tab-upload")?.click()}>{upload ? "Replace uploaded CSV" : "Upload a CSV"}</button>
        {uploadErr && <div className="err" role="alert">{uploadErr}</div>}
        <Field label="How many rows?" hint="How many fake records to create."><input type="number" min={1} max={20000} value={rows} onChange={(e) => setRows(Math.max(1, +e.target.value || 1))} /></Field>
        <button className="btn primary" onClick={download} disabled={dlBusy}>{dlBusy ? "Preparing…" : "Download this data (CSV)"}</button>
        <p className="muted" role="status" aria-live="polite">{saved ? "Downloaded and saved to History." : "Downloads are saved to History."}</p>
        <Panel title="Rules" badge={rulesOk.length} defaultOpen>
          <p className="muted">Plain words, like "age must be at least 25". Rows that break a rule are fixed or replaced.</p>
          <RulesEditor key={dataKey} value={ruleText} onChange={setRuleText} dataset="customers" sample={sample === "upload" ? undefined : sample}
            data={sample === "upload" ? upload?.rows : undefined} onValid={setRulesOk} examples={RULE_EXAMPLES[sample] ?? []} />
          <Field label="If a row breaks a rule"><select value={ruleMode} onChange={(e) => setRuleMode(e.target.value as typeof ruleMode)}>
            <option value="hybrid">Fix it, or replace it (recommended)</option><option value="repair">Only fix it</option><option value="reject">Throw it away and make a new one</option></select></Field>
        </Panel>
        <Panel title="Messy values" badge={Object.values(nulls).filter((v) => v > 0).length + Object.values(outliers).filter((v) => v > 0).length}>
          <p className="muted">Optional. "Missing" leaves a value empty; "odd" pushes a number far outside the normal range. Marked in the preview.</p>
          <div className="colrow head"><span>column</span><span title="Percent of values left empty">missing %</span><span title="Percent of numbers made unusually large or small">odd %</span></div>
          {cols.map((c) => (
            <div className="colrow" key={c}>
              <span title={demo?.dtypes[c]}>{c}</span>
              <input aria-label={`${c} null rate`} type="number" min={0} max={100} step={1} value={Math.round((nulls[c] ?? 0) * 100)}
                onChange={(e) => setNulls({ ...nulls, [c]: Math.min(1, Math.max(0, +e.target.value / 100)) })} />
              <input aria-label={`${c} outlier rate`} type="number" min={0} max={100} step={1} value={Math.round((outliers[c] ?? 0) * 100)}
                disabled={!isNumeric(demo?.dtypes[c] ?? "") || demo?.dtypes[c] === "bool"}
                onChange={(e) => setOutliers({ ...outliers, [c]: Math.min(1, Math.max(0, +e.target.value / 100)) })} />
            </div>
          ))}
        </Panel>
        <Panel title="Tricky test cases" badge={Object.values(edge).filter((v) => v > 0).length}>
          <p className="muted">Optionally mix in awkward rows (% of rows) to test how other systems cope.</p>
          {EDGE_PACKS.map(([k, label]) => (
            <div className="colrow" style={{ gridTemplateColumns: "1fr 64px" }} key={k}>
              <span title={label}>{label}</span>
              <input aria-label={`${k} rate`} type="number" min={0} max={100} step={1} value={Math.round((edge[k] ?? 0) * 100)} onChange={(e) => setEdge({ ...edge, [k]: Math.min(1, Math.max(0, +e.target.value / 100)) })} />
            </div>
          ))}
        </Panel>
        <Panel title="Advanced">
        <Field label="Repeat code" hint="Same number gives exactly the same data again. Change it for a fresh set."><input type="number" value={seed} onChange={(e) => setSeed(+e.target.value || 0)} /></Field>
        <Field label="How should the data be made?" hint="If unsure, keep the first choice. It is fast and works well for most tables.">
          <select value={method} onChange={(e) => setMethod(e.target.value as typeof method)}>
            <option value="gaussian_copula">Standard: fast, good for most tables (recommended)</option>
            <option value="ctgan">Advanced AI model: slower, needs an extra install</option>
          </select>
        </Field>
        <Field label="How odd should odd values be?" hint="Only used if you ask for odd values (outliers) below. They are numbers far away from the normal range, like an age of 190.">
          <select value={outlierMethod} onChange={(e) => setOutlierMethod(e.target.value as typeof outlierMethod)}>
            <option value="iqr">Clearly odd (statistical rule, recommended)</option>
            <option value="zscore">Very odd (far from the average)</option>
            <option value="scale">Extreme (5 to 10 times the normal value)</option>
          </select>
        </Field>
        </Panel>
      </aside>
    </>
  );
}

function Quality({ f }: { f: TabularResponse["fidelity"] }) {
  const ok = (p: number) => <span className={`badge ${p > 0.01 ? "ok" : "bad"}`}>{p > 0.01 ? "match" : "differs"}</span>;
  return (
    <>
      <div className="cards">
        <div className="card"><div className="big">{f.mean_ks_statistic?.toFixed(3) ?? "–"}</div><div className="label">mean KS statistic (lower is better)</div></div>
        <div className="card"><div className="big">{f.correlation_gap.toFixed(3)}</div><div className="label">mean |Δ Spearman correlation|</div></div>
        <div className="card"><div className="big">{f.dtypes_match ? "Yes" : "No"}</div><div className="label">dtypes preserved</div></div>
      </div>
      <h2>Numeric marginals (KS test)</h2>
      <div className="tablewrap"><table><thead><tr><th>column</th><th>statistic</th><th>p-value</th><th></th></tr></thead>
        <tbody>{Object.entries(f.ks).map(([c, v]) => <tr key={c}><td>{c}</td><td>{v.statistic.toFixed(3)}</td><td>{v.p_value.toFixed(3)}</td><td>{ok(v.p_value)}</td></tr>)}</tbody></table></div>
      <h3>Category frequencies (chi-square)</h3>
      <div className="tablewrap"><table><thead><tr><th>column</th><th>χ²</th><th>p-value</th><th>max freq diff</th><th></th></tr></thead>
        <tbody>{Object.entries(f.chi_square).map(([c, v]) => <tr key={c}><td>{c}</td><td>{v.statistic.toFixed(2)}</td><td>{v.p_value.toFixed(3)}</td><td>{pct(v.max_abs_freq_diff * 100, 2)}</td><td>{ok(v.p_value)}</td></tr>)}</tbody></table></div>
    </>
  );
}

function Tstr({ r }: { r: TstrResponse }) {
  return (
    <>
      <div className="cards" style={{ marginTop: 24 }}>
        <div className="card"><div className="big">{pct(r.summary.mean_gap_pct)}</div><div className="label">mean {r.summary.primary_metric} gap vs real baseline</div></div>
        <div className="card"><div className="big">{r.task}</div><div className="label">auto-detected task</div></div>
      </div>
      <div className="tablewrap"><table>
        <thead><tr><th>model</th><th>metric</th><th>real→real</th><th>synthetic→real</th><th>gap</th></tr></thead>
        <tbody>{Object.entries(r.models).flatMap(([m, v]) => Object.keys(v.baseline_real).map((k) => (
          <tr key={m + k}><td>{m}</td><td>{k}</td><td>{v.baseline_real[k]?.toFixed(3) ?? "–"}</td><td>{v.synthetic[k]?.toFixed(3) ?? "–"}</td><td>{pct(v.gap_pct[k])}</td></tr>
        )))}</tbody>
      </table></div>
    </>
  );
}
