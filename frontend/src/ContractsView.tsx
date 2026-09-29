import { useState } from "react";
import { get, post, saveBlob, type Row } from "./api";
import { DataTable, Field, Panel, Tabs, Term, useLive } from "./ui";

interface Pack { name: string; title: string; description: string; default_locale: string; default_rows: number; tables: string[]; expectations: number }
interface Result {
  expectation_config: { expectation_type: string; table: string; kwargs: Record<string, unknown>; meta?: { description?: string } };
  success: boolean;
  result: { element_count?: number; unexpected_count?: number; unexpected_percent?: number; partial_unexpected_list?: unknown[]; observed_value?: unknown };
  exception_info: { exception_message: string } | null;
}
interface Report { success: boolean; suite_name: string; statistics: { evaluated_expectations: number; successful_expectations: number; unsuccessful_expectations: number; success_percent: number; by_table: Record<string, { evaluated: number; successful: number }> }; results: Result[] }
interface Run { pack: string; rows: Record<string, number>; report: Report; preview: Record<string, Row[]> }

const short = (t: string) => t.replace("expect_", "").replace(/_/g, " ");

export function ContractsWorkspace() {
  const packs = useLive(() => get<Pack[]>("/api/contracts/packs"), [], 0);
  const [name, setName] = useState("banking");
  const [rows, setRows] = useState(300);
  const [seed, setSeed] = useState(1);
  const [run, setRun] = useState<Run | null>(null);
  const [tab, setTab] = useState<string>("");
  const [onlyFailed, setOnlyFailed] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const pack = packs.data?.find((p) => p.name === name);

  const go = async () => {
    setBusy(true); setErr(null);
    try { const r = await post<Run>("/api/contracts/run", { pack: name, rows, seed }); setRun(r); setTab(Object.keys(r.preview)[0]); }
    catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  };
  const exportGe = async () => {
    const j = await get<Record<string, unknown>>(`/api/contracts/${name}/great-expectations`);
    saveBlob(new Blob([JSON.stringify(j, null, 2)], { type: "application/json" }), `${name}-expectations.json`);
  };
  const st = run?.report.statistics;
  const shown = run?.report.results.filter((r) => !onlyFailed || !r.success) ?? [];
  return (
    <>
      <main className="center">
        <h1>Contracts</h1>
        <p className="lede">Starter packs ship with a <Term k="Data contract">data contract</Term>. Generated data is checked against it and you get a report, in the style of <Term k="Great Expectations">Great Expectations</Term>.</p>
        <div className="cards">
          {packs.data?.map((p) => (
            <button key={p.name} className="card" style={{ textAlign: "left", cursor: "pointer", borderColor: p.name === name ? "var(--accent)" : undefined }} aria-pressed={p.name === name} onClick={() => setName(p.name)}>
              <div className="big" style={{ fontSize: 20 }}>{p.title}</div>
              <div className="label">{p.description}</div>
              <div className="label">{p.tables.join(", ")} · {p.expectations} checks</div>
            </button>
          ))}
        </div>
        <div className="row2" style={{ maxWidth: 520 }}>
          <Field label="Rows"><input type="number" min={10} max={20000} value={rows} onChange={(e) => setRows(Math.min(20000, Math.max(10, +e.target.value || 10)))} /></Field>
          <Field label={<Term k="Seed" />}><input type="number" value={seed} onChange={(e) => setSeed(+e.target.value || 0)} /></Field>
        </div>
        <button className="btn primary" onClick={go} disabled={busy}>{busy ? "Generating and checking…" : `Generate ${pack?.title ?? ""} data and validate`}</button>{" "}
        <button className="btn" onClick={exportGe}>Export as Great Expectations suites</button>
        {err && <div className="err" role="alert">{err}</div>}

        {run && st && (
          <section style={{ marginTop: 24 }} aria-live="polite">
            <div className="cards">
              <div className="card"><div className="big" style={{ color: run.report.success ? "var(--ok)" : "var(--bad)" }}>{run.report.success ? "PASSED" : "FAILED"}</div><div className="label">{st.successful_expectations} of {st.evaluated_expectations} checks met ({st.success_percent}%)</div></div>
              {Object.entries(st.by_table).map(([t, s]) => <div className="card" key={t}><div className="big">{s.successful}/{s.evaluated}</div><div className="label">{t} · {(run.rows[t] ?? 0).toLocaleString()} rows</div></div>)}
            </div>
            <label><input type="checkbox" checked={onlyFailed} onChange={(e) => setOnlyFailed(e.target.checked)} /> Show only failed checks</label>
            <div className="tablewrap" style={{ marginTop: 8 }}><table>
              <thead><tr><th>result</th><th>table</th><th>check</th><th>column / detail</th><th>unexpected</th></tr></thead>
              <tbody>{shown.map((r, i) => {
                const c = r.expectation_config, k = c.kwargs;
                const col = String(k.column ?? (k.column_A ? `${k.column_A} vs ${k.column_B}` : "") );
                return (
                  <tr key={i}>
                    <td className={r.success ? "result-pass" : "result-fail"}>{r.success ? "PASS" : "FAIL"}</td><td>{c.table}</td>
                    <td title={c.meta?.description}>{short(c.expectation_type)}</td>
                    <td className="mono">{r.exception_info?.exception_message ?? col}</td>
                    <td>{r.result.unexpected_count != null ? `${r.result.unexpected_count} (${r.result.unexpected_percent}%)` : r.result.observed_value != null ? String(r.result.observed_value) : ""}
                      {r.result.partial_unexpected_list?.length ? <span className="muted"> e.g. {r.result.partial_unexpected_list.slice(0, 3).join(", ")}</span> : null}</td>
                  </tr>
                );
              })}</tbody></table></div>
            <h2 style={{ marginTop: 24 }}>Data preview</h2>
            <Tabs tabs={Object.keys(run.preview) as readonly string[]} value={tab} onChange={setTab} />
            {tab && run.preview[tab] && <DataTable columns={Object.keys(run.preview[tab][0] ?? {})} rows={run.preview[tab]} />}
          </section>
        )}
      </main>
      <aside className="config">
        <h2>About contracts</h2>
        <Panel title="What is checked" defaultOpen>
          <p className="muted">Not empty, unique keys, value ranges and sets, patterns, column types, row counts, dates in order, <Term k="Foreign key">foreign keys</Term> that exist, and cross-table totals.</p>
        </Panel>
        <Panel title="Use it on your own data">
          <p className="muted">Send tables and a pack (or your own expectations) to <code>POST /api/contracts/validate</code>. Add a pack by dropping a JSON file into <code>sdp/contracts/packs</code> (or <code>SDP_PACK_DIR</code>).</p>
        </Panel>
      </aside>
    </>
  );
}
