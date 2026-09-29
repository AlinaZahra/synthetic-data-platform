import { useEffect, useRef, useState } from "react";
import { get, post, saveBlob } from "./api";
import { Advanced, Field, Panel, Tabs, Term, useLive } from "./ui";

const TABS = ["Export", "Database", "Large run"] as const;
const SOURCES = [
  { label: "Banking pack", src: { kind: "pack", name: "banking" } },
  { label: "E-commerce pack", src: { kind: "pack", name: "ecommerce" } },
  { label: "Healthcare pack", src: { kind: "pack", name: "healthcare" } },
  { label: "Demo: shop (relational)", src: { kind: "demo", name: "shop_full" } },
  { label: "Demo: customers (one table)", src: { kind: "demo", name: "customers" } },
] as const;
const FORMAT_HELP: Record<string, string> = {
  csv: "one file per table (zipped if several)", json: "all tables in one JSON document", jsonl: "one JSON object per line, single table", sql: "schema-preserving dump: types, keys, parents first",
  pdf: "readable report with schema and a preview of the rows", zip: "a bundle of the formats you choose",
};

function SourcePicker({ idx, setIdx, rows, setRows }: { idx: number; setIdx: (i: number) => void; rows: number; setRows: (n: number) => void }) {
  return (
    <div className="row2">
      <Field label="Data"><select value={idx} onChange={(e) => setIdx(+e.target.value)}>{SOURCES.map((s, i) => <option key={s.label} value={i}>{s.label}</option>)}</select></Field>
      <Field label="Rows (main table)"><input type="number" min={10} max={20000} value={rows} onChange={(e) => setRows(Math.min(20000, Math.max(10, +e.target.value || 10)))} /></Field>
    </div>
  );
}

function ExportPanel() {
  const formats = useLive(() => get<{ name: string; extension: string }[]>("/api/export/formats"), [], 0).data;
  const [idx, setIdx] = useState(0);
  const [rows, setRows] = useState(200);
  const [format, setFormat] = useState("sql");
  const [dialect, setDialect] = useState("postgres");
  const [bom, setBom] = useState(false);
  const [bundle, setBundle] = useState<string[]>(["csv", "json", "sql"]);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const download = async () => {
    setBusy(true); setErr(null);
    try {
      const options: Record<string, unknown> = { dialect, bom };
      if (format === "zip") options.formats = bundle;
      const res = await fetch("/api/export", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: { ...SOURCES[idx].src, rows, seed: 1 }, format, options }) });
      if (!res.ok) { const d = await res.json().catch(() => ({})); throw new Error(typeof d.detail === "string" ? d.detail : res.statusText); }
      const cd = res.headers.get("content-disposition") ?? "";
      saveBlob(await res.blob(), /filename="([^"]+)"/.exec(cd)?.[1] ?? `export.${format}`);
    } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  };
  return (
    <div className="stack">
      <p className="lede">Choose data and a format. SQL dumps keep column types, primary and foreign keys, and load parents before children.</p>
      <SourcePicker idx={idx} setIdx={setIdx} rows={rows} setRows={setRows} />
      <Field label="Format">
        <select value={format} onChange={(e) => setFormat(e.target.value)}>{(formats ?? []).map((f) => <option key={f.name} value={f.name}>{f.name.toUpperCase()}: {FORMAT_HELP[f.name] ?? f.extension}</option>)}</select>
      </Field>
      <Advanced>
        {format === "sql" && <Field label="SQL dialect"><select value={dialect} onChange={(e) => setDialect(e.target.value)}>{["postgres", "mysql", "sqlite", "ansi"].map((d) => <option key={d}>{d}</option>)}</select></Field>}
        {(format === "csv" || format === "json" || format === "jsonl") && <label><input type="checkbox" checked={bom} onChange={(e) => setBom(e.target.checked)} /> UTF-8 byte-order mark (needed for Excel with Urdu, Arabic or Chinese)</label>}
        {format === "zip" && <fieldset style={{ border: 0, padding: 0 }}><legend className="muted">Formats inside the ZIP</legend>
          {["csv", "json", "jsonl", "sql", "pdf"].map((f) => <label key={f} style={{ marginRight: 12 }}><input type="checkbox" checked={bundle.includes(f)} onChange={(e) => setBundle((b) => e.target.checked ? [...b, f] : b.filter((x) => x !== f))} /> {f}</label>)}</fieldset>}
      </Advanced>
      <button className="btn primary" onClick={download} disabled={busy}>{busy ? "Preparing…" : "Download"}</button>
      {err && <div className="err" role="alert">{err}</div>}
    </div>
  );
}

interface LoadResult { target: string; dialect: string; mode: string; dry_run: boolean; ok: boolean; committed: boolean; rolled_back: boolean; strategy: string;
  tables: Record<string, number>; verified: Record<string, number>; statements: string[]; errors: string[]; notes: string[]; seconds: number }

function DatabasePanel() {
  const status = useLive(() => get<{ ui_enabled: boolean; allowed_hosts: string[] }>("/api/connect/status"), [], 0).data;
  const [idx, setIdx] = useState(0);
  const [rows, setRows] = useState(100);
  const [url, setUrl] = useState("sqlite:///demo.db");
  const [mode, setMode] = useState("create");
  const [dry, setDry] = useState(true);
  const [res, setRes] = useState<LoadResult | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const run = async () => {
    setBusy(true); setErr(null); setRes(null);
    try { setRes(await post<LoadResult>("/api/connect/load", { url, mode, dry_run: dry, source: { ...SOURCES[idx].src, rows, seed: 1 } })); }
    catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  };
  return (
    <div className="stack">
      <p className="lede">Load straight into PostgreSQL, MySQL, MongoDB (or SQLite). A <Term k="Dry run">dry run</Term> does the whole load and checks row counts, then rolls it back.</p>
      {status && !status.ui_enabled && <div className="banner">Loading from the browser is switched off on this server. Start it with <code>SDP_CONNECTORS_UI=1</code>, or use <code>POST /connect/load</code> with an API key. Allowed hosts: {status.allowed_hosts.join(", ")}.</div>}
      <SourcePicker idx={idx} setIdx={setIdx} rows={rows} setRows={setRows} />
      <Field label="Connection URL"><input value={url} autoComplete="off" spellCheck={false} onChange={(e) => setUrl(e.target.value)} placeholder="postgresql://user:password@localhost:5432/db" /></Field>
      <Advanced>
        <Field label="If the table already exists"><select value={mode} onChange={(e) => setMode(e.target.value)}>
          <option value="create">create (stop if it exists)</option><option value="append">append rows</option><option value="replace">replace (drop and recreate)</option></select></Field>
      </Advanced>
      <label><input type="checkbox" checked={dry} onChange={(e) => setDry(e.target.checked)} /> Dry run (nothing is kept)</label>
      <div><button className="btn primary" onClick={run} disabled={busy || !url.trim()}>{busy ? "Working…" : dry ? "Run dry run" : "Load for real"}</button></div>
      {!dry && <p className="muted">This will commit rows to <code>{url.replace(/:\/\/([^:@/]+):[^@]*@/, "://$1:***@")}</code>.</p>}
      {err && <div className="err" role="alert">{err}</div>}
      {res && (
        <div className="card" role="status" aria-live="polite">
          <b className={res.ok ? "result-pass" : "result-fail"}>{res.ok ? (res.committed ? "Committed" : "Dry run passed, rolled back") : "Failed, nothing kept"}</b>
          <span className="muted"> · {res.dialect} · {res.strategy.replace("_", " ")} · {res.seconds}s</span>
          <table className="diff"><thead><tr><th>table</th><th>rows written</th><th>counted in database</th></tr></thead>
            <tbody>{Object.entries(res.tables).map(([t, n]) => <tr key={t}><td>{t}</td><td>{n.toLocaleString()}</td><td>{res.verified[t]?.toLocaleString() ?? "–"}</td></tr>)}</tbody></table>
          {res.errors.map((e) => <p key={e} className="err">{e}</p>)}
          {res.notes.map((n) => <p key={n} className="muted">{n}</p>)}
          {res.statements.length > 0 && <details><summary>First statements</summary><pre className="json">{res.statements.join("\n")}</pre></details>}
        </div>
      )}
    </div>
  );
}

interface Progress { status: string; phase: string | null; done: number; total: number | null; percent: number; elapsed_s: number | null; items_per_s: number | null; eta_s: number | null; cancel_requested: boolean; finished: boolean }
interface Manifest { dataset?: { throughput?: { rows: number; seconds: number; rows_per_second: number; mb_per_second: number; peak_rss_mb: number | null; chunks: number; bytes: number; executor: string; max_rows_in_memory: number; file: string } } }

function LargePanel() {
  const [rows, setRows] = useState(1_000_000);
  const [chunk, setChunk] = useState(100_000);
  const [workers, setWorkers] = useState(4);
  const [format, setFormat] = useState("csv");
  const [compress, setCompress] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [prog, setProg] = useState<Progress | null>(null);
  const [tp, setTp] = useState<NonNullable<Manifest["dataset"]>["throughput"] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (!jobId) return;
    let stop = false;
    const tick = async () => {
      try {
        const p = await get<Progress>(`/api/history/${jobId}/progress`);
        if (stop) return;
        setProg(p);
        if (p.finished) { try { setTp((await get<Manifest>(`/api/history/${jobId}`)).dataset?.throughput ?? null); } catch { /* no report */ } return; }
      } catch (e) { if (!stop) setErr((e as Error).message); return; }
      timer.current = setTimeout(tick, 700);
    };
    void tick();
    return () => { stop = true; if (timer.current) clearTimeout(timer.current); };
  }, [jobId]);

  const start = async () => {
    setErr(null); setTp(null); setProg(null);
    try { setJobId((await post<{ id: string }>("/api/history/generate", { kind: "large", params: { rows, chunk_rows: chunk, workers, format, compress, seed: 1 }, wait: false })).id); }
    catch (e) { setErr((e as Error).message); }
  };
  const running = !!prog && !prog.finished;
  return (
    <div className="stack">
      <p className="lede">Millions of rows without freezing the page or your memory: rows are produced in <Term k="Chunk">chunks</Term>, in parallel, and streamed to disk. The file is identical whatever the number of workers.</p>
      <div className="row2">
        <Field label="Rows (up to 100,000,000)"><input type="number" min={1000} max={100000000} step={100000} value={rows} disabled={running} onChange={(e) => setRows(Math.min(100_000_000, Math.max(1000, +e.target.value || 1000)))} /></Field>
        <Field label="Format"><select value={format} disabled={running} onChange={(e) => setFormat(e.target.value)}>{["csv", "jsonl", "json", "sql"].map((f) => <option key={f}>{f}</option>)}</select></Field>
      </div>
      <Advanced>
        <div className="row2">
          <Field label="Rows per chunk"><input type="number" min={100} max={1000000} step={10000} value={chunk} disabled={running} onChange={(e) => setChunk(Math.min(1_000_000, Math.max(100, +e.target.value || 100)))} /></Field>
          <Field label="Workers"><input type="number" min={1} max={16} value={workers} disabled={running} onChange={(e) => setWorkers(Math.min(16, Math.max(1, +e.target.value || 1)))} /></Field>
        </div>
        <label><input type="checkbox" checked={compress} disabled={running} onChange={(e) => setCompress(e.target.checked)} /> Compress (gzip)</label>
        <p className="muted">From one million rows the work moves to separate processes, which is about 2.4x faster on this machine.</p>
      </Advanced>
      <div>
        <button className="btn primary" onClick={start} disabled={running}>{running ? "Running…" : `Generate ${rows.toLocaleString()} rows`}</button>{" "}
        {running && <button className="btn" onClick={() => post(`/api/history/${jobId}/cancel`, {})} disabled={prog?.cancel_requested}>{prog?.cancel_requested ? "Cancelling…" : "Cancel"}</button>}
      </div>
      {err && <div className="err" role="alert">{err}</div>}
      {prog && (
        <div className="card" role="status" aria-live="polite">
          <div className="progress-head"><b>{prog.status === "succeeded" ? "Done" : prog.status === "cancelled" ? "Cancelled. The file holds what was written." : prog.status === "failed" ? "Failed" : prog.status === "queued" ? "Waiting in the queue…" : "Generating"}</b>
            <span className="muted">{prog.done.toLocaleString()} / {prog.total?.toLocaleString() ?? "?"} · {Math.round(prog.percent)}%{prog.items_per_s ? ` · ${Math.round(prog.items_per_s).toLocaleString()} rows/s` : ""}</span></div>
          <div className="progress" role="progressbar" aria-valuenow={Math.round(prog.percent)} aria-valuemin={0} aria-valuemax={100}><i style={{ width: `${prog.percent}%` }} className={prog.status === "cancelled" || prog.status === "failed" ? "stopped" : undefined} /></div>
          {tp && (
            <div className="cards" style={{ marginTop: 16 }}>
              <div className="card"><div className="big">{Math.round(tp.rows_per_second).toLocaleString()}</div><div className="label">rows per second</div></div>
              <div className="card"><div className="big">{tp.mb_per_second}</div><div className="label">MB per second ({(tp.bytes / 1e6).toFixed(0)} MB total)</div></div>
              <div className="card"><div className="big">{tp.peak_rss_mb ?? "n/a"}</div><div className="label">peak memory MB (window {tp.max_rows_in_memory.toLocaleString()} rows)</div></div>
              <div className="card"><div className="big">{tp.seconds}s</div><div className="label">{tp.chunks} chunks · {tp.executor}</div></div>
            </div>
          )}
          {prog.finished && prog.status !== "failed" && <a className="btn" href={`/api/history/${jobId}/files/${tp?.file ?? `synthetic.${format}`}`} download>Download file</a>}
        </div>
      )}
    </div>
  );
}

export function ExportWorkspace() {
  const [tab, setTab] = useState<(typeof TABS)[number]>("Export");
  return (
    <>
      <main className="center">
        <h1>Export & load</h1>
        <Tabs tabs={TABS} value={tab} onChange={setTab} />
        {tab === "Export" && <ExportPanel />}
        {tab === "Database" && <DatabasePanel />}
        {tab === "Large run" && <LargePanel />}
      </main>
      <aside className="config">
        <h2>Good to know</h2>
        <Panel title="Exports">
          <p className="muted">Hidden bookkeeping columns are never exported. SQL dumps are one transaction, with text safely escaped.</p>
        </Panel>
        <Panel title="Database loads">
          <p className="muted">Only hosts on the server's allowlist (localhost by default) can be reached, and passwords are never shown back.</p>
        </Panel>
      </aside>
    </>
  );
}
