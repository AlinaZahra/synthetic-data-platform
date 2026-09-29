import { useEffect, useState } from "react";
import { get, post } from "./api";
import { PreviewFrame, Skeleton } from "./Preview";
import { Field, Panel } from "./ui";

interface Job {
  id: string; kind: string; status: string; created_at: string; seed: number | null; score: number | null; label: string | null;
  version: number; root_id: string; parent_id: string | null; reproduced: boolean | null; code_changed: boolean | null;
  output_hash: string | null; files: string[]; error: string | null;
  progress?: { done: number; total: number | null; phase: string | null; failed: number } | null; cancel_requested?: boolean;
}
const live = (j: Job) => j.status === "queued" || j.status === "running" || j.status === "cancelling";
const EXAMPLES: Record<string, object> = {
  tabular: { rows: 500, seed: 1, rules: ["age must be at least 25"], edge_cases: { typos: 0.02 } },
  relational: { dataset: "shop_full", seed: 1, rules: ["orders.discount <= 0.3"] },
  nl: { text: "1,000 Pakistani bank customers, 3% fraud, 6 months of history" },
  document: { doc_type: "payslip", count: 2000, spec: { locale: "en-GB" } },
};

const repro = (j: Job) => j.reproduced === true ? <span className="badge ok">reproduced{j.code_changed ? " (code changed)" : ""}</span>
  : j.reproduced === false ? <span className="badge bad">output differs{j.code_changed ? " · code changed" : ""}</span> : <span className="badge">{j.parent_id ? "new version" : "original"}</span>;

export function HistoryWorkspace() {
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [kind, setKind] = useState("tabular");
  const [params, setParams] = useState(JSON.stringify(EXAMPLES.tabular, null, 2));
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState<string | null>(null);
  const [manifest, setManifest] = useState<unknown>(null);
  const [scores, setScores] = useState<Record<string, string>>({});
  const [tick, setTick] = useState(0);

  const load = async () => {
    try { setJobs(await get<Job[]>("/api/history")); setError(null); } catch (e) { setError((e as Error).message); }
  };
  useEffect(() => { void load(); }, [tick]);
  // poll while anything is still running
  useEffect(() => {
    if (!jobs?.some(live)) return;
    const t = setTimeout(() => setTick((x) => x + 1), 1000);
    return () => clearTimeout(t);
  }, [jobs]);

  const start = async () => {
    setBusy(true); setError(null);
    try {
      await post("/api/history/generate", { kind, params: JSON.parse(params), wait: false });
      setTick((x) => x + 1);
    } catch (e) { setError(e instanceof SyntaxError ? `Parameters are not valid JSON: ${e.message}` : (e as Error).message); }
    finally { setBusy(false); }
  };
  const rerun = async (id: string, overrides?: object) => {
    setError(null);
    try { await post(`/api/history/${id}/rerun`, { wait: false, overrides: overrides ?? {} }); setTick((x) => x + 1); } catch (e) { setError((e as Error).message); }
  };
  const showManifest = async (id: string) => {
    if (open === id) { setOpen(null); return; }
    setOpen(id); setManifest(null);
    setManifest(await get(`/api/history/${id}`));
  };
  const doScore = async (id: string) => {
    setScores((s) => ({ ...s, [id]: "scoring…" }));
    try { const r = await get<{ score: number; label: string; kind: string }>(`/api/history/${id}/score`); setScores((s) => ({ ...s, [id]: `${r.score.toFixed(0)} · ${r.label}` })); setTick((x) => x + 1); }
    catch (e) { setScores((s) => ({ ...s, [id]: (e as Error).message.slice(0, 60) })); }
  };

  return (
    <>
      <main className="center">
        <h1>History</h1>
        <p className="lede">Every run keeps its request, seed, schema, model version, output hashes and scores. Rerun any job from its manifest to prove it reproduces.</p>
        {error && <div className="err" role="alert">{error}</div>}
        <PreviewFrame loading={!jobs} error={null} hasData={!!jobs?.length} emptyTitle="No jobs yet" emptyText="Start one on the right, or press “Save to history” at the end of the guided flow."
          skeleton={<Skeleton rows={5} cols={6} />}>
          <div className="tablewrap"><table>
            <thead><tr><th>job</th><th>kind</th><th>version</th><th>status</th><th>seed</th><th>score</th><th>lineage</th><th></th></tr></thead>
            <tbody>{jobs?.map((j) => (
              <>
                <tr key={j.id}>
                  <td><code>{j.id}</code><div className="muted">{new Date(j.created_at).toLocaleString()}</div></td>
                  <td>{j.kind}</td><td>v{j.version}</td>
                  <td>
                    <span className={`badge ${j.status === "succeeded" ? "ok" : j.status === "failed" ? "bad" : ""}`}>{j.status}</span>
                    {live(j) && j.progress?.total ? (
                      <div style={{ marginTop: 4 }}><span className="progress mini" role="progressbar" aria-valuenow={Math.round(100 * j.progress.done / j.progress.total)}><i style={{ width: `${100 * j.progress.done / j.progress.total}%` }} /></span>
                        <span className="muted"> {j.progress.done.toLocaleString()}/{j.progress.total.toLocaleString()}</span></div>
                    ) : null}
                    {j.status === "cancelled" && j.progress?.total ? <div className="muted">{j.progress.done.toLocaleString()} of {j.progress.total.toLocaleString()} done</div> : null}
                    {j.error && <div className="muted" title={j.error}>{j.error.slice(0, 50)}</div>}
                  </td>
                  <td>{j.seed ?? "–"}</td>
                  <td>{scores[j.id] ?? (j.score != null ? `${j.score.toFixed(0)} · ${j.label}` : j.status === "succeeded" ? <button className="btn" onClick={() => doScore(j.id)}>Score</button> : "–")}</td>
                  <td>{repro(j)}</td>
                  <td style={{ whiteSpace: "nowrap" }}>
                    {live(j)
                      ? <button className="btn" disabled={!!j.cancel_requested} onClick={async () => { try { await post(`/api/history/${j.id}/cancel`, {}); setTick((x) => x + 1); } catch (e) { setError((e as Error).message); } }}>{j.cancel_requested ? "Cancelling…" : "Cancel"}</button>
                      : <button className="btn" onClick={() => rerun(j.id)} title="Replay the stored request and compare output hashes">Rerun from manifest</button>}{" "}
                    {(j.status === "succeeded" || j.status === "cancelled") && <a className="btn" href={`/api/history/${j.id}/download.zip`} download>ZIP</a>}{" "}
                    <button className="btn" onClick={() => showManifest(j.id)}>{open === j.id ? "Hide" : "Manifest"}</button>
                  </td>
                </tr>
                {open === j.id && (
                  <tr key={j.id + "m"}><td colSpan={8}>
                    <div className="muted">output hash <code>{j.output_hash?.slice(0, 16) ?? "–"}</code> · files: {j.files.join(", ") || "–"} · <a href={`/api/history/${j.id}/files/${j.files[0] ?? ""}`}>download first file</a></div>
                    <pre className="json">{manifest ? JSON.stringify(manifest, null, 2).slice(0, 6000) : "Loading…"}</pre>
                  </td></tr>
                )}
              </>
            ))}</tbody>
          </table></div>
        </PreviewFrame>
      </main>
      <aside className="config">
        <h2>New job</h2>
        <Field label="Kind"><select value={kind} onChange={(e) => { setKind(e.target.value); setParams(JSON.stringify(EXAMPLES[e.target.value], null, 2)); }}>
          <option value="tabular">Tabular</option><option value="relational">Relational</option><option value="nl">Described in words</option><option value="document">Documents</option></select></Field>
        <Field label="Parameters (JSON)"><textarea rows={10} value={params} onChange={(e) => setParams(e.target.value)} style={{ width: "100%", fontFamily: "ui-monospace, Consolas, monospace", fontSize: 12 }} /></Field>
        <button className="btn primary" onClick={start} disabled={busy}>{busy ? "Starting…" : "Run job"}</button>
        <Panel title="How reruns work">
          <p className="muted">A rerun with the same request and code must give identical output. If the code changed, the row says so instead of silently accepting a difference.</p>
        </Panel>
        <Panel title="API and command line">
          <p className="muted">The same jobs are available at <code>POST /generate</code>, <code>GET /jobs/&#123;id&#125;</code>, <code>GET /score/&#123;id&#125;</code> (API key) and via <code>python -m sdp.cli</code>.</p>
        </Panel>
      </aside>
    </>
  );
}
