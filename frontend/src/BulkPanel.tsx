import { useEffect, useRef, useState } from "react";
import { get, post, saveBlob } from "./api";
import { Field } from "./ui";

interface Progress {
  job_id: string; status: string; phase: string | null; done: number; total: number | null; percent: number; failed: number;
  elapsed_s: number | null; items_per_s: number | null; eta_s: number | null; cancel_requested: boolean; finished: boolean;
}
interface Report { total: number; succeeded: number; failed: number; skipped: number; cancelled?: boolean; by_stage: Record<string, number>;
  failures: { index: number; stage: string; error_type: string; message: string }[] }

const fmtTime = (s: number | null) => (s == null ? "–" : s < 90 ? `${Math.round(s)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`);

/** Runs a bulk generation as a background job: the page stays responsive, shows live progress, can cancel, and offers a ZIP. */
export function BulkPanel({ docType, spec, seed }: { docType: string; spec: Record<string, unknown>; seed: number }) {
  const [count, setCount] = useState(1000);
  const [jobId, setJobId] = useState<string | null>(null);
  const [prog, setProg] = useState<Progress | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // poll while a job is in flight; stop as soon as it reaches a terminal state
  useEffect(() => {
    if (!jobId) return;
    let stop = false;
    const tick = async () => {
      try {
        const p = await get<Progress>(`/api/history/${jobId}/progress`);
        if (stop) return;
        setProg(p);
        if (p.finished) {
          try { setReport(JSON.parse(await (await fetch(`/api/history/${jobId}/files/report.json`)).text())); } catch { setReport(null); }
          return;
        }
      } catch (e) { if (!stop) setErr((e as Error).message); return; }
      timer.current = setTimeout(tick, 600);
    };
    void tick();
    return () => { stop = true; if (timer.current) clearTimeout(timer.current); };
  }, [jobId]);

  const start = async () => {
    setStarting(true); setErr(null); setReport(null); setProg(null);
    try {
      const r = await post<{ id: string }>("/api/history/generate", { kind: "document", params: { doc_type: docType, count, seed, spec: { ...spec, seed: undefined } }, wait: false });
      setJobId(r.id);
    } catch (e) { setErr((e as Error).message); } finally { setStarting(false); }
  };
  const cancel = async () => {
    if (!jobId) return;
    try { await post(`/api/history/${jobId}/cancel`, {}); } catch (e) { setErr((e as Error).message); }
  };
  const zip = async () => {
    const res = await fetch(`/api/history/${jobId}/download.zip`);
    if (!res.ok) { setErr(await res.text()); return; }
    saveBlob(await res.blob(), `documents-${jobId}.zip`);
  };

  const running = !!prog && !prog.finished;
  const label = !prog ? "" : prog.status === "queued" ? "Waiting in the queue…" : prog.status === "cancelling" ? "Cancelling… finishing the current chunk"
    : prog.status === "cancelled" ? "Cancelled. What finished is kept." : prog.status === "failed" ? "Failed" : prog.status === "succeeded" ? "Done" : `Generating (${prog.phase ?? "working"})`;

  return (
    <div>
      <p className="lede">Runs in the background (validate → generate → render → post-validate → export) with retries and per-document isolation. You can keep working, cancel at any time, and download everything as a ZIP.</p>
      <div className="row2" style={{ maxWidth: 420 }}>
        <Field label="Number of documents (up to 20,000)"><input type="number" min={1} max={20000} value={count} disabled={running} onChange={(e) => setCount(Math.min(20000, Math.max(1, +e.target.value || 1)))} /></Field>
      </div>
      <button className="btn primary" onClick={start} disabled={starting || running}>{starting ? "Starting…" : `Start ${count.toLocaleString()} ${docType.replace("_", " ")}s`}</button>{" "}
      {running && <button className="btn" onClick={cancel} disabled={prog?.cancel_requested}>{prog?.cancel_requested ? "Cancelling…" : "Cancel"}</button>}
      {err && <div className="err" role="alert">{err}</div>}

      {prog && (
        <div className="card" style={{ marginTop: 24 }} role="status" aria-live="polite">
          <div className="progress-head"><b>{label}</b><span className="muted">{prog.done.toLocaleString()} / {prog.total?.toLocaleString() ?? "?"} · {Math.round(prog.percent)}%</span></div>
          <div className="progress" role="progressbar" aria-valuenow={Math.round(prog.percent)} aria-valuemin={0} aria-valuemax={100}>
            <i style={{ width: `${prog.percent}%` }} className={prog.status === "cancelled" || prog.status === "failed" ? "stopped" : undefined} />
          </div>
          <div className="muted" style={{ marginTop: 8 }}>
            {prog.failed > 0 && <span style={{ color: "var(--bad)" }}>{prog.failed} failed · </span>}
            {prog.items_per_s ? `${prog.items_per_s.toFixed(1)} docs/s · ` : ""}elapsed {fmtTime(prog.elapsed_s)}{running && prog.eta_s != null ? ` · about ${fmtTime(prog.eta_s)} left` : ""}
          </div>
        </div>
      )}

      {prog?.finished && (
        <>
          {report && (
            <div className="cards" style={{ marginTop: 16 }}>
              <div className="card"><div className="big" style={{ color: "var(--ok)" }}>{report.succeeded.toLocaleString()}</div><div className="label">documents produced</div></div>
              <div className="card"><div className="big" style={{ color: report.failed ? "var(--bad)" : undefined }}>{report.failed}</div><div className="label">failed (isolated)</div></div>
              {report.cancelled && <div className="card"><div className="big">{(report.total - report.succeeded - report.failed - report.skipped).toLocaleString()}</div><div className="label">not started (cancelled)</div></div>}
            </div>
          )}
          {prog.status !== "failed" && <button className="btn primary" onClick={zip}>Download ZIP</button>}
          {report && report.failures.length > 0 && (
            <div className="tablewrap" style={{ marginTop: 16 }}><table><thead><tr><th>#</th><th>stage</th><th>error</th><th>message</th></tr></thead>
              <tbody>{report.failures.slice(0, 50).map((f) => <tr key={f.index}><td>{f.index}</td><td>{f.stage}</td><td>{f.error_type}</td><td>{f.message}</td></tr>)}</tbody></table></div>
          )}
          {prog.status === "failed" && <p className="muted">Open <b>History</b> for the error message and manifest.</p>}
        </>
      )}
    </div>
  );
}
