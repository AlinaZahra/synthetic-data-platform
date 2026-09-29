import { postBlob, saveBlob, type TrustReport } from "./api";

export const tone = (s: number) => (s >= 85 ? "var(--ok)" : s >= 70 ? "var(--accent)" : "var(--bad)");

/** Visual layer only: strong = teal, middling = blue, weak = red. The numbers and wording come straight from the report. */
const meterTone = (s: number) => (s >= 85 ? "var(--trust-strong)" : s >= 70 ? "var(--accent)" : "var(--bad)");

/** A thin horizontal indicator. The exact value is printed next to it by the caller. */
export function Meter({ value, label }: { value: number; label: string }) {
  const v = Math.max(0, Math.min(100, value));
  return (
    <span className="meter" role="progressbar" aria-label={label} aria-valuenow={Math.round(v)} aria-valuemin={0} aria-valuemax={100}>
      <i style={{ width: `${v}%`, background: meterTone(value) }} />
    </span>
  );
}

/** A subtle ring around the headline score. */
export function Ring({ score, children }: { score: number; children: React.ReactNode }) {
  const r = 52, c = 2 * Math.PI * r, v = Math.max(0, Math.min(100, score));
  return (
    <div className="ring" aria-hidden={false}>
      <svg viewBox="0 0 120 120" width="132" height="132" aria-hidden="true">
        <circle cx="60" cy="60" r={r} fill="none" stroke="var(--subtle)" strokeWidth="6" />
        <circle cx="60" cy="60" r={r} fill="none" stroke={meterTone(score)} strokeWidth="6" strokeLinecap="round"
          strokeDasharray={`${(v / 100) * c} ${c}`} transform="rotate(-90 60 60)" />
      </svg>
      <div className="ring-inner">{children}</div>
    </div>
  );
}

export interface Quality {
  score: number; label: string; note: string;
  components: { key: string; label: string; score: number; weight: number; summary: string }[];
}

/** Quality score for a dataset described in words (no real data exists, so no fidelity or privacy). Same look as the Trust Score card. */
export function QualityCard({ quality }: { quality: Quality }) {
  return (
    <article className="card trust" aria-label="Quality score">
      <header className="trust-head">
        <Ring score={quality.score}>
          <div className="trust-num" style={{ color: meterTone(quality.score) }}>{quality.score.toFixed(0)}<small>/100</small></div>
        </Ring>
        <div>
          <div className="trust-label">{quality.label}</div>
          <p className="trust-verdict">Quality score for this described dataset.</p>
        </div>
      </header>
      {quality.components.map((c) => (
        <div className="sub" key={c.key}>
          <div className="sub-row">
            <span className="sub-name">{c.label}</span>
            {c.weight > 0 ? <span className="badge">weight {Math.round(c.weight * 100)}%</span> : <span className="badge">info</span>}
            <b className="sub-val">{c.score.toFixed(0)}</b>
          </div>
          <Meter value={c.score} label={`${c.label} score`} />
          <p className="muted" style={{ margin: "6px 0 0" }}>{c.summary}</p>
        </div>
      ))}
      <p className="muted" style={{ marginTop: 16 }}>{quality.note}</p>
    </article>
  );
}

/** One card: headline number, verdict, three weighted sub-scores that expand to their reasons, exports. */
export function TrustCard({ report }: { report: TrustReport }) {
  const exportJson = () => saveBlob(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }), "trust-score.json");
  const exportPdf = async () => saveBlob(await postBlob("/api/trust/pdf", { report }), "trust-score.pdf");
  const cols = Object.entries(report.details.fidelity_columns ?? {}).filter(([, v]) => v.score != null) as [string, { score: number; test: string; statistic?: number }][];
  return (
    <article className="card trust">
      <header className="trust-head">
        <Ring score={report.trust_score}>
          <div className="trust-num" style={{ color: meterTone(report.trust_score) }}>{report.trust_score.toFixed(0)}<small>/100</small></div>
        </Ring>
        <div>
          <div className="trust-label">{report.label}</div>
          <p className="trust-verdict">{report.verdict}</p>
        </div>
      </header>
      {report.sub_scores.map((s) => (
        <details key={s.key} className="sub">
          <summary>
            <span className="sub-row">
              <span className="sub-name">{s.label}</span>
              <span className="badge">weight {Math.round(s.weight * 100)}%</span>
              <b className="sub-val">{s.score.toFixed(0)}</b>
            </span>
            <Meter value={s.score} label={`${s.label} score`} />
          </summary>
          <div className="sub-body">
            {s.components.map((c) => (
              <div className="comp" key={c.key}>
                <span>{c.label}</span><span className="muted">{c.summary}</span><span className="comp-val">{c.score.toFixed(0)}</span>
                <Meter value={c.score} label={`${c.label} score`} />
              </div>
            ))}
            {s.key === "fidelity" && cols.length > 0 && (
              <details className="cols"><summary>Per-column fidelity</summary>
                <div className="tablewrap"><table><thead><tr><th>column</th><th>test</th><th>statistic</th><th>score</th></tr></thead>
                  <tbody>{[...cols].sort((a, b) => a[1].score - b[1].score).map(([c, v]) => (
                    <tr key={c}><td>{c}</td><td>{v.test}</td><td>{v.statistic?.toFixed(3) ?? "–"}</td><td>{v.score.toFixed(0)}</td></tr>
                  ))}</tbody></table></div>
              </details>
            )}
          </div>
        </details>
      ))}
      {report.gates.length > 0 && <div className="gates"><b>Review before sharing</b><ul>{report.gates.map((g) => <li key={g}>{g}</li>)}</ul></div>}
      <footer className="trust-foot">
        <span className="muted">{report.title} · {new Date(report.generated_at).toLocaleString()}</span>
        <span><button className="btn" onClick={exportJson}>Export JSON</button> <button className="btn" onClick={exportPdf}>Export PDF</button></span>
      </footer>
    </article>
  );
}
