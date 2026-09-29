import { useState } from "react";
import { get, post, type NLResult, type ParseResponse, type LocaleInfo, type Row } from "./api";
import { localeName } from "./locales";
import { QualityCard } from "./TrustCard";
import { VisualizePanel } from "./viz/LazyVisualize";
import { DataTable, Field, Panel, Tabs, Term, useLive } from "./ui";

interface Reply { lang: string; rtl: boolean; headline: string; lines: string[]; errors: string[]; warnings: string[] }
interface MLResponse { language: string; canonical_text: string; mapped: string[]; result: ParseResponse; reply: Reply }
interface ChatOut {
  applied: number; message: string; questions: string[]; source: string; ops: Record<string, unknown>[]; state: { config: Record<string, unknown>; ops: Record<string, unknown>[] };
  before: Record<string, unknown>; after: Record<string, unknown>; diff: { path: string; before: unknown; after: unknown }[];
  regenerated: Record<string, { rows_before: number; rows_after: number; columns_changed: string[]; untouched: boolean }>;
  preview: { table: string; rows: Row[] } | null;
}

const EXAMPLES = [
  "5,000 Pakistani bank customers, 3% fraud, 12 months of history",
  "2k Indian bank customers, 1.5% fraud, 6 months of history",
  "800 French e-commerce customers, 0.5% chargebacks",
  "Quiero 3.000 clientes bancarios pakistaníes, 3% de fraude y 6 meses de historial",
  "生成5000名中国银行客户，百分之3欺诈，12个月的历史",
  "٥٠٠٠ عملاء البنك سعوديين، ٣٪ احتيال",
];
const CHAT_EXAMPLES = ["double customers from Lahore", "fraud 8%", "add more outliers to balance", "halve the customers", "3 months of history"];

export function AssistantWorkspace() {
  const [text, setText] = useState(EXAMPLES[0]);
  const [seed, setSeed] = useState(0);
  const [parsed, setParsed] = useState<ParseResponse | null>(null);
  const [result, setResult] = useState<NLResult | null>(null);
  const [busy, setBusy] = useState<"" | "parse" | "generate">("");
  const [error, setError] = useState<string | null>(null);
  const [table, setTable] = useState<string | null>(null);
  const locales = useLive(() => get<LocaleInfo[]>("/api/locales"), [], 0).data;
  const [ml, setMl] = useState<MLResponse | null>(null);
  const [showQuality, setShowQuality] = useState(false);
  const [resTab, setResTab] = useState<"Preview" | "Visualize">("Preview");
  const [chatText, setChatText] = useState("");
  const [chatState, setChatState] = useState<ChatOut["state"] | null>(null);
  const [chatLog, setChatLog] = useState<{ text: string; out: ChatOut }[]>([]);
  const [chatBusy, setChatBusy] = useState(false);
  const [chatPreview, setChatPreview] = useState<ChatOut["preview"]>(null);
  const [useLlm, setUseLlm] = useState(false);
  const llm = useLive(() => get<{ available: boolean }>("/api/llm/status"), [], 0).data;

  const refine = async (t: string) => {
    if (!chatState || !t.trim()) return;
    setChatBusy(true); setError(null);
    try {
      const out = await post<ChatOut>("/api/chat/edit", { state: chatState, text: t, use_llm: useLlm });
      setChatLog((l) => [...l, { text: t, out }]);
      if (out.applied) { setChatState(out.state); setChatPreview(out.preview); }
      setChatText("");
    } catch (e) { setError((e as Error).message); }
    finally { setChatBusy(false); }
  };

  const parse = async () => {
    setBusy("parse"); setError(null); setResult(null); setChatLog([]); setChatState(null); setChatPreview(null);
    try {
      const r = await post<MLResponse>("/api/nl/parse-multilingual", { text, seed });
      setMl(r); setParsed(r.result);
    }
    catch (e) { setError((e as Error).message); }
    finally { setBusy(""); }
  };
  const generate = async () => {
    if (!parsed?.config) return;
    setBusy("generate"); setError(null);
    try {
      const r = await post<NLResult>("/api/nl/generate", { config: parsed.config, confirmed: true });
      setResult(r); setTable(Object.keys(r.preview)[0]); setShowQuality(false); setResTab("Preview");
      setChatState({ config: parsed.config, ops: [] }); setChatPreview(null); setChatLog([]);
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(""); }
  };

  const v = result?.validation;
  const pct = (x: number) => `${x.toFixed(x === 100 ? 0 : 1)}%`;
  return (
    <>
      <main className="center">
        <h1>Assistant</h1>
        <p className="lede">Describe the dataset in a sentence. You’ll see exactly what was understood before anything is generated.</p>
        <div className="chat">
          <textarea aria-label="Describe your dataset" value={text} rows={2} onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); parse(); } }} />
          <button className="btn primary" onClick={parse} disabled={busy !== "" || !text.trim()}>{busy === "parse" ? "Reading…" : "Parse"}</button>
        </div>
        <div className="chips">{EXAMPLES.map((x) => <button key={x} className="chip" onClick={() => setText(x)}>{x}</button>)}</div>
        {error && <div className="err">{error}</div>}

        {ml && ml.language !== "en" && (
          <div className="reply" dir={ml.reply.rtl ? "rtl" : "ltr"} lang={ml.language}>
            <b>{ml.reply.headline}</b>
            <ul>{ml.reply.lines.map((l) => <li key={l}>{l}</li>)}</ul>
            {ml.reply.warnings.map((w) => <p key={w} className="muted">⚠ {w}</p>)}
            <p className="muted" dir="ltr">Detected <b>{ml.language}</b>. Understood as: <span className="mono">{ml.canonical_text}</span></p>
          </div>
        )}
        {parsed && !parsed.ok && (
          <div className="card" style={{ marginTop: 24 }} dir={ml?.reply.rtl ? "rtl" : "ltr"} lang={ml?.language}>
            <b>{ml?.language && ml.language !== "en" ? ml.reply.errors[0] : "I need a bit more."}</b>
            {(!ml || ml.language === "en") && <ul>{parsed.errors.map((e) => <li key={e}>{e}</li>)}</ul>}
          </div>
        )}
        {parsed?.ok && parsed.config && (
          <section className="card" style={{ marginTop: 24 }}>
            <h2>Here’s what I understood</h2>
            <table><tbody>{parsed.explanation.map((e) => (
              <tr key={e.field}><td className="muted">{e.field}</td><td><b>{e.field === "locale" ? localeName(e.value) : e.value}</b></td><td className="muted">from “{e.source}”</td></tr>
            ))}</tbody></table>
            {parsed.warnings.map((w) => <p key={w} className="muted">⚠ {w}</p>)}
            <details style={{ margin: "12px 0" }}><summary>Full config (schema, distributions, rules)</summary>
              <pre className="json">{JSON.stringify(parsed.config, null, 2)}</pre></details>
            <button className="btn primary" onClick={generate} disabled={busy !== ""}>{busy === "generate" ? "Generating…" : "Confirm and generate"}</button>
          </section>
        )}

        {result && v && (
          <section style={{ marginTop: 24 }}>
            <h2>Generated and checked</h2>
            <div className="cards">
              <div className="card"><div className="big">{pct(v.locale.valid_pct)}</div><div className="label">records <Term k="Locale validity">locale-valid</Term></div></div>
              <div className="card"><div className="big" style={{ color: v.integrity.total_violations ? "var(--bad)" : "var(--ok)" }}>{v.integrity.total_violations}</div><div className="label">integrity violations</div></div>
              <div className="card"><div className="big">{pct(v.constraints.pass_pct)}</div><div className="label">rows passing {v.constraints.n_rules} rules</div></div>
              {v.flag && <div className="card"><div className="big">{(v.flag.achieved * 100).toFixed(1)}%</div><div className="label">{v.flag.name} (asked {(v.flag.requested * 100).toFixed(1)}%, {v.flag.count} rows)</div></div>}
            </div>
            <Tabs tabs={["Preview", "Visualize"] as const} value={resTab} onChange={setResTab} />
            {resTab === "Visualize" && <VisualizePanel datasetId={result.history_id ?? null} kind="nl" emptyNote="This run was not saved to History, so there is nothing to chart. Turn autosave on (SDP_AUTOSAVE) and generate again." />}
            {resTab === "Preview" && <Tabs tabs={Object.keys(result.preview) as readonly string[]} value={table ?? ""} onChange={setTable} />}
            {resTab === "Preview" && table && <DataTable columns={result.columns[table]} rows={result.preview[table]} />}
            <p className="muted">{Object.entries(v.rows).map(([k, n]) => `${k}: ${n.toLocaleString()} rows`).join(" · ")}. All values are fictional.</p>

            <div style={{ margin: "24px 0" }}>
              {!showQuality && <button className="btn" onClick={() => setShowQuality(true)} disabled={!result.quality}>Compute Trust Score</button>}
              {showQuality && result.quality && <QualityCard quality={result.quality} />}
            </div>

            <h2>Refine by chat</h2>
            <p className="lede">Ask for a change in plain words (any supported language). Only the affected parts are regenerated, and you see before and after.</p>
            <div className="chat">
              <textarea aria-label="Describe a change" rows={1} value={chatText} placeholder="double customers from Lahore" onChange={(e) => setChatText(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void refine(chatText); } }} />
              <button className="btn primary" disabled={chatBusy || !chatText.trim()} onClick={() => refine(chatText)}>{chatBusy ? "Applying…" : "Apply"}</button>
            </div>
            <div className="chips">{CHAT_EXAMPLES.map((x) => <button key={x} className="chip" onClick={() => refine(x)} disabled={chatBusy}>{x}</button>)}</div>
            {llm?.available && <label className="muted" style={{ display: "block", marginTop: 8 }}><input type="checkbox" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} /> Let Claude interpret requests the rules cannot (sends column names and category values)</label>}
            <div role="log" aria-live="polite">
              {chatLog.map((c, i) => (
                <div key={i} className="card" style={{ marginTop: 12 }}>
                  <div><b>“{c.text}”</b> <span className="pill">{c.out.source}</span></div>
                  <p>{c.out.message}</p>
                  {c.out.ops.map((o, k) => <div key={k} className="mono">{JSON.stringify(o)}</div>)}
                  {c.out.diff.length > 0 && (
                    <table className="diff"><thead><tr><th>changed</th><th>before</th><th>after</th></tr></thead>
                      <tbody>{c.out.diff.map((d) => <tr key={d.path}><td>{d.path}</td><td className="before">{String(d.before ?? "–")}</td><td className="after">{String(d.after ?? "–")}</td></tr>)}</tbody></table>
                  )}
                  {Object.keys(c.out.regenerated).length > 0 && (
                    <p className="muted">Regenerated: {Object.entries(c.out.regenerated).map(([t, r]) => r.untouched ? `${t} (unchanged)` : `${t} (${r.rows_before}→${r.rows_after} rows${r.columns_changed.length ? `, columns ${r.columns_changed.join(", ")}` : ""})`).join(" · ")}</p>
                  )}
                </div>
              ))}
            </div>
            {chatPreview && <><h3>Preview after edits · {chatPreview.table}</h3><DataTable columns={Object.keys(chatPreview.rows[0] ?? {})} rows={chatPreview.rows} /></>}
          </section>
        )}
      </main>
      <aside className="config">
        <h2>Settings</h2>
        <Panel title="Countries you can ask for" badge={locales?.length}>
          <p className="muted">Mention a country or language in your request. Open one to see sample values.</p>
          {locales?.map((l) => (
            <details key={l.code} style={{ marginBottom: 6 }}>
              <summary><b>{localeName(l.code)}</b> <span className="muted">{l.currency}</span></summary>
              <div className="muted" style={{ paddingLeft: 12 }}>{l.sample.name}<br />{l.sample.phone}<br />{l.sample.national_id}<br />{l.sample.amount}</div>
            </details>
          ))}
        </Panel>
        <Panel title="Advanced">
          <Field label="Repeat code" hint="Same number gives exactly the same data again."><input type="number" value={seed} onChange={(e) => setSeed(+e.target.value || 0)} /></Field>
        </Panel>
      </aside>
    </>
  );
}
