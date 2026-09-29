import { useState } from "react";
import { get, post, type Row } from "./api";
import { parseCsv } from "./csv";
import { Advanced, DataTable, Field, Panel, Tabs, Term, useLive } from "./ui";

const TABS = ["Understand a sample", "Realistic text", "Language edge cases"] as const;
const SAMPLE = `cust_id,email,signup,last_seen,balance,tier,age,note
1,aiden.park@example.com,03/03/2024,20/04/2024,"$1,204.50",gold,34,asked about the new savings plan
2,maya.roy@example.com,14/03/2024,02/05/2024,"$88.10",basic,27,called about a card replacement
3,li.wei@example.com,21/03/2024,11/04/2024,"$5,930.00",silver,45,wants to increase the credit limit
4,noor.khan@example.com,09/04/2024,25/05/2024,"$412.75",basic,52,complained about a late statement
5,sara.ali@example.com,17/04/2024,30/05/2024,"$2,760.20",gold,39,interested in a business account
6,omar.hadi@example.com,25/04/2024,18/06/2024,"$19.99",basic,23,forgot online banking password
7,eva.stone@example.com,02/05/2024,29/06/2024,"$7,310.00",silver,61,asked about mortgage rates
8,raj.iyer@example.com,10/05/2024,12/06/2024,"$640.00",basic,31,reported an unknown transaction`;

interface Col { name: string; semantic_type: string; confidence: number; nullable: boolean; unique: boolean; date_format: string | null; currency: string | null; allowed_values: string[] | null; source: string; reason: string }
interface Constraint { type: string; table: string; columns: string[]; params: Record<string, unknown>; description: string; source: string; holds_on_sample: boolean }
interface Rel { child_table: string; child_column: string; parent_table: string; parent_column: string; confidence: number; source: string }
interface Proposal { tables: Record<string, Col[]>; relationships: Rel[]; constraints: Constraint[]; warnings: string[]; used_llm: boolean; confirmed: boolean; proposal_hash: string }
interface InferOut { proposal: Proposal; rules: Record<string, string[]>; llm_available: boolean }
const TYPES = ["person_name", "first_name", "last_name", "email", "phone", "national_id", "identifier", "currency_amount", "currency_code", "date", "datetime", "percentage", "boolean", "category", "integer", "decimal", "free_text", "address", "city", "country", "postal_code", "url", "ip_address", "uuid", "other"];

function Understand() {
  const [csv, setCsv] = useState(SAMPLE);
  const [useLlm, setUseLlm] = useState(false);
  const [out, setOut] = useState<InferOut | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const llm = useLive(() => get<{ available: boolean }>("/api/llm/status"), [], 0).data;

  const infer = async () => {
    setBusy(true); setError(null);
    try {
      const { rows } = parseCsv(csv, 50);
      setOut(await post<InferOut>("/api/schema/infer", { records: rows, use_llm: useLlm }));
    } catch (e) { setError((e as Error).message); setOut(null); }
    finally { setBusy(false); }
  };
  const edit = async (edits: object[]) => {
    if (!out) return;
    try {
      const r = await post<{ proposal: Proposal; rules: Record<string, string[]> }>("/api/schema/edit", { proposal: out.proposal, edits });
      setOut({ ...out, proposal: r.proposal, rules: r.rules });
    } catch (e) { setError((e as Error).message); }
  };
  const p = out?.proposal;
  const table = p ? Object.keys(p.tables)[0] : "";
  return (
    <div className="stack">
      <p className="lede">Paste 20 to 50 rows of CSV. The platform proposes what each column means, its format, and likely rules. You confirm or edit before anything is used.</p>
      <textarea className="mono" rows={9} aria-label="CSV sample" value={csv} onChange={(e) => setCsv(e.target.value)} />
      <Advanced title="Options">
        <label><input type="checkbox" checked={useLlm} disabled={!llm?.available} onChange={(e) => setUseLlm(e.target.checked)} /> Refine with Claude
          {!llm?.available && <span className="muted"> (set ANTHROPIC_API_KEY to enable; local inference is always on)</span>}</label>
        {useLlm && <p className="muted">This sends the column names and up to 50 sample rows to the Claude API. Every suggestion is checked against your sample and dropped if the data contradicts it.</p>}
      </Advanced>
      <button className="btn primary" onClick={infer} disabled={busy || !csv.trim()}>{busy ? "Analysing…" : "Infer schema"}</button>
      {error && <div className="err" role="alert">{error}</div>}
      {p && (
        <section aria-live="polite">
          <h2>Proposed schema {p.confirmed ? <span className="pill user">confirmed</span> : <span className="pill">draft</span>}</h2>
          {p.used_llm && <span className="pill llm">Claude refined</span>}
          {p.warnings.map((w) => <p key={w} className="muted">⚠ {w}</p>)}
          <div className="tablewrap"><table>
            <thead><tr><th>column</th><th><Term k="Semantic type">type</Term></th><th>format</th><th>confidence</th><th>why</th></tr></thead>
            <tbody>{p.tables[table].map((c) => (
              <tr key={c.name}>
                <td><b>{c.name}</b>{c.unique && <span className="pill">unique</span>}{!c.nullable && <span className="pill">required</span>}</td>
                <td><select aria-label={`Type of ${c.name}`} value={c.semantic_type} onChange={(e) => edit([{ action: "set_type", table, column: c.name, value: e.target.value }])}>
                  {TYPES.map((t) => <option key={t}>{t}</option>)}</select>
                  {c.source !== "heuristic" && <span className={`pill ${c.source}`}>{c.source}</span>}</td>
                <td className="mono">{c.date_format ?? c.currency ?? (c.allowed_values ? c.allowed_values.slice(0, 4).join(", ") : "")}</td>
                <td>{Math.round(c.confidence * 100)}%</td><td className="muted">{c.reason}</td>
              </tr>
            ))}</tbody></table></div>
          <h3>Constraints</h3>
          {p.constraints.filter((c) => c.type !== "not_null").map((c) => {
            const i = p.constraints.indexOf(c);
            return (
              <div key={i} className="edge">
                <span className={c.holds_on_sample ? "result-pass" : "result-fail"}>{c.holds_on_sample ? "holds" : "violated in sample"}</span>
                <span className="grow">{c.description || `${c.type} ${c.columns.join(", ")}`}</span>
                {c.source !== "heuristic" && <span className={`pill ${c.source}`}>{c.source}</span>}
                <button className="btn" onClick={() => edit([{ action: "remove_constraint", value: i }])} aria-label={`Remove constraint: ${c.description}`}>Remove</button>
              </div>
            );
          })}
          {p.relationships.length > 0 && <><h3><Term k="Foreign key">Relationships</Term></h3>{p.relationships.map((r, i) => (
            <div key={i} className="edge"><code>{r.child_table}.{r.child_column} → {r.parent_table}.{r.parent_column}</code><span className="grow muted">{Math.round(r.confidence * 100)}% sure</span>
              <button className="btn" onClick={() => edit([{ action: "remove_relationship", value: i }])}>Remove</button></div>))}</>}
          <h3>Rules for the generator</h3>
          <pre className="json">{Object.entries(out!.rules).flatMap(([t, r]) => r.map((x) => `${t}: ${x}`)).join("\n") || "none"}</pre>
          <button className="btn primary" onClick={() => edit([{ action: "confirm" }])} disabled={p.confirmed}>{p.confirmed ? "Confirmed" : "Confirm this schema"}</button>
          <span className="muted"> · hash {p.proposal_hash}</span>
        </section>
      )}
    </div>
  );
}

interface Synth { values: string[]; sources: string[]; stats: Record<string, number>; warnings: string[] }
function RealisticText() {
  const [kind, setKind] = useState("review");
  const [locale, setLocale] = useState("en-US");
  const [n, setN] = useState(8);
  const [seed, setSeed] = useState(1);
  const [useLlm, setUseLlm] = useState(false);
  const [out, setOut] = useState<Synth | null>(null);
  const [error, setError] = useState<string | null>(null);
  const llm = useLive(() => get<{ available: boolean }>("/api/llm/status"), [], 0).data;
  const contexts = Array.from({ length: n }, (_, i) => kind === "ticket" ? { priority: ["low", "medium", "high"][i % 3], product: "card" } : { rating: (i % 5) + 1, product: "kettle", city: "Lahore" });
  const run = async () => {
    setError(null);
    try { setOut(await post<Synth>("/api/content/synthesize", { kind, n, locale, seed, use_llm: useLlm, contexts: kind === "name" || kind === "address" ? undefined : contexts })); }
    catch (e) { setError((e as Error).message); }
  };
  return (
    <div className="stack">
      <p className="lede">Names, addresses and free text that fit the locale and each row. With Claude, requests are batched and cached; without it, local generators produce the same shapes.</p>
      <div className="row2">
        <Field label="Kind"><select value={kind} onChange={(e) => setKind(e.target.value)}>{["name", "address", "review", "description", "ticket"].map((k) => <option key={k}>{k}</option>)}</select></Field>
        <Field label="Locale"><select value={locale} onChange={(e) => setLocale(e.target.value)}>{["en-US", "en-GB", "ur-PK", "ar", "hi", "es", "fr", "zh"].map((k) => <option key={k}>{k}</option>)}</select></Field>
      </div>
      <Advanced>
        <div className="row2"><Field label="Rows"><input type="number" min={1} max={200} value={n} onChange={(e) => setN(Math.max(1, +e.target.value || 1))} /></Field>
          <Field label={<Term k="Seed" />}><input type="number" value={seed} onChange={(e) => setSeed(+e.target.value || 0)} /></Field></div>
        <label><input type="checkbox" checked={useLlm} disabled={!llm?.available} onChange={(e) => setUseLlm(e.target.checked)} /> Use Claude {!llm?.available && <span className="muted">(not configured)</span>}</label>
      </Advanced>
      <button className="btn primary" onClick={run}>Generate</button>
      {error && <div className="err" role="alert">{error}</div>}
      {out && <>
        <p className="muted">{Object.entries(out.stats).map(([k, v]) => `${k}: ${v}`).join(" · ")}</p>
        {out.warnings.map((w) => <p key={w} className="muted">⚠ {w}</p>)}
        <div className="tablewrap"><table><thead><tr><th>#</th><th>text</th><th>source</th></tr></thead>
          <tbody>{out.values.map((v, i) => <tr key={i}><td>{i + 1}</td><td dir="auto" style={{ whiteSpace: "normal" }}>{v}</td><td><span className={`pill ${out.sources[i] === "llm" ? "llm" : ""}`}>{out.sources[i]}</span></td></tr>)}</tbody></table></div>
      </>}
    </div>
  );
}

interface CaseRow extends Row { pack: string; case: string; value: string; length: number; scripts: string; is_nfc: boolean; apostrophes: string }
function LanguageEdge() {
  const packs = ["unicode_diacritics", "unicode_normalization", "apostrophes", "long_names", "mixed_scripts", "rtl_digits"];
  const [pack, setPack] = useState("apostrophes");
  const data = useLive(() => get<{ rows: CaseRow[] }>(`/api/locale/language-cases?packs=${pack}&per_case=2&seed=1`), [pack], 0);
  return (
    <div className="stack">
      <p className="lede">Strings that break real systems. Add a pack to any dataset with <code>edge_cases</code> and rows are tagged so you know which ones are meant to be awkward.</p>
      <Tabs tabs={packs as unknown as readonly string[]} value={pack} onChange={setPack} />
      {data.error && <div className="err">{data.error}</div>}
      <div className="tablewrap"><table>
        <thead><tr><th>case</th><th>value</th><th>chars</th><th>scripts</th><th><Term k="NFC">NFC</Term></th></tr></thead>
        <tbody>{data.data?.rows.map((r, i) => (
          <tr key={i}><td>{r.case}</td><td dir="auto" style={{ whiteSpace: "normal", maxWidth: 420, overflowWrap: "anywhere" }}>{r.value}</td><td>{r.length}</td><td>{r.scripts}</td><td>{r.is_nfc ? "yes" : "no"}</td></tr>
        ))}</tbody></table></div>
      <p className="muted">Look-alike letters from other alphabets are <Term k="Homoglyph">homoglyphs</Term>; the mixed_scripts pack includes them on purpose.</p>
    </div>
  );
}

export function SchemaWorkspace() {
  const [tab, setTab] = useState<(typeof TABS)[number]>("Understand a sample");
  return (
    <>
      <main className="center">
        <h1>Schema & text</h1>
        <Tabs tabs={TABS} value={tab} onChange={setTab} />
        {tab === "Understand a sample" && <Understand />}
        {tab === "Realistic text" && <RealisticText />}
        {tab === "Language edge cases" && <LanguageEdge />}
      </main>
      <aside className="config">
        <h2>About this page</h2>
        <Panel title="How it works">
          <p className="muted">Everything runs offline by default. Claude is optional and never trusted blindly: its answers are checked against your data and replaced by local generators if the API is unavailable.</p>
        </Panel>
        <Panel title="Privacy">
          <p className="muted">Nothing leaves this machine unless you tick “Use Claude”. Then only column names and a capped sample are sent.</p>
        </Panel>
      </aside>
    </>
  );
}
