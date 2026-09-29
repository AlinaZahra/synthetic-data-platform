import { useMemo, useState } from "react";
import { post, postBlob, saveBlob, type CardinalityCfg, type FK, type Graph, type RelationalFullResponse } from "./api";
import { ERDiagram } from "./ERDiagram";
import { PreviewFrame, Skeleton } from "./Preview";
import { RulesEditor } from "./RulesEditor";
import { VisualizePanel } from "./viz/LazyVisualize";
import { DataTable, Field, Panel, Tabs, pct, useLive } from "./ui";

const TABS = ["Graph", "Tables", "Scorecard", "Visualize"] as const;
interface Inferred { graph: Graph; errors: string[]; order: string[]; suggested_rules: string[] }

export function RelationalWorkspace() {
  const [tab, setTab] = useState<(typeof TABS)[number]>("Graph");
  const inferred = useLive(() => post<Inferred>("/api/relational/infer", { dataset: "shop_full" }), [], 0).data;
  const [ruleText, setRuleText] = useState<string | null>(null);
  const [rulesOk, setRulesOk] = useState<string[]>([]);
  const [enforce, setEnforce] = useState(true);
  const [retry, setRetry] = useState(0);
  const rulesValue = ruleText ?? (inferred?.suggested_rules ?? []).join("\n");
  const [edited, setEdited] = useState<Graph | null>(null);
  const graph = edited ?? inferred?.graph ?? null;
  const [scale, setScale] = useState(1);
  const [seed, setSeed] = useState(0);
  const [card, setCard] = useState<Record<string, CardinalityCfg>>({});
  const [selected, setSelected] = useState<string | null>(null);
  const [validation, setValidation] = useState<string[]>([]);

  const { data, error, busy } = useLive(
    (signal) => (graph ? post<RelationalFullResponse>("/api/relational/generate", { dataset: "shop_full", graph, scale, seed, cardinality: card, rules: rulesOk, enforce }, signal) : Promise.resolve(null)),
    [JSON.stringify(graph), scale, seed, JSON.stringify(card), JSON.stringify(rulesOk), enforce, retry],
  );

  const removeFk = (key: string) => graph && setEdited({ ...graph, foreign_keys: graph.foreign_keys.filter((f) => f.key !== key) });
  const addFk = async (child: string, col: string, parent: string) => {
    if (!graph) return;
    const p = graph.tables.find((t) => t.name === parent)!;
    const fk: FK = { key: `${child}(${col})->${parent}`, child_table: child, child_columns: [col], parent_table: parent,
      parent_columns: p.primary_key, cardinality: "1:N", nullable: true, confidence: 1, source: "user", stats: {} };
    if (graph.foreign_keys.some((f) => f.key === fk.key)) return;
    const next = { ...graph, foreign_keys: [...graph.foreign_keys, fk] };
    const v = await post<{ errors: string[] }>("/api/relational/validate", { graph: next });
    setValidation(v.errors);
    if (!v.errors.length) setEdited(next);
  };

  const [dlBusy, setDlBusy] = useState(false);
  const [saved, setSaved] = useState(false);
  const [dlErr, setDlErr] = useState<string | null>(null);
  const download = async () => {
    setDlBusy(true); setSaved(false); setDlErr(null);
    try { saveBlob(await postBlob("/api/export/relational", { dataset: "shop_full", graph, scale, seed, cardinality: card, rules: rulesOk, enforce }), "synthetic-tables.zip"); setSaved(true); }
    catch (e) { setDlErr((e as Error).message); }
    finally { setDlBusy(false); }
  };

  const sc = data?.scorecard;
  const table = selected ?? data?.order[0] ?? null;
  return (
    <>
      <main className="center">
        <h1>Relational <span className={`status ${busy ? "" : "live"}`}>{busy ? "updating…" : "live"}</span></h1>
        <p className="lede">Infer keys and relationships, edit them, then generate parents first and children against valid parent keys.</p>
        <div style={{ marginBottom: 16 }}>
          <button className="btn primary" onClick={download} disabled={dlBusy || !graph}>{dlBusy ? "Preparing…" : "Download these tables (ZIP)"}</button>{" "}
          <span className="muted" role="status" aria-live="polite">{saved ? "Downloaded and saved to History. " : "Every download is saved to History. "}</span>
          {dlErr && <span className="err">{dlErr}</span>}
        </div>
        <Tabs tabs={TABS} value={tab} onChange={setTab} />
        {!data && <PreviewFrame loading={busy} error={error} hasData={false} onRetry={() => setRetry((r) => r + 1)} skeleton={<Skeleton rows={6} cols={4} />}><></></PreviewFrame>}
        {data && error && <div className="err" role="alert">{error} <button className="btn" onClick={() => setRetry((r) => r + 1)}>Retry</button></div>}
        {tab === "Graph" && graph && (
          <>
            <ERDiagram graph={graph} rowCounts={sc?.row_counts} />
            <h2 style={{ marginTop: 24 }}>Generation order</h2>
            <p className="lede">{(data?.order ?? inferred?.order ?? []).join("  →  ")}</p>
            <h2>Relationships</h2>
            {graph.foreign_keys.map((f) => (
              <div className="edge" key={f.key}>
                <code>{f.child_table}.{f.child_columns.join(",")}</code> → <code>{f.parent_table}.{f.parent_columns.join(",")}</code>
                <span className="badge">{f.cardinality}</span>
                {f.nullable && <span className="badge">nullable</span>}
                {f.child_table === f.parent_table && <span className="badge">self-reference</span>}
                <span className="grow" />
                <span className="badge">{Math.round(f.confidence * 100)}% · {f.source}</span>
                <button className="btn" onClick={() => removeFk(f.key)} aria-label={`remove ${f.key}`}>Remove</button>
              </div>
            ))}
            {graph.many_to_many?.length ? <p className="lede">N:N via junction: {graph.many_to_many.map((m) => `${m.left} ↔ ${m.right} (${m.junction})`).join(", ")}</p> : null}
            <AddFk graph={graph} onAdd={addFk} />
            {validation.map((v) => <div className="err" key={v}>{v}</div>)}
            {edited && <button className="btn" onClick={() => { setEdited(null); setValidation([]); }}>Reset to inferred</button>}
          </>
        )}
        {tab === "Tables" && data && table && (
          <>
            <div className="tabs">{data.order.map((t) => <button key={t} aria-selected={t === table} onClick={() => setSelected(t)}>{t} ({sc?.row_counts[t]})</button>)}</div>
            <DataTable columns={data.columns[table]} rows={data.preview[table]} />
          </>
        )}
        {tab === "Visualize" && <VisualizePanel datasetId="shop_full" kind="relational" />}
        {tab === "Scorecard" && sc && (
          <>
            <div className="cards">
              <div className="card"><div className="big" style={{ color: sc.integrity.ok ? "var(--ok)" : "var(--bad)" }}>{sc.integrity.total_violations}</div><div className="label">integrity violations (target 0)</div></div>
              <div className="card"><div className="big">{sc.mean_cardinality_match == null ? "–" : `${Math.round(sc.mean_cardinality_match * 100)}%`}</div><div className="label">children-per-parent match</div></div>
              {data?.relation_metrics && <div className="card"><div className="big">{data.relation_metrics.score.toFixed(0)}</div><div className="label">relation-preservation score</div></div>}
              {sc.rules && <div className="card"><div className="big">{pct(sc.rules.reconciliation_pass_rate, 0)}</div><div className="label">rules hold (naive: {pct(sc.rules.pass_rate_before, 0)})</div></div>}
            </div>
            {data?.relation_metrics && (<><h3>Relation preservation</h3>
              {Object.entries(data.relation_metrics.components).filter(([, c]) => c.score !== null).map(([k, c]) => (
                <div className="metric" key={k}><span>{k.replace("_", " ")}</span><span className="bar wide"><i style={{ width: `${c.score}%` }} /></span><b>{c.score!.toFixed(0)}</b><span className="muted">{c.summary}</span></div>
              ))}</>)}
            {sc.rules && (<><h3>Business rules: naive vs enforced</h3>
              <div className="tablewrap"><table><thead><tr><th>rule</th><th>violations before</th><th>after</th><th>how</th></tr></thead>
                <tbody>{sc.rules.rules.map((r) => <tr key={r.name}><td><code>{r.dsl}</code></td><td>{r.violations_before}</td><td>{r.violations_after}</td><td>{r.unrepairable ?? (r.strategy || "already held")}</td></tr>)}</tbody></table></div></>)}
            <div className="tablewrap"><table>
              <thead><tr><th>violation type</th><th>count</th></tr></thead>
              <tbody>{Object.entries(sc.integrity.by_kind).map(([k, v]) => <tr key={k}><td>{k}</td><td>{v}</td></tr>)}</tbody>
            </table></div>
            <h3>Children per parent</h3>
            <div className="tablewrap"><table>
              <thead><tr><th>relationship</th><th>target</th><th>mean real</th><th>mean synthetic</th><th>match</th></tr></thead>
              <tbody>{Object.entries(sc.cardinality).map(([k, v]) => (
                <tr key={k}><td>{k}</td><td>{v.target}</td><td>{v.mean_real.toFixed(2)}</td><td>{v.mean_synth.toFixed(2)}</td>
                  <td><span className="bar"><i style={{ width: `${v.match_score * 100}%` }} /></span> {Math.round(v.match_score * 100)}%</td></tr>
              ))}</tbody>
            </table></div>
          </>
        )}
      </main>
      <aside className="config">
        <h2>Settings</h2>
        <Field label="Size (× the real data)" hint="2 means twice as many rows as the sample data."><input type="number" min={0.1} max={20} step={0.5} value={scale} onChange={(e) => setScale(Math.max(0.1, +e.target.value || 1))} /></Field>
        <Panel title="Rules across tables" badge={rulesOk.length} defaultOpen>
          <RulesEditor value={rulesValue} onChange={setRuleText} dataset="shop_full" onValid={setRulesOk} />
          <label className="field" style={{ marginTop: 12 }}><span>Enforce the rules (off = ignore them)</span><input type="checkbox" checked={enforce} onChange={(e) => setEnforce(e.target.checked)} /></label>
        </Panel>
        <Panel title="Children per parent" badge={Object.values(card).filter((c) => c.kind !== "learned").length}>
        <p className="muted">How many orders per customer, items per order, and so on. "Learned" copies the real data.</p>
        {graph?.foreign_keys.filter((f) => f.child_table !== f.parent_table).map((f) => {
          const c = card[f.key] ?? { kind: "learned" };
          const set = (next: CardinalityCfg) => setCard({ ...card, [f.key]: next });
          return (
            <div key={f.key} style={{ marginBottom: 16 }}>
              <Field label={`${f.child_table} per ${f.parent_table}`}>
                <select value={c.kind} onChange={(e) => set({ kind: e.target.value as CardinalityCfg["kind"] })}>
                  <option value="learned">Learned from real data</option><option value="poisson">Poisson</option>
                  <option value="zipf">Zipf</option><option value="fixed">Fixed</option>
                </select>
              </Field>
              {c.kind === "poisson" && <Field label="mean (λ)"><input type="number" min={0.1} step={0.5} value={c.lam ?? 3} onChange={(e) => set({ ...c, lam: +e.target.value || 3 })} /></Field>}
              {c.kind === "zipf" && <Field label="exponent (a > 1)"><input type="number" min={1.05} step={0.1} value={c.a ?? 2} onChange={(e) => set({ ...c, a: Math.max(1.05, +e.target.value || 2) })} /></Field>}
              {c.kind === "fixed" && <Field label="children"><input type="number" min={0} value={c.value ?? 1} onChange={(e) => set({ ...c, value: Math.max(0, +e.target.value || 0) })} /></Field>}
            </div>
          );
        })}
        </Panel>
        <Panel title="Advanced">
          <Field label="Repeat code" hint="Same number gives exactly the same data again."><input type="number" value={seed} onChange={(e) => setSeed(+e.target.value || 0)} /></Field>
        </Panel>
      </aside>
    </>
  );
}

function AddFk({ graph, onAdd }: { graph: Graph; onAdd: (child: string, col: string, parent: string) => void }) {
  const [child, setChild] = useState(graph.tables[0]?.name ?? "");
  const cols = useMemo(() => graph.tables.find((t) => t.name === child)?.columns.map((c) => c.name) ?? [], [graph, child]);
  const [col, setCol] = useState("");
  const [parent, setParent] = useState(graph.tables[0]?.name ?? "");
  return (
    <>
      <h3>Add relationship</h3>
      <div className="row2" style={{ gridTemplateColumns: "1fr 1fr 1fr auto", alignItems: "end" }}>
        <Field label="child table"><select value={child} onChange={(e) => { setChild(e.target.value); setCol(""); }}>{graph.tables.map((t) => <option key={t.name}>{t.name}</option>)}</select></Field>
        <Field label="column"><select value={col || cols[0]} onChange={(e) => setCol(e.target.value)}>{cols.map((c) => <option key={c}>{c}</option>)}</select></Field>
        <Field label="parent table"><select value={parent} onChange={(e) => setParent(e.target.value)}>{graph.tables.map((t) => <option key={t.name}>{t.name}</option>)}</select></Field>
        <button className="btn" style={{ marginBottom: 16 }} onClick={() => onAdd(child, col || cols[0], parent)}>Add</button>
      </div>
    </>
  );
}
