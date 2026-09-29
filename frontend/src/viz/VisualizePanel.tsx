import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { get, post, postBlob, saveBlob } from "../api";
import { ChartCard } from "./ChartCard";
import { BatchBody, CardinalityBody, CategoricalBody, DistributionBody, GaugeBody, HeatmapsBody, LocaleBody, PrivacyBody, TstrBody } from "./charts";
import { ERFlow } from "./ERFlow";
import type { Batch, Cardinality, Catalog, CatalogItem, Categorical, Correlation, Distribution, LocaleV, Privacy, Tstr } from "./types";
import { useViz } from "./useViz";
import "./viz.css";

type Kind = "tabular" | "relational" | "nl" | "documents";
interface ScoreResult { score: number; label: string; kind: string; report?: { sub_scores?: unknown[] } | null }

const FALLBACK_TYPES: Record<Kind, string[]> = {
  tabular: ["distribution", "categorical", "correlation", "tstr", "privacy_distance"],
  relational: ["distribution", "categorical", "correlation", "cardinality"],
  nl: ["distribution", "categorical", "cardinality", "locale_validity"],
  documents: ["batch_summary"],
};

function Select({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (v: string) => void }) {
  if (options.length < 2) return null;
  return (
    <select aria-label={label} title={label} value={value} onChange={(e) => onChange(e.target.value)}>
      {options.map((o) => <option key={o} value={o}>{o}</option>)}
    </select>
  );
}

/**
 * The Visualize tab: real-vs-synthetic charts for a built-in dataset or a saved run, all drawn from small summaries (never rows).
 * `datasetId` is the workspace's dataset (a built-in name such as "customers" / "shop_full", or a saved run id).
 */
export function VisualizePanel({ datasetId, kind, emptyNote }: { datasetId: string | null; kind: Kind; emptyNote?: string }) {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [picked, setPicked] = useState<string | null>(null);
  const [seed, setSeed] = useState(0);
  useEffect(() => { get<Catalog>("/api/visualize").then(setCatalog).catch(() => setCatalog(null)); }, []);
  useEffect(() => { setPicked(null); }, [datasetId]);

  const id = picked ?? datasetId;
  const item: CatalogItem | undefined = useMemo(() => [...(catalog?.datasets ?? []), ...(catalog?.saved_runs ?? [])].find((d) => d.id === id), [catalog, id]);
  const effKind = (item?.kind as Kind | undefined) ?? kind;
  const types = item?.types ?? FALLBACK_TYPES[effKind];
  const has = (t: string) => types.includes(t);

  // what is on screen, so the scorecard PDF embeds exactly these charts
  const seen = useRef<Record<string, unknown>>({});
  const collect = useCallback((t: string, p: unknown) => { seen.current[t] = p; }, []);
  useEffect(() => { seen.current = {}; }, [id, seed]);

  const [table, setTable] = useState<string | undefined>(undefined);
  const [dCol, setDCol] = useState<string | undefined>(undefined);
  const [cCol, setCCol] = useState<string | undefined>(undefined);
  const [target, setTarget] = useState<string | undefined>(undefined);
  const [fk, setFk] = useState<string | undefined>(undefined);
  useEffect(() => { setTable(undefined); setDCol(undefined); setCCol(undefined); setTarget(undefined); setFk(undefined); }, [id]);
  useEffect(() => { setDCol(undefined); setCCol(undefined); }, [table]);

  const on = !!id;
  const base = { seed };
  const dist = useViz<Distribution>(id, "distribution", { ...base, table, column: dCol }, on && has("distribution"), collect);
  const cat = useViz<Categorical>(id, "categorical", { ...base, table, column: cCol }, on && has("categorical"), collect);
  const cor = useViz<Correlation>(id, "correlation", { ...base, table }, on && has("correlation"), collect);
  const tst = useViz<Tstr>(id, "tstr", { ...base, target }, on && has("tstr"), collect);
  const car = useViz<Cardinality>(id, "cardinality", { ...base, fk }, on && has("cardinality"), collect);
  const pri = useViz<Privacy>(id, "privacy_distance", base, on && has("privacy_distance"), collect);
  const loc = useViz<LocaleV>(id, "locale_validity", base, on && has("locale_validity"), collect);
  const bat = useViz<Batch>(id, "batch_summary", base, on && has("batch_summary"), collect);

  // ------------------------------------------------------------ Trust Score gauge
  const [score, setScore] = useState<ScoreResult | null>(null);
  const [scoreBusy, setScoreBusy] = useState(false);
  const [scoreErr, setScoreErr] = useState<string | null>(null);
  const [pdfBusy, setPdfBusy] = useState(false);
  useEffect(() => { setScore(null); setScoreErr(null); }, [id]);
  const scoreable = !!item && (item.source === "job" ? effKind !== "documents" : effKind === "tabular" || effKind === "relational");
  const computeScore = async () => {
    if (!item) return;
    setScoreBusy(true); setScoreErr(null);
    try {
      if (item.source === "job") setScore(await get<ScoreResult>(`/api/history/${item.id}/score`));
      else if (effKind === "relational") { const r = await post<{ trust_score: number; label: string; sub_scores: unknown[] }>("/api/relational/trust", { dataset: "shop_full", seed }); setScore({ score: r.trust_score, label: r.label, kind: "trust", report: r as never }); }
      else { const r = await post<{ trust_score: number; label: string; sub_scores: unknown[] }>("/api/trust/report", { dataset: "demo", sample: item.id, rows: 1000, seed }); setScore({ score: r.trust_score, label: r.label, kind: "trust", report: r as never }); }
    } catch (e) { setScoreErr((e as Error).message); } finally { setScoreBusy(false); }
  };
  const pdf = async () => {
    if (!score?.report) return;
    setPdfBusy(true);
    try { saveBlob(await postBlob("/api/trust/pdf", { report: score.report, visuals: seen.current }), `${id}-scorecard-with-charts.pdf`); }
    catch (e) { setScoreErr((e as Error).message); } finally { setPdfBusy(false); }
  };

  if (!id) return <div className="state"><b>Nothing to visualize yet</b><p>{emptyNote ?? "Generate some data first."}</p></div>;

  const groups = catalog ? [["Built-in", catalog.datasets], ["Saved runs", catalog.saved_runs]] as const : [];
  const notReal = (d: { has_real: boolean } | null) => (d && !d.has_real ? "No real data was used for this dataset, so only the synthetic side is drawn." : null);
  const er = car.data?.graph;

  return (
    <div>
      <div className="vtoolbar">
        <label className="field"><span>Dataset</span>
          <select value={id} onChange={(e) => setPicked(e.target.value)}>
            {!item && <option value={id}>{id}</option>}
            {groups.map(([g, list]) => list.length ? <optgroup key={g} label={g}>{list.map((d) => <option key={d.id} value={d.id}>{d.title}</option>)}</optgroup> : null)}
          </select></label>
        <button className="btn" onClick={() => setSeed((s) => s + 1)} title="Generate a fresh synthetic sample with a different seed">New sample (seed {seed})</button>
        <span className="muted">Charts show summaries only (bars, shares, counts), never individual rows.</span>
      </div>

      <div className="viz-grid">
        {scoreable && (
          <ChartCard title="Trust Score" fileName={`${id}-trust-score`} legend={false} loading={scoreBusy} error={scoreErr} onRetry={computeScore}
            caption={score ? (score.kind === "trust" ? "One number for “can I use this synthetic data?”: how closely it matches the real data, how private it is, and whether it is valid." : "Quality of the described dataset (validity of values, rules and links).") : null}
            empty={!score && !scoreBusy && !scoreErr ? "Compute the score to see the gauge." : null}
            controls={<>
              <button className="btn" onClick={computeScore} disabled={scoreBusy}>{score ? "Recompute" : "Compute Trust Score"}</button>
              {score?.report && Array.isArray(score.report.sub_scores) && <button className="btn" onClick={pdf} disabled={pdfBusy} title="Scorecard PDF with the charts shown below">{pdfBusy ? "Building PDF…" : "PDF with charts"}</button>}
            </>}>
            {score && <GaugeBody score={score.score} label={score.label} />}
          </ChartCard>
        )}

        {has("distribution") && (
          <ChartCard title="Distribution" fileName={`${id}-distribution-${dist.data?.column ?? ""}`} loading={dist.loading} waiting={dist.waiting} error={dist.error} onRetry={dist.retryable ? dist.reload : undefined}
            caption={dist.data?.caption} legend={!!dist.data?.has_real} empty={notReal(dist.data) && !dist.data?.bins.length ? notReal(dist.data) : null}
            controls={<>{dist.data && <Select label="Table" value={dist.data.table} options={dist.data.tables} onChange={setTable} />}
              {dist.data && <Select label="Column" value={dist.data.column} options={dist.data.columns} onChange={setDCol} />}</>}>
            {dist.data && <DistributionBody d={dist.data} />}
          </ChartCard>
        )}

        {has("categorical") && (
          <ChartCard title="Category mix" fileName={`${id}-categories-${cat.data?.column ?? ""}`} loading={cat.loading} waiting={cat.waiting} error={cat.error} onRetry={cat.retryable ? cat.reload : undefined}
            caption={cat.data?.caption} legend={!!cat.data?.has_real}
            controls={<>{cat.data && <Select label="Table" value={cat.data.table} options={cat.data.tables} onChange={setTable} />}
              {cat.data && <Select label="Column" value={cat.data.column} options={cat.data.columns} onChange={setCCol} />}</>}>
            {cat.data && <CategoricalBody d={cat.data} />}
          </ChartCard>
        )}

        {has("correlation") && (
          <ChartCard title="Correlations" fileName={`${id}-correlations`} wide legend={false} loading={cor.loading} waiting={cor.waiting} error={cor.error} onRetry={cor.retryable ? cor.reload : undefined} caption={cor.data?.caption}
            controls={cor.data && <Select label="Table" value={cor.data.table} options={cor.data.tables} onChange={setTable} />}>
            {cor.data && <HeatmapsBody d={cor.data} />}
          </ChartCard>
        )}

        {has("tstr") && (
          <ChartCard title="Usefulness for machine learning (TSTR)" fileName={`${id}-tstr`} legend={false} loading={tst.loading} waiting={tst.waiting} error={tst.error} onRetry={tst.retryable ? tst.reload : undefined} caption={tst.data?.caption}
            controls={tst.data && <Select label="Column to predict" value={tst.data.target} options={tst.data.targets} onChange={setTarget} />}>
            {tst.data && <TstrBody d={tst.data} />}
          </ChartCard>
        )}

        {has("privacy_distance") && (
          <ChartCard title="Privacy: distance to the closest real record" fileName={`${id}-privacy`} loading={pri.loading} waiting={pri.waiting} error={pri.error} onRetry={pri.retryable ? pri.reload : undefined}
            caption={pri.data?.caption} realLabel="Real (unseen)">
            {pri.data && <PrivacyBody d={pri.data} />}
          </ChartCard>
        )}

        {has("cardinality") && (
          <ChartCard title={car.data ? `${car.data.child} per ${car.data.parent.replace(/s$/, "")}` : "Rows per parent"} fileName={`${id}-cardinality`} loading={car.loading} waiting={car.waiting} error={car.error} onRetry={car.retryable ? car.reload : undefined}
            caption={car.data?.caption} legend={!!car.data?.has_real}
            controls={<>{car.data && <Select label="Relationship" value={car.data.fk} options={car.data.fks} onChange={setFk} />}
              {car.data && <span className={`integrity ${car.data.integrity.ok ? "ok" : "bad"}`} role="status">{car.data.integrity.ok ? "✓" : "✕"} Integrity: {car.data.integrity.total_violations} violation{car.data.integrity.total_violations === 1 ? "" : "s"}</span>}</>}>
            {car.data && <CardinalityBody d={car.data} />}
          </ChartCard>
        )}

        {has("cardinality") && (
          <ChartCard title="Tables and relationships" fileName={`${id}-er-diagram`} legend={false} loading={car.loading} waiting={car.waiting} error={car.error} onRetry={car.retryable ? car.reload : undefined}
            caption={car.data ? `Each box is a table and each arrow is a link from a parent to its children. Arrows turn red when some child rows point to a parent that does not exist (orphans). Total broken links: ${car.data.integrity.total_violations}.` : null}
            controls={car.data && <span className={`integrity ${car.data.integrity.ok ? "ok" : "bad"}`}>{car.data.integrity.ok ? "✓ all links valid" : `✕ ${car.data.integrity.total_violations} broken`}</span>}>
            {er && <ERFlow nodes={er.nodes} edges={er.edges} />}
          </ChartCard>
        )}

        {has("locale_validity") && (
          <ChartCard title="Valid for the country" fileName={`${id}-locale`} legend={false} loading={loc.loading} waiting={loc.waiting} error={loc.error} onRetry={loc.retryable ? loc.reload : undefined} caption={loc.data?.caption}>
            {loc.data && <LocaleBody d={loc.data} />}
          </ChartCard>
        )}

        {has("batch_summary") && (
          <ChartCard title="Document batch summary" fileName={`${id}-batch`} wide legend={false} loading={bat.loading} waiting={bat.waiting} error={bat.error} onRetry={bat.retryable ? bat.reload : undefined} caption={bat.data?.caption}>
            {bat.data && <BatchBody d={bat.data} />}
          </ChartCard>
        )}
      </div>
    </div>
  );
}
