import { Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, PolarAngleAxis, RadialBar, RadialBarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Batch, Bin, Categorical, Correlation, Distribution, LocaleV, Privacy, Tstr } from "./types";

const AXIS = { fontSize: 11, fill: "var(--muted)" } as const;
const pctFmt = (v: unknown) => `${Number(v).toFixed(1)}%`;
const tone = (s: number) => (s >= 85 ? "var(--syn)" : s >= 70 ? "var(--real)" : "var(--bad)");

/** Two series side by side (real vs synthetic), shares shown as percentages. Real is omitted when the dataset has none. */
function OverlayBars({ rows, height = 240, xTitle, realName = "Real", synName = "Synthetic", angled = false }: {
  rows: { name: string; range?: string; real: number | null; synthetic: number }[]; height?: number; xTitle?: string; realName?: string; synName?: string; angled?: boolean }) {
  const hasReal = rows.some((r) => r.real !== null);
  const data = rows.map((r) => ({ ...r, real: r.real === null ? undefined : r.real * 100, synthetic: r.synthetic * 100 }));
  return (
    <div style={{ width: "100%", height }} role="img" aria-label={xTitle ?? "Bar chart"}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: angled ? 34 : 18, left: 0 }} barGap={1} barCategoryGap="12%">
          <CartesianGrid stroke="var(--viz-grid)" vertical={false} />
          <XAxis dataKey="name" tick={AXIS} interval="preserveStartEnd" minTickGap={14} angle={angled ? -28 : 0} textAnchor={angled ? "end" : "middle"} height={angled ? 48 : 26}
            label={xTitle ? { value: xTitle, position: "insideBottom", offset: -6, fill: "var(--muted)", fontSize: 11 } : undefined} />
          <YAxis tick={AXIS} tickFormatter={(v) => `${Math.round(Number(v))}%`} width={42} />
          <Tooltip formatter={(v, n) => [pctFmt(v), n]} labelFormatter={(l, p) => p?.[0]?.payload?.range ?? l} contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", fontSize: 12 }} />
          {hasReal && <Bar dataKey="real" name={realName} className="viz-bar-real" isAnimationActive={false} />}
          <Bar dataKey="synthetic" name={synName} className="viz-bar-syn" isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function DistributionBody({ d }: { d: Distribution }) {
  return <OverlayBars rows={d.bins.map((b: Bin) => ({ name: b.label, range: `${b.x0} to ${b.x1}`, real: b.real, synthetic: b.synthetic }))} xTitle={d.column} />;
}

export function CategoricalBody({ d }: { d: Categorical }) {
  return <OverlayBars rows={d.categories.map((c) => ({ name: c.label, range: c.real_count === null ? `${c.synthetic_count} rows` : `${c.real_count} real rows · ${c.synthetic_count} synthetic rows`, real: c.real, synthetic: c.synthetic }))}
    xTitle={d.column} angled={d.categories.length > 6} />;
}

export function CardinalityBody({ d }: { d: { bins: { label: string; real: number | null; synthetic: number }[]; child: string; parent: string } }) {
  return <OverlayBars rows={d.bins.map((b) => ({ name: b.label, range: `${b.label} ${d.child} per ${d.parent.replace(/s$/, "")}`, real: b.real, synthetic: b.synthetic }))} xTitle={`${d.child} per ${d.parent.replace(/s$/, "")}`} />;
}

export function PrivacyBody({ d }: { d: Privacy }) {
  return <OverlayBars rows={d.bins.map((b) => ({ name: b.label, range: `distance ${b.x0} to ${b.x1}`, real: b.real, synthetic: b.synthetic }))} xTitle="distance to the closest real record (0 = a copy)"
    realName={d.series_labels.real} synName={d.series_labels.synthetic} />;
}

export function TstrBody({ d }: { d: Tstr }) {
  const data = d.models.map((m) => ({ name: m.model.replace(/_/g, " "), real: m.real ?? 0, synthetic: m.synthetic ?? 0, gap: m.gap_pct }));
  return (
    <div style={{ width: "100%", height: 240 }} role="img" aria-label={`Model quality, trained on real versus synthetic data. ${d.metric_label}`}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 8, left: 0 }} barGap={4}>
          <CartesianGrid stroke="var(--viz-grid)" vertical={false} />
          <XAxis dataKey="name" tick={AXIS} />
          <YAxis tick={AXIS} domain={[0, 1]} width={36} />
          <Tooltip formatter={(v, n) => [Number(v).toFixed(3), n]} contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", fontSize: 12 }} />
          <Legend wrapperStyle={{ fontSize: 12 }} formatter={(v) => (v === "real" ? "Trained on real" : "Trained on synthetic")} />
          <Bar dataKey="real" name="real" className="viz-bar-real" isAnimationActive={false} />
          <Bar dataKey="synthetic" name="synthetic" className="viz-bar-syn" isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ------------------------------------------------------------------ heatmaps
/** Correlation grid: teal for positive, slate for negative, intensity = strength. Recharts has no heatmap, so this is plain CSS grid. */
function Heat({ title, m, cols, diff }: { title: string; m: number[][]; cols: string[]; diff?: boolean }) {
  const n = m.length;
  const color = (v: number) => {
    const a = Math.min(1, Math.abs(v));
    const base = diff ? "var(--viz-warn)" : v >= 0 ? "var(--syn)" : "var(--real)";
    return `color-mix(in srgb, ${base} ${Math.round(a * 100)}%, var(--surface))`;
  };
  return (
    <figure className="heat" style={{ margin: 0 }}>
      <figcaption>{title}</figcaption>
      <div className="heat-grid" style={{ gridTemplateColumns: `repeat(${n}, 1fr)` }} role="img" aria-label={`${title} correlation grid of ${n} columns`}>
        {m.flatMap((row, i) => row.map((v, j) => (
          <div key={`${i}-${j}`} className="heat-cell" style={{ background: color(v) }} title={`${cols[i]} × ${cols[j]}: ${v.toFixed(2)}`} />
        )))}
      </div>
      <div className="heat-scale"><span>{diff ? "same" : "−1"}</span><i style={{ background: diff ? "linear-gradient(90deg, var(--surface), var(--viz-warn))" : "linear-gradient(90deg, var(--real), var(--surface), var(--syn))" }} /><span>{diff ? "different" : "+1"}</span></div>
    </figure>
  );
}

export function HeatmapsBody({ d }: { d: Correlation }) {
  return (
    <>
      <div className="heats">
        <Heat title="Real" m={d.real_matrix} cols={d.columns} />
        <Heat title="Synthetic" m={d.synthetic_matrix} cols={d.columns} />
        <Heat title="Difference (synthetic − real)" m={d.diff_matrix} cols={d.columns} diff />
      </div>
      <div className="heat-labels">Rows and columns, in order: {d.columns.join(" · ")}</div>
    </>
  );
}

// ------------------------------------------------------------------ locale
export function LocaleBody({ d }: { d: LocaleV }) {
  const data = d.checks.map((c) => ({ name: c.label, pass: c.pass_pct, checked: c.checked }));
  return (
    <div style={{ width: "100%", height: Math.max(180, 44 * data.length + 30) }} role="img" aria-label="Share of values that are valid for the country, by check">
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 30, bottom: 4, left: 10 }}>
          <CartesianGrid stroke="var(--viz-grid)" horizontal={false} />
          <XAxis type="number" domain={[0, 100]} tick={AXIS} tickFormatter={(v) => `${v}%`} />
          <YAxis type="category" dataKey="name" tick={AXIS} width={150} />
          <Tooltip formatter={(v, _n, p) => [`${Number(v).toFixed(1)}% of ${p.payload.checked.toLocaleString()} checked`, "valid"]} contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", fontSize: 12 }} />
          <Bar dataKey="pass" isAnimationActive={false} label={{ position: "right", fill: "var(--text)", fontSize: 11, formatter: (v: unknown) => `${Number(v).toFixed(0)}%` }}>
            {data.map((r) => <Cell key={r.name} fill={tone(r.pass)} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ------------------------------------------------------------- document batch
export function BatchBody({ d }: { d: Batch }) {
  const good = d.succeeded + d.skipped;
  const pie = [{ name: "Produced and reconciled", value: good, fill: "var(--syn)" }, ...(d.failed ? [{ name: "Failed", value: d.failed, fill: "var(--viz-warn)" }] : [])];
  const stages = Object.entries(d.by_stage).map(([name, value]) => ({ name, value }));
  return (
    <div>
      <div className="vstats" style={{ marginBottom: 8 }}>
        <div className="vstat"><div className="n">{d.total}</div><div className="l">documents</div></div>
        <div className="vstat"><div className="n" style={{ color: "var(--syn)" }}>{d.succeeded}</div><div className="l">succeeded</div></div>
        <div className="vstat"><div className="n" style={{ color: d.failed ? "var(--viz-warn)" : undefined }}>{d.failed}</div><div className="l">failed</div></div>
        <div className="vstat"><div className="n">{d.reconciliation_rate_pct.toFixed(1)}%</div><div className="l">reconcile exactly</div></div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)", gap: 12, alignItems: "center" }}>
        <div style={{ height: 200 }} role="img" aria-label={`${good} of ${d.total} documents succeeded`}>
          <ResponsiveContainer>
            <PieChart>
              <Pie data={pie} dataKey="value" nameKey="name" innerRadius="62%" outerRadius="90%" startAngle={90} endAngle={-270} isAnimationActive={false} stroke="var(--surface)">
                {pie.map((p) => <Cell key={p.name} fill={p.fill} />)}
              </Pie>
              <Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", fontSize: 12 }} />
              <text x="50%" y="48%" textAnchor="middle" style={{ fontSize: 22, fontWeight: 600, fill: "var(--text)" }}>{d.success_rate_pct.toFixed(0)}%</text>
              <text x="50%" y="60%" textAnchor="middle" style={{ fontSize: 11, fill: "var(--muted)" }}>success</text>
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div>
          {stages.length ? (
            <>
              <div className="vcard-caption" style={{ marginBottom: 4 }}>Where failures happened</div>
              <div style={{ height: 130 }}>
                <ResponsiveContainer>
                  <BarChart data={stages} layout="vertical" margin={{ left: 10, right: 20 }}>
                    <XAxis type="number" allowDecimals={false} tick={AXIS} />
                    <YAxis type="category" dataKey="name" tick={AXIS} width={90} />
                    <Tooltip contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", fontSize: 12 }} />
                    <Bar dataKey="value" name="failed documents" fill="var(--viz-warn)" isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          ) : <div className="vcard-caption">No failures. Every document was produced and every total reconciled.</div>}
          {d.failures.slice(0, 2).map((f) => <div key={f.index} className="vcard-caption" style={{ marginTop: 4 }}>#{f.index + 1} {f.stage}: {f.message}</div>)}
        </div>
      </div>
    </div>
  );
}

// -------------------------------------------------------------------- gauge
/** Trust Score gauge: a half-circle that fills to the score. Same tone rules as the Trust Score card (teal strong, blue middling, red weak). */
export function GaugeBody({ score, label }: { score: number; label: string }) {
  const fill = score >= 85 ? "var(--syn)" : score >= 70 ? "var(--real)" : "var(--bad)";
  return (
    <div style={{ width: "100%", height: 200, position: "relative" }} role="img" aria-label={`Trust Score ${Math.round(score)} out of 100, ${label}`}>
      <ResponsiveContainer>
        <RadialBarChart data={[{ v: score }]} startAngle={180} endAngle={0} innerRadius="70%" outerRadius="100%" cx="50%" cy="82%">
          <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
          <RadialBar dataKey="v" cornerRadius={8} background={{ fill: "var(--viz-grid)" }} fill={fill} isAnimationActive={false} />
        </RadialBarChart>
      </ResponsiveContainer>
      <div style={{ position: "absolute", left: 0, right: 0, bottom: 14, textAlign: "center" }}>
        <div style={{ fontSize: 34, fontWeight: 600, lineHeight: 1, color: fill }}>{Math.round(score)}<small style={{ fontSize: 12, color: "var(--muted)", marginLeft: 3 }}>/100</small></div>
        <div style={{ fontSize: 12, fontWeight: 700, letterSpacing: ".08em", textTransform: "uppercase" }}>{label}</div>
      </div>
    </div>
  );
}
