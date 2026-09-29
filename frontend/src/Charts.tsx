import { useState } from "react";
import type { Overlay } from "./api";

const W = 260, H = 116, PAD = { l: 6, r: 6, t: 18, b: 22 };
const short = (v: number | string) => (typeof v === "number" ? (Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(1)}k` : Number.isInteger(v) ? String(v) : v.toFixed(1)) : String(v));

/** Minimal overlay: synthetic as filled bars, real as an outline. Same scale for both, so gaps are visible at a glance. */
export function OverlayChart({ name, d, showReal = true, showSyn = true }: { name: string; d: Overlay[string]; showReal?: boolean; showSyn?: boolean }) {
  const n = d.real.length;
  const max = Math.max(...d.real, ...d.synthetic, 1e-9);
  const iw = W - PAD.l - PAD.r, ih = H - PAD.t - PAD.b;
  const bw = iw / n;
  const y = (v: number) => PAD.t + ih - (v / max) * ih;
  const categorical = d.kind === "categorical";
  const label = (i: number) => (categorical ? d.categories![i] : "");
  return (
    <figure className="chart">
      <figcaption>{name}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${name}: real versus synthetic distribution`}>
        {[0, 0.5, 1].map((t) => <line key={t} x1={PAD.l} x2={W - PAD.r} y1={y(max * t)} y2={y(max * t)} className="grid" />)}
        {showSyn && d.synthetic.map((v, i) => (
          <rect key={`s${i}`} x={PAD.l + i * bw + (categorical ? bw * 0.5 : 0.5)} width={categorical ? bw * 0.4 : Math.max(0.5, bw - 1)} y={y(v)} height={PAD.t + ih - y(v)} className="bar-syn">
            <title>{`${categorical ? label(i) : `${short(d.edges![i])} – ${short(d.edges![i + 1])}`}: real ${(d.real[i] * 100).toFixed(1)}%, synthetic ${(v * 100).toFixed(1)}%`}</title>
          </rect>
        ))}
        {showReal && (categorical
          ? d.real.map((v, i) => <rect key={`r${i}`} x={PAD.l + i * bw + bw * 0.08} width={bw * 0.4} y={y(v)} height={PAD.t + ih - y(v)} className="bar-real" />)
          : <polyline className="line-real" fill="none" points={d.real.flatMap((v, i) => [`${PAD.l + i * bw},${y(v)}`, `${PAD.l + (i + 1) * bw},${y(v)}`]).join(" ")} />)}
        {categorical
          ? d.categories!.slice(0, 5).map((c, i) => <text key={c} x={PAD.l + i * bw + bw / 2} y={H - 8} textAnchor="middle" className="axis">{c.length > 8 ? c.slice(0, 7) + "…" : c}</text>)
          : (<><text x={PAD.l} y={H - 8} className="axis">{short(d.edges![0])}</text><text x={W - PAD.r} y={H - 8} textAnchor="end" className="axis">{short(d.edges![n])}</text></>)}
      </svg>
    </figure>
  );
}

export function ChartGrid({ overlay, initial = 6 }: { overlay: Overlay; initial?: number }) {
  const [all, setAll] = useState(false);
  const [showReal, setShowReal] = useState(true);
  const [showSyn, setShowSyn] = useState(true);
  const names = Object.keys(overlay);
  if (!names.length) return <div className="state"><b>No comparable columns</b></div>;
  // at least one series stays visible: hiding both would leave empty charts
  const toggleReal = () => { if (showReal && !showSyn) setShowSyn(true); setShowReal(!showReal); };
  const toggleSyn = () => { if (showSyn && !showReal) setShowReal(true); setShowSyn(!showSyn); };
  return (
    <>
      <div className="legend" role="group" aria-label="Show or hide a series">
        <button type="button" className={`lg${showReal ? "" : " off"}`} aria-pressed={showReal} onClick={toggleReal}><i className="sw real" /> real</button>
        <button type="button" className={`lg${showSyn ? "" : " off"}`} aria-pressed={showSyn} onClick={toggleSyn}><i className="sw syn" /> synthetic</button>
        <span className="muted">Click to show or hide</span>
      </div>
      <div className="chart-grid">{(all ? names : names.slice(0, initial)).map((n) => <OverlayChart key={n} name={n} d={overlay[n]} showReal={showReal} showSyn={showSyn} />)}</div>
      {names.length > initial && <button className="btn" onClick={() => setAll(!all)}>{all ? "Show fewer" : `Show all ${names.length} columns`}</button>}
    </>
  );
}
