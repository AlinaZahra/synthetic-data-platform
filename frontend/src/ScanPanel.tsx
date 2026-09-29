import { useState } from "react";
import { get, post, postBlob, saveBlob } from "./api";
import { PreviewFrame, Skeleton } from "./Preview";
import { Field, useLive } from "./ui";

interface Field_ { key: string; text: string; value: string | null; bbox: number[]; polygon: number[][]; kind: "text" | "handwritten" | "stamp" }
interface ScanPage { page: number; width: number; height: number; image_base64: string; fields: Field_[]; augmentations: Record<string, unknown> }
interface ScanResp { mime: string; n_pages: number; pages: ScanPage[] }
interface Presets { presets: Record<string, Record<string, number | boolean>>; defaults: Record<string, number | boolean | string | null> }

const num = (v: unknown, d: number) => (typeof v === "number" ? v : d);

/** Scan-style realism controls + a preview with the ground-truth boxes drawn over the image. */
export function ScanPanel({ docType, spec }: { docType: string; spec: Record<string, unknown> }) {
  const presets = useLive(() => get<Presets>("/api/documents/scan/presets"), [], 0).data;
  const [preset, setPreset] = useState("office_scan");
  const [o, setO] = useState<Record<string, number | boolean | string>>({ seed: 1, dpi: 100, handwrite_total: false });
  const [boxes, setBoxes] = useState(true);
  const [page, setPage] = useState(0);
  const [err, setErr] = useState<string | null>(null);

  const { handwrite_total, ...rest } = o;
  const scan = { ...rest, ...(handwrite_total ? { handwrite_fields: ["total"] } : {}) };
  const req = { doc_type: docType, spec, preset, scan };
  const { data, error, busy } = useLive((s) => post<ScanResp>("/api/documents/scan", req, s), [JSON.stringify(req)], 500);
  const [retry, setRetry] = useState(0);
  void retry;

  const set = (k: string, v: number | boolean | string) => setO((x) => ({ ...x, [k]: v }));
  const val = (k: string) => (k in o ? o[k] : presets?.presets[preset]?.[k] ?? presets?.defaults[k]);
  const slider = (k: string, label: string, min: number, max: number, step: number) => (
    <Field label={`${label}: ${num(val(k), 0)}`}><input type="range" min={min} max={max} step={step} value={num(val(k), 0)} onChange={(e) => set(k, +e.target.value)} /></Field>
  );
  const p = data?.pages[Math.min(page, (data?.pages.length ?? 1) - 1)];
  const color = { text: "#16a34a", handwritten: "#2563eb", stamp: "#dc2626" };
  const zip = async () => { setErr(null); try { saveBlob(await postBlob("/api/documents/scan/zip", req), "scan-with-labels.zip"); } catch (e) { setErr((e as Error).message); } };

  return (
    <div className="scan">
      <div className="scan-controls">
        <Field label="Look"><select value={preset} onChange={(e) => { setPreset(e.target.value); setO({ seed: num(o.seed, 1), dpi: 100 }); }}>
          {Object.keys(presets?.presets ?? { office_scan: 1 }).map((k) => <option key={k}>{k}</option>)}</select></Field>
        {slider("rotate_deg", "Tilt (max °)", 0, 8, 0.1)}
        {slider("skew", "Skew", 0, 0.1, 0.005)}
        {slider("blur_radius", "Blur", 0, 3, 0.1)}
        {slider("noise_sigma", "Noise", 0, 40, 1)}
        {slider("jpeg_quality", "JPEG quality", 5, 100, 5)}
        {slider("low_res_scale", "Resolution scale", 0.2, 1, 0.05)}
        {slider("stamps", "Stamps", 0, 4, 1)}
        <label className="field"><span>Handwritten signature and note</span><input type="checkbox" checked={!!val("handwriting")} onChange={(e) => set("handwriting", e.target.checked)} /></label>
        <label className="field"><span>Re-draw the total by hand</span><input type="checkbox" checked={!!handwrite_total} onChange={(e) => set("handwrite_total", e.target.checked)} /></label>
        <Field label="Seed"><input type="number" value={num(o.seed, 1)} onChange={(e) => set("seed", +e.target.value || 0)} /></Field>
      </div>
      <div className="scan-view">
        {err && <div className="err" role="alert">{err}</div>}
        <PreviewFrame loading={busy} error={error} hasData={!!p} onRetry={() => setRetry((r) => r + 1)} emptyTitle="Rendering…" skeleton={<Skeleton rows={12} cols={2} />}>
          {p && (
            <>
              <div className="scan-bar">
                <label><input type="checkbox" checked={boxes} onChange={(e) => setBoxes(e.target.checked)} /> Show label boxes</label>
                <span className="muted">{p.fields.length} labelled fields · {p.width}×{p.height}px</span>
                {data!.n_pages > 1 && <span>{data!.pages.map((_, i) => <button key={i} className="btn" aria-pressed={i === page} onClick={() => setPage(i)}>{i + 1}</button>)}</span>}
                <button className="btn primary" onClick={zip}>Download images + labels</button>
              </div>
              <div className="scan-img" style={{ aspectRatio: `${p.width} / ${p.height}` }}>
                <img alt="Scanned document preview" src={`data:${data!.mime};base64,${p.image_base64}`} />
                {boxes && (
                  <svg viewBox={`0 0 ${p.width} ${p.height}`} preserveAspectRatio="none" aria-hidden>
                    {p.fields.map((f, i) => <polygon key={i} points={f.polygon.map((q) => q.join(",")).join(" ")} fill="none" stroke={color[f.kind]} strokeWidth={Math.max(1, p.width / 500)}><title>{`${f.key}: ${f.text}`}</title></polygon>)}
                  </svg>
                )}
              </div>
              <div className="legend"><span><i className="sw" style={{ background: color.text }} /> printed text</span><span><i className="sw" style={{ background: color.handwritten }} /> handwritten</span><span><i className="sw" style={{ background: color.stamp }} /> stamp</span></div>
            </>
          )}
        </PreviewFrame>
      </div>
    </div>
  );
}
