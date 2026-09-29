import { useRef, useState, type ReactNode } from "react";
import { toPng } from "html-to-image";

interface Props {
  title: string;
  caption?: string | null;
  loading?: boolean;
  waiting?: boolean;
  error?: string | null;
  empty?: string | null;          // a plain-English reason there is nothing to draw
  onRetry?: () => void;
  fileName: string;               // PNG name without extension
  controls?: ReactNode;           // selectors shown in the card header
  legend?: boolean;
  realLabel?: string;
  synLabel?: string;
  wide?: boolean;
  children?: ReactNode;
}

/** One chart with a title, plain-English caption, a PNG button, and loading / empty / error states. */
export function ChartCard({ title, caption, loading, waiting, error, empty, onRetry, fileName, controls, legend = true, realLabel = "Real", synLabel = "Synthetic", wide, children }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [busy, setBusy] = useState(false);
  const [exportErr, setExportErr] = useState<string | null>(null);

  const savePng = async () => {
    if (!ref.current) return;
    setBusy(true); setExportErr(null);
    try {
      const bg = getComputedStyle(document.body).backgroundColor || "#ffffff";
      const url = await toPng(ref.current, { pixelRatio: 2, backgroundColor: bg, cacheBust: true,
        filter: (n) => !(n instanceof HTMLElement && (n.classList.contains("react-flow__controls") || n.classList.contains("vcard-noexport"))) });
      const a = document.createElement("a");
      a.href = url; a.download = `${fileName}.png`; a.click();
    } catch (e) { setExportErr(`Could not export the image: ${(e as Error).message}`); }
    finally { setBusy(false); }
  };

  const ready = !loading && !error && !empty && !!children;
  return (
    <section className={`vcard${wide ? " wide" : ""}`} aria-label={title}>
      <div className="vcard-head">
        <h3>{title}</h3>
        {controls}
        {ready && <button type="button" className="vbar-btn" onClick={savePng} disabled={busy} aria-label={`Save ${title} as a PNG image`}>{busy ? "Saving…" : "Save PNG"}</button>}
      </div>
      <div ref={ref} style={{ background: "var(--surface)" }}>
        {ready && legend && <div className="vlegend" style={{ marginBottom: 4 }}><span className="r"><i />{realLabel}</span><span className="s"><i />{synLabel}</span></div>}
        <div className="vcard-body">
          {loading && <div role="status" aria-live="polite" className="vcard-state">{waiting ? "The saved run is still being generated. This will appear when it finishes…" : <div className="vskel" aria-label="Loading chart" />}</div>}
          {!loading && error && (
            <div role="alert" className="vcard-state err"><div>{error}</div>{onRetry && <button type="button" className="btn" onClick={onRetry}>Try again</button>}</div>
          )}
          {!loading && !error && empty && <div className="vcard-state"><div><b>Nothing to show</b><div>{empty}</div></div></div>}
          {ready && children}
        </div>
        {ready && caption && <p className="vcard-caption">{caption}</p>}
      </div>
      {exportErr && <div className="err" role="alert">{exportErr}</div>}
    </section>
  );
}
