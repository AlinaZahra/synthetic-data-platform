import { useEffect, useRef, useState, type ReactNode } from "react";
import type { Row } from "./api";

export function Field({ label, children, hint }: { label: ReactNode; children: ReactNode; hint?: ReactNode }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint && <small className="hint">{hint}</small>}
    </label>
  );
}

export function Tabs<T extends string>({ tabs, value, onChange }: { tabs: readonly T[]; value: T; onChange: (t: T) => void }) {
  // WAI-ARIA tabs: one tab stop, arrow keys / Home / End move between tabs
  const onKey = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const i = tabs.indexOf(value);
    const j = e.key === "ArrowRight" ? (i + 1) % tabs.length : e.key === "ArrowLeft" ? (i - 1 + tabs.length) % tabs.length : e.key === "Home" ? 0 : e.key === "End" ? tabs.length - 1 : -1;
    if (j < 0) return;
    e.preventDefault();
    onChange(tabs[j]);
    (e.currentTarget.querySelectorAll("button")[j] as HTMLButtonElement | undefined)?.focus();
  };
  return (
    <div className="tabs" role="tablist" onKeyDown={onKey}>
      {tabs.map((t) => (
        <button key={t} role="tab" aria-selected={t === value} tabIndex={t === value ? 0 : -1} onClick={() => onChange(t)}>{t}</button>
      ))}
    </div>
  );
}

export const GLOSSARY: Record<string, string> = {
  TSTR: "Train on Synthetic, Test on Real: a model trained on synthetic data is scored on real data. Close to the real-data baseline means the synthetic data kept the useful patterns.",
  "Trust Score": "One 0-100 number: 40% how closely the data matches the real one (fidelity), 30% privacy, 30% validity and coverage.",
  Fidelity: "How closely the synthetic columns and their relationships match the real data.",
  Seed: "A number that fixes the random choices. The same seed and settings always give the same data.",
  Outlier: "A value far outside the normal range (beyond 1.5 interquartile ranges of the middle half of the data).",
  "Dry run": "Does everything, including checks, inside a transaction that is rolled back at the end. Nothing is kept.",
  Lineage: "The recorded request, seed, code version and output hashes of a job, so it can be reproduced and audited.",
  "Data contract": "A list of checks (not empty, unique, in range, matches a pattern, valid foreign key...) that data must pass.",
  "Great Expectations": "An open-source data quality tool. Contracts here use its expectation names and can be exported to it.",
  NFC: "Unicode normal form where accented letters are single characters. NFD splits them into letter plus accent. They look the same but differ byte by byte.",
  Homoglyph: "A character from another alphabet that looks like a Latin letter (for example Cyrillic a), used in look-alike names.",
  "Foreign key": "A column whose values must exist in another table's key column, like orders.customer_id pointing to customers.",
  "Semantic type": "What a column means (email, phone, date, currency...) rather than just how it is stored.",
  "Locale validity": "Whether names, phones, IDs, dates and currency follow the rules of the chosen country and language.",
  Chunk: "A block of rows generated and written at a time, so millions of rows never sit in memory at once.",
};

/** A technical term with a tooltip that works on hover, keyboard focus and tap. */
export function Term({ k, children }: { k: string; children?: ReactNode }) {
  const [open, setOpen] = useState(false);
  const id = "tip-" + k.replace(/\W+/g, "-");
  return (
    <button type="button" className="term" aria-describedby={id} aria-expanded={open} onClick={() => setOpen((o) => !o)} onBlur={() => setOpen(false)}
      onKeyDown={(e) => { if (e.key === "Escape") setOpen(false); }}>
      {children ?? k}
      <span className="tip" role="tooltip" id={id}>{GLOSSARY[k] ?? k}</span>
    </button>
  );
}

/** A collapsible group in a side panel. `badge` shows how many settings inside are active, so nothing changes silently when it is closed. */
export function Panel({ title, badge, defaultOpen = false, children }: { title: string; badge?: number | string | null; defaultOpen?: boolean; children: ReactNode }) {
  return (
    <details className="panel" open={defaultOpen}>
      <summary><span>{title}</span>{badge ? <span className="pbadge">{badge}</span> : null}</summary>
      <div className="panel-body">{children}</div>
    </details>
  );
}

/** Progressive disclosure: advanced settings stay collapsed until asked for. */
export function Advanced({ title = "Advanced settings", children }: { title?: string; children: ReactNode }) {
  return <details className="adv"><summary>{title}</summary>{children}</details>;
}

export function DataTable({ columns, rows, marks, rowMark }: { columns: string[]; rows: Row[]; marks?: (col: string, i: number) => "null" | "outlier" | undefined; rowMark?: (i: number) => string | undefined }) {
  return (
    <div className="tablewrap">
      <table>
        <thead><tr>{columns.map((c) => <th key={c} dir="auto">{c}</th>)}</tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className={rowMark?.(i) ? "edge-row" : undefined} title={rowMark?.(i)}>
              {columns.map((c) => {
                const v = r[c];
                const isNull = v === null || v === undefined;
                const mark = isNull ? "null" : marks?.(c, i);
                return <td key={c} className={mark} dir="auto">{isNull ? "null" : fmt(v)}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function fmt(v: unknown): string {
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(2);
  if (typeof v === "string" && /^\d{4}-\d\d-\d\dT/.test(v)) return v.slice(0, 10);
  return String(v);
}

export const pct = (x: number | null | undefined, d = 1) => (x == null ? "n/a" : `${x.toFixed(d)}%`);

/** Runs `job` (debounced) whenever `deps` change; aborts the previous request. */
export function useLive<T>(job: (signal: AbortSignal) => Promise<T>, deps: unknown[], delay = 350) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const jobRef = useRef(job);
  jobRef.current = job;
  useEffect(() => {
    const ctl = new AbortController();
    setBusy(true);
    const t = setTimeout(() => {
      jobRef.current(ctl.signal)
        .then((d) => { setData(d); setError(null); })
        .catch((e: Error) => { if (e.name !== "AbortError") setError(e.message); })
        .finally(() => { if (!ctl.signal.aborted) setBusy(false); });
    }, delay);
    return () => { clearTimeout(t); ctl.abort(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return { data, error, busy };
}
