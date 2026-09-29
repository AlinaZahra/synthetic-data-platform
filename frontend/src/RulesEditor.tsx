import { useEffect } from "react";
import { post, type RulesCompile } from "./api";
import { useLive } from "./ui";

interface Props {
  value: string;
  onChange: (v: string) => void;
  dataset: "customers" | "shop_full" | "shop" | "inline";
  tables?: Record<string, unknown[]>;
  /** dataset="customers" only: check against another built-in table, or against these rows (one uploaded table). */
  sample?: string;
  data?: unknown[];
  examples?: string[];
  /** Called with the rules that compiled, so the caller only ever sends valid ones. */
  onValid: (rules: string[]) => void;
}

export function RulesEditor({ value, onChange, dataset, tables, sample, data: rows, examples = [], onValid }: Props) {
  const lines = value.split("\n").map((l) => l.trim()).filter(Boolean);
  const { data, error, busy } = useLive(
    (signal) => (lines.length ? post<RulesCompile>("/api/rules/compile", { rules: lines, dataset, tables, sample, data: rows }, signal) : Promise.resolve({ results: [], columns: [] } as RulesCompile)),
    [lines.join("\n"), dataset, sample, rows?.length], 400);

  useEffect(() => {
    onValid(data ? data.results.filter((r) => r.ok).map((r) => r.input) : []);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  return (
    <div className="rules">
      <textarea aria-label="Business rules" rows={Math.min(8, Math.max(3, lines.length + 1))} value={value}
        placeholder={"One rule per line, in plain English or DSL\ne.g. discount must be at most 30%\n      delivery_date > order_date"}
        onChange={(e) => onChange(e.target.value)} />
      {error && <div className="err">{error}</div>}
      {data?.results.map((r, i) => (
        <div key={i} className={`rule ${r.ok ? "ok" : "bad"}`}>
          <span className="mark">{r.ok ? "✓" : "✕"}</span>
          <div>
            {r.ok ? (<><code>{r.dsl}</code>{r.translated_from_plain_language && <span className="badge">from plain language</span>}
              <span className="muted"> · checks {r.owner}</span></>) : (<><span className="muted">{r.input}</span><div className="err">{r.error}</div></>)}
          </div>
        </div>
      ))}
      {busy && lines.length > 0 && !data && <div className="muted">Checking rules…</div>}
      {examples.length > 0 && (
        <div className="chips">{examples.map((x) => (
          <button key={x} className="chip" onClick={() => onChange(value.includes(x) ? value : (value ? value + "\n" : "") + x)}>+ {x}</button>
        ))}</div>
      )}
    </div>
  );
}
