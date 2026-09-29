export type Row = Record<string, unknown>;

export async function post<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(typeof detail.detail === "string" ? detail.detail : JSON.stringify(detail.detail ?? res.statusText));
  }
  return res.json() as Promise<T>;
}

export async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(res.statusText);
  return res.json() as Promise<T>;
}

// ---- response shapes (mirror backend/sdp/api/main.py) ----
export interface InjectionRecord {
  column: string; action: "null" | "outlier"; requested_rate: number; count: number;
  method: string | null; row_indices: number[];
}
export interface TabularResponse {
  seed: number; n_rows: number; columns: string[]; dtypes: Record<string, string>;
  preview: Row[]; injection_log: InjectionRecord[];
  fidelity: {
    ks: Record<string, { statistic: number; p_value: number }>;
    chi_square: Record<string, { statistic: number; p_value: number; max_abs_freq_diff: number }>;
    correlation_gap: number; mean_ks_statistic: number | null; dtypes_match: boolean;
  };
}
export interface TstrResponse {
  target: string; task: string;
  models: Record<string, { baseline_real: Record<string, number | null>; synthetic: Record<string, number | null>; gap_pct: Record<string, number | null> }>;
  summary: { primary_metric: string; mean_gap_pct: number | null };
}
export interface FK {
  key: string; child_table: string; child_columns: string[]; parent_table: string; parent_columns: string[];
  cardinality: "1:1" | "1:N"; nullable: boolean; confidence: number; source: string; stats: Record<string, number>;
}
export interface Graph {
  tables: { name: string; columns: { name: string; dtype: string; nullable: boolean }[]; primary_key: string[]; row_count: number | null }[];
  foreign_keys: FK[]; many_to_many?: { junction: string; left: string; right: string }[];
}
export interface CardinalityCfg { kind: "learned" | "poisson" | "zipf" | "fixed"; lam?: number; a?: number; value?: number }
export interface RelationalResponse {
  order: string[]; columns: Record<string, string[]>; preview: Record<string, Row[]>;
  scorecard: {
    row_counts: Record<string, number>; mean_cardinality_match: number | null;
    integrity: { ok: boolean; total_violations: number; by_kind: Record<string, number> };
    cardinality: Record<string, { match_score: number; mean_real: number; mean_synth: number; target: string }>;
    notes: string[];
  };
}

export async function postBlob(path: string, body: unknown): Promise<Blob> {
  const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(await res.text());
  return res.blob();
}

export function saveBlob(blob: Blob, name: string) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

// ---- trust score
export interface TrustComponent { key: string; label: string; score: number; weight: number; summary: string }
export interface TrustReport {
  title: string; generated_at: string; trust_score: number; label: string; verdict: string; gates: string[];
  sub_scores: { key: string; label: string; score: number; weight: number; components: TrustComponent[] }[];
  details: { fidelity_columns: Record<string, { score: number | null; test: string; statistic?: number; kind: string }> } & Record<string, unknown>;
}

// ---- assistant
export interface ParseResponse {
  ok: boolean; errors: string[]; warnings: string[]; explanation: { field: string; value: string; source: string }[];
  config: Record<string, unknown> | null; config_hash: string | null;
}
export interface NLResult {
  history_id?: string | null;
  quality?: { score: number; label: string; note: string; components: { key: string; label: string; score: number; weight: number; summary: string }[] };
  config_hash: string; columns: Record<string, string[]>; preview: Record<string, Row[]>;
  validation: {
    rows: Record<string, number>;
    integrity: { total_violations: number };
    constraints: { pass_pct: number; n_rules: number };
    locale: { valid_pct: number; n_rows: number };
    flag?: { name: string; requested: number; achieved: number; count: number };
  };
}

// ---- documents / locales
export interface LocaleInfo {
  code: string; country: string; script: string; currency: string; tax_label: string; date_format: string;
  direction: "ltr" | "rtl"; font_script: string; native_title: string; native_digits: boolean; complex_shaping: boolean;
  sample: { name: string; phone: string; national_id: string; address: string; amount: string };
}
export interface Invoice {
  invoice_number: string; issue_date: string; currency: string; locale: string; region: string | null; tax_inclusive: boolean;
  seller: { name: string; address: string }; customer: { name: string; address: string };
  lines: { description: string; quantity: string; unit_price: string; line_total: string; tax_label: string; tax_rate: string }[];
  tax_summary: { tax_label: string; tax_rate: string; taxable_amount: string; tax: string }[];
  subtotal: string; tax_total: string; total: string;
}
export interface BatchReport {
  report: { job_id: string; total: number; succeeded: number; failed: number; skipped: number; duration_s: number;
    failures: { index: number; doc: string; stage: string; error_type: string; message: string; attempts: number }[];
    by_stage: Record<string, number> };
}

// ---- rules
export interface CompiledRule { ok: boolean; input: string; dsl?: string; translated_from_plain_language?: boolean; owner?: string; kind?: string; error?: string }
export interface RulesCompile { results: CompiledRule[]; columns: string[] }
export interface RuleResult {
  name: string; dsl: string; owner: string; kind: string; checked: number; violations_before: number; violations_after: number;
  pass_rate_before: number; pass_rate_after: number; strategy: string; unrepairable: string | null;
}
export interface RulesReport {
  rules: RuleResult[]; violations_before: number; violations_after: number; pass_rate_before: number;
  reconciliation_pass_rate: number; derived_pass_rate_before: number; derived_pass_rate: number;
}
export interface TabularRulesReport extends RulesReport { mode: string; rows_dropped: number; naive_violations: Record<string, number> }

// ---- overlay charts
export type Overlay = Record<string, { kind: "numeric" | "datetime" | "categorical"; edges?: (number | string)[]; categories?: string[]; real: number[]; synthetic: number[] }>;

// ---- relation metrics
export interface RelationMetrics {
  score: number;
  components: Record<string, { score: number | null; weight: number; summary: string; details: unknown }>;
}
export interface RelationalFullResponse extends RelationalResponse {
  graph: Graph; relation_metrics?: RelationMetrics;
  scorecard: RelationalResponse["scorecard"] & { rules: RulesReport | null };
}

// ---- documents
export interface DocLayout {
  doc_type: string; direction: "ltr" | "rtl"; title: string; number: string;
  header_start: { text: string; bold: boolean }[]; header_end: string[]; block_label: string; block_lines: string[];
  columns: { label: string; width: number; align: "start" | "end" }[]; rows: string[][];
  totals: { label: string; value: string; strong: boolean }[]; notes: string[]; script: string;
}
export interface DocPreview { document: Record<string, unknown>; layout: DocLayout | null; html?: string | null; engine?: string; reconciliation_errors: string[] }
export interface DocTypeInfo { name: string; title: string; engine: "reportlab" | "html"; params: string[] }
export interface DocTypes { types: string[]; details: DocTypeInfo[] }
