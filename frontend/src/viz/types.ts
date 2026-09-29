// Shapes of the Visualize endpoint (backend/sdp/visualize.py). Only summaries: bins, shares, matrices, counts.
export interface Base { dataset_id: string; type: string; kind: string; title: string; source: string; has_real: boolean; caption: string }
export interface Bin { label: string; x0?: string; x1?: string; real: number | null; synthetic: number }
export interface Stats { n?: number; min?: number | string; median?: number | string; mean?: number | string; max?: number | string; std?: number }
export interface Distribution extends Base { table: string; column: string; columns: string[]; tables: string[]; kind_of_column: string; bins: Bin[]; overlap_pct: number | null;
  stats: { real: Stats | null; synthetic: Stats } }
export interface Category { label: string; real: number | null; synthetic: number; real_count: number | null; synthetic_count: number }
export interface Categorical extends Base { table: string; column: string; columns: string[]; tables: string[]; categories: Category[]; match_pct: number | null }
export interface Correlation extends Base { table: string; tables: string[]; columns: string[]; real_matrix: number[][]; synthetic_matrix: number[][]; diff_matrix: number[][];
  mean_abs_diff: number; max_abs_diff: number; worst_pairs: { a: string; b: string; real: number; synthetic: number }[] }
export interface Tstr extends Base { target: string; targets: string[]; task: string; metric: string; metric_label: string; mean_gap_pct: number | null;
  models: { model: string; real: number | null; synthetic: number | null; gap_pct: number | null }[]; n_train_real: number; n_test_real: number }
export interface GraphNode { id: string; rows: number; columns: number; primary_key: string[] }
export interface GraphEdge { id: string; source: string; target: string; label: string; cardinality: string; orphans: number }
export interface Cardinality extends Base { fk: string; fks: string[]; child: string; parent: string; bins: { k: number; label: string; real: number | null; synthetic: number }[];
  mean_children: { real: number | null; synthetic: number }; orphans: { real: number | null; synthetic: number };
  integrity: { ok: boolean; total_violations: number; by_kind: Record<string, number>; rows_checked: number }; graph: { nodes: GraphNode[]; edges: GraphEdge[] } }
export interface Privacy extends Base { bins: Bin[]; median: { real: number; synthetic: number }; ratio: number; exact_copies: number; rows_compared: number; verdict: string;
  series_labels: { real: string; synthetic: string } }
export interface LocaleV extends Base { locale: string; valid_pct: number; n_rows: number; n_invalid_rows: number; checks: { check: string; label: string; checked: number; failed: number; pass_pct: number }[] }
export interface Batch extends Base { total: number; succeeded: number; failed: number; skipped: number; success_rate_pct: number; reconciliation_rate_pct: number;
  by_stage: Record<string, number>; by_error: Record<string, number>; failures: { index: number; stage: string; error_type: string; message: string }[]; duration_s: number; docs_per_s: number | null }
export interface CatalogItem { id: string; title: string; kind: string; source: string; types: string[] }
export interface Catalog { types: string[]; datasets: CatalogItem[]; saved_runs: CatalogItem[] }
