import { useMemo, useState } from "react";
import type { FK, Graph } from "./api";

const NODE_W = 190, HEAD = 26, ROW = 17, GAP_X = 110, GAP_Y = 28, MAX_ROWS = 8;

interface Node { name: string; x: number; y: number; h: number; cols: { name: string; pk: boolean; fk: boolean }[]; extra: number; layer: number }

function layout(graph: Graph): Node[] {
  const parents = new Map<string, string[]>();
  graph.tables.forEach((t) => parents.set(t.name, []));
  graph.foreign_keys.filter((f) => f.child_table !== f.parent_table).forEach((f) => parents.get(f.child_table)?.push(f.parent_table));
  const memo = new Map<string, number>();
  const depth = (t: string, seen: Set<string> = new Set()): number => {
    if (memo.has(t)) return memo.get(t)!;
    if (seen.has(t)) return 0;
    seen.add(t);
    const d = parents.get(t)!.length ? 1 + Math.max(...parents.get(t)!.map((p) => depth(p, seen))) : 0;
    memo.set(t, d);
    return d;
  };
  const byLayer = new Map<number, Node[]>();
  graph.tables.forEach((t) => {
    const fkCols = new Set(graph.foreign_keys.filter((f) => f.child_table === t.name).flatMap((f) => f.child_columns));
    const all = t.columns.map((c) => ({ name: c.name, pk: t.primary_key.includes(c.name), fk: fkCols.has(c.name) }));
    // keys first so relationships always land on a visible row
    const cols = [...all.filter((c) => c.pk || c.fk), ...all.filter((c) => !c.pk && !c.fk)].slice(0, MAX_ROWS);
    const layer = depth(t.name);
    const n: Node = { name: t.name, x: 0, y: 0, h: HEAD + cols.length * ROW + 8 + (all.length > cols.length ? ROW : 0), cols, extra: all.length - cols.length, layer };
    byLayer.set(layer, [...(byLayer.get(layer) ?? []), n]);
  });
  const nodes: Node[] = [];
  [...byLayer.keys()].sort((a, b) => a - b).forEach((l) => {
    let y = 12;
    byLayer.get(l)!.forEach((n) => { n.x = 12 + l * (NODE_W + GAP_X); n.y = y; y += n.h + GAP_Y; nodes.push(n); });
  });
  return nodes;
}

export function ERDiagram({ graph, rowCounts, onSelect }: { graph: Graph; rowCounts?: Record<string, number>; onSelect?: (t: string | null) => void }) {
  const nodes = useMemo(() => layout(graph), [graph]);
  const [selected, setSelected] = useState<string | null>(null);
  const [hover, setHover] = useState<string | null>(null);
  const by = new Map(nodes.map((n) => [n.name, n]));
  const width = Math.max(...nodes.map((n) => n.x + NODE_W), 300) + 12;
  const height = Math.max(...nodes.map((n) => n.y + n.h), 100) + 12;
  const focus = hover ?? selected;
  const connected = (f: FK) => focus !== null && (f.child_table === focus || f.parent_table === focus);
  const rowY = (n: Node, col: string) => n.y + HEAD + Math.max(0, n.cols.findIndex((c) => c.name === col)) * ROW + ROW / 2;
  const pick = (t: string | null) => { setSelected(t); onSelect?.(t); };
  const sel = graph.tables.find((t) => t.name === selected);
  const rels = graph.foreign_keys.filter((f) => f.child_table === selected || f.parent_table === selected);

  return (
    <div className="er">
      <svg viewBox={`0 0 ${width} ${height}`} role="group" aria-label="Entity relationship diagram" onClick={() => pick(null)}>
        {graph.foreign_keys.map((f) => {
          const c = by.get(f.child_table)!, p = by.get(f.parent_table)!;
          const active = connected(f);
          const cls = `edge-line${active ? " on" : focus ? " dim" : ""}`;
          if (c === p) {
            const y1 = rowY(c, f.child_columns[0]), y2 = rowY(p, f.parent_columns[0]);
            return <path key={f.key} className={cls} d={`M${c.x + NODE_W},${y1} C${c.x + NODE_W + 46},${y1} ${c.x + NODE_W + 46},${y2} ${p.x + NODE_W},${y2}`} fill="none"><title>{f.key}</title></path>;
          }
          const x1 = p.x + NODE_W, y1 = rowY(p, f.parent_columns[0]), x2 = c.x, y2 = rowY(c, f.child_columns[0]);
          const mx = (x1 + x2) / 2;
          return (
            <g key={f.key} className={cls}>
              <path d={`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`} fill="none"><title>{f.key}</title></path>
              <text x={x1 + 8} y={y1 - 5} className="card">1</text>
              <text x={x2 - 8} y={y2 - 5} textAnchor="end" className="card">{f.cardinality === "1:1" ? "1" : "N"}</text>
            </g>
          );
        })}
        {nodes.map((n) => {
          const on = focus === n.name || (focus !== null && graph.foreign_keys.some((f) => connected(f) && (f.child_table === n.name || f.parent_table === n.name)));
          return (
            <g key={n.name} className={`er-node${selected === n.name ? " sel" : ""}${focus && !on ? " dim" : ""}`} transform={`translate(${n.x},${n.y})`}
              tabIndex={0} role="button" aria-pressed={selected === n.name} aria-label={`Table ${n.name}`}
              onClick={(e) => { e.stopPropagation(); pick(selected === n.name ? null : n.name); }}
              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(selected === n.name ? null : n.name); } }}
              onMouseEnter={() => setHover(n.name)} onMouseLeave={() => setHover(null)} onFocus={() => setHover(n.name)} onBlur={() => setHover(null)}>
              <rect width={NODE_W} height={n.h} rx={8} className="er-box" />
              <text x={12} y={17} className="er-title">{n.name}</text>
              {rowCounts?.[n.name] != null && <text x={NODE_W - 10} y={17} textAnchor="end" className="er-rows">{rowCounts[n.name].toLocaleString()} rows</text>}
              <line x1={0} x2={NODE_W} y1={HEAD} y2={HEAD} className="er-rule" />
              {n.cols.map((c, i) => (
                <g key={c.name} transform={`translate(0,${HEAD + i * ROW})`}>
                  <text x={12} y={ROW - 5} className={c.pk ? "er-col pk" : "er-col"}>{c.name}</text>
                  {(c.pk || c.fk) && <text x={NODE_W - 10} y={ROW - 5} textAnchor="end" className="er-tag">{c.pk && c.fk ? "PK·FK" : c.pk ? "PK" : "FK"}</text>}
                </g>
              ))}
              {n.extra > 0 && <text x={12} y={HEAD + n.cols.length * ROW + ROW - 5} className="er-more">+{n.extra} more columns</text>}
            </g>
          );
        })}
      </svg>
      {sel ? (
        <div className="er-info">
          <b>{sel.name}</b>{rowCounts?.[sel.name] != null && <span className="muted"> · {rowCounts[sel.name].toLocaleString()} rows</span>}
          <div className="muted">Primary key: {sel.primary_key.join(", ") || "none"} · {sel.columns.length} columns</div>
          {rels.map((f) => (
            <div key={f.key} className="er-rel">
              <code>{f.child_table}.{f.child_columns.join(",")}</code> → <code>{f.parent_table}.{f.parent_columns.join(",")}</code>
              <span className="badge">{f.cardinality}</span>
              {f.stats?.mean_children != null && <span className="muted"> {f.stats.mean_children.toFixed(1)} children per parent on average</span>}
            </div>
          ))}
        </div>
      ) : <div className="er-info muted">Click a table to see its keys and relationships.</div>}
    </div>
  );
}
