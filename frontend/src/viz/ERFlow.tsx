import { useMemo } from "react";
import { Background, Controls, Handle, MarkerType, Position, ReactFlow, type Edge, type Node, type NodeProps } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { GraphEdge, GraphNode } from "./types";

type TableData = { label: string; rows: number; columns: number; pk: string[]; bad: boolean };

function TableNode({ data }: NodeProps<Node<TableData>>) {
  return (
    <div className="ernode" style={data.bad ? { borderColor: "var(--bad)" } : undefined}>
      <Handle type="target" position={Position.Top} style={{ opacity: 0 }} />
      <b>{data.label}</b>
      <small>{data.rows.toLocaleString()} rows · {data.columns} columns</small>
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0 }} />
    </div>
  );
}
const nodeTypes = { table: TableNode };

/** Parents on top, children below (a table sits one level under its deepest parent). */
function layout(nodes: GraphNode[], edges: GraphEdge[]): Record<string, { x: number; y: number }> {
  const level: Record<string, number> = Object.fromEntries(nodes.map((n) => [n.id, 0]));
  for (let pass = 0; pass < nodes.length; pass++) {
    for (const e of edges) if (e.source !== e.target) level[e.target] = Math.max(level[e.target], level[e.source] + 1);
  }
  const rows: Record<number, string[]> = {};
  nodes.forEach((n) => (rows[level[n.id]] ??= []).push(n.id));
  const pos: Record<string, { x: number; y: number }> = {};
  Object.entries(rows).forEach(([lv, ids]) => ids.forEach((id, i) => { pos[id] = { x: (i - (ids.length - 1) / 2) * 200, y: Number(lv) * 120 }; }));
  return pos;
}

/** Interactive ER diagram: tables as nodes, foreign keys as arrows labelled with the key column and its orphan count (red when above zero). */
export function ERFlow({ nodes, edges }: { nodes: GraphNode[]; edges: GraphEdge[] }) {
  const { rfNodes, rfEdges } = useMemo(() => {
    const pos = layout(nodes, edges);
    const badTables = new Set(edges.filter((e) => e.orphans > 0).map((e) => e.target));
    const rfNodes: Node<TableData>[] = nodes.map((n) => ({ id: n.id, type: "table", position: pos[n.id], data: { label: n.id, rows: n.rows, columns: n.columns, pk: n.primary_key, bad: badTables.has(n.id) } }));
    const rfEdges: Edge[] = edges.map((e) => ({
      id: e.id, source: e.source, target: e.target, label: `${e.label} · ${e.orphans} orphan${e.orphans === 1 ? "" : "s"}`,
      markerEnd: { type: MarkerType.ArrowClosed, color: e.orphans ? "var(--bad)" : "var(--syn)" },
      style: { stroke: e.orphans ? "var(--bad)" : "var(--syn)", strokeWidth: 1.6 }, labelBgStyle: { fill: "var(--surface)" }, labelStyle: { fontSize: 10 },
    }));
    return { rfNodes, rfEdges };
  }, [nodes, edges]);
  return (
    <div className="erflow" role="img" aria-label={`Relationship diagram of ${nodes.length} tables`}>
      <ReactFlow nodes={rfNodes} edges={rfEdges} nodeTypes={nodeTypes} fitView fitViewOptions={{ padding: 0.25 }} nodesConnectable={false} elementsSelectable={false} minZoom={0.4} maxZoom={1.6}
        proOptions={{ hideAttribution: true }}>
        <Background gap={20} color="var(--viz-grid)" />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
