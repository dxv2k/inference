// Topological-depth-based grid layout. Column = depth from inputs, row = order within depth.

import type { CanvasNode, CanvasEdge } from "../types.ts";

const COL_W = 280;
const ROW_H = 140;
const COL_INPUT_X = 40;
const COL_OUTPUT_PAD = 80;
const ROW_PAD = 40;

export function autoLayout(nodes: CanvasNode[], edges: CanvasEdge[]): CanvasNode[] {
  if (nodes.length === 0) return nodes;

  // Build predecessor map
  const preds: Record<string, string[]> = {};
  for (const n of nodes) preds[n.id] = [];
  for (const e of edges) {
    if (preds[e.target]) preds[e.target].push(e.source);
  }

  // Compute depth via BFS. Input nodes start at depth 0.
  const depth: Record<string, number> = {};
  for (const n of nodes) {
    if (n.type === "input") depth[n.id] = 0;
  }

  // Iterative relax until stable (handles arbitrary DAG topologies)
  let changed = true;
  let safety = 0;
  while (changed && safety < 100) {
    changed = false;
    safety++;
    for (const n of nodes) {
      if (n.type === "input") continue;
      const pds = preds[n.id];
      if (pds.length === 0) {
        if (depth[n.id] === undefined) {
          depth[n.id] = 0;
          changed = true;
        }
        continue;
      }
      let maxD = -1;
      let allKnown = true;
      for (const p of pds) {
        if (depth[p] === undefined) {
          allKnown = false;
          break;
        }
        if (depth[p] > maxD) maxD = depth[p];
      }
      if (allKnown) {
        const newD = maxD + 1;
        if (depth[n.id] !== newD) {
          depth[n.id] = newD;
          changed = true;
        }
      }
    }
  }
  // Fallback for cyclic / unreachable nodes
  for (const n of nodes) {
    if (depth[n.id] === undefined) depth[n.id] = 1;
  }

  // Force output nodes to the largest column
  let maxDepth = 0;
  for (const n of nodes) if (depth[n.id] > maxDepth) maxDepth = depth[n.id];
  for (const n of nodes) {
    if (n.type === "output") depth[n.id] = maxDepth + 1;
  }

  // Group by column
  const byCol: Record<number, string[]> = {};
  for (const n of nodes) {
    const d = depth[n.id];
    if (!byCol[d]) byCol[d] = [];
    byCol[d].push(n.id);
  }

  // Stable order within column: by original node order
  const order: Record<string, number> = {};
  nodes.forEach((n, i) => (order[n.id] = i));
  for (const col of Object.keys(byCol)) {
    byCol[+col].sort((a, b) => order[a] - order[b]);
  }

  // Assign positions
  const idMap: Record<string, { x: number; y: number }> = {};
  for (const col of Object.keys(byCol)) {
    const c = +col;
    byCol[c].forEach((id, idx) => {
      const x = COL_INPUT_X + c * COL_W;
      const y = ROW_PAD + idx * ROW_H;
      idMap[id] = { x, y };
    });
  }

  return nodes.map((n) => ({ ...n, position: idMap[n.id] ?? n.position }));
}

export { COL_W, ROW_H, COL_INPUT_X, COL_OUTPUT_PAD };
