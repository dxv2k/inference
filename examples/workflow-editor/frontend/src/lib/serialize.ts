// Canvas (nodes + edges) -> workflow JSON. PLAN.md §5.

import type {
  CanvasNode,
  CanvasEdge,
  Manifest,
  ManifestBlock,
  WorkflowSpec,
  WorkflowStep,
  WorkflowInput,
  WorkflowOutput,
} from "../types.ts";
import { formatInputSelector, formatStepSelector } from "./selectors.ts";

export type SerializeResult = {
  spec: WorkflowSpec;
  errors: string[];
};

function findBlock(manifest: Manifest, type: string): ManifestBlock | undefined {
  return manifest.blocks.find(
    (b) => b.type === type || (b.aliases && b.aliases.includes(type)),
  );
}

/**
 * Topological sort of block nodes. Inputs are considered "depth 0";
 * a block comes after every block it depends on (via edges from its target
 * handles to other blocks' source handles).
 *
 * Returns block nodes in topological order; throws on cycles.
 */
function topoOrderBlocks(nodes: CanvasNode[], edges: CanvasEdge[]): CanvasNode[] {
  const blocks = nodes.filter((n) => n.type === "block");
  const idToNode = new Map(nodes.map((n) => [n.id, n] as const));
  const preds = new Map<string, Set<string>>();
  for (const b of blocks) preds.set(b.id, new Set());
  for (const e of edges) {
    const src = idToNode.get(e.source);
    const tgt = idToNode.get(e.target);
    if (!src || !tgt) continue;
    if (tgt.type !== "block") continue;
    if (src.type !== "block") continue; // only block-to-block matters for ordering
    preds.get(tgt.id)!.add(src.id);
  }

  // Kahn's algorithm — preserve original block insertion order for ties.
  const originalOrder = new Map(blocks.map((b, i) => [b.id, i] as const));
  const result: CanvasNode[] = [];
  const remaining = new Map<string, Set<string>>();
  for (const [k, v] of preds) remaining.set(k, new Set(v));

  while (remaining.size > 0) {
    // Pop ONE ready node at a time, picking the smallest-originalOrder among
    // ready ones. This preserves the original spec's relative ordering of
    // independent (parallel) branches — critical for round-trip equality.
    let pick: string | null = null;
    for (const [id, pset] of remaining) {
      if (pset.size !== 0) continue;
      if (pick === null || originalOrder.get(id)! < originalOrder.get(pick)!) {
        pick = id;
      }
    }
    if (pick === null) {
      const remainingIds = Array.from(remaining.keys()).join(", ");
      throw new Error(`workflow has a cycle (unresolved: ${remainingIds})`);
    }
    result.push(idToNode.get(pick)! as CanvasNode);
    remaining.delete(pick);
    for (const [, pset] of remaining) pset.delete(pick);
  }
  return result;
}

export function canvasToWorkflowJSON(
  nodes: CanvasNode[],
  edges: CanvasEdge[],
  manifest: Manifest,
): SerializeResult {
  const errors: string[] = [];

  // 1) Inputs — emit in node insertion order
  const inputs: WorkflowInput[] = [];
  for (const n of nodes) {
    if (n.type !== "input") continue;
    const d = n.data;
    const entry: WorkflowInput = { type: d.inputKind, name: d.name };
    if ("default_value" in d && d.default_value !== undefined) {
      entry.default_value = d.default_value;
    }
    inputs.push(entry);
  }

  // 2) Build incoming-edges map keyed by target node id
  const incoming: Record<string, CanvasEdge[]> = {};
  for (const e of edges) {
    if (!incoming[e.target]) incoming[e.target] = [];
    incoming[e.target].push(e);
  }
  const idToNode = new Map(nodes.map((n) => [n.id, n] as const));

  // 3) Steps — topological order; emit in manifest input order
  let stepsOrdered: CanvasNode[];
  try {
    stepsOrdered = topoOrderBlocks(nodes, edges);
  } catch (e: any) {
    errors.push(e.message || "topological sort failed");
    stepsOrdered = nodes.filter((n) => n.type === "block");
  }

  const steps: WorkflowStep[] = [];
  for (const n of stepsOrdered) {
    if (n.type !== "block") continue;
    const d = n.data;
    const step: WorkflowStep = { type: d.blockType, name: d.stepName };
    const block = findBlock(manifest, d.blockType);
    const fieldOrder = block ? block.inputs.map((i) => i.name) : Object.keys(d.params);

    // Compute selector value for each input field that has an incoming edge.
    // Edge target handle is "in:<fieldName>"; use that to map.
    const edgeByField: Record<string, CanvasEdge> = {};
    for (const e of incoming[n.id] ?? []) {
      if (!e.targetHandle.startsWith("in:")) continue;
      const field = e.targetHandle.slice(3);
      edgeByField[field] = e;
    }

    // Emit fields in manifest order. Then add unknown literal params at the end.
    const emitted = new Set<string>();
    for (const fieldName of fieldOrder) {
      const edge = edgeByField[fieldName];
      if (edge) {
        const src = idToNode.get(edge.source);
        if (!src) {
          errors.push(`step ${d.stepName}.${fieldName}: dangling edge`);
          continue;
        }
        if (src.type === "input") {
          step[fieldName] = formatInputSelector(src.data.name);
        } else if (src.type === "block") {
          const field = edge.sourceHandle.startsWith("out:")
            ? edge.sourceHandle.slice(4)
            : edge.sourceHandle;
          step[fieldName] = formatStepSelector(src.data.stepName, field);
        } else {
          errors.push(`step ${d.stepName}.${fieldName}: edge source is an output node`);
          continue;
        }
        emitted.add(fieldName);
      } else if (fieldName in d.params) {
        step[fieldName] = d.params[fieldName];
        emitted.add(fieldName);
      } else {
        // Not wired and no literal — only an error if required AND no default
        const inputDecl = block?.inputs.find((i) => i.name === fieldName);
        if (inputDecl && inputDecl.required) {
          errors.push(`step ${d.stepName}.${fieldName}: required input not connected and no literal value`);
        }
        // otherwise omit — engine will use default
      }
    }
    // Carry over any params that aren't in the manifest (user-set unknown fields)
    for (const [k, v] of Object.entries(d.params)) {
      if (!emitted.has(k)) {
        step[k] = v;
      }
    }
    steps.push(step);
  }

  // 4) Outputs — emit in node insertion order
  const outputs: WorkflowOutput[] = [];
  for (const n of nodes) {
    if (n.type !== "output") continue;
    const inc = incoming[n.id] ?? [];
    if (inc.length === 0) {
      errors.push(`output ${n.data.name}: not connected`);
      continue;
    }
    const e = inc[0];
    const src = idToNode.get(e.source);
    if (!src) {
      errors.push(`output ${n.data.name}: dangling edge`);
      continue;
    }
    let selector: string;
    if (src.type === "input") {
      selector = formatInputSelector(src.data.name);
    } else if (src.type === "block") {
      const field = e.sourceHandle.startsWith("out:")
        ? e.sourceHandle.slice(4)
        : e.sourceHandle;
      selector = formatStepSelector(src.data.stepName, field);
    } else {
      errors.push(`output ${n.data.name}: edge source is another output node`);
      continue;
    }
    outputs.push({ type: "JsonField", name: n.data.name, selector });
  }

  const spec: WorkflowSpec = {
    version: "1.0",
    inputs,
    steps,
    outputs,
  };
  return { spec, errors };
}
