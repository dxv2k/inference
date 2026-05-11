// Workflow JSON -> canvas (nodes + edges). PLAN.md §5.

import type {
  CanvasNode,
  CanvasEdge,
  Manifest,
  ManifestBlock,
  WorkflowSpec,
} from "../types.ts";
import { isSelector, parseSelector } from "./selectors.ts";
import { autoLayout } from "./layout.ts";

export type DeserializeResult = {
  nodes: CanvasNode[];
  edges: CanvasEdge[];
  warnings: string[];
};

function findBlock(manifest: Manifest, type: string): ManifestBlock | undefined {
  return manifest.blocks.find(
    (b) => b.type === type || (b.aliases && b.aliases.includes(type)),
  );
}

export function workflowJSONToCanvas(
  spec: WorkflowSpec,
  manifest: Manifest,
): DeserializeResult {
  const warnings: string[] = [];
  const nodes: CanvasNode[] = [];
  const edges: CanvasEdge[] = [];

  // 1) Inputs -> InputNode
  const inputNodeIdByName: Record<string, string> = {};
  spec.inputs.forEach((inp, i) => {
    const id = `input_${i}_${inp.name}`;
    inputNodeIdByName[inp.name] = id;
    const data: any = {
      inputKind: inp.type === "WorkflowImage" ? "WorkflowImage" : "WorkflowParameter",
      name: inp.name,
    };
    if ("default_value" in inp) data.default_value = inp.default_value;
    nodes.push({
      id,
      type: "input",
      position: { x: 0, y: 0 },
      data,
    });
  });

  // 2) Steps -> BlockNode. Capture step-name -> node-id mapping.
  const stepNodeIdByName: Record<string, string> = {};
  spec.steps.forEach((step, i) => {
    const id = `step_${i}_${step.name}`;
    stepNodeIdByName[step.name] = id;
    const block = findBlock(manifest, step.type);
    // params: everything NOT a top-level scalar selector (and NOT type/name)
    const params: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(step)) {
      if (k === "type" || k === "name") continue;
      // Only treat a value as an edge if it's a scalar selector string.
      // Dict/list values that internally contain selectors (e.g.
      // prompt_parameters in the vlm workflow) stay as literals — see PLAN §5
      // "Selector wiring rules" + brief's note about vlm.
      if (isSelector(v)) {
        // edge will be made below; skip
      } else {
        params[k] = v;
      }
    }
    nodes.push({
      id,
      type: "block",
      position: { x: 0, y: 0 },
      data: {
        blockType: step.type,
        stepName: step.name,
        params,
      },
    });
    if (!block) {
      warnings.push(`step ${step.name}: block type "${step.type}" not in manifest`);
    }
  });

  // 3) For each step field that IS a scalar selector, create an edge
  spec.steps.forEach((step) => {
    const targetId = stepNodeIdByName[step.name];
    for (const [k, v] of Object.entries(step)) {
      if (k === "type" || k === "name") continue;
      if (!isSelector(v)) continue;
      const parsed = parseSelector(v);
      if (!parsed) {
        warnings.push(`step ${step.name}: cannot parse selector ${String(v)}`);
        continue;
      }
      if (parsed.kind === "input") {
        const srcId = inputNodeIdByName[parsed.name];
        if (!srcId) {
          warnings.push(
            `step ${step.name}.${k}: refers to $inputs.${parsed.name} but no such input declared`,
          );
          continue;
        }
        edges.push({
          id: `e_${srcId}_to_${targetId}_${k}`,
          source: srcId,
          sourceHandle: "out:value",
          target: targetId,
          targetHandle: `in:${k}`,
        });
      } else {
        const srcId = stepNodeIdByName[parsed.name];
        if (!srcId) {
          warnings.push(
            `step ${step.name}.${k}: refers to $steps.${parsed.name}.${parsed.field} but no such step`,
          );
          continue;
        }
        edges.push({
          id: `e_${srcId}_to_${targetId}_${k}`,
          source: srcId,
          sourceHandle: `out:${parsed.field}`,
          target: targetId,
          targetHandle: `in:${k}`,
        });
      }
    }
  });

  // 4) Outputs -> OutputNode + edge from the selector source
  spec.outputs.forEach((out, i) => {
    const id = `output_${i}_${out.name}`;
    nodes.push({
      id,
      type: "output",
      position: { x: 0, y: 0 },
      data: { name: out.name },
    });
    const parsed = parseSelector(out.selector);
    if (!parsed) {
      warnings.push(`output ${out.name}: cannot parse selector ${out.selector}`);
      return;
    }
    if (parsed.kind === "input") {
      const srcId = inputNodeIdByName[parsed.name];
      if (!srcId) {
        warnings.push(`output ${out.name}: refers to $inputs.${parsed.name} but no such input`);
        return;
      }
      edges.push({
        id: `e_${srcId}_to_${id}`,
        source: srcId,
        sourceHandle: "out:value",
        target: id,
        targetHandle: "in:value",
      });
    } else {
      const srcId = stepNodeIdByName[parsed.name];
      if (!srcId) {
        warnings.push(`output ${out.name}: refers to $steps.${parsed.name} but no such step`);
        return;
      }
      edges.push({
        id: `e_${srcId}_to_${id}`,
        source: srcId,
        sourceHandle: `out:${parsed.field}`,
        target: id,
        targetHandle: "in:value",
      });
    }
  });

  // 5) Auto-layout
  const laidOut = autoLayout(nodes, edges);

  return { nodes: laidOut, edges, warnings };
}
