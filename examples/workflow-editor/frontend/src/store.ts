// Zustand store: canvas state + manifest + run result + dispatchers.

import { create } from "zustand";
import { applyNodeChanges as rfApplyNodeChanges, applyEdgeChanges as rfApplyEdgeChanges } from "reactflow";
import type {
  CanvasNode,
  CanvasEdge,
  Manifest,
  WorkflowSpec,
  RunResponse,
  BlockNodeData,
} from "./types";
import { workflowJSONToCanvas } from "./lib/deserialize";
import { canvasToWorkflowJSON } from "./lib/serialize";
import { autoLayout } from "./lib/layout";

let _nextId = 1;
export function nextId(prefix = "node"): string {
  return `${prefix}_${_nextId++}`;
}

type State = {
  manifest: Manifest | null;
  nodes: CanvasNode[];
  edges: CanvasEdge[];
  selectedNodeId: string | null;
  lastResult: RunResponse | null;
  workflowName: string;
  validateStatus: { ok: boolean; error?: string } | null;
  runtimeParams: Record<string, unknown>;
};

type Actions = {
  setManifest: (m: Manifest) => void;
  setNodes: (n: CanvasNode[]) => void;
  setEdges: (e: CanvasEdge[]) => void;
  applyNodeChanges: (changes: any[]) => void;
  applyEdgeChanges: (changes: any[]) => void;
  addBlock: (blockType: string, position: { x: number; y: number }) => void;
  addInput: (inputKind: "WorkflowImage" | "WorkflowParameter", name: string) => void;
  addOutput: (name: string) => void;
  deleteNode: (id: string) => void;
  connect: (e: { source: string; sourceHandle: string; target: string; targetHandle: string }) => void;
  disconnect: (edgeId: string) => void;
  setParam: (nodeId: string, field: string, value: unknown) => void;
  setStepName: (nodeId: string, newName: string) => void;
  setInputName: (nodeId: string, newName: string) => void;
  setInputDefault: (nodeId: string, value: unknown) => void;
  setOutputName: (nodeId: string, newName: string) => void;
  setSelected: (id: string | null) => void;
  loadSpec: (spec: WorkflowSpec, name: string) => void;
  clear: () => void;
  setLastResult: (r: RunResponse | null) => void;
  setWorkflowName: (n: string) => void;
  setValidateStatus: (s: { ok: boolean; error?: string } | null) => void;
  setRuntimeParams: (p: Record<string, unknown>) => void;
  getCurrentSpec: () => { spec: WorkflowSpec; errors: string[] };
};

export const useStore = create<State & Actions>((set, get) => ({
  manifest: null,
  nodes: [],
  edges: [],
  selectedNodeId: null,
  lastResult: null,
  workflowName: "",
  validateStatus: null,
  runtimeParams: {},

  setManifest: (m) => set({ manifest: m }),
  setNodes: (n) => set({ nodes: n }),
  setEdges: (e) => set({ edges: e }),

  applyNodeChanges: (changes) => {
    set((s) => ({ nodes: rfApplyNodeChanges(changes, s.nodes as any) as CanvasNode[] }));
  },
  applyEdgeChanges: (changes) => {
    set((s) => ({ edges: rfApplyEdgeChanges(changes, s.edges as any) as CanvasEdge[] }));
  },

  addBlock: (blockType, position) => {
    const manifest = get().manifest;
    if (!manifest) return;
    const block = manifest.blocks.find((b) => b.type === blockType);
    if (!block) return;
    const id = nextId("step");
    // step name: short slug of block display name, deduplicated
    const slug = (block.display_name || block.type)
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_|_$/g, "")
      .slice(0, 24) || "step";
    const existing = new Set(
      get().nodes.filter((n) => n.type === "block").map((n) => (n.data as BlockNodeData).stepName),
    );
    let stepName = slug;
    let i = 1;
    while (existing.has(stepName)) {
      i++;
      stepName = `${slug}_${i}`;
    }
    const newNode: CanvasNode = {
      id,
      type: "block",
      position,
      data: { blockType, stepName, params: {} },
    };
    set((s) => ({ nodes: [...s.nodes, newNode], selectedNodeId: id, validateStatus: null }));
  },

  addInput: (inputKind, name) => {
    const id = nextId("input");
    set((s) => ({
      nodes: [
        ...s.nodes,
        { id, type: "input", position: { x: 50, y: 50 + s.nodes.length * 30 }, data: { inputKind, name } },
      ],
      selectedNodeId: id,
      validateStatus: null,
    }));
  },

  addOutput: (name) => {
    const id = nextId("output");
    set((s) => ({
      nodes: [
        ...s.nodes,
        { id, type: "output", position: { x: 800, y: 50 + s.nodes.length * 30 }, data: { name } },
      ],
      selectedNodeId: id,
      validateStatus: null,
    }));
  },

  deleteNode: (id) =>
    set((s) => ({
      nodes: s.nodes.filter((n) => n.id !== id),
      edges: s.edges.filter((e) => e.source !== id && e.target !== id),
      selectedNodeId: s.selectedNodeId === id ? null : s.selectedNodeId,
      validateStatus: null,
    })),

  connect: (e) =>
    set((s) => {
      // Remove any existing edge targeting the same input handle (only one wire allowed)
      const filtered = s.edges.filter(
        (x) => !(x.target === e.target && x.targetHandle === e.targetHandle),
      );
      return {
        edges: [
          ...filtered,
          {
            id: `e_${e.source}_${e.sourceHandle}_${e.target}_${e.targetHandle}_${Date.now()}`,
            ...e,
          },
        ],
        validateStatus: null,
      };
    }),

  disconnect: (edgeId) =>
    set((s) => ({ edges: s.edges.filter((e) => e.id !== edgeId), validateStatus: null })),

  setParam: (nodeId, field, value) =>
    set((s) => ({
      nodes: s.nodes.map((n) => {
        if (n.id !== nodeId || n.type !== "block") return n;
        return {
          ...n,
          data: {
            ...n.data,
            params: { ...n.data.params, [field]: value },
          },
        };
      }),
      validateStatus: null,
    })),

  setStepName: (nodeId, newName) =>
    set((s) => ({
      nodes: s.nodes.map((n) =>
        n.id === nodeId && n.type === "block"
          ? { ...n, data: { ...n.data, stepName: newName } }
          : n,
      ),
      validateStatus: null,
    })),

  setInputName: (nodeId, newName) =>
    set((s) => ({
      nodes: s.nodes.map((n) =>
        n.id === nodeId && n.type === "input"
          ? { ...n, data: { ...n.data, name: newName } }
          : n,
      ),
      validateStatus: null,
    })),

  setInputDefault: (nodeId, value) =>
    set((s) => ({
      nodes: s.nodes.map((n) =>
        n.id === nodeId && n.type === "input"
          ? { ...n, data: { ...n.data, default_value: value } }
          : n,
      ),
      validateStatus: null,
    })),

  setOutputName: (nodeId, newName) =>
    set((s) => ({
      nodes: s.nodes.map((n) =>
        n.id === nodeId && n.type === "output"
          ? { ...n, data: { ...n.data, name: newName } }
          : n,
      ),
      validateStatus: null,
    })),

  setSelected: (id) => set({ selectedNodeId: id }),

  loadSpec: (spec, name) => {
    const m = get().manifest;
    if (!m) return;
    const { nodes, edges, warnings } = workflowJSONToCanvas(spec, m);
    if (warnings.length) {
      console.warn("[loadSpec] warnings:", warnings);
    }
    set({
      nodes: autoLayout(nodes, edges),
      edges,
      workflowName: name,
      selectedNodeId: null,
      validateStatus: null,
      lastResult: null,
    });
  },

  clear: () =>
    set({
      nodes: [],
      edges: [],
      selectedNodeId: null,
      lastResult: null,
      validateStatus: null,
      workflowName: "",
    }),

  setLastResult: (r) => set({ lastResult: r }),
  setWorkflowName: (n) => set({ workflowName: n }),
  setValidateStatus: (s) => set({ validateStatus: s }),
  setRuntimeParams: (p) => set({ runtimeParams: p }),

  getCurrentSpec: () => {
    const { nodes, edges, manifest } = get();
    if (!manifest) return { spec: { version: "1.0", inputs: [], steps: [], outputs: [] }, errors: ["manifest not loaded"] };
    return canvasToWorkflowJSON(nodes, edges, manifest);
  },
}));
