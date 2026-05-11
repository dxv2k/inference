// Mirror of backend response shapes (PLAN.md §3) + canvas types (PLAN.md §5).

// ---- Manifest ----

export type ManifestInput = {
  name: string;
  required: boolean;
  description?: string;
  default?: unknown;
  accepts_selector: boolean;
  selector_kinds: string[];
  is_list_element?: boolean;
  is_dict_element?: boolean;
  primitive_annotation?: string;
  primitive_kinds: string[];
  enum?: string[];
};

export type ManifestOutput = {
  name: string;
  kinds: string[];
};

export type ManifestBlock = {
  type: string;
  aliases: string[];
  display_name: string;
  category: string;
  block_source: string;
  short_description: string;
  long_description?: string;
  inputs: ManifestInput[];
  outputs: ManifestOutput[];
  execution_engine_compatibility?: string;
  accepts_batch_input?: boolean;
  parameters_accepting_batches?: string[];
};

export type Manifest = {
  generated_at: string;
  total_available: number;
  blocks: ManifestBlock[];
};

// ---- Workflow JSON spec (Roboflow Inference) ----

export type WorkflowInput = {
  type: "WorkflowImage" | "WorkflowParameter" | string;
  name: string;
  default_value?: unknown;
};

export type WorkflowStep = {
  type: string;
  name: string;
  [field: string]: unknown;
};

export type WorkflowOutput = {
  type: "JsonField" | string;
  name: string;
  selector: string;
};

export type WorkflowSpec = {
  version: string;
  inputs: WorkflowInput[];
  steps: WorkflowStep[];
  outputs: WorkflowOutput[];
};

// ---- Builtin / saved listings ----

export type BuiltinEntry = {
  id: string;
  name: string;
  description?: string;
  default_runtime_params?: Record<string, unknown>;
  requires?: string;
  requires_env?: string[];
};

export type SavedEntry = {
  name: string;
  modified_at: string;
};

// ---- Validate / run responses ----

export type ValidateResponse =
  | { ok: true }
  | { ok: false; error: string; context?: string };

export type RunOutputItem = {
  annotated?: string;
  detections?: {
    xyxy?: number[][];
    class_id?: number[];
    confidence?: number[];
    class_name?: string[];
    tracker_id?: number[];
  };
  raw_text?: string;
  [k: string]: unknown;
};

export type RunResponse =
  | { ok: true; elapsed_ms: number; outputs: RunOutputItem[] }
  | { ok: false; error: string; elapsed_ms: number };

// ---- Canvas node + edge shapes ----

export type BlockNodeData = {
  blockType: string;
  stepName: string;
  params: Record<string, unknown>;
};

export type InputNodeData = {
  inputKind: "WorkflowImage" | "WorkflowParameter";
  name: string;
  default_value?: unknown;
};

export type OutputNodeData = {
  name: string;
};

export type CanvasNode =
  | { id: string; type: "block"; position: { x: number; y: number }; data: BlockNodeData }
  | { id: string; type: "input"; position: { x: number; y: number }; data: InputNodeData }
  | { id: string; type: "output"; position: { x: number; y: number }; data: OutputNodeData };

export type CanvasEdge = {
  id: string;
  source: string;
  sourceHandle: string;
  target: string;
  targetHandle: string;
};
