# Workflow Editor — Master Spec

A standalone visual workflow editor for the Roboflow Inference engine. Drag blocks onto a canvas, wire them with `$steps.X.Y` selectors, save to JSON, click **Run**, get results back.

> **Out of scope for v1:** auth, multi-user, real-time collab, RTSP/streaming, undo/redo, type-checking selector kinds in the UI, drag-to-create-input, mobile layout, workflow versioning, S3-backed projects. The success criterion is "drag, wire, save, run" on a single image. Everything else is gold-plating.

---

## 1. Directory layout

All paths below are relative to `examples/workflow-editor/`.

```
workflow-editor/
├── PLAN.md                          # this file
├── BUILDER_BRIEFS.md                # per-agent assignments
├── README.md                        # end-user how-to (Builder C)
├── .env.example                     # OPENROUTER_API_KEY=...  (Builder C)
├── .gitignore                       # node_modules, .venv, dist/  (Builder C)
├── start.sh                         # one-shot launcher (Builder C)
├── pyproject.toml                   # backend deps via uv (Builder C)
├── blocks_manifest.json             # ALREADY GENERATED — frontend reads this
├── _introspect_blocks.py            # regenerates blocks_manifest.json
├── prebuilt_workflows.json          # 5 starter workflows (planner ships)
├── backend/                         # Builder A
│   ├── __init__.py
│   ├── main.py                      # FastAPI app, also serves /frontend/dist
│   ├── engine_runner.py             # thin wrapper around real_engine_runner
│   ├── workflows_store.py           # save/load JSON files in saved_workflows/
│   ├── schemas.py                   # pydantic request/response models
│   └── routes/
│       ├── __init__.py
│       ├── blocks.py
│       ├── workflows.py
│       └── run.py
├── frontend/                        # Builder B
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── index.html
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── api.ts                   # all fetch() calls to the backend
│   │   ├── types.ts                 # mirror of backend schemas
│   │   ├── components/
│   │   │   ├── Palette.tsx          # left-rail block list
│   │   │   ├── Canvas.tsx           # ReactFlow canvas
│   │   │   ├── BlockNode.tsx        # custom node renderer
│   │   │   ├── Inspector.tsx        # right-rail per-step config
│   │   │   ├── Toolbar.tsx          # Load/Save/Run/Clear buttons
│   │   │   └── RunPanel.tsx         # image upload + results display
│   │   ├── lib/
│   │   │   ├── serialize.ts         # canvas → workflow JSON
│   │   │   ├── deserialize.ts       # workflow JSON → canvas
│   │   │   └── selectors.ts         # $inputs.X / $steps.Y.Z parsing
│   │   └── styles.css
│   └── dist/                        # build output, served by FastAPI
└── saved_workflows/                 # written by POST /api/workflows/save
    └── .gitkeep
```

---

## 2. Run model — single port

**Decision: single FastAPI process on port `7873`.** Builder B builds the React app to `frontend/dist/`; FastAPI serves it as static files at `/`. API lives under `/api/*`. Rationale:

- One process, one URL, no CORS to think about.
- The user runs `./start.sh` and opens `http://localhost:7873`.
- Vite dev server is available (`pnpm dev` on 5173) for frontend builders during development. The dev server proxies `/api` → `:7873`. But the *shipped* artifact is single-port.

Port choice: `7873` (the existing demo uses 7861; pick something out of the way).

---

## 3. Backend API surface

All endpoints return `application/json` unless noted. All `4xx` responses follow `{"error": "...", "detail": "..."}`. The engine is **shared, lazily-initialised, per-workflow**, exactly as `real_engine_runner.py` does it — engines are cached by a hash of the workflow JSON so reruns are fast.

### `GET /api/health`
Liveness check.
- **Response 200:** `{"status": "ok", "plugins_loaded": ["local_yolo_plugin", ...]}`

### `GET /api/blocks`
Returns the introspected block catalog.
- **Response 200:** the raw contents of `blocks_manifest.json` (regenerated on backend boot if older than the file's mtime on `_introspect_blocks.py`, otherwise served as-is).

### `GET /api/workflows/builtin`
Lists the five pre-built starter workflows.
- **Response 200:** `{"workflows": [{"id": "speed", "name": "Speed Estimation", "description": "..."}, ...]}`

### `GET /api/workflows/builtin/{id}`
Returns one pre-built workflow as a parsed JSON dict.
- **Path params:** `id ∈ {speed, smart, autoannotate, sam3_autoannotate, vlm_only}`
- **Response 200:** `{"id": "...", "name": "...", "workflow": {<the spec dict>}}`
- **404** if id unknown.

### `GET /api/workflows/saved`
Lists user-saved workflows (filenames in `saved_workflows/`).
- **Response 200:** `{"workflows": [{"name": "my-thing", "modified_at": "..."}]}`

### `GET /api/workflows/saved/{name}`
Returns a saved workflow.
- **Response 200:** `{"name": "...", "workflow": {<spec>}}`

### `POST /api/workflows/save`
Saves the given spec to `saved_workflows/{name}.json`. Overwrites without prompting.
- **Body:** `{"name": "my-thing", "workflow": {<spec dict>}}`
- **Response 200:** `{"name": "...", "path": "saved_workflows/my-thing.json"}`
- **400** if `name` is empty/has slashes/has `..`.

### `POST /api/workflows/validate`
Sanity-checks a spec by attempting `ExecutionEngine.init(workflow_definition=spec, ...)` — does **not** run it. Catches missing inputs, bad selectors, unknown block types.
- **Body:** `{"workflow": {<spec>}}`
- **Response 200 (valid):** `{"ok": true}`
- **Response 200 (invalid):** `{"ok": false, "error": "...", "context": "..."}` (we use 200 even for invalid because the editor wants to show the error, not a 4xx)

### `POST /api/workflows/run`
Runs the spec against a single uploaded image (or a path under `sample_images/`).
- **Body:** multipart/form-data with fields:
  - `workflow` (string, JSON-encoded spec)
  - `runtime_params` (string, JSON-encoded dict of extra runtime parameters — e.g. `{"conf": 0.3, "prompts": ["car"]}`. Optional, default `{}`.)
  - **One of**:
    - `image` (file upload, jpg/png) — becomes the WorkflowImage input
    - OR `image_path` (form string, relative to `examples/local-workflow-demo/sample_images/`)
- **Response 200:**
  ```json
  {
    "ok": true,
    "elapsed_ms": 142,
    "outputs": [
      {
        "annotated": "<base64 PNG, if the workflow has an output of kind=image>",
        "detections": {"xyxy": [...], "class_id": [...], "confidence": [...], "class_name": [...], "tracker_id": [...]},
        "raw_text": "...",
        "_meta": {"shape": [720, 1280, 3]}
      }
    ]
  }
  ```
- **Response 200 (failure):** `{"ok": false, "error": "...", "elapsed_ms": 0}`
- **413** if upload > 10 MB.

Implementation note for `/run`:
- Use `engine_runner.get_or_init_engine(workflow_spec)` — internally hash the JSON, cache by hash.
- Wrap the image with `wrap_frame()` from `real_engine_runner`.
- Pass `{"image": [wrapped], **runtime_params}` to `engine.run(...)`.
- For each output, detect type and serialize:
  - `WorkflowImageData` → base64-encoded PNG (`numpy_image` → cv2.imencode `.png`).
  - `sv.Detections` → dict with `xyxy`/`class_id`/`confidence`/`class_name`/`tracker_id` lists.
  - Plain strings/numbers/lists pass through.
  - Unknown → `str(value)`.

---

## 4. Frontend stack

**Decision: React 18 + ReactFlow + Vite + TypeScript.**

Pinned majors (latest stable in each):
```jsonc
{
  "dependencies": {
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "reactflow": "^11.11.0",
    "zustand": "^4.5.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.0",
    "typescript": "^5.4.0",
    "vite": "^5.3.0"
  }
}
```

Why these:
- **ReactFlow** is the canonical drag-wire-canvas lib. Custom nodes via `nodeTypes={{block: BlockNode}}`. Edges represent selector wires.
- **Zustand** for the canvas store (`nodes`, `edges`, `selectedNodeId`, dispatch functions). Lighter than Redux, simpler than Context for a single store.
- **No** Tailwind, no UI kit — keep it small. Plain CSS in `styles.css`.

### Component tree

```
<App>
  <Toolbar />                                # top bar: Load/Save/Run/Clear, workflow name
  <main>
    <Palette blocks={manifest.blocks} />     # left rail (200px, scrollable)
    <Canvas />                               # ReactFlow filling the centre
    <Inspector selectedNode={...} />         # right rail (320px)
  </main>
  <RunPanel />                               # bottom drawer: image upload + last result
</App>
```

### Props / data flow

- `Palette` renders categories from `blocks_manifest.json`. Each block tile is `draggable` with `dataTransfer.setData("application/x-block-type", block.type)`.
- `Canvas` listens to `onDrop`, creates a node with `data: {blockType, params: {}}`. Wires (`edges`) are created by dragging from a node's source handle (one per output) to a target handle (one per input).
- `BlockNode` (custom node) renders the block name + a list of inputs (target handles, left side) + outputs (source handles, right side). Handles use `id={`in:${inputName}`}` / `id={`out:${outputName}`}` so the serializer can map them to selectors.
- `Inspector` shows the selected node's manifest entry and lets the user edit `params` for each non-selector-connected input (text field, number, dropdown for enums, JSON textarea for list/dict).

---

## 5. Data model — canvas ↔ workflow JSON

This is the most important section. The serializer/deserializer round-trip must be lossless.

### Canvas node shape (React state)
```ts
type CanvasNode = {
  id: string;                  // ReactFlow node id, e.g. "node_1"
  type: "block";               // custom node type
  position: {x: number; y: number};
  data: {
    blockType: string;         // e.g. "roboflow_core/trackers_bytetrack@v1"
    stepName: string;          // e.g. "track" — becomes the JSON `name` field
    params: Record<string, unknown>;   // literal values for inputs NOT wired to another node
  };
};
```

There's also a special non-block node for `WorkflowImage` and `WorkflowParameter` inputs:
```ts
type InputNode = {
  id: string;
  type: "input";
  data: {
    inputKind: "WorkflowImage" | "WorkflowParameter";
    name: string;                  // e.g. "image", "conf"
    default_value?: unknown;
  };
};
```
And one for outputs:
```ts
type OutputNode = {
  id: string;
  type: "output";
  data: {
    name: string;                  // e.g. "annotated"
    // selector is computed from the incoming edge
  };
};
```

### Canvas edge shape
```ts
type CanvasEdge = {
  id: string;
  source: string;                  // node id
  sourceHandle: string;            // "out:predictions"  (or "out:value" for input nodes)
  target: string;                  // node id
  targetHandle: string;            // "in:detections"
};
```

### Serialization (`lib/serialize.ts`)

Algorithm:
1. **Inputs**: collect every `InputNode`, emit `{"type": inputKind, "name": data.name, "default_value": data.default_value?}`.
2. **Steps**: for each `CanvasNode` (in topological order — see step 5):
   - Start with `{"type": data.blockType, "name": data.stepName}`.
   - For each declared input field on the block (look up from manifest by `blockType`):
     - If there is an incoming edge `e` whose `targetHandle === "in:" + inputName`:
       - If `e.source` is an `InputNode` → write `"$inputs." + sourceNode.data.name`.
       - Else (it's another block) → write `"$steps." + sourceNode.data.stepName + "." + sourceHandle.replace("out:", "")`.
     - Else if `data.params[inputName]` is set → write the literal.
     - Else if the input is `required: false` → omit the key.
     - Else → serialization error (collect, show in UI before save).
3. **Outputs**: for each `OutputNode`, find the single incoming edge; emit `{"type": "JsonField", "name": data.name, "selector": "$steps." + sourceStep.stepName + "." + outputName}`.
4. **Top-level**: `{"version": "1.0", "inputs": [...], "steps": [...], "outputs": [...]}`.
5. **Topological sort**: walk the edge graph from input nodes; emit each block after all its predecessors. If a cycle is detected, raise a serialization error ("workflow has a cycle").

### Deserialization (`lib/deserialize.ts`)

Inverse of the above:
1. For each `input` entry, create an `InputNode` (auto-position in a column on the left).
2. For each `step`, create a `CanvasNode` with `stepName=step.name`, `blockType=step.type`, and `params = {field: value for field, value in step.items() if not is_selector(value)}`.
3. For each step field whose value is a selector string `$inputs.X` or `$steps.X.Y`:
   - Resolve to the source node, add an edge with the correct handle ids.
4. For each `output` entry, create an `OutputNode` and add the incoming edge.
5. Layout: simple grid auto-layout based on topological depth (so a freshly-loaded workflow looks readable). Real users can drag them around afterwards.

### Selector wiring rules

A block input's value in the JSON is one of:
- A literal (string, number, list, dict, null) — from `Inspector` text input.
- `$inputs.X` — when the input handle is connected to an `InputNode` whose `data.name === "X"`.
- `$steps.X.Y` — when the input handle is connected to a block node `X` (its `stepName`) on output handle `Y`.

UI constraint for v1: **any input that has an incoming edge is hidden in the Inspector** (because its value is determined by the wire). The user disconnects the wire to type a literal again.

---

## 6. Block-manifest extraction

The catalog the frontend uses lives at `blocks_manifest.json`. It is generated by `_introspect_blocks.py` which:

1. Mirrors `real_engine_runner.py`'s sys.path + `WORKFLOWS_PLUGINS` setup so our three custom plugins (`local_yolo_plugin`, `yolo_world_plugin`, `sam3_plugin`) are visible.
2. Calls `describe_available_blocks(dynamic_blocks=[])` from `inference.core.workflows.execution_engine.introspection.blocks_loader` — this is the same call the Roboflow UI uses internally.
3. For each block in the curated allowlist (24 entries — see the script for the exact list), captures:
   - `type` (manifest_type_identifier), `aliases`, `display_name`, `category`, `block_source`
   - `short_description`, `long_description`
   - `inputs`: for each manifest field, derives `{name, required, description, default, accepts_selector, selector_kinds, primitive_annotation, primitive_kinds, enum?, is_list_element, is_dict_element}` by walking the Pydantic JSON schema AND `parse_block_manifest(...)`'s selector map.
   - `outputs`: from `BlockDescription.outputs_manifest` (each `{name, kinds[]}`).
   - `accepts_batch_input`, `parameters_accepting_batches`.
4. Writes `blocks_manifest.json` with the 24 chosen blocks, sorted by category then name.

**Backend boot regenerates this file if `_introspect_blocks.py` is newer than the JSON** (avoid stale catalogs after a plugin edit).

The 24 chosen blocks (the curated subset of 197 available):

| Category | Block type |
|---|---|
| model | `local_models/ultralytics_yolo@v1` |
| model | `local_models/yolo_world@v1` |
| model | `local_models/sam3@v1` |
| model | `triton/yolo@v1` |
| model | `roboflow_core/openai_compatible@v1` |
| transformation | `roboflow_core/byte_tracker@v1` |
| transformation | `roboflow_core/trackers_bytetrack@v1` |
| transformation | `roboflow_core/detections_filter@v1` |
| transformation | `roboflow_core/dynamic_crop@v1` |
| analytics | `roboflow_core/velocity@v1` |
| analytics | `roboflow_core/time_in_zone@v2` |
| analytics | `roboflow_core/line_counter@v2` |
| visualization | `roboflow_core/bounding_box_visualization@v1` |
| visualization | `roboflow_core/label_visualization@v1` |
| visualization | `roboflow_core/polygon_zone_visualization@v1` |
| visualization | `roboflow_core/mask_visualization@v1` |
| visualization | `roboflow_core/line_counter_visualization@v1` |
| fusion | `roboflow_core/detections_classes_replacement@v1` |
| formatter | `roboflow_core/expression@v1` |
| formatter | `roboflow_core/property_definition@v1` |
| formatter | `roboflow_core/first_non_empty_or_default@v1` |
| formatter | `roboflow_core/csv_formatter@v1` |
| flow_control | `roboflow_core/continue_if@v1` |
| sink | `roboflow_core/webhook_sink@v1` |

---

## 7. Pre-built workflow seeds

`backend/engine_runner.py` exposes the 5 factories from `real_engine_runner.py`:
- `speed` → `make_speed_workflow("pytorch")`
- `smart` → `make_smart_workflow("pytorch")`
- `autoannotate` → `make_autoannotate_workflow()`
- `sam3_autoannotate` → `make_sam3_autoannotate_workflow()` (404 if SAM3 not installed)
- `vlm_only` → `make_vlm_only_workflow()`

Static metadata (id → display name + description) lives in `prebuilt_workflows.json` so the frontend can render the load dialog without calling Python.

---

## 8. Reused code from `examples/local-workflow-demo/`

Backend imports (no copies, just `sys.path.insert` to the sibling dir):
- `real_engine_runner` — for `_engine_for`, `wrap_frame`, `_make_init_parameters`, all the `make_*_workflow` factories, and the persistent `ThreadPoolExecutor`.
- The three plugin modules (`local_yolo_plugin`, `yolo_world_plugin`, `sam3_plugin`) — auto-discovered via `WORKFLOWS_PLUGINS`.

The editor must NOT modify any file under `local-workflow-demo/`.

---

## 9. Smoke test

After all three builders ship:
1. `cd examples/workflow-editor && ./start.sh`
2. Browser: `http://localhost:7873`.
3. Click **Load → Speed Estimation**. Canvas shows the 5-step graph wired up.
4. Click **Run**. Upload `sample_images/cars.jpg` (from the demo). Result shows annotated image + a detections table.
5. Drag a `Detections Filter` block from the palette onto the canvas. Wire its `predictions` input to the YOLO block's `predictions` output. Wire it to ByteTracker's `detections` input. Click **Save as `speed-filtered`**. Confirm a new file `saved_workflows/speed-filtered.json` exists with the filter step inserted.
6. Click **Run** again. Result reflects the filter.

If steps 3–6 all work, the editor is done for v1.
