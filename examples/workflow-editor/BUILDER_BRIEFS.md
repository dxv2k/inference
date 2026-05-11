# Builder Briefs

Three independent builder agents. Each brief is self-contained — assume your builder does not see the other briefs or the master PLAN.md. (They can read it, but it's not required.)

All file paths are relative to `examples/workflow-editor/`.

Shared constraint: **DO NOT touch any file under `examples/local-workflow-demo/`**. You may import from it via `sys.path`, but never edit it.

---

## Builder A — Backend (FastAPI + engine runner)

### What you own

Everything under `backend/`, plus the launch wiring in `start.sh` for the backend half.

### Files to create

| File | Purpose |
|---|---|
| `backend/__init__.py` | Empty package marker. |
| `backend/main.py` | FastAPI app object. Mounts `/api/*` routers. Serves `frontend/dist/` as static files at `/`. Runs `_introspect_blocks.py` at startup if `blocks_manifest.json` is missing or older than the script. |
| `backend/schemas.py` | Pydantic models for the request/response shapes below. |
| `backend/engine_runner.py` | Imports `real_engine_runner` from `../local-workflow-demo/` (via `sys.path.insert`). Exposes: `get_engine_for_spec(spec: dict) -> ExecutionEngine` (hash-cached), `wrap_image_for_engine(rgb: np.ndarray) -> WorkflowImageData`, `serialize_output(value) -> jsonable`, `load_prebuilt(id: str) -> dict`. |
| `backend/workflows_store.py` | `list_saved() -> list[dict]`, `load_saved(name) -> dict`, `save(name, spec) -> Path`. Hard-rejects `name` if it contains `/`, `\`, or `..`. |
| `backend/routes/__init__.py` | Empty. |
| `backend/routes/blocks.py` | `GET /api/blocks` → returns `blocks_manifest.json`. `GET /api/health`. |
| `backend/routes/workflows.py` | `GET /api/workflows/builtin`, `GET /api/workflows/builtin/{id}`, `GET /api/workflows/saved`, `GET /api/workflows/saved/{name}`, `POST /api/workflows/save`, `POST /api/workflows/validate`. |
| `backend/routes/run.py` | `POST /api/workflows/run` (multipart). |

### Endpoint contracts (exact)

**`GET /api/health`** →
```json
{"status": "ok", "plugins_loaded": ["local_yolo_plugin", "yolo_world_plugin", ...]}
```

**`GET /api/blocks`** → contents of `blocks_manifest.json` verbatim.

**`GET /api/workflows/builtin`** → contents of `prebuilt_workflows.json` verbatim.

**`GET /api/workflows/builtin/{id}`** →
```json
{"id": "speed", "name": "Speed Estimation", "workflow": {<spec dict from factory>}, "default_runtime_params": {...}}
```
- 404 if id unknown.
- Call the factory function named in `prebuilt_workflows.json[id].factory` on the `real_engine_runner` module to get the dict.

**`GET /api/workflows/saved`** →
```json
{"workflows": [{"name": "my-thing", "modified_at": "2026-05-11T12:00:00Z"}]}
```

**`GET /api/workflows/saved/{name}`** →
```json
{"name": "my-thing", "workflow": {<spec>}}
```
- 404 if not found.

**`POST /api/workflows/save`** body:
```json
{"name": "my-thing", "workflow": {<spec>}}
```
Response:
```json
{"name": "my-thing", "path": "saved_workflows/my-thing.json"}
```
- 400 if `name` is empty / has path traversal characters.

**`POST /api/workflows/validate`** body:
```json
{"workflow": {<spec>}}
```
Response (always 200):
```json
{"ok": true}
```
or
```json
{"ok": false, "error": "human-readable msg", "context": "workflow_compilation"}
```
- Implementation: try `ExecutionEngine.init(workflow_definition=spec, init_parameters=_make_init_parameters())`. Catch `WorkflowSyntaxError`, `WorkflowDefinitionError`, etc. — all are subclasses of Roboflow's `WorkflowError` family. Don't actually run it.

**`POST /api/workflows/run`** multipart:
- `workflow`: form field, JSON-encoded spec string
- `runtime_params`: form field, JSON-encoded dict string (optional, default `"{}"`)
- `image`: file upload (jpg/png), OR `image_path`: form string (relative to `../local-workflow-demo/sample_images/`)

Response (always 200):
```json
{
  "ok": true,
  "elapsed_ms": 142,
  "outputs": [
    {
      "annotated": "<base64 PNG>",
      "detections": {
        "xyxy": [[10, 20, 100, 200]],
        "class_id": [0],
        "confidence": [0.92],
        "class_name": ["person"],
        "tracker_id": [1]
      }
    }
  ]
}
```
or `{"ok": false, "error": "...", "elapsed_ms": 0}` on failure.

### How to access the ExecutionEngine

```python
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
DEMO = HERE.parent.parent / "local-workflow-demo"
sys.path.insert(0, str(DEMO))

# Must set WORKFLOWS_PLUGINS BEFORE importing inference. real_engine_runner
# already does this at module-import time, so just import it first:
import real_engine_runner as rer

# Reuse rer._make_init_parameters() and rer._PERSISTENT_EXECUTOR.
# For ad-hoc engines (user-built spec), build your own cache:
import hashlib, json
from inference.core.workflows.execution_engine.core import ExecutionEngine

_AD_HOC_ENGINES: dict[str, ExecutionEngine] = {}

def get_engine_for_spec(spec: dict) -> ExecutionEngine:
    key = hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()
    if key not in _AD_HOC_ENGINES:
        _AD_HOC_ENGINES[key] = ExecutionEngine.init(
            workflow_definition=spec,
            init_parameters=rer._make_init_parameters(),
            max_concurrent_steps=1,
            executor=rer._PERSISTENT_EXECUTOR,
        )
    return _AD_HOC_ENGINES[key]
```

### Output serialization helper

For each value in `engine.run(...)[0]`:
- `WorkflowImageData` (or has `.numpy_image`) → cv2.imencode(".png", BGR-converted np.ndarray) → base64.
- `sv.Detections` → dict with keys `xyxy` (list of [x1,y1,x2,y2]), `class_id`, `confidence`, `class_name` (from `detections.data["class_name"]` if present), `tracker_id` (from `detections.data["tracker_id"]` if present).
- np.ndarray → list (via `.tolist()`).
- Other → if JSON-serializable, pass through; else `str(value)`.

### What NOT to do

- Do not stream / SSE / websockets — request/response only.
- Do not implement auth.
- Do not write an OpenAPI schema editor.
- Do not add CORS — same-origin works because FastAPI serves the frontend too.
- Do not change the port from 7873.
- Do not edit anything under `examples/local-workflow-demo/`.

### Launch command

In `backend/main.py`:
```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=7873, reload=False)
```

### Smoke test (run these before reporting done)

```bash
cd examples/workflow-editor
uv run python -m backend.main &
sleep 3
curl -s localhost:7873/api/health | jq
curl -s localhost:7873/api/blocks | jq '.blocks | length'      # expect 24
curl -s localhost:7873/api/workflows/builtin | jq '.workflows | length'  # expect 5
curl -s localhost:7873/api/workflows/builtin/speed | jq '.workflow.steps | length'  # expect 5
curl -s -X POST localhost:7873/api/workflows/validate \
  -H 'content-type: application/json' \
  -d '{"workflow": {"version":"1.0","inputs":[{"type":"WorkflowImage","name":"image"}],"steps":[],"outputs":[]}}' | jq
# Run a real workflow:
curl -s -X POST localhost:7873/api/workflows/run \
  -F "workflow=$(curl -s localhost:7873/api/workflows/builtin/speed | jq -c .workflow)" \
  -F 'runtime_params={"conf":0.3,"pixels_per_meter":12.5,"weights":"yolov8n.pt","device":"cuda"}' \
  -F "image=@../local-workflow-demo/sample_images/$(ls ../local-workflow-demo/sample_images | head -1)" | jq '.ok, .elapsed_ms'
```
Last call must return `true` and a millisecond count.

---

## Builder B — Frontend (React + ReactFlow)

### What you own

Everything under `frontend/`.

### Files to create

| File | Purpose |
|---|---|
| `frontend/package.json` | Deps + scripts. |
| `frontend/vite.config.ts` | Dev server proxies `/api` → `http://localhost:7873`. Build output goes to `dist/`. |
| `frontend/tsconfig.json` | Strict TS, React JSX. |
| `frontend/index.html` | Root HTML, mounts `<div id="root">`. |
| `frontend/src/main.tsx` | ReactDOM root. |
| `frontend/src/App.tsx` | Layout shell (Toolbar / Palette / Canvas / Inspector / RunPanel). |
| `frontend/src/types.ts` | Mirror of the backend's response shapes + canvas types (see PLAN §5). |
| `frontend/src/api.ts` | All `fetch()` helpers: `getBlocks()`, `getBuiltinList()`, `getBuiltin(id)`, `getSaved()`, `loadSaved(name)`, `save(name, spec)`, `validate(spec)`, `run(spec, runtimeParams, imageFile)`. |
| `frontend/src/store.ts` | Zustand store: `{nodes, edges, manifest, selectedNodeId, lastResult, …}` + dispatchers (`addBlock`, `deleteNode`, `connect`, `setParam`, `loadSpec`, `clear`, `setSelected`). |
| `frontend/src/components/Palette.tsx` | Left rail. Groups blocks by `category`. Each tile is `<div draggable onDragStart=...>`. |
| `frontend/src/components/Canvas.tsx` | ReactFlow component. Handles `onDrop` for new blocks. Wires `nodeTypes={{block: BlockNode, input: InputNode, output: OutputNode}}`. |
| `frontend/src/components/BlockNode.tsx` | Custom node renderer (manifest-driven). Renders `<Handle type="target" id={"in:"+name}>` per input and `<Handle type="source" id={"out:"+name}>` per output. |
| `frontend/src/components/InputNode.tsx` | A node representing a WorkflowImage / WorkflowParameter input. Has a single output handle `out:value`. |
| `frontend/src/components/OutputNode.tsx` | A node representing a JsonField output. Has a single input handle `in:value`. |
| `frontend/src/components/Inspector.tsx` | Right rail. Reads `selectedNode` from store; for each input field NOT currently wired (no edge to its handle), renders an editor (text/number/textarea/select-for-enum). |
| `frontend/src/components/Toolbar.tsx` | Buttons: `Load` (dropdown of builtin + saved), `Save` (prompts for name), `Validate`, `Run` (toggles RunPanel), `Clear`. |
| `frontend/src/components/RunPanel.tsx` | Bottom drawer. File input for image, run button, displays returned base64 images + a `<details>` of the JSON response. |
| `frontend/src/lib/serialize.ts` | `canvasToWorkflowJSON(nodes, edges, manifest): {spec: dict, errors: string[]}` — implements PLAN §5 serialization. |
| `frontend/src/lib/deserialize.ts` | `workflowJSONToCanvas(spec, manifest): {nodes, edges}` — implements PLAN §5 deserialization with auto-layout. |
| `frontend/src/lib/selectors.ts` | `isSelector(v)`, `parseSelector(s)` (returns `{kind: "input"|"step", name, field?}`), `formatStepSelector(stepName, field)`, `formatInputSelector(name)`. |
| `frontend/src/lib/layout.ts` | Topological-depth-based grid layout. Column = depth, row = within-column index. |
| `frontend/src/styles.css` | Plain CSS, no framework. |

### Dependencies (pinned majors)

```json
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

`npm` or `pnpm` is fine — pick one and document in README (Builder C).

### Backend API contract

Read PLAN.md §3. The frontend should expose those calls verbatim in `src/api.ts`. The base URL is `""` (empty — same origin in production; vite proxy in dev).

### Serialization (re-read PLAN §5 — this is the load-bearing part)

The serializer/deserializer round-trip must be lossless for all 5 builtin workflows. Test by loading each builtin and immediately comparing the re-serialized JSON to the original (with normalised key ordering). If they differ, fix the serializer.

Key edge cases:
- Some inputs are `WorkflowImageSelector` (image-only) vs general `Selector` (any kind). The manifest's `selector_kinds` tells you what's allowed. For v1 don't enforce kinds in the UI; the backend's `/validate` will catch errors.
- `keep_classes` in `make_smart_workflow` is wired as `$inputs.keep_classes`, not a literal — make sure the deserializer creates the InputNode and the edge.
- The `vlm_only` workflow uses a literal `system_prompt` (long string) and a literal `prompt` (jinja-ish template). Keep them as literals in `params` — they don't become edges.

### What NOT to do

- Do not add a graph minimap (it'd be cute, but it's gold-plating).
- Do not add undo/redo.
- Do not add real-time collaborative editing.
- Do not implement type-checking of selector kinds — the backend's validate endpoint does it.
- Do not bundle a UI framework (no MUI / Chakra / Tailwind). Plain CSS, ~300 lines max.

### Smoke test

```bash
cd examples/workflow-editor/frontend
pnpm install     # or npm install
pnpm dev         # starts vite on 5173, proxies /api to :7873
# Open localhost:5173, verify the palette shows 24 blocks grouped by 8 categories.
# Verify Load → Speed Estimation populates the canvas with 5 blocks + edges.
# Verify Save creates a file (curl the backend to confirm).
# Verify Run with a sample image returns an annotated PNG in <RunPanel/>.
pnpm build       # produces dist/
ls dist/         # index.html + assets/
```

---

## Builder C — Glue, DevOps, docs

### What you own

The cross-cutting wiring: README, launcher, Python project file, sample env, .gitignore. You also produce the `start.sh` that builds the frontend and starts the backend.

### Files to create

| File | Purpose |
|---|---|
| `pyproject.toml` | uv project. Deps: `fastapi`, `uvicorn[standard]`, `python-multipart`, `pydantic>=2`, `numpy`, `opencv-python`, `supervision`, `ultralytics`, `python-dotenv`. Also `inference[examples]` from the parent repo (`{path = "../..", editable = true}`). Match Python version (`>=3.10`). |
| `.env.example` | `OPENROUTER_API_KEY=...` plus `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1`. |
| `.gitignore` | `__pycache__/`, `.venv/`, `node_modules/`, `frontend/dist/`, `saved_workflows/*.json` (but **keep** `saved_workflows/.gitkeep`), `.env`. |
| `start.sh` | Bash script. See below. |
| `README.md` | User-facing how-to. Must include: prerequisites, install steps, launch command, where the editor lives, brief description of the 5 builtin workflows, and known limitations. Keep under 150 lines. |
| `saved_workflows/.gitkeep` | Empty file. |

### `start.sh` template

```bash
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# 1. Frontend build (skip if dist/ exists and is newer than src/)
if [ ! -d frontend/dist ] || [ "$(find frontend/src -newer frontend/dist 2>/dev/null | head -1)" ]; then
  echo "[start.sh] building frontend..."
  (cd frontend && (command -v pnpm >/dev/null && pnpm install --frozen-lockfile && pnpm build) \
                 || (npm install && npm run build))
fi

# 2. Regenerate blocks_manifest.json if introspect script is newer
if [ _introspect_blocks.py -nt blocks_manifest.json ]; then
  echo "[start.sh] regenerating blocks_manifest.json..."
  uv run python _introspect_blocks.py
fi

# 3. Backend
echo "[start.sh] starting backend on :7873..."
exec uv run python -m backend.main
```

Make it executable (`chmod +x start.sh`).

### `pyproject.toml` skeleton

```toml
[project]
name = "workflow-editor"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "fastapi>=0.111",
  "uvicorn[standard]>=0.30",
  "python-multipart>=0.0.9",
  "pydantic>=2.7",
  "python-dotenv>=1.0",
  "numpy>=1.26",
  "opencv-python>=4.9",
  "supervision>=0.21",
  "ultralytics>=8.2",
  "inference[examples]",   # from parent repo, see [tool.uv.sources]
]

[tool.uv.sources]
inference = {path = "../..", editable = true}

[tool.uv]
package = false
```

### README structure

```
# Workflow Editor

Visual editor for Roboflow Inference workflows. Drag, wire, save, run — locally, no API key needed.

## Quick start
   ./start.sh
   open http://localhost:7873

## What's inside
- 24 pre-introspected blocks (palette)
- 5 builtin workflows (speed / smart / autoannotate / sam3 / vlm)
- Save/Load to saved_workflows/*.json

## Environment
   cp .env.example .env
   # put OPENROUTER_API_KEY if you want to use the VLM block

## Known limitations (v1)
- single-image runs only (no video / streaming)
- no auth
- selector kinds not type-checked in UI (backend validates)
```

### What NOT to do

- Do not write any of the React or FastAPI code yourself — those are owned by Builders A and B.
- Do not bake in absolute paths. Use `$HERE`-relative paths in scripts.
- Do not include `node_modules/` or `frontend/dist/` in git.

### Smoke test

```bash
cd examples/workflow-editor
./start.sh   # should build frontend (or skip), regen manifest (or skip), then run backend
# In another shell:
curl -s localhost:7873/api/health | jq .status   # expect "ok"
curl -s localhost:7873/ | head -3                # expect HTML (frontend dist served)
```

---

## Coordination between builders

- **Manifest file**: Builders A and B both read `blocks_manifest.json`. Builder A serves it via `GET /api/blocks`. Builder B fetches from there at app startup. Neither builder should write to the file (the introspect script + start.sh own it).
- **Schemas**: Builder A defines them in Python (`backend/schemas.py`); Builder B mirrors them in TypeScript (`frontend/src/types.ts`). If a mismatch is found mid-build, the canonical source is PLAN.md §3.
- **Port 7873**: hard-coded in three places — `backend/main.py`, `frontend/vite.config.ts` (proxy target), `start.sh` (echo). Do not change.
- **Frontend dist path**: FastAPI in `backend/main.py` mounts `StaticFiles(directory="frontend/dist", html=True)` at `/`. The dist must exist before backend boots (Builder C's `start.sh` ensures this).
