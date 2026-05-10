# Where this demo touches the Roboflow workflow engine

The demo's integration with `inference.core.workflows.*` lives in **three files**, totaling
~400 lines. Everything else (`app.py`, `workflow.py`, the RTSP simulator) is UI / glue / a
pedagogical reference engine.

## The three integration points

```
┌─ app.py (Gradio UI) ──────────────────────────────────────────────┐
│   Tabs 4, 5, 6 → call into ↓                                      │
└────────────────┬──────────────────────────────────────────────────┘
                 │
┌─ real_engine_runner.py ─────────┐    ┌─ rtsp_runner.py ──────────┐
│  • set WORKFLOWS_PLUGINS env    │    │  • set WORKFLOWS_PLUGINS  │
│  • workflow JSON specs          │    │  • workflow JSON spec     │
│  • ExecutionEngine.init(...)    │    │  • InferencePipeline      │
│  • engine.run(...) per frame    │    │      .init_with_workflow  │
└──────────────┬──────────────────┘    └──────────┬────────────────┘
               │                                  │
               │   reads/calls                    │   reads/calls
               ▼                                  ▼
┌─ inference.core.workflows.execution_engine ─────────────────────────┐
│   • ExecutionEngine.init()                                          │
│   • engine.run(runtime_parameters)                                  │
│   • discovers plugin via WORKFLOWS_PLUGINS=local_yolo_plugin   ←────┐│
└─────────────────────────────────────────────────────────────────────┘│
                                                                       │
┌─ local_yolo_plugin.py ────────────────────────────────────────────┐  │
│   • inherits WorkflowBlock + WorkflowBlockManifest                │──┘
│   • load_blocks() returns [LocalYoloBlockV1]                      │
│   • run(image: WorkflowImageData) → {"predictions": sv.Detections}│
└───────────────────────────────────────────────────────────────────┘
```

## The whole bridge in three lines

```python
# 1. Plugin discovery hook — must be set before importing inference
os.environ["WORKFLOWS_PLUGINS"] = "local_yolo_plugin"

# 2. Build engine from a JSON workflow spec
engine = ExecutionEngine.init(workflow_definition=SPEC, init_parameters=...)

# 3. Run it per frame
result = engine.run(runtime_parameters={"image": [WorkflowImageData(...)]})
```

That's it. Everything else is plumbing.

---

## File 1 — `local_yolo_plugin.py` (custom block)

The plugin that lets the real engine use a locally-loaded Ultralytics YOLO model
instead of going through Roboflow's model registry. ~80 lines.

| Lines | What |
|---|---|
| `36-41` | imports from `inference.core.workflows.*` |
| `67-92` | `LocalYoloManifest` — Pydantic spec, declares `type: "local_models/ultralytics_yolo@v1"` |
| `101-156` | `LocalYoloBlockV1.run()` — takes `WorkflowImageData`, returns `{"predictions": sv.Detections}` |
| `158-159` | `load_blocks()` — entry point the engine calls when `WORKFLOWS_PLUGINS` env var includes this module |

## File 2 — `real_engine_runner.py` (drives the engine)

| Lines | What |
|---|---|
| `25` | `os.environ.setdefault("WORKFLOWS_PLUGINS", "local_yolo_plugin")` ← **the one line** that hooks the plugin in |
| `34` | `from inference.core.workflows.execution_engine.core import ExecutionEngine` |
| `35-40` | imports `WorkflowImageData`, `VideoMetadata`, `ImageParentMetadata`, `OriginCoordinatesSystem` |
| `43-93` | `REAL_SPEED_WORKFLOW` — speed-estimation JSON spec (5 steps) |
| `96-167` | `SMART_CAMERA_WORKFLOW` — surveillance JSON spec (6 steps) |
| `175-180` | `ExecutionEngine.init(workflow_definition=REAL_SPEED_WORKFLOW, init_parameters=...)` ← **the actual engine handoff** (used by Tab 4) |
| `184-189` | `init_smart_engine()` — same call with `SMART_CAMERA_WORKFLOW` (Tab 5) |
| `192-211` | `wrap_frame()` — turns a numpy frame into `WorkflowImageData` with `VideoMetadata` (so ByteTracker / Velocity see fps + frame_number) |
| `218-243` | `run_engine_on_frame()` — calls `engine.run(runtime_parameters=...)` once per frame |

## File 3 — `rtsp_runner.py` (production RTSP path)

Different primitive — `InferencePipeline` wraps the engine in a long-running stream loop with reconnect/watchdog.

| Lines | What |
|---|---|
| `30` | `os.environ.setdefault("WORKFLOWS_PLUGINS", "local_yolo_plugin")` |
| `39-42` | `from inference.core.interfaces.stream.inference_pipeline import InferencePipeline` |
| `58-110` | `make_rtsp_workflow()` — JSON spec including `roboflow_core/webhook_sink@v1` step |
| `120-138` | `InferencePipeline.init_with_workflow(video_reference=rtsp_url, workflow_specification=spec, ...)` ← **the production-grade entry point** |

---

## How `app.py` uses them

`app.py` doesn't import `inference` directly. It only touches our two runner modules:

| Lines | What |
|---|---|
| `24` | `import real_engine_runner as rer` |
| `382-394` | `get_real_engine()` / `get_smart_engine()` — cached engine handles (lazy init) |
| `486-494` | `stream_real_engine()` (Tab 4) calls `rer.run_engine_on_frame(engine, ...)` |
| `586-594` | `stream_real_smart()` (Tab 5) calls `rer.run_smart_on_frame(engine, ...)` |
| `644-657` | `stream_rtsp()` (Tab 6) routes between the two depending on workflow toggle |

## What's NOT integrated with the real engine

- **Tabs 1–3** use `workflow.py` — a 330-line hand-rolled engine kept for pedagogical contrast.
  Stages are plain functions `(state, frame, params) → state`. Same conceptual shape, different
  implementation, no Roboflow code paths.

---

## Reading order if you want to copy this pattern for your own project

1. **`local_yolo_plugin.py`** in this folder — see the minimal block contract
2. **`inference/core/workflows/prototypes/block.py`** in the inference repo — the base classes
3. **`inference/core/workflows/core_steps/formatters/first_non_empty_or_default/v1.py`** — simplest possible block in the wild
4. **`real_engine_runner.py`** in this folder — see how the engine is built and called
5. **`tests/workflows/integration_tests/execution/`** in the inference repo — gold-standard integration test patterns
