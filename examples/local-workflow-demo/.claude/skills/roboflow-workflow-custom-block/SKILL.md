---
name: roboflow-workflow-custom-block
description: Integrate custom inference logic (your own model, your own backend, your own pre/post-processing) into a Roboflow Inference Workflow without forking the inference repo. Use when the user wants to plug a local PyTorch/ONNX/TensorRT/Triton/HF/SDK model into a Workflow JSON, write a custom WorkflowBlock, drive ExecutionEngine programmatically, run Workflows on video frames or RTSP without a Roboflow API key, or asks about WORKFLOWS_PLUGINS, load_blocks, sv.Detections compatibility with ByteTracker/Velocity/visualization blocks, WorkflowImageData wrapping, or persistent ThreadPoolExecutor for the engine.
---

# Roboflow Workflow Custom Block

Plug your own model or custom logic into a Roboflow Inference workflow with a
single Python file and one env var. No fork of the `inference` repo. No
Roboflow API key. Downstream blocks (ByteTracker, Velocity, TimeInZone,
BoundingBoxVisualization, webhook_sink) keep working unchanged.

## The whole bridge in three lines

```python
# 1. Plugin discovery hook — set BEFORE importing inference.*
os.environ["WORKFLOWS_PLUGINS"] = "my_plugin_module"

# 2. Build engine from a JSON workflow spec
engine = ExecutionEngine.init(
    workflow_definition=spec,
    init_parameters=init_params,
    executor=PERSISTENT_EXECUTOR,   # see "Persistent executor" below — required for perf
)

# 3. Run it per frame
result = engine.run(runtime_parameters={"image": [WorkflowImageData(...)]})
```

The plugin module is any Python module importable from `sys.path` that exposes:

```python
def load_blocks() -> List[Type[WorkflowBlock]]:
    return [MyBlockV1]
```

Everything else is elaboration.

## When to use this skill

Use when the user is:
- Writing a custom `WorkflowBlock` that emits `sv.Detections` into a Roboflow workflow
- Driving `ExecutionEngine.init(...) / engine.run(...)` from their own script (Gradio, FastAPI, batch job)
- Wrapping a local model — Ultralytics YOLO checkpoint, ONNX/TensorRT engine, HuggingFace Transformers model, Triton client, or any custom inference function — so a Roboflow workflow can call it
- Hitting batching issues (silent fallback to single-frame), or perf cliffs they don't understand
- Asking about `WORKFLOWS_PLUGINS`, `load_blocks`, `WorkflowImageData`, `Batch[...]`, `attach_parents_coordinates_to_sv_detections`, or how to make ByteTracker/Velocity see their detections

The reference demo for this skill lives at
`examples/local-workflow-demo/` in the inference repo
(`local_yolo_plugin.py`, `real_engine_runner.py`, `INTEGRATION.md`,
`BACKENDS.md`).

## Workflow

### Step 1 — Read the contract once

Before writing any code, load `references/block-contract.md`. It documents:
- The manifest fields (`type: Literal["..."]`, `WorkflowImageSelector`, `Selector(kind=[...])`)
- Single-frame vs batched `run()` signatures
- The exact `sv.Detections.data` keys downstream blocks require
- The `get_parameters_accepting_batches()` footgun

### Step 2 — Write the plugin module

Use `assets/example_plugin.py` as a starting template. Edit:
1. The `type: Literal[...]` identifier (this is what appears in workflow JSON)
2. The manifest fields (your model's params)
3. The `run()` body (your inference call → `sv.Detections`)
4. `load_blocks()` returns your block class

Rules (the ones that bite):
- **Always declare batched inputs** (`images: Batch[WorkflowImageData]`) and
  `get_parameters_accepting_batches() -> ["images"]`. Without **both**, the
  engine silently iterates frame-by-frame and batching never reaches your
  model.
- **Always populate** these `sv.Detections.data` keys: `class_name`,
  `detection_id`, `image_dimensions`, `prediction_type`. Without them,
  ByteTracker/Velocity/visualizations either crash or produce wrong output.
- **Always call** `attach_parents_coordinates_to_sv_detections(detections=..., image=img)`
  before returning. This sets parent metadata so downstream blocks can map
  detections back to the source frame.
- **Cache the model object** at module scope. Block instances are constructed
  often; loading weights every time is wasteful.

### Step 3 — Write the driver

Load `references/driver-pattern.md` for `ExecutionEngine.init` details, frame
wrapping, and the persistent-executor performance fix.

Minimum driver shape:

```python
import os
# Set BEFORE importing anything from inference.*
os.environ.setdefault("WORKFLOWS_PLUGINS", "my_plugin_module")

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from inference.core.cache import cache  # noqa: F401  (warms cache module)
from inference.core.managers.base import ModelManager
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata, OriginCoordinatesSystem, VideoMetadata, WorkflowImageData,
)

WORKFLOW_SPEC = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {"type": "my_namespace/my_block@v1", "name": "detect", "images": "$inputs.image"},
    ],
    "outputs": [{"type": "JsonField", "name": "predictions",
                 "selector": "$steps.detect.predictions"}],
}

# Persistent executor — REQUIRED for good perf on GPU. See driver-pattern.md.
EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="wf-engine")

ENGINE = ExecutionEngine.init(
    workflow_definition=WORKFLOW_SPEC,
    init_parameters={
        "workflows_core.model_manager": ModelManager(model_registry=RoboflowModelRegistry({})),
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    },
    executor=EXECUTOR,
)
```

Then per frame, wrap the numpy frame and call `engine.run`:

```python
def wrap_frame(frame_rgb, video_id, frame_number, fps):
    h, w = frame_rgb.shape[:2]
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(
            parent_id="image",
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=0, left_top_y=0, origin_width=w, origin_height=h),
        ),
        numpy_image=frame_rgb,
        video_metadata=VideoMetadata(
            video_identifier=video_id,         # ByteTracker keys per-stream state on this
            frame_number=frame_number,         # Velocity needs this for dt
            frame_timestamp=datetime.now(),
            fps=fps,
            comes_from_video_file=True,
        ),
    )

img = wrap_frame(frame_rgb, "cam0", frame_idx, fps=30.0)
result = ENGINE.run(runtime_parameters={"image": [img]})
```

### Step 4 — Smoke test

```python
import cv2
cap = cv2.VideoCapture("path/to/video.mp4")
for i in range(10):
    ok, bgr = cap.read()
    if not ok: break
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = ENGINE.run(runtime_parameters={"image": [wrap_frame(rgb, "cam0", i, 30.0)]})
    print(out[0]["predictions"])
```

If `engine.run` raises `PluginLoadingError` or "block type not found", the
plugin wasn't discovered — `WORKFLOWS_PLUGINS` was set after the import, or
the module isn't on `sys.path`.

## Hard gotchas (read these before debugging)

1. **`WORKFLOWS_PLUGINS` is read at import time.** Set the env var **before**
   any `from inference.*` import. The canonical pattern:
   ```python
   os.environ.setdefault("WORKFLOWS_PLUGINS", "my_plugin")
   from inference.core.workflows.execution_engine.core import ExecutionEngine  # noqa: E402
   ```

2. **Pass a persistent `ThreadPoolExecutor`** to `ExecutionEngine.init`. The
   default creates a fresh pool per `engine.run()` call. On CUDA, this resets
   cuDNN per-thread heuristics every frame and adds ~85 ms — turning a 17 ms
   forward pass into 100 ms. One executor at module scope, shared across
   engines, fixes it.

3. **Cache engines, don't rebuild per frame.** ByteTracker / Velocity /
   TimeInZone keep per-instance state (track IDs, dwell timers, smoothed
   velocities). Build the engine once and reuse it. Rebuild only when the
   workflow JSON or backend choice changes.

4. **Batch-awareness requires *both* signals.** Plural `images` parameter
   declared as `WorkflowImageSelector` **and**
   `get_parameters_accepting_batches() -> ["images"]`. Drop either and the
   engine silently iterates frame-by-frame — your batched code path is
   never exercised.

5. **Dotted-path plugins work.** For sub-package layout, set
   `WORKFLOWS_PLUGINS=optimize.triton.triton_yolo_plugin`. The directory
   tree just needs to be importable from `sys.path`.

6. **`init_parameters` is mandatory** even for offline workflows. Pass at
   minimum a `ModelManager` (with empty registry), `api_key=None`, and
   `step_execution_mode=StepExecutionMode.LOCAL`. The engine's init code path
   touches `model_manager` even when no Roboflow model is in the spec.

7. **Switching backends is a one-line workflow JSON edit**, not a code
   rewrite. If you have two plugins (e.g. local PyTorch and Triton client)
   that both emit the same `OBJECT_DETECTION_PREDICTION_KIND`, swapping the
   detector step's `type:` to the other plugin's identifier is enough.
   Downstream blocks don't care.

## Resources

- `references/block-contract.md` — read before writing the block. Manifest
  schema, kind types, batched-vs-single signatures, the `sv.Detections`
  metadata table, parent-coordinates pattern.
- `references/driver-pattern.md` — read before writing the driver.
  `ExecutionEngine.init` signature, persistent executor, frame wrapping,
  engine caching, runtime parameter shape, plugin packaging tricks.
- `assets/example_plugin.py` — minimal ~80 LOC plugin you can copy and
  modify. Replace the inference call, keep the metadata population.

## Anti-patterns to avoid

- Importing `inference.*` at the top of the file before setting `WORKFLOWS_PLUGINS`
- Building `ExecutionEngine` inside a per-frame function
- Returning bare `np.ndarray` boxes instead of `sv.Detections`
- Forgetting `attach_parents_coordinates_to_sv_detections` (silent: things
  "work" until you add a Crop or zone block upstream)
- Putting plugin code inline in the driver script — make it a real
  importable module so `WORKFLOWS_PLUGINS=<dotted.name>` resolves
