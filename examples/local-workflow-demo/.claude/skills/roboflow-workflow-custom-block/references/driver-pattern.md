# Driver Pattern Reference

How to construct and drive `ExecutionEngine` from your own script (Gradio app,
FastAPI endpoint, batch job, RTSP worker). Covers the perf gotchas that turn
a 17 ms forward into a 100 ms round-trip.

## Table of contents

- [ExecutionEngine.init signature](#executionengineinit-signature)
- [Persistent ThreadPoolExecutor (the cuDNN reset gotcha)](#persistent-threadpoolexecutor-the-cudnn-reset-gotcha)
- [init_parameters: what's actually required](#init_parameters-whats-actually-required)
- [Engine caching for stateful workflows](#engine-caching-for-stateful-workflows)
- [Wrapping numpy frames as WorkflowImageData](#wrapping-numpy-frames-as-workflowimagedata)
- [runtime_parameters shape](#runtime_parameters-shape)
- [Plugin discovery & packaging](#plugin-discovery--packaging)
- [InferencePipeline (production RTSP path)](#inferencepipeline-production-rtsp-path)

## ExecutionEngine.init signature

```python
ExecutionEngine.init(
    workflow_definition: dict,                  # the workflow JSON spec
    init_parameters: Optional[Dict[str, Any]] = None,
    max_concurrent_steps: int = 1,              # parallelism within a single run
    prevent_local_images_loading: bool = False, # set True for untrusted server contexts
    workflow_id: Optional[str] = None,
    profiler: Optional[WorkflowsProfiler] = None,
    executor: Optional[ThreadPoolExecutor] = None,   # pass a persistent one!
    step_error_handler: Optional[Union[str, Callable]] = DEFAULT_WORKFLOWS_STEP_ERROR_HANDLER,
) -> "ExecutionEngine"
```

Returns an engine instance with one method:

```python
engine.run(
    runtime_parameters: Dict[str, Any],
    fps: float = 0,
    _is_preview: bool = False,
    serialize_results: bool = False,
) -> List[Dict[str, Any]]
```

The result is a **list, one element per image in the input batch**. Each
element is a dict keyed by the names declared in the workflow's `outputs`
section.

## Persistent ThreadPoolExecutor (the cuDNN reset gotcha)

If you don't pass `executor=...` to `ExecutionEngine.init`, the engine creates
a fresh `ThreadPoolExecutor` for every `engine.run()` call. On CUDA, every
new worker thread starts with cold cuDNN per-thread heuristic state. cuDNN
re-runs its convolution algorithm benchmark on the first forward pass —
adding ~85 ms per `engine.run()` call on top of the actual compute.

For a yolov8n model whose forward pass is ~17 ms, this turns each frame into
a ~100 ms call. Simple fix:

```python
from concurrent.futures import ThreadPoolExecutor

# One executor at module scope, shared across all engines in the process.
_PERSISTENT_EXECUTOR = ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="wf-engine",
)

ENGINE = ExecutionEngine.init(
    workflow_definition=spec,
    init_parameters=...,
    executor=_PERSISTENT_EXECUTOR,
)
```

Threads are reused across calls, cuDNN state stays warm, the ~85 ms vanishes.

## init_parameters: what's actually required

The engine resolves blocks' `get_init_parameters()` from this dict. For an
offline workflow (no Roboflow API, no Roboflow models) the minimum is:

```python
from inference.core.cache import cache  # noqa: F401  (import side-effect: warms cache module)
from inference.core.managers.base import ModelManager
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.core.workflows.core_steps.common.entities import StepExecutionMode

init_parameters = {
    "workflows_core.model_manager":   ModelManager(model_registry=RoboflowModelRegistry({})),
    "workflows_core.api_key":         None,
    "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
}
```

Even a workflow that uses zero Roboflow models still needs `model_manager` —
the engine constructs core blocks that look it up.

If your custom block needs values from `init_parameters` (e.g. you want the
engine to inject a shared resource), declare them in
`MyBlockV1.get_init_parameters()`:

```python
class MyBlockV1(WorkflowBlock):
    def __init__(self, my_shared_thing):
        self.my_shared_thing = my_shared_thing

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["my_namespace.my_shared_thing"]    # key in init_parameters dict
```

Then pass `"my_namespace.my_shared_thing": <obj>` in `init_parameters`. Most
custom blocks don't need this — keep `get_init_parameters() -> []` and load
shared state in module-level caches.

## Engine caching for stateful workflows

Several core blocks are stateful **per engine instance**:

- `roboflow_core/trackers_bytetrack@v1` — keeps tracks keyed by
  `video_metadata.video_identifier`
- `roboflow_core/velocity@v1` — needs prior frame to compute velocity
- `roboflow_core/time_in_zone@v2` — accumulates dwell timers per track
- `roboflow_core/line_counter@v1` — accumulates crossings

Rebuilding the engine per frame **resets all this state**. Cache engines
keyed by `(workflow_name, backend, ...whatever varies)`:

```python
_ENGINE_CACHE: dict[tuple, ExecutionEngine] = {}

def get_engine(workflow_name: str, backend: str) -> ExecutionEngine:
    key = (workflow_name, backend)
    if key not in _ENGINE_CACHE:
        spec = build_spec(workflow_name, backend)
        _ENGINE_CACHE[key] = ExecutionEngine.init(
            workflow_definition=spec,
            init_parameters=_make_init_parameters(),
            executor=_PERSISTENT_EXECUTOR,
        )
    return _ENGINE_CACHE[key]

def reset_engine_cache() -> None:
    """Call when the workflow JSON has actually changed."""
    _ENGINE_CACHE.clear()
```

If multiple cameras share an engine, **distinguish them via
`video_identifier`**, not separate engines (ByteTracker keys per-stream
state internally).

## Wrapping numpy frames as WorkflowImageData

For video workflows, the metadata you attach matters:

```python
from datetime import datetime
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata, OriginCoordinatesSystem, VideoMetadata, WorkflowImageData,
)

def wrap_frame(frame_rgb, video_id: str, frame_number: int, fps: float):
    h, w = frame_rgb.shape[:2]
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(
            parent_id="image",      # MUST match the input name in the workflow spec
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=0, left_top_y=0,
                origin_width=w, origin_height=h,
            ),
        ),
        numpy_image=frame_rgb,      # RGB uint8 (H, W, 3)
        video_metadata=VideoMetadata(
            video_identifier=video_id,             # ByteTracker keys state on this
            frame_number=frame_number,             # Velocity needs this for dt
            frame_timestamp=datetime.now(),
            fps=fps,                               # also used by Velocity
            comes_from_video_file=True,            # vs streamed
        ),
    )
```

Notes:
- `numpy_image` should be **RGB**, not BGR. OpenCV reads BGR — convert before wrapping.
- `video_identifier` must be stable per stream. Use `"camera_1"`, an RTSP URL
  hash, etc. Do not regenerate per call.
- `frame_number` should monotonically increase per stream. Velocity uses
  `(frame_number_now - frame_number_prev) / fps` for `dt`.
- For still images (no temporal context), `video_metadata=None` is fine —
  but ByteTracker / Velocity / TimeInZone require `video_metadata`.

## runtime_parameters shape

`engine.run(runtime_parameters=...)` takes a dict matching the workflow's
`inputs`. For the workflow:

```python
{
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "conf", "default_value": 0.30},
        {"type": "WorkflowParameter", "name": "weights"},
    ],
    "steps": [...],
    "outputs": [...],
}
```

You'd call:

```python
result = engine.run(runtime_parameters={
    "image": [wrapped_image_1, wrapped_image_2, ...],   # list, even for batch=1
    "conf": 0.45,
    "weights": "yolov8n.pt",
})
# result is a list with len(images) elements:
# [{"output_name_1": ..., "output_name_2": ...}, ...]
```

Rules:
- `image` is always wrapped in a list (the engine treats this as the batch dim)
- Parameters with `default_value` are optional; others must be provided
- Passing a key not in `inputs` raises a validation error — gate on the
  active backend if you swap step types

## Plugin discovery & packaging

`WORKFLOWS_PLUGINS` is comma-separated. Each entry is a Python module path
that's importable from `sys.path`:

```bash
WORKFLOWS_PLUGINS=my_plugin
WORKFLOWS_PLUGINS=my_plugin,another_plugin
WORKFLOWS_PLUGINS=optimize.triton.triton_yolo_plugin    # dotted path works
```

In Python:

```python
import os, sys
from pathlib import Path

# Make your plugin's directory importable BEFORE the inference imports
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Build plugin list, optionally gating on an importable dependency
plugins = ["my_plugin"]
try:
    import tritonclient.grpc  # noqa: F401
    plugins.append("optimize.triton.triton_yolo_plugin")
except ImportError:
    pass
os.environ.setdefault("WORKFLOWS_PLUGINS", ",".join(plugins))

# NOW it's safe to import inference
from inference.core.workflows.execution_engine.core import ExecutionEngine  # noqa: E402
```

If a plugin in `WORKFLOWS_PLUGINS` fails to import, the engine raises
`PluginLoadingError` at engine construction. **Don't list plugins whose
deps aren't installed.**

The plugin module itself just exposes:

```python
def load_blocks() -> List[Type[WorkflowBlock]]:
    return [MyBlockV1, MyOtherBlockV1]
```

Optional siblings (rare for custom blocks): `load_kinds()`,
`load_initializers()`, kind serializers/deserializers.

## InferencePipeline (production RTSP path)

For long-running RTSP/video-file workers with reconnect, watchdog, and a
sink (webhook, file, custom), use `InferencePipeline` instead of driving
`ExecutionEngine.run()` yourself:

```python
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline

pipeline = InferencePipeline.init_with_workflow(
    video_reference="rtsp://camera/stream",
    workflow_specification=spec,
    workflows_parameters={"weights": "yolov8n.pt", "device": "cuda"},
    on_prediction=lambda preds, frame: ...,    # your sink
    max_fps=15.0,
)
pipeline.start()
pipeline.join()
```

`InferencePipeline.init_with_workflow` constructs an `ExecutionEngine`
internally, runs the engine in its own thread, and feeds frames from the
video source. Same plugin-discovery rules apply — set `WORKFLOWS_PLUGINS`
before the import.

For multi-camera deployments where N pipelines share one model, see
`BACKENDS.md` in the demo (Triton path) — `InferencePipeline` + a Triton
client plugin block is the canonical setup.
