# Local Workflow Demo — fully offline, no Roboflow API key

End-to-end Gradio demo that runs the **real** Roboflow `ExecutionEngine` against
a locally-loaded Ultralytics YOLO model. Six tabs cover progressively richer
scenarios — from a hand-rolled pipeline you can read in 200 lines, all the way
to RTSP streams firing webhook alerts to your own API.

```
┌─── Tabs 1–3 ─────────────┐    ┌─── Tabs 4–6 ───────────────────────┐
│  Hand-rolled engine      │    │  REAL Roboflow ExecutionEngine     │
│  (workflow.py, ~330 LOC) │    │  driven by local_yolo_plugin       │
│  Educational reference   │    │  Production-shape: real blocks     │
└──────────────────────────┘    └────────────────────────────────────┘
```

| Tab | Engine | Workflow | Use-case |
|---|---|---|---|
| 1 | hand-rolled | single image | quick test |
| 2 | hand-rolled | smart-camera | surveillance · counting · zone alerts |
| 3 | hand-rolled | speed-estimation | per-object km/h |
| **4** | **real `ExecutionEngine`** | LocalYOLO + ByteTrack + Velocity | speed estimation |
| **5** | **real `ExecutionEngine`** | LocalYOLO + ByteTrack + TimeInZone | smart camera |
| **6** | **real `ExecutionEngine`** | RTSP → workflow → webhook | production pattern |

## How it works

The trick: a tiny custom `WorkflowBlock` (`local_yolo_plugin.py`, ~80 LOC)
wraps `YOLO("yolov8n.pt")` and emits Roboflow-compatible `sv.Detections`.
It's loaded into the engine via the `WORKFLOWS_PLUGINS` env var. From there
on, every other block (`roboflow_core/trackers_bytetrack@v1`,
`roboflow_core/velocity@v1`, `roboflow_core/time_in_zone@v2`,
`roboflow_core/webhook_sink@v1`, etc.) is the **real, unmodified** Roboflow
block — no API key required.

```
local_models/ultralytics_yolo@v1   ← our plugin (wraps yolov8n.pt)
            ↓
roboflow_core/trackers_bytetrack@v1   ← real
            ↓
roboflow_core/velocity@v1   OR   roboflow_core/time_in_zone@v2   ← real
            ↓
bounding_box_visualization@v1  +  label_visualization@v1   ← real
```

## File layout

```
local-workflow-demo/
├── app.py                  # Gradio UI, 6 tabs
├── workflow.py             # hand-rolled pedagogical engine (Tabs 1-3)
├── local_yolo_plugin.py    # custom WorkflowBlock wrapping local YOLO
├── real_engine_runner.py   # drives ExecutionEngine for Tabs 4 & 5
├── rtsp_runner.py          # InferencePipeline-based driver for production RTSP
├── rtsp_sim/               # local RTSP server simulator (MediaMTX + ffmpeg)
│   ├── mediamtx.yml
│   └── start_rtsp_sim.sh
├── sample_video/
│   ├── sample.mp4          # people-in-room (smart-camera demo)
│   └── traffic_cctv.mp4    # highway CCTV (speed-estimation demo)
├── pyproject.toml          # uv-managed, torch + cuda121 + gradio + ultralytics
└── uv.lock
```

## Quickstart

```bash
# 1. Install (uv handles torch+cuda121 + ultralytics + gradio + the inference repo).
uv sync

# 2. Make sure the inference repo itself is installed in editable mode.
#    (Required for `from inference.core.workflows.execution_engine.core import ExecutionEngine`.)
uv pip install -e ../..

# 3. Run.
uv run python app.py
# → http://0.0.0.0:7872
```

Tabs 1–3 work immediately. Tabs 4–6 require the inference package to be
importable — the `WORKFLOWS_PLUGINS=local_yolo_plugin` env var is set
internally by `real_engine_runner.py` / `rtsp_runner.py`, so you don't need
to export it yourself.

## Tab 6: RTSP without a real camera

```bash
# Start a local RTSP server that loops the sample traffic video.
bash rtsp_sim/start_rtsp_sim.sh
# → rtsp://localhost:8554/traffic
```

This downloads MediaMTX (single Go binary, ~29 MB — fetched on first run; not
checked into the repo) and uses ffmpeg to push the traffic video on infinite
loop. You can now paste `rtsp://localhost:8554/traffic` into Tab 6 and use
the simulator just like a real RTSP camera.

In Tab 6 you can choose:
- **Speed Estimation** workflow → alerts when smoothed speed exceeds a km/h threshold
- **Smart Camera** workflow → alerts when `time_in_zone` exceeds a seconds threshold

Each alert can POST a JSON payload to your own webhook URL. Speed alerts are
deduplicated 5s per track to prevent spam.

## Headless / production deployment

For one-process-per-camera production deployment, skip the Gradio UI and use
`rtsp_runner.py` directly:

```python
from rtsp_runner import make_rtsp_workflow, run_pipeline

spec = make_rtsp_workflow(
    alert_url="https://your-api.example.com/alerts",
    keep_classes=["car", "truck", "bus", "motorbike"],
    zone=[[100,100],[540,100],[540,260],[100,260]],
    dwell_threshold_s=3.0,
)
pipeline = run_pipeline(
    rtsp_url="rtsp://user:pass@cam.lan:554/stream",
    workflow_spec=spec,
    max_fps=6.0,   # throttle GPU; cameras stream at 25-30 fps
)
pipeline.join()    # blocks; auto-reconnects on stream drop via watchdog
```

Or scale to N cameras in a single process:

```python
from rtsp_runner import run_multi_camera

run_multi_camera(
    cameras={"front": "rtsp://...", "back": "rtsp://..."},
    alert_url="https://your-api.example.com/alerts",
    keep_classes=["person", "car"],
)
```

Underneath, `rtsp_runner` uses `InferencePipeline.init_with_workflow(...)` —
the same primitive the inference HTTP server uses. Buffer is set to
`ADAPTIVE_DROP_OLDEST` + `EAGER` so RTSP back-pressure never accumulates.

## What is NOT touched

- ✗ `api.roboflow.com` — never called
- ✗ Roboflow `model_id` lookup — bypassed
- ✗ Roboflow Universe model downloads — bypassed (we load `.pt` from disk)
- ✗ Roboflow's hosted inference server — not used

The only network traffic on first install is `pip install` from PyPI.
After that, this demo runs fully air-gapped.

## Speeding it up — ONNX, TensorRT, Triton

The custom `LocalYoloBlockV1` is the only place the demo touches a model.
Swapping to ONNX, TensorRT, or Triton is a matter of changing what that one
block does — every workflow JSON, every Gradio tab stays unchanged.

- See `BACKENDS.md` for the comparison and decision tree.
- See `optimize/triton/README.md` for a runnable Triton starting point.

## See also

- `../stream-examples/` — older stream-processing examples
- `../inference-dashboard-example/` — analytics dashboard example
- `../../inference/core/interfaces/stream/inference_pipeline.py` — the
  production stream runner
- `../../inference/core/workflows/core_steps/` — full block library
