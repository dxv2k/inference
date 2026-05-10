# Switching the YOLO backend

The custom `LocalYoloBlockV1` (`local_yolo_plugin.py`) is the **only** place
the demo touches a model. Switching to ONNX, TensorRT, or Triton is a matter
of changing what that one block does — every other workflow JSON spec, every
downstream block (ByteTracker, Velocity, TimeInZone, webhook_sink), and
every Gradio tab stays unchanged.

## Three options at a glance

| Backend | Effort | Speedup | When to use |
|---|---|---|---|
| **ONNX-Runtime** | ⭐ change one path | ~1.2–1.5× on GPU, much bigger on CPU | Cross-platform deploy, no GPU, or as a stepping stone to TensorRT |
| **TensorRT** | ⭐⭐ export per GPU class | ~2.5–4× FP16, ~5× INT8 | Single-host GPU production. The big speed win. |
| **Triton Inference Server** | ⭐⭐⭐ new plugin block | depends on backend (TRT under the hood) | Multi-camera or multi-process deployments — share one GPU model across N workers |

---

## Path 1 — ONNX (5 min)

Ultralytics' `YOLO()` is multi-backend (`ultralytics/nn/autobackend.py`). It
dispatches by file extension. Our plugin already passes `weights` straight to
`YOLO(...)`, so swapping format is just an export step:

```bash
# Export once
uv run yolo export model=yolov8n.pt format=onnx imgsz=640 simplify=True
# → yolov8n.onnx
```

In Tab 4/5/6 (or directly in the workflow JSON) change the `weights` runtime
parameter from `yolov8n.pt` → `yolov8n.onnx`. Done. No code changes.

---

## Path 2 — TensorRT (15 min)

Same trick:

```bash
uv run yolo export model=yolov8n.pt format=engine imgsz=640 half=True device=0
# → yolov8n.engine
```

Then `weights="yolov8n.engine"`. Caveats:

- TensorRT engines are **GPU-architecture-specific**. Re-export for each GPU
  class (Ampere ≠ Hopper ≠ Jetson). Don't ship `.engine` files in your
  container image.
- First export takes 5–15 minutes (kernel auto-tuning). Cache the file.
- Workflow JSON, Gradio UI, ByteTrack/Velocity downstream — all unchanged.

For yolov8n at 640×640 on a typical RTX-class GPU:
PyTorch FP32 ~12 ms → TensorRT FP16 ~3–5 ms.

---

## Path 3 — Triton (~half-day)

Triton is the production answer for **multi-camera deployments**. The model
lives in a separate process (often a separate container or host), and our
workflow worker connects via gRPC. Benefits:

- **One model copy in GPU memory**, shared across N camera workers
- **Dynamic batching** — Triton aggregates inflight requests from all
  cameras and serves them in batched GPU calls (much higher GPU utilization)
- **Hot model swap** — drop a new `.plan` into `model_repository/` and bump
  the version, no worker restart needed
- **Independent scaling** — workers scale horizontally; the GPU server is
  one box you tune separately

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Triton Inference Server (one process owns the GPU)          │
│  • model.plan (TensorRT FP16) + dynamic batching             │
│  • gRPC :8001  /  HTTP :8000                                 │
└──────────────────────────────┬───────────────────────────────┘
                               │ gRPC infer requests (batched)
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
┌─ Worker 1 ──────┐   ┌─ Worker 2 ──────┐   ┌─ Worker N ──────┐
│ InferencePipe-  │   │ InferencePipe-  │   │ InferencePipe-  │
│   line(rtsp://1)│   │   line(rtsp://2)│   │   line(rtsp://N)│
│ + workflow      │   │ + workflow      │   │ + workflow      │
│ + triton_plugin │   │ + triton_plugin │   │ + triton_plugin │
│ + webhook_sink  │   │ + webhook_sink  │   │ + webhook_sink  │
└─────────────────┘   └─────────────────┘   └─────────────────┘
```

### How the integration looks

The workflow JSON spec changes by **one step type**:

```diff
- {"type": "local_models/ultralytics_yolo@v1", "name": "detect", ...}
+ {"type": "triton/yolo@v1", "name": "detect",
+  "image": "$inputs.image",
+  "triton_url": "localhost:8001",
+  "model_name": "yolov8n_trt"}
```

Everything downstream is the same. The new block does pre-process →
gRPC infer → post-process (NMS) → emit `sv.Detections` with the same
metadata as `LocalYoloBlockV1`.

See `optimize/triton/` for a runnable starting point:

```
optimize/triton/
├── README.md                         quickstart
├── triton_yolo_plugin.py             the Triton client block
├── export_to_onnx.sh                 export yolov8n.pt → ONNX
├── start_triton.sh                   docker run nvcr.io/nvidia/tritonserver:...
└── model_repository/
    └── yolov8n_onnx/
        ├── config.pbtxt              dims, dynamic batching, instance_group
        └── 1/                        drop yolov8n.onnx here
```

## Recommended progression

For the local-workflow-demo, the smart progression is:

1. **Today** — keep PyTorch (`yolov8n.pt`) for development. ~18 ms/frame is fine.
2. **Next** — try ONNX (5 min, no risk). If on CPU box, this is your prod path.
3. **Production single-GPU** — TensorRT engine. Big win, almost free.
4. **Production multi-camera** — Triton. Plugin block lives in `optimize/triton/`.

Don't jump straight to Triton if you only have one camera. The complexity
isn't worth it until you're sharing a GPU across workers.
