# Triton path — quickstart

Start small: get the Triton-served YOLO running on the same box, talking to the
existing demo. Once that works end-to-end, you can move Triton to a separate
host and run multiple worker processes against it.

```
optimize/triton/
├── triton_yolo_plugin.py             ← drop-in replacement for local_yolo_plugin
├── export_to_onnx.sh                 ← yolov8n.pt → model_repository/yolov8n_onnx/1/model.onnx
├── start_triton.sh                   ← docker run nvcr.io/nvidia/tritonserver
└── model_repository/yolov8n_onnx/
    ├── config.pbtxt                  ← dims, dynamic batching, GPU instance
    └── 1/                            ← model.onnx goes here (auto-created by export script)
```

## 1. Export the model

```bash
cd ../..  # demo root
bash optimize/triton/export_to_onnx.sh
# → optimize/triton/model_repository/yolov8n_onnx/1/model.onnx
```

## 2. Start Triton (Docker)

```bash
bash optimize/triton/start_triton.sh
# Waits for /v2/health/ready, prints endpoints when up.
# Default host ports (override with HTTP_PORT / GRPC_PORT / METRICS_PORT env):
#   gRPC on :18001  ← use this in the demo's UI
#   HTTP on :18000
#   Prometheus metrics on :18002
```

Smoke check:
```bash
curl -s http://localhost:18000/v2/models/yolov8n_onnx | python -m json.tool
# Should report platform "onnxruntime_onnx" and versions ["1"]
```

## 2a. Performance note (single-frame vs multi-camera)

For **one camera at a time**, Triton is *slower* than the PyTorch path
(~64 ms vs ~22 ms in our box) — the gRPC round-trip + Python pre/post
(letterbox + NMS) eats more than the model itself. Triton's win is
**dynamic batching across multiple cameras**: 4 cameras through Triton
all share one batched GPU call (~22 ms total), while the PyTorch path
would do 4×22 = 88 ms in serial.

Don't switch the toggle to Triton expecting a single-camera speedup —
switch it when you start running `run_multi_camera({...})` from
`rtsp_runner.py`.

## 3. Install the Triton client

```bash
cd ../..   # demo root
uv pip install 'tritonclient[grpc]'
```

## 4. Switch the demo to use Triton

The plugin loader (`WORKFLOWS_PLUGINS` env var) only loads the modules listed.
**To activate the Triton block, change the env var in `real_engine_runner.py`
and `rtsp_runner.py`:**

```diff
- os.environ.setdefault("WORKFLOWS_PLUGINS", "local_yolo_plugin")
+ os.environ.setdefault("WORKFLOWS_PLUGINS", "optimize.triton.triton_yolo_plugin")
```

Then change one step in your workflow JSON (`real_engine_runner.py`):

```diff
  {
-     "type": "local_models/ultralytics_yolo@v1",
+     "type": "triton/yolo@v1",
      "name": "detect",
      "image": "$inputs.image",
-     "weights": "$inputs.weights",
-     "device": "$inputs.device",
+     "triton_url": "localhost:8001",
+     "model_name": "yolov8n_onnx",
      "confidence": "$inputs.conf",
      "keep_classes": "$inputs.keep_classes"
  },
```

Restart `python app.py`. Tabs 4-6 now run inference through Triton instead of
loading yolov8n.pt locally.

## 5. (Later) Convert ONNX → TensorRT inside Triton

Once the ONNX path works, switch the same model to TensorRT for the speed win:

```bash
# Drop into the Triton container
docker exec -it triton-yolo bash
cd /models/yolov8n_onnx/1
trtexec --onnx=model.onnx --saveEngine=model.plan --fp16 --explicitBatch
# Edit ../config.pbtxt: platform: "onnxruntime_onnx" → "tensorrt_plan"
# Rename model.onnx → model.plan
exit
docker restart triton-yolo
```

No changes needed in the demo — the plugin doesn't care which backend Triton
uses internally. Just bump the `model_name` if you want to keep both side by
side.

## 6. Multi-camera

Once Triton is up and serving, the demo's `rtsp_runner.run_multi_camera({...})`
already works — each camera gets its own `InferencePipeline` worker, all of
them sharing the one model copy in Triton's GPU memory. Triton's dynamic
batching aggregates concurrent requests automatically.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `state: UNAVAILABLE` for the model | `config.pbtxt` dims don't match the exported ONNX. Check `polygraphy inspect model 1/model.onnx`. |
| Detections look wrong / shifted | letterbox padding / scale undo is off. The plugin assumes square `imgsz` — match it to your export. |
| Empty detections always | check `input_name` and `output_name` against the actual ONNX (Ultralytics uses `images` / `output0` by default but custom exports may differ). |
| 5–15 min hang on first inference | TensorRT auto-tuning. Subsequent calls cache. |
| GPU OOM | drop `instance_group.count` to 1, or reduce `preferred_batch_size`. |

## What stays unchanged

- `workflow.py` (hand-rolled engine)
- `app.py` (Gradio UI)
- All downstream blocks: ByteTracker, Velocity, TimeInZone, visualizations, webhook_sink
- The RTSP simulator (`rtsp_sim/`)
- The workflow JSON specs except for the one detector step

This is the whole point of the plugin pattern — backends are swappable, the
business workflow is not.
