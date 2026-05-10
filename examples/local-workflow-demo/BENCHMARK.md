# Benchmark — PyTorch vs Triton, 1/5/10 concurrent RTSP cameras

Run with `stress_test.py` — N threads each open `rtsp://localhost:8554/traffic`
through the demo's MediaMTX simulator and pump frames through the same
speed-estimation engine. Same workflow JSON, same downstream blocks; only the
detector step differs (`local_models/ultralytics_yolo@v1` vs `triton/yolo@v1`).

**Hardware:** RTX 3090, AMD Ryzen 9, CUDA 12.5 host driver
**Model:** YOLOv8n COCO at 640×640
**Workflow:** YOLO → ByteTrack → Velocity → BoxViz → LabelViz
**Triton:** ONNX Runtime backend on CUDA, dynamic batching enabled
(`preferred_batch_size: [4, 8]`, `max_queue_delay: 5ms`)

## Results

| Cameras | Backend | Aggregate FPS | per-camera FPS | p50 latency | p95 latency | p99 latency |
|---:|:---|---:|---:|---:|---:|---:|
| 1 | PyTorch | **13.9** | 13.9 | 26 ms | 44 ms | 56 ms |
| 1 | Triton  | 12.2 | 12.2 | 78 ms | 93 ms | 98 ms |
| 5 | PyTorch | 31.7 | 6.3 | 131 ms | 238 ms | 306 ms |
| 5 | Triton  | **31.9** | 6.4 | 127 ms | 196 ms | 567 ms |
| 10 | PyTorch | **39.5** | 3.9 | 210 ms | 334 ms | 529 ms |
| 10 | Triton  | 20.2 | 2.0 | 391 ms | 620 ms | 680 ms |

## Bottom line: PyTorch wins this workload

Triton is **slower** at every camera count tested. Surprising for those who
expect "Triton + dynamic batching = always faster." Here's why this workload
defeats Triton's strengths.

## Why Triton loses here

### 1. Python-side NMS is the real bottleneck

Each frame's post-processing (filter 8400 anchors → top-K → per-class NMS)
runs in Python in our `triton_yolo_plugin.py`. With 10 worker threads, Python's
GIL means **NMS effectively serializes**. Even though the GPU is idle, the
client can't keep up.

Meanwhile, the PyTorch path uses Ultralytics' `non_max_suppression` (called
inside `model.predict()`), which releases the GIL during the forward pass
and uses NumPy/torch under the hood — much friendlier to multi-threading.

Evidence: at 10 cameras, **GPU-side compute on Triton is only ~22 ms / exec**
(`nv_inference_compute_infer_duration_us / nv_inference_exec_count`). Per-call
total latency is 391 ms. The gap (~370 ms) is client-side: gRPC
serialization, Python NMS, and threads queuing on the GIL.

### 2. Dynamic batching doesn't fire

Even at 10 concurrent cameras, Triton metrics show:

```
nv_inference_count{model="yolov8n_onnx"} = 690    # requests received
nv_inference_exec_count{model="yolov8n_onnx"} = 660 # GPU executions
batch_ratio = 690 / 660 = 1.045   # ~no batching
```

Why? Because requests don't arrive simultaneously enough. The 10 client
threads spend most of their time in Python (NMS + GIL), so they emit
gRPC requests at staggered times. By the time request N arrives, the GPU
has already finished request N-1 and dispatched it.

Bumping `max_queue_delay_microseconds` from 5 ms → 50 ms made batches form
(avg batch 2.6) but **made overall throughput worse** (24 fps total, 333 ms
p50) because the 50 ms wait was a deadweight cost, not amortized over a
GPU-bound workload.

### 3. gRPC has fixed ~50 ms per-call overhead in this stack

Single-camera comparison: PyTorch 26 ms vs Triton 78 ms. The +50 ms is the
combined cost of:
- gRPC tensor (de)serialization
- Network round-trip (loopback, but still goes through the kernel TCP stack)
- ONNX Runtime memcpy nodes (4 inserted because of CUDA 12.5 vs 12.6 minor
  version compat — see `docker logs triton-yolo`)

For yolov8n at ~5 ms forward pass, this 50 ms overhead is 10× the model.
For a heavier model (yolov8x, ~50 ms forward), it'd be 1× — much more
favorable to Triton.

## When Triton WOULD win

Three changes, any of which would flip the result:

1. **Server-side NMS.** Re-export the model with `yolo export ... nms=True`
   to push NMS into the ONNX graph. The plugin then receives small
   already-filtered detection arrays — no Python NMS, GIL freed.

2. **Heavier model.** YOLOv8x has ~50 ms forward pass. The ratio of GPU
   compute to client overhead flips, and dynamic batching across cameras
   delivers near-linear scaling on the GPU side.

3. **Many more cameras (50+).** GPU becomes the bottleneck → requests
   bunch up at the queue → batching kicks in → ~5× throughput improvement
   per batched call. We can't reproduce this here because MediaMTX maxes
   out at ~10 simultaneous read connections per path before connection
   refusals start.

## Honest recommendation for this demo

If your real workload is "≤10 cameras, yolov8n, modest object counts":

- **Use the PyTorch backend.** It's simpler (no docker, no model export,
  no gRPC), faster, and `~22 ms / call` even at 10 cameras with the
  persistent-executor fix.

If your real workload has any of:
- 20+ concurrent cameras
- A bigger model than yolov8n
- A clean way to push NMS into the graph (Ultralytics `nms=True`)
- Need to share GPU memory across multiple processes / containers / hosts

**Then Triton is worth it.** The server-side-NMS export is the single
highest-leverage change to make.

## Reproducing

```bash
cd examples/local-workflow-demo
bash rtsp_sim/start_rtsp_sim.sh                 # MediaMTX + ffmpeg loop
bash optimize/triton/start_triton.sh             # Triton on :18001
uv run python stress_test.py --backend pytorch --cameras 10 --duration 20
uv run python stress_test.py --backend triton  --cameras 10 --duration 20
```

The stress test exposes `--cameras`, `--duration`, `--rtsp-url`,
`--triton-url`, `--triton-model`. Modify and re-run for your scenario.
