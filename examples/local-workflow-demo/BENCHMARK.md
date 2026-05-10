# Benchmark — PyTorch vs Triton, batched and unbatched

> **Update:** the original v1 of this doc concluded "PyTorch wins because Triton
> batching doesn't fire." Looking again with fresh eyes (h/t reviewer): both
> pieces of that test were wrong — the plugins weren't batch-aware, and the
> stress test ran N independent decoders/inferences instead of multiplexing.
> The fixes flipped the question — Triton batching now works fine; the new
> bottleneck is the **gRPC marshaling of fp32 tensors** (~40 MB per call
> at batch=8). PyTorch still wins for this model + hardware, but for clearer
> reasons.

## Setup

- **Hardware:** RTX 3090, AMD Ryzen 9, CUDA 12.5 host driver
- **Model:** YOLOv8n COCO, exported from `yolov8n.pt`
- **Workflow:** YOLO → ByteTrack → Velocity → BoxViz → LabelViz
- **Triton:** ONNX Runtime backend on CUDA, dynamic batching enabled
  (`max_batch_size: 32`, `preferred_batch_size: [8, 16]`,
  `max_queue_delay: 5ms`)
- **Triton model:** exported with `nms=True` so NMS is **inside** the graph
  (output shape `[batch, 300, 6]`). Eliminates the Python-NMS GIL bottleneck.

## Methodology fix

Before the refactor the plugins declared single-frame `image: WorkflowImageData`,
so the workflow engine just iterated batched inputs frame-by-frame. **No
batching ever reached Triton's server.** The fix:

```python
class LocalYoloManifest(WorkflowBlockManifest):
    images: WorkflowImageSelector = Field(...)   # plural

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

class LocalYoloBlockV1(WorkflowBlock):
    def run(self, images: Batch[WorkflowImageData], ...) -> BlockResult:
        # process all N frames in one call → return List[dict] of length N
```

Same change in `triton_yolo_plugin.py`, but there it actually matters: the
plugin now stacks N preprocessed tensors into one `(N, 3, 640, 640)` numpy
array and sends **one** gRPC request. Triton dynamic batching can then
aggregate multiple clients' batches into single GPU executions.

Verified via Triton metrics:

```
nv_inference_count    = 507    # frames inferred
nv_inference_exec_count = 78   # GPU executions  
batch ratio = 507 / 78 = 6.5   # ✓ batching is working
```

## Results — direct batched benchmark (single process, N frames per call)

```python
# What this measures: one engine.run({"image": [N frames], ...})
# Equivalent to: InferencePipeline driving N RTSP sources with
# batch_collection_timeout set so frames arrive together.
```

| Batch | PyTorch call | PyTorch / frame | PyTorch fps | Triton call | Triton / frame | Triton fps |
|---:|---:|---:|---:|---:|---:|---:|
| 1  |   17.9 ms |  17.9 ms |  56 |   56.0 ms |  56.0 ms | 18 |
| 2  |   27.2 ms |  13.6 ms |  74 |  124.7 ms |  62.4 ms | 16 |
| 4  |   33.3 ms |   8.3 ms | 120 |  307.5 ms |  76.9 ms | 13 |
| 8  |   50.0 ms |   6.3 ms | 160 |  712.7 ms |  89.1 ms | 11 |
| 16 |   86.6 ms |   5.4 ms | 185 | 1529.1 ms |  95.6 ms | 10 |

PyTorch scales near-linearly with batch size: 56 → 185 fps (3.3× from
batch=1 to batch=16). Triton scales **negatively** — bigger batches make it
worse per frame. Why?

## Where Triton's time actually goes

`cProfile` of a 10-iteration batch=8 Triton run, total 14.0 sec:

```
9.96 s  time.sleep              # gRPC blocking wait for response
0.68 s  SerializeToString       # protobuf serialize the request
0.65 s  grpc/_channel.py        # gRPC channel internals
0.47 s  numpy.ascontiguousarray # memory layout for the request tensor
0.32 s  ndarray.tobytes         # numpy → bytes for protobuf
0.30 s  ndarray.astype          # dtype casts
```

**Per batch=8 call: ~1.0 sec waiting on gRPC** — and that's loopback.

Tensor sizes:
- Request: `8 × 3 × 640 × 640 × 4 bytes = 39.3 MB` (fp32 NCHW)
- Response: `8 × 300 × 6 × 4 bytes = 56 KB`

39 MB / 1 sec = **40 MB/s effective gRPC throughput on loopback** — surprisingly
slow. Two layers stacked: protobuf serialize/copy in our process, then
TCP through the kernel, then deserialize on the Triton side.

Meanwhile Triton-side per request:

```
nv_inference_request_duration_us / nv_inference_count = 24.7 ms / frame
nv_inference_compute_infer_duration_us / exec_count   = 143 ms / batch (~6.5 frames)
```

So Triton's GPU+queue does each frame in ~25 ms total. Our client overhead
(serialize + gRPC + deserialize) is **3-5× that**.

## Why PyTorch wins this workload

PyTorch's batched `model.predict(frame_list)`:
- One torch tensor allocation, contiguous in memory already
- Forward pass amortizes overhead across the batch (GPU launch latency)
- Ultralytics' C++ NMS runs after, releases the GIL during forward
- No serialization, no network, no tensor copy

For a tiny model (yolov8n at ~5 ms forward at batch=1, ~80 ms at batch=16),
the model itself is faster than gRPC marshaling its inputs.

## When Triton WOULD win (concrete recipes)

Three changes, ranked by effort:

### 1. Triton **shared-memory** I/O (~half-day, biggest lever)

Triton's `cuda_shared_memory` and `system_shared_memory` APIs eliminate
the gRPC tensor copy. Client puts frames into a CUDA buffer; server reads
directly. Documented in `triton-inference-server/server`'s
`docs/protocol/extension_shared_memory.md`. This would drop the 1 sec/call
gRPC wait to ~1 ms.

For a single-host deploy (one Triton + N workers on same box), this is
the right answer. For multi-host (Triton on a dedicated GPU server) it
doesn't help.

### 2. Heavier model where GPU is the bottleneck (free, just swap weights)

YOLOv8n forward is ~5 ms — the gRPC overhead dominates. YOLOv8x forward
is ~50 ms — the gRPC overhead becomes 1× the model time instead of 10×.
Triton starts winning when model time ≫ marshaling time.

```bash
yolo export model=yolov8x.pt format=onnx imgsz=640 simplify=True dynamic=True nms=True
```

### 3. Send uint8 HWC instead of fp32 NCHW (~quarter-day)

Quartering the request payload from 39 MB → 9.6 MB at batch=8 by sending
the raw uint8 frames and doing normalize/transpose **inside** the model
(Ultralytics export supports this with a small graph prepended). Combined
with batching: ~4× the throughput we measured.

## Where the demo lands

Even after the proper-batching fix, **PyTorch is the right backend for this
demo's workload** (yolov8n, single-host, < 32 cameras). Triton's full value
unlocks at scale where shared-memory transport pays off, or with bigger
models where GPU compute dominates over serialization.

The plugin pattern still wins: switching to a different backend was 3 lines
in the workflow JSON, not a code rewrite. When you do hit a workload where
Triton's strengths matter, you flip a radio in the UI and you're done.

## Reproducing

```bash
cd examples/local-workflow-demo

# 1. Export with server-side NMS
bash optimize/triton/export_to_onnx.sh

# 2. Start Triton
bash optimize/triton/start_triton.sh

# 3. Single-process batched benchmark (the table above):
uv run python -c "
import cv2, numpy as np, time
from real_engine_runner import init_engine, wrap_frame
cap = cv2.VideoCapture('sample_video/traffic_cctv.mp4')
ok, frame = cap.read(); rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
for backend in ['pytorch', 'triton']:
    eng = init_engine(backend)
    rt = {'conf': 0.30, 'pixels_per_meter': 12.5}
    if backend == 'triton':
        rt.update({'triton_url': 'localhost:18001', 'triton_model': 'yolov8n_onnx'})
    else:
        rt.update({'weights': 'yolov8n.pt', 'device': 'cuda'})
    for batch_size in [1, 2, 4, 8, 16]:
        imgs = [wrap_frame(rgb, f'cam{i}', 1, 30.0) for i in range(batch_size)]
        for _ in range(3): eng.run(runtime_parameters={'image': imgs, **rt})
        ts = []
        for i in range(10):
            imgs = [wrap_frame(rgb, f'cam{j}', 100+i, 30.0) for j in range(batch_size)]
            s = time.perf_counter()
            eng.run(runtime_parameters={'image': imgs, **rt})
            ts.append((time.perf_counter()-s)*1000)
        print(f'{backend:7s} batch={batch_size:2d} per-frame={np.mean(ts)/batch_size:5.1f} ms')
"
```

## Triton metrics reference (for your own profiling)

```bash
curl -s http://localhost:18002/metrics | grep yolov8n_onnx
```

Useful ratios:
- `nv_inference_count / nv_inference_exec_count` — average batch size achieved
- `nv_inference_compute_infer_duration_us / nv_inference_exec_count` — GPU per-batch compute
- `nv_inference_request_duration_us / nv_inference_count` — Triton-side per-frame total
- Difference between client-measured per-call latency and Triton-side latency = client overhead (mostly gRPC marshaling)
