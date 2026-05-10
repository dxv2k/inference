"""
Stress test: N concurrent RTSP consumers running the speed-estimation
workflow against the SAME engine. Compares the PyTorch backend (local
yolov8n.pt) and the Triton backend (yolov8n_onnx via gRPC).

Each camera is a Python thread that:
  1. cv2.VideoCapture(rtsp_url)  with CAP_PROP_BUFFERSIZE=1
  2. Pulls frames (skipping every other) and feeds them to engine.run(...)
  3. Records per-call latency

Headline numbers reported at the end:
  • total frames processed
  • aggregate throughput (frames/s across all cameras)
  • per-frame latency p50 / p95 / p99 across all cameras

Usage:
    bash rtsp_sim/start_rtsp_sim.sh                 # ensure RTSP simulator is up
    bash optimize/triton/start_triton.sh            # only needed for triton runs
    uv run python stress_test.py --backend pytorch --cameras 10 --duration 30
    uv run python stress_test.py --backend triton  --cameras 10 --duration 30
"""

from __future__ import annotations

import argparse
import threading
import time
from collections import defaultdict

import cv2
import numpy as np

import real_engine_runner as rer


def consumer(
    cam_id: int,
    rtsp_url: str,
    engine,
    backend: str,
    triton_url: str,
    triton_model: str,
    duration_s: float,
    latencies: list[float],
    counters: dict,
    barrier: threading.Barrier,
) -> None:
    """One camera worker. Records latencies into the shared list."""
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        counters[f"cam{cam_id}_open_fail"] += 1
        barrier.wait()  # don't deadlock other cameras
        return

    barrier.wait()  # all cameras start measuring at the same instant
    # stop_at is set HERE so connection time isn't counted against the budget.
    stop_at = time.time() + duration_s
    video_id = f"stress-{backend}-{cam_id}"
    frame_idx = 0
    failed_reads = 0

    while time.time() < stop_at:
        ok, frame_bgr = cap.read()
        if not ok:
            failed_reads += 1
            if failed_reads >= 30:
                break
            time.sleep(0.01)
            continue
        failed_reads = 0
        frame_idx += 1
        if frame_idx % 2 != 0:   # match the demo's frame skip
            continue
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[1] > 960:
            s = 960 / rgb.shape[1]
            rgb = cv2.resize(rgb, (int(rgb.shape[1] * s), int(rgb.shape[0] * s)))

        try:
            t0 = time.perf_counter()
            rer.run_engine_on_frame(
                engine, rgb,
                video_id=video_id,
                frame_number=frame_idx,
                fps=30.0,
                pixels_per_meter=12.5,
                confidence=0.30,
                backend=backend,
                weights="yolov8n.pt",
                device="cuda",
                triton_url=triton_url,
                triton_model=triton_model,
            )
            latencies.append((time.perf_counter() - t0) * 1000)
            counters[f"cam{cam_id}_frames"] += 1
        except Exception as e:
            counters[f"cam{cam_id}_errors"] += 1
            if counters[f"cam{cam_id}_errors"] <= 3:
                print(f"[cam{cam_id}] error: {type(e).__name__}: {e}")

    cap.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["pytorch", "triton"], default="pytorch")
    ap.add_argument("--cameras", type=int, default=10)
    ap.add_argument("--duration", type=float, default=20.0,
                    help="Seconds of measurement window (after warmup).")
    ap.add_argument("--rtsp-url", default="rtsp://localhost:8554/traffic")
    ap.add_argument("--triton-url", default="localhost:18001")
    ap.add_argument("--triton-model", default="yolov8n_onnx")
    args = ap.parse_args()

    print(f"=== Stress test ===")
    print(f"  backend       : {args.backend}")
    print(f"  cameras       : {args.cameras}")
    print(f"  duration      : {args.duration}s")
    print(f"  rtsp_url      : {args.rtsp_url}")
    if args.backend == "triton":
        print(f"  triton_url    : {args.triton_url}")
        print(f"  triton_model  : {args.triton_model}")

    print(f"\nInitializing engine (backend={args.backend}) ...")
    engine = rer.init_engine(args.backend)

    # Warmup: 5 frames through the engine on the main thread so model is loaded
    # and any first-call overhead (cuDNN heuristics, gRPC connection) is paid.
    print("Warming up ...")
    cap = cv2.VideoCapture(args.rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(5):
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rer.run_engine_on_frame(
            engine, rgb, video_id="warmup", frame_number=0, fps=30.0,
            pixels_per_meter=12.5, confidence=0.30,
            backend=args.backend, weights="yolov8n.pt", device="cuda",
            triton_url=args.triton_url, triton_model=args.triton_model,
        )
    cap.release()
    print("Warmup done.")

    latencies: list[float] = []
    counters: dict[str, int] = defaultdict(int)
    barrier = threading.Barrier(args.cameras + 1)

    threads = []
    for i in range(args.cameras):
        t = threading.Thread(
            target=consumer,
            args=(i, args.rtsp_url, engine, args.backend, args.triton_url,
                  args.triton_model, args.duration, latencies, counters, barrier),
            daemon=True,
        )
        t.start()
        threads.append(t)

    print(f"\nWaiting for {args.cameras} cameras to connect ...")
    barrier.wait()
    t_start = time.time()
    print(f"All cameras connected. Measuring for {args.duration}s ...")

    for t in threads:
        t.join()

    wall = time.time() - t_start
    total_frames = sum(v for k, v in counters.items() if k.endswith("_frames"))
    errors = sum(v for k, v in counters.items() if k.endswith("_errors"))
    open_fails = sum(v for k, v in counters.items() if k.endswith("_open_fail"))

    print(f"\n=== Results: {args.backend} × {args.cameras} cameras ===")
    print(f"  wall time          : {wall:.2f} s")
    print(f"  total frames       : {total_frames}")
    print(f"  aggregate FPS      : {total_frames / wall:.1f}  (across all cameras)")
    print(f"  per-camera FPS     : {total_frames / wall / args.cameras:.1f}  (avg)")
    print(f"  open failures      : {open_fails}")
    print(f"  inference errors   : {errors}")
    if latencies:
        arr = np.asarray(latencies)
        print(f"  per-call latency   : "
              f"mean {arr.mean():.1f} ms  ·  "
              f"p50 {np.percentile(arr, 50):.1f}  ·  "
              f"p95 {np.percentile(arr, 95):.1f}  ·  "
              f"p99 {np.percentile(arr, 99):.1f}  ·  "
              f"max {arr.max():.1f}")


if __name__ == "__main__":
    main()
