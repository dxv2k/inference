"""
CV Workflow Demo — local GPU, no cloud.

Tabs:
  1. Workflow on a single image — drag/drop, see annotated output and the JSON spec.
  2. Live stream — multi-stage pipeline applied to video at ~10-15 FPS, with
     stage toggles, count-over-time chart, and alert log.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

import workflow as wf
import real_engine_runner as rer

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
def _font(name: str, size: int):
    try:
        return ImageFont.truetype(f"{FONT_DIR}/{name}", size)
    except Exception:
        return ImageFont.load_default()


def render_workflow_diagram(spec: dict) -> np.ndarray:
    """Render the workflow as a horizontal flow diagram of color-coded boxes."""
    stages = spec.get("stages", [])
    if not stages:
        return np.full((140, 600, 3), 248, dtype=np.uint8)

    box_w, box_h, gap, pad = 210, 140, 56, 28
    width = pad * 2 + len(stages) * box_w + (len(stages) - 1) * gap
    height = box_h + pad * 2 + 64

    img = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(img)

    f_title = _font("DejaVuSans-Bold.ttf", 18)
    f_label = _font("DejaVuSans-Bold.ttf", 15)
    f_type  = _font("DejaVuSansMono.ttf", 11)
    f_param = _font("DejaVuSans.ttf", 10)

    draw.text((pad, 12), spec.get("name", "workflow"), fill=(36, 38, 56), font=f_title)
    draw.text((pad + 320, 16),
              f"v{spec.get('version', '?')}  ·  {len(stages)} stages  ·  left → right per frame",
              fill=(150, 150, 170), font=f_param)

    # Color palette: (border, fill) per stage type
    palette = {
        "object_detection": ((37, 99, 235),   (235, 244, 255)),   # blue
        "class_filter":     ((124, 58, 237),  (244, 240, 255)),   # violet
        "iou_tracker":      ((22, 163, 74),   (235, 252, 240)),   # green
        "polygon_zone":     ((217, 119, 6),   (255, 247, 230)),   # amber
        "counter":          ((14, 165, 233),  (231, 246, 255)),   # sky
        "threshold_alert":  ((220, 38, 38),   (255, 235, 235)),   # red
        "speed_estimator":  ((16, 185, 129),  (209, 250, 229)),   # emerald
        "annotate":         ((71, 85, 105),   (244, 246, 250)),   # slate
    }

    y0 = 56
    for i, stage in enumerate(stages):
        x = pad + i * (box_w + gap)
        stype = stage.get("type", "")
        edge, fill = palette.get(stype, ((90, 90, 110), (240, 240, 245)))
        draw.rounded_rectangle((x, y0, x + box_w, y0 + box_h),
                               radius=14, fill=fill, outline=edge, width=2)
        draw.text((x + 14, y0 + 12),  stage.get("name", "?"), fill=edge, font=f_label)
        draw.text((x + 14, y0 + 36),  stype, fill=(80, 80, 110), font=f_type)

        # Params: show 3 most relevant
        params = stage.get("params") or {}
        items = []
        for k, v in params.items():
            if v is None:
                continue
            if isinstance(v, list):
                sval = "[" + ", ".join(str(x) for x in v[:2]) + ("…" if len(v) > 2 else "") + "]"
            elif isinstance(v, float):
                sval = f"{v:g}"
            else:
                sval = str(v)
            if len(sval) > 24:
                sval = sval[:21] + "…"
            items.append(f"{k} = {sval}")
        for j, line in enumerate(items[:4]):
            draw.text((x + 14, y0 + 60 + j * 16), line, fill=(60, 65, 90), font=f_param)

        # Arrow to next
        if i < len(stages) - 1:
            ax_start = x + box_w + 6
            ax_end = ax_start + gap - 12
            ay = y0 + box_h // 2
            draw.line([(ax_start, ay), (ax_end - 6, ay)], fill=(160, 168, 188), width=2)
            draw.polygon(
                [(ax_end, ay), (ax_end - 8, ay - 6), (ax_end - 8, ay + 6)],
                fill=(160, 168, 188),
            )

    return np.array(img)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "yolov8n.pt"
DEMO_TITLE = "Local CV Workflow Demo"
SAMPLE_VIDEO = "sample_video/sample.mp4"           # people in room — for smart-camera tab
TRAFFIC_VIDEO = "sample_video/traffic_cctv.mp4"   # highway CCTV — for speed-estimation tab
SAMPLE_DIR = Path("sample_images")
COCO_CLASSES_OF_INTEREST = ["person", "car", "bicycle", "motorbike", "bus", "truck",
                             "dog", "cat", "backpack", "handbag", "suitcase", "cell phone",
                             "laptop", "chair", "bottle", "cup"]

print(f"[demo] device={DEVICE}, gpus={torch.cuda.device_count() if DEVICE == 'cuda' else 0}")
model = YOLO(MODEL_PATH)
model.to(DEVICE)
print(f"[demo] model loaded: {MODEL_PATH}")
CTX = {"model": model, "device": DEVICE, "class_names": model.names}


def workflow_with_overrides(keep_classes, conf, threshold, draw_trails) -> dict:
    spec = json.loads(json.dumps(wf.DEFAULT_WORKFLOW))
    for stage in spec["stages"]:
        if stage["type"] == "object_detection":
            stage["params"]["conf"] = float(conf)
        elif stage["type"] == "class_filter":
            stage["params"]["keep"] = list(keep_classes) if keep_classes else None
        elif stage["type"] == "threshold_alert":
            stage["params"]["threshold"] = int(threshold)
        elif stage["type"] == "annotate":
            stage["params"]["draw_trails"] = bool(draw_trails)
    return spec


def stage_summary_md(state: wf.WorkflowState, total_ms: int) -> str:
    counts = state.counts or {}
    by_class = counts.get("by_class", {})
    by_class_rows = "\n".join(f"| {c} | {n} |" for c, n in sorted(by_class.items(), key=lambda x: -x[1])) or "| — | 0 |"
    stage_rows = "\n".join(f"| {n} | {ms} ms |" for n, ms in state.stage_latencies_ms.items()) or "| — | 0 ms |"
    alerts_md = "_no alerts_"
    if state.alerts:
        last = list(state.alerts)[-3:]
        alerts_md = "\n".join(
            f"- `{time.strftime('%H:%M:%S', time.localtime(a['ts']))}` — {a['msg']}"
            for a in reversed(last)
        )
    return f"""### Frame results — {total_ms} ms total on {DEVICE.upper()}

**Active tracks:** {counts.get('total', 0)} &nbsp;·&nbsp;
**In zone:** {counts.get('in_zone', 0)} &nbsp;·&nbsp;
**Unique seen since start:** {counts.get('ever_seen', 0)}

#### By class
| Class | Count |
|---|---|
{by_class_rows}

#### Stage latencies
| Stage | Time |
|---|---|
{stage_rows}

#### Recent alerts
{alerts_md}
"""


def run_single_frame(image, keep_classes, conf, threshold, draw_trails):
    spec = workflow_with_overrides(keep_classes, conf, threshold, draw_trails)
    diagram = render_workflow_diagram(spec)
    if image is None:
        return None, "_drop an image to start_", diagram, json.dumps(spec, indent=2)
    state = wf.reset_state()
    t0 = time.perf_counter()
    state = wf.run_workflow(state, image, spec, CTX)
    total_ms = int((time.perf_counter() - t0) * 1000)
    return (state.annotated if state.annotated is not None else image,
            stage_summary_md(state, total_ms),
            diagram,
            json.dumps(spec, indent=2))


def render_count_chart(history):
    W, H = 720, 220
    img = np.full((H, W, 3), 250, dtype=np.uint8)
    if not history:
        cv2.putText(img, "Count over time (waiting for data)", (12, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)
        return img
    times = [h[0] for h in history]
    totals = [h[1] for h in history]
    in_zones = [h[2] for h in history]
    t0, t1 = times[0], times[-1] if times[-1] > times[0] else times[0] + 1
    y_max = max(max(totals + in_zones, default=0), 5)
    pad = 32
    plot_w, plot_h = W - 2 * pad, H - 2 * pad

    def to_xy(t, y):
        x = pad + int((t - t0) / max(t1 - t0, 1e-6) * plot_w)
        yy = pad + plot_h - int(y / y_max * plot_h)
        return x, yy

    cv2.line(img, (pad, pad), (pad, pad + plot_h), (180, 180, 180), 1)
    cv2.line(img, (pad, pad + plot_h), (pad + plot_w, pad + plot_h), (180, 180, 180), 1)
    for i in (1, 2, 3):
        y = pad + plot_h - int(i / 4 * plot_h)
        cv2.line(img, (pad, y), (pad + plot_w, y), (230, 230, 230), 1)
        cv2.putText(img, str(int(y_max * i / 4)), (4, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)
    pts_total = np.array([to_xy(t, y) for t, y in zip(times, totals)], dtype=np.int32)
    cv2.polylines(img, [pts_total], False, (200, 100, 30), 2, cv2.LINE_AA)
    pts_zone = np.array([to_xy(t, y) for t, y in zip(times, in_zones)], dtype=np.int32)
    cv2.polylines(img, [pts_zone], False, (40, 60, 220), 2, cv2.LINE_AA)
    cv2.rectangle(img, (W - 200, 8), (W - 12, 56), (255, 255, 255), -1)
    cv2.rectangle(img, (W - 200, 8), (W - 12, 56), (200, 200, 200), 1)
    cv2.line(img, (W - 192, 22), (W - 168, 22), (200, 100, 30), 2)
    cv2.putText(img, "Active total", (W - 160, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.line(img, (W - 192, 44), (W - 168, 44), (40, 60, 220), 2)
    cv2.putText(img, "In zone",      (W - 160, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def stream_camera(source_choice, keep_classes, conf, threshold, draw_trails, enabled_stage_names):
    if source_choice == "Webcam (device 0)":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(SAMPLE_VIDEO)
    if not cap.isOpened():
        empty_chart = render_count_chart([])
        empty_diagram = render_workflow_diagram(workflow_with_overrides(keep_classes, conf, threshold, draw_trails))
        yield None, f"_cannot open source: {source_choice}_", empty_diagram, "{}", empty_chart
        return

    spec = workflow_with_overrides(keep_classes, conf, threshold, draw_trails)
    diagram = render_workflow_diagram(spec)
    state = wf.reset_state()
    enabled = set(enabled_stage_names) if enabled_stage_names else None

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame_idx += 1
        if frame_idx % 2 != 0:
            continue
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if w > 960:
            s = 960 / w
            rgb = cv2.resize(rgb, (int(w * s), int(h * s)))
        t0 = time.perf_counter()
        state = wf.run_workflow(state, rgb, spec, CTX, enabled=enabled)
        total_ms = int((time.perf_counter() - t0) * 1000)
        chart = render_count_chart(list(state.history_count))
        yield (state.annotated if state.annotated is not None else rgb,
               stage_summary_md(state, total_ms),
               diagram,
               json.dumps(spec, indent=2),
               chart)


def speed_workflow_with_overrides(keep_classes, conf, fps_hint, pixel_to_meter, smooth_n) -> dict:
    import json as _json
    spec = _json.loads(_json.dumps(wf.SPEED_WORKFLOW))
    for stage in spec["stages"]:
        if stage["type"] == "object_detection":
            stage["params"]["conf"] = float(conf)
        elif stage["type"] == "class_filter":
            stage["params"]["keep"] = list(keep_classes) if keep_classes else None
        elif stage["type"] == "speed_estimator":
            stage["params"]["fps"] = float(fps_hint)
            stage["params"]["pixel_to_meter"] = float(pixel_to_meter)
            stage["params"]["smoothing_frames"] = int(smooth_n)
    return spec


def speed_summary_md(state: wf.WorkflowState, total_ms: int) -> str:
    speeds = state.speeds or {}
    active_tracks = {tid: tr for tid, tr in state.tracks.items() if tr.missed == 0}
    rows = []
    for tid in sorted(active_tracks.keys()):
        tr = active_tracks[tid]
        spd = speeds.get(tid, 0.0)
        rows.append(f"| #{tid} | {tr.cls} | {spd:.1f} |")
    table = "\n".join(rows) if rows else "| — | — | 0 |"
    stage_rows = "\n".join(f"| {n} | {ms} ms |" for n, ms in state.stage_latencies_ms.items()) or "| — | 0 ms |"
    moving = sum(1 for s in speeds.values() if s > 1.0)
    avg_spd = (sum(speeds.values()) / len(speeds)) if speeds else 0.0
    return f"""### Speed estimation — {total_ms} ms on {DEVICE.upper()}

**Tracked objects:** {len(active_tracks)} &nbsp;·&nbsp;
**Moving (>1 km/h):** {moving} &nbsp;·&nbsp;
**Avg speed:** {avg_spd:.1f} km/h

#### Per-object speeds
| ID | Class | Speed (km/h) |
|---|---|---|
{table}

#### Stage latencies
| Stage | Time |
|---|---|
{stage_rows}
"""


def render_speed_chart(state: wf.WorkflowState) -> np.ndarray:
    """Bar chart of current per-track speeds."""
    W, H = 720, 220
    img = np.full((H, W, 3), 250, dtype=np.uint8)
    speeds = {tid: spd for tid, spd in state.speeds.items()
              if state.tracks.get(tid) and state.tracks[tid].missed == 0}
    if not speeds:
        cv2.putText(img, "Speed chart (waiting for tracks)", (12, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)
        return img
    pad = 36
    max_spd = max(max(speeds.values(), default=0), 10)
    bar_w = min(60, max(8, (W - 2 * pad) // max(len(speeds), 1) - 4))
    plot_h = H - 2 * pad
    x_cursor = pad
    for tid in sorted(speeds.keys()):
        spd = speeds[tid]
        bar_h = int(spd / max_spd * plot_h)
        x1 = x_cursor
        x2 = x1 + bar_w
        y1 = pad + plot_h - bar_h
        y2 = pad + plot_h
        # color gradient: green (slow) → red (fast)
        r = min(255, int(spd / max_spd * 510))
        g = min(255, int((1 - spd / max_spd) * 510))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, g, r), -1)
        cv2.putText(img, f"{spd:.0f}", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1, cv2.LINE_AA)
        cv2.putText(img, f"#{tid}", (x1, pad + plot_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1, cv2.LINE_AA)
        x_cursor += bar_w + 4
    cv2.line(img, (pad, pad), (pad, pad + plot_h), (180, 180, 180), 1)
    cv2.line(img, (pad, pad + plot_h), (W - pad, pad + plot_h), (180, 180, 180), 1)
    cv2.putText(img, "km/h", (4, pad + plot_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)
    return img


def stream_speed(source_choice, keep_classes, conf, fps_hint, pixel_to_meter, smooth_n):
    if source_choice == "Webcam (device 0)":
        cap = cv2.VideoCapture(0)
    elif source_choice == "Traffic CCTV sample":
        cap = cv2.VideoCapture(TRAFFIC_VIDEO if Path(TRAFFIC_VIDEO).exists() else SAMPLE_VIDEO)
    else:
        cap = cv2.VideoCapture(SAMPLE_VIDEO)
    if not cap.isOpened():
        empty_chart = render_speed_chart(wf.WorkflowState())
        empty_diagram = render_workflow_diagram(wf.SPEED_WORKFLOW)
        yield None, "_cannot open source_", empty_diagram, "{}", empty_chart
        return

    spec = speed_workflow_with_overrides(keep_classes, conf, fps_hint, pixel_to_meter, smooth_n)
    diagram = render_workflow_diagram(spec)
    state = wf.reset_state()

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame_idx += 1
        if frame_idx % 2 != 0:
            continue
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if w > 960:
            s = 960 / w
            rgb = cv2.resize(rgb, (int(w * s), int(h * s)))
        t0 = time.perf_counter()
        state = wf.run_workflow(state, rgb, spec, CTX)
        total_ms = int((time.perf_counter() - t0) * 1000)
        chart = render_speed_chart(state)
        yield (state.annotated if state.annotated is not None else rgb,
               speed_summary_md(state, total_ms),
               diagram,
               json.dumps(spec, indent=2),
               chart)


# Backend choices visible in the UI. "Triton" is only listed if tritonclient
# is importable AND the WORKFLOWS_PLUGINS env included the triton plugin.
BACKEND_CHOICES = ["PyTorch (local YOLOv8n)"] + (
    ["Triton (gRPC)"] if rer.TRITON_AVAILABLE else []
)


def _backend_id(label: str) -> str:
    return "triton" if "Triton" in label else "pytorch"


def get_real_engine(backend_label: str = BACKEND_CHOICES[0]):
    """Lazily init (and cache) one engine per backend for the speed workflow."""
    return rer.init_engine(backend=_backend_id(backend_label))


def get_smart_engine(backend_label: str = BACKEND_CHOICES[0]):
    """Lazily init (and cache) one engine per backend for the smart-camera workflow."""
    return rer.init_smart_engine(backend=_backend_id(backend_label))


def real_engine_summary_md(detections, total_ms: int, backend_label: str = "PyTorch") -> str:
    n = len(detections) if detections is not None else 0
    speeds_ms = list(detections.data.get("smoothed_speed", []))[:n] if hasattr(detections, "data") else []
    if not speeds_ms:
        speeds_ms = list(detections.data.get("speed", []))[:n] if hasattr(detections, "data") else []
    tracker_ids = list(detections.tracker_id) if hasattr(detections, "tracker_id") and detections.tracker_id is not None else [None] * n
    cls = list(detections.data.get("class_name", []))[:n] if hasattr(detections, "data") else [""] * n
    rows = []
    for i in range(n):
        spd_kmh = float(speeds_ms[i]) * 3.6 if i < len(speeds_ms) else 0.0
        tid = tracker_ids[i] if i < len(tracker_ids) else "?"
        rows.append(f"| #{tid} | {cls[i] if i < len(cls) else '?'} | {spd_kmh:.1f} |")
    table = "\n".join(rows) if rows else "| — | — | 0 |"
    moving = sum(1 for s in speeds_ms if float(s) * 3.6 > 1.0)
    avg_kmh = (sum(float(s) for s in speeds_ms) / len(speeds_ms) * 3.6) if speeds_ms else 0.0
    detector_block = "triton/yolo@v1" if "Triton" in backend_label else "local_models/ultralytics_yolo@v1"
    return f"""### Real Roboflow Engine — {total_ms} ms / frame  ·  Backend: **{backend_label}**

**Tracked objects:** {n} &nbsp;·&nbsp;
**Moving (>1 km/h):** {moving} &nbsp;·&nbsp;
**Avg speed:** {avg_kmh:.1f} km/h

#### Per-track output (from `roboflow_core/velocity@v1`)
| ID | Class | Speed (km/h) |
|---|---|---|
{table}

_Pipeline: `{detector_block}` → `roboflow_core/trackers_bytetrack@v1` → `roboflow_core/velocity@v1` → `bounding_box_visualization@v1` → `label_visualization@v1`_
"""


def render_real_workflow_diagram(backend: str = "pytorch") -> np.ndarray:
    detector_label = "triton/yolo@v1" if backend == "triton" else "local_models/ultralytics_yolo@v1"
    detector_params = ({"block": detector_label, "url": "$inputs.triton_url"} if backend == "triton"
                       else {"block": detector_label, "weights": "yolov8n.pt"})
    spec = {
        "version": "1.0",
        "name": f"real-roboflow-engine-{backend}",
        "stages": [
            {"name": "detect",   "type": "object_detection", "params": detector_params},
            {"name": "track",    "type": "iou_tracker",      "params": {"block": "roboflow_core/trackers_bytetrack@v1"}},
            {"name": "speed",    "type": "speed_estimator",  "params": {"block": "roboflow_core/velocity@v1"}},
            {"name": "boxes",    "type": "annotate",         "params": {"block": "bounding_box_visualization@v1"}},
            {"name": "labels",   "type": "annotate",         "params": {"block": "label_visualization@v1"}},
        ],
    }
    return render_workflow_diagram(spec)


def stream_real_engine(source_choice, conf, pixels_per_meter, backend, triton_url, triton_model):
    backend_id = _backend_id(backend)
    try:
        engine = get_real_engine(backend)
    except RuntimeError as e:
        yield None, f"_{e}_", render_real_workflow_diagram(), "{}"
        return
    if source_choice == "Webcam (device 0)":
        cap = cv2.VideoCapture(0)
    elif source_choice == "Sample video (loops)":
        cap = cv2.VideoCapture(SAMPLE_VIDEO)
    else:
        cap = cv2.VideoCapture(TRAFFIC_VIDEO if Path(TRAFFIC_VIDEO).exists() else SAMPLE_VIDEO)
    spec_dict = rer.make_speed_workflow(backend_id)
    diagram = render_real_workflow_diagram(backend_id)
    spec_json = json.dumps(spec_dict, indent=2)
    if not cap.isOpened():
        yield None, "_cannot open source_", diagram, spec_json
        return

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    video_id = f"video-{backend_id}-{int(time.time())}"
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame_idx += 1
        if frame_idx % 2 != 0:
            continue
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if w > 960:
            s = 960 / w
            rgb = cv2.resize(rgb, (int(w * s), int(h * s)))
        try:
            annotated, detections, total_ms = rer.run_engine_on_frame(
                engine, rgb,
                video_id=video_id,
                frame_number=frame_idx,
                fps=fps,
                pixels_per_meter=float(pixels_per_meter),
                confidence=float(conf),
                backend=backend_id,
                weights="yolov8n.pt",
                device=DEVICE,
                triton_url=str(triton_url or "localhost:8001"),
                triton_model=str(triton_model or "yolov8n_onnx"),
            )
        except Exception as e:
            yield None, f"_engine error ({backend_id}): {e}_", diagram, spec_json
            return
        yield annotated, real_engine_summary_md(detections, total_ms, backend), diagram, spec_json


def render_smart_workflow_diagram(backend: str = "pytorch") -> np.ndarray:
    detector_label = "triton/yolo@v1" if backend == "triton" else "local_models/ultralytics_yolo@v1"
    detector_params = ({"block": detector_label, "url": "$inputs.triton_url"} if backend == "triton"
                       else {"block": detector_label, "weights": "yolov8n.pt"})
    spec = {
        "version": "1.0",
        "name": f"real-roboflow-smart-camera-{backend}",
        "stages": [
            {"name": "detect",   "type": "object_detection", "params": detector_params},
            {"name": "track",    "type": "iou_tracker",      "params": {"block": "roboflow_core/trackers_bytetrack@v1"}},
            {"name": "zone",     "type": "polygon_zone",     "params": {"block": "roboflow_core/time_in_zone@v2"}},
            {"name": "zone_viz", "type": "annotate",         "params": {"block": "polygon_zone_visualization@v1"}},
            {"name": "boxes",    "type": "annotate",         "params": {"block": "bounding_box_visualization@v1"}},
            {"name": "labels",   "type": "annotate",         "params": {"block": "label_visualization@v1"}},
        ],
    }
    return render_workflow_diagram(spec)


def smart_real_summary_md(detections, total_ms: int, backend_label: str = "PyTorch") -> str:
    n = len(detections) if detections is not None else 0
    times = list(detections.data.get("time_in_zone", []))[:n] if hasattr(detections, "data") else []
    classes = list(detections.data.get("class_name", []))[:n] if hasattr(detections, "data") else []
    tracker_ids = list(detections.tracker_id) if hasattr(detections, "tracker_id") and detections.tracker_id is not None else [None] * n
    in_zone = sum(1 for t in times if float(t) > 0)
    rows = []
    for i in range(n):
        t = float(times[i]) if i < len(times) else 0.0
        rows.append(f"| #{tracker_ids[i] if i<len(tracker_ids) else '?'} | {classes[i] if i<len(classes) else '?'} | {t:.2f}s |")
    table = "\n".join(rows) if rows else "| — | — | 0 |"
    detector_block = "triton/yolo@v1" if "Triton" in backend_label else "local_models/ultralytics_yolo@v1"
    return f"""### Real Roboflow Engine (Smart Camera) — {total_ms} ms / frame  ·  Backend: **{backend_label}**

**Tracked objects:** {n} &nbsp;·&nbsp; **In zone:** {in_zone}

#### Per-track output (from `roboflow_core/time_in_zone@v2`)
| ID | Class | Time in zone |
|---|---|---|
{table}

_Pipeline: `{detector_block}` → `trackers_bytetrack@v1` → `time_in_zone@v2` → `polygon_zone_visualization@v1` → `bounding_box_visualization@v1` → `label_visualization@v1`_
"""


def _default_zone_for_frame(w: int, h: int) -> list[list[int]]:
    """Center rectangle covering 60% width, 50% height — good default."""
    cx, cy = w // 2, h // 2
    dx, dy = int(w * 0.30), int(h * 0.25)
    return [[cx - dx, cy - dy], [cx + dx, cy - dy],
            [cx + dx, cy + dy], [cx - dx, cy + dy]]


def stream_rtsp(rtsp_url, workflow_choice, conf, keep_classes_sel,
                alert_url, dwell_thresh_s, speed_thresh_kmh, ppm,
                backend, triton_url, triton_model):
    if not rtsp_url or not rtsp_url.strip():
        yield None, "_paste an RTSP URL above_", "_no alerts_"
        return
    is_speed = workflow_choice == "Speed Estimation"
    backend_id = _backend_id(backend)
    try:
        engine = get_real_engine(backend) if is_speed else get_smart_engine(backend)
    except RuntimeError as e:
        yield None, f"_{e}_", "_no alerts_"
        return

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        yield None, f"_cannot open RTSP URL: {rtsp_url}_", "_no alerts_"
        return

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    video_id = f"rtsp-{int(time.time())}"
    frame_idx = 0
    zone = None
    alerts_log: list[str] = []
    consecutive_failures = 0
    last_alerted_track: dict[int, float] = {}  # track_id → ts of last alert
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            consecutive_failures += 1
            if consecutive_failures >= 30:
                yield None, "_RTSP stream lost (30 consecutive read failures)_", "\n".join(alerts_log[-10:])
                return
            time.sleep(0.05)
            continue
        consecutive_failures = 0
        frame_idx += 1
        if frame_idx % 2 != 0:
            continue
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if w > 960:
            s = 960 / w
            rgb = cv2.resize(rgb, (int(w * s), int(h * s)))
            h, w = rgb.shape[:2]
        if zone is None:
            zone = _default_zone_for_frame(w, h)

        try:
            backend_kwargs = dict(
                backend=backend_id, weights="yolov8n.pt", device=DEVICE,
                triton_url=str(triton_url or "localhost:8001"),
                triton_model=str(triton_model or "yolov8n_onnx"),
            )
            if is_speed:
                annotated, dets, total_ms = rer.run_engine_on_frame(
                    engine, rgb,
                    video_id=video_id, frame_number=frame_idx, fps=fps,
                    pixels_per_meter=float(ppm), confidence=float(conf),
                    **backend_kwargs,
                )
            else:
                annotated, dets, total_ms = rer.run_smart_on_frame(
                    engine, rgb,
                    video_id=video_id, frame_number=frame_idx, fps=fps,
                    zone=zone, confidence=float(conf),
                    keep_classes=list(keep_classes_sel) if keep_classes_sel else None,
                    **backend_kwargs,
                )
        except Exception as e:
            yield None, f"_engine error ({backend_id}): {e}_", "\n".join(alerts_log[-10:])
            return

        ts_iso = time.strftime("%H:%M:%S")
        if is_speed:
            speeds_ms = list(dets.data.get("smoothed_speed", [])) if hasattr(dets, "data") else []
            if not speeds_ms:
                speeds_ms = list(dets.data.get("speed", [])) if hasattr(dets, "data") else []
            tracker_ids = list(dets.tracker_id) if hasattr(dets, "tracker_id") and dets.tracker_id is not None else []
            classes = list(dets.data.get("class_name", [])) if hasattr(dets, "data") else []
            speeds_kmh = [float(s) * 3.6 for s in speeds_ms]
            max_kmh = max(speeds_kmh, default=0.0)
            n_active = len(speeds_kmh)
            n_moving = sum(1 for s in speeds_kmh if s > 1.0)
            n_speeding = sum(1 for s in speeds_kmh if s > float(speed_thresh_kmh))

            now = time.time()
            for i, kmh in enumerate(speeds_kmh):
                if kmh <= float(speed_thresh_kmh):
                    continue
                tid = int(tracker_ids[i]) if i < len(tracker_ids) else -1
                if now - last_alerted_track.get(tid, 0) < 5.0:
                    continue  # 5s cooldown per track
                last_alerted_track[tid] = now
                cls = classes[i] if i < len(classes) else "?"
                line = f"`{ts_iso}` SPEEDING: track #{tid} {cls} @ **{kmh:.1f} km/h** (>{speed_thresh_kmh})"
                if alert_url and alert_url.strip():
                    try:
                        import requests
                        requests.post(alert_url.strip(), json={
                            "camera_id": video_id, "alert_type": "speeding",
                            "track_id": tid, "class": cls, "speed_kmh": round(kmh, 1),
                            "threshold_kmh": float(speed_thresh_kmh), "ts": ts_iso,
                        }, timeout=1.5)
                        line += "  → POSTed"
                    except Exception as e:
                        line += f"  → POST failed: {type(e).__name__}"
                alerts_log.append(line)
            alerts_log = alerts_log[-30:]
            summary = (f"### RTSP × Speed — {total_ms} ms / frame  ·  Backend: **{backend}**\n"
                       f"**Tracked:** {n_active} &nbsp;·&nbsp; **Moving:** {n_moving} &nbsp;·&nbsp; "
                       f"**Speeding (>{speed_thresh_kmh:.0f} km/h):** {n_speeding} &nbsp;·&nbsp; "
                       f"**Max:** {max_kmh:.1f} km/h\n\n"
                       f"_Pipeline: `local_yolo → bytetrack → velocity → boxes → labels`. "
                       f"Speeds smoothed by velocity block. Alerts dedup'd 5s per track._")
        else:
            times = list(dets.data.get("time_in_zone", [])) if hasattr(dets, "data") else []
            tracker_ids = list(dets.tracker_id) if hasattr(dets, "tracker_id") and dets.tracker_id is not None else []
            max_dwell = max((float(t) for t in times), default=0.0)
            in_zone = sum(1 for t in times if float(t) > 0.0)
            if max_dwell > float(dwell_thresh_s):
                line = f"`{ts_iso}` DWELL: in_zone={in_zone} max_dwell={max_dwell:.1f}s (>{dwell_thresh_s})"
                if alert_url and alert_url.strip():
                    try:
                        import requests
                        requests.post(alert_url.strip(), json={
                            "camera_id": video_id, "alert_type": "dwell",
                            "in_zone": in_zone, "max_dwell_s": round(max_dwell, 2),
                            "threshold_s": float(dwell_thresh_s), "ts": ts_iso,
                        }, timeout=1.5)
                        line += "  → POSTed"
                    except Exception as e:
                        line += f"  → POST failed: {type(e).__name__}"
                alerts_log.append(line)
                alerts_log = alerts_log[-30:]
            summary = (f"### RTSP × Smart Camera — {total_ms} ms / frame  ·  Backend: **{backend}**\n"
                       f"**In-zone tracks:** {in_zone} &nbsp;·&nbsp; **Max dwell:** {max_dwell:.1f}s\n\n"
                       f"_Pipeline: `local_yolo → bytetrack → time_in_zone → zone_viz → boxes → labels`._")

        yield annotated, summary, "\n".join(f"- {a}" for a in reversed(alerts_log[-15:])) or "_no alerts yet_"


def stream_real_smart(source_choice, conf, keep_classes_sel, backend, triton_url, triton_model):
    backend_id = _backend_id(backend)
    try:
        engine = get_smart_engine(backend)
    except RuntimeError as e:
        yield None, f"_{e}_", render_smart_workflow_diagram(backend_id), "{}"
        return
    if source_choice == "Webcam (device 0)":
        cap = cv2.VideoCapture(0)
    elif source_choice == "Traffic CCTV sample":
        cap = cv2.VideoCapture(TRAFFIC_VIDEO if Path(TRAFFIC_VIDEO).exists() else SAMPLE_VIDEO)
    else:
        cap = cv2.VideoCapture(SAMPLE_VIDEO)
    spec_dict = rer.make_smart_workflow(backend_id)
    diagram = render_smart_workflow_diagram(backend_id)
    spec_json = json.dumps(spec_dict, indent=2)
    if not cap.isOpened():
        yield None, "_cannot open source_", diagram, spec_json
        return

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    video_id = f"smart-{backend_id}-{int(time.time())}"
    frame_idx = 0
    zone = None
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame_idx += 1
        if frame_idx % 2 != 0:
            continue
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if w > 960:
            s = 960 / w
            rgb = cv2.resize(rgb, (int(w * s), int(h * s)))
            h, w = rgb.shape[:2]
        if zone is None:
            zone = _default_zone_for_frame(w, h)
        try:
            annotated, detections, total_ms = rer.run_smart_on_frame(
                engine, rgb,
                video_id=video_id,
                frame_number=frame_idx,
                fps=fps,
                zone=zone,
                confidence=float(conf),
                keep_classes=list(keep_classes_sel) if keep_classes_sel else None,
                backend=backend_id,
                weights="yolov8n.pt",
                device=DEVICE,
                triton_url=str(triton_url or "localhost:8001"),
                triton_model=str(triton_model or "yolov8n_onnx"),
            )
        except Exception as e:
            yield None, f"_engine error ({backend_id}): {e}_", diagram, spec_json
            return
        yield annotated, smart_real_summary_md(detections, total_ms, backend), diagram, spec_json


def render_autoannotate_diagram() -> np.ndarray:
    spec = {
        "version": "1.0",
        "name": "real-roboflow-autoannotate",
        "stages": [
            {"name": "detect", "type": "object_detection",
             "params": {"block": "local_models/yolo_world@v1",
                        "weights": "yolov8s-world.pt", "prompts": "$inputs.prompts"}},
            {"name": "boxes", "type": "annotate",
             "params": {"block": "bounding_box_visualization@v1"}},
            {"name": "labels", "type": "annotate",
             "params": {"block": "label_visualization@v1"}},
        ],
    }
    return render_workflow_diagram(spec)


def _detections_to_coco(detections, image_shape: tuple[int, int],
                       image_filename: str = "image.jpg") -> dict:
    """sv.Detections -> COCO JSON dict (single-image annotation)."""
    h, w = image_shape[:2]
    classes = list(detections.data.get("class_name", []))
    cat_map = {}
    annotations = []
    for i in range(len(detections)):
        cls = classes[i] if i < len(classes) else "obj"
        if cls not in cat_map:
            cat_map[cls] = len(cat_map) + 1
        x1, y1, x2, y2 = [float(v) for v in detections.xyxy[i]]
        annotations.append({
            "id": i + 1,
            "image_id": 1,
            "category_id": cat_map[cls],
            "bbox": [x1, y1, x2 - x1, y2 - y1],   # COCO is xywh
            "area": float((x2 - x1) * (y2 - y1)),
            "iscrowd": 0,
            "score": float(detections.confidence[i]) if detections.confidence is not None else 1.0,
        })
    return {
        "images": [{"id": 1, "file_name": image_filename, "width": int(w), "height": int(h)}],
        "categories": [{"id": cid, "name": name} for name, cid in cat_map.items()],
        "annotations": annotations,
    }


def _detections_to_yolo_txt(detections, image_shape: tuple[int, int],
                            prompts: list[str]) -> str:
    """sv.Detections -> YOLO TXT (one row per detection):
       `class_id cx cy w h`   (all normalized 0-1, cx/cy = center)."""
    h, w = image_shape[:2]
    classes = list(detections.data.get("class_name", []))
    prompt_index = {name: idx for idx, name in enumerate(prompts)}
    lines = []
    for i in range(len(detections)):
        cls = classes[i] if i < len(classes) else "obj"
        cls_id = prompt_index.get(cls, 0)
        x1, y1, x2, y2 = [float(v) for v in detections.xyxy[i]]
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines)


def run_autoannotate(image, prompts_text, confidence, weights):
    if image is None:
        return None, "_drop an image to auto-annotate_", "", "", render_autoannotate_diagram(), ""
    prompts = [p.strip() for p in prompts_text.split(",") if p.strip()]
    if not prompts:
        return image, "_enter at least one class prompt_", "", "", render_autoannotate_diagram(), ""

    engine = rer.init_autoannotate_engine()
    try:
        annotated, detections, total_ms = rer.run_autoannotate_on_image(
            engine, image, prompts=prompts, confidence=float(confidence),
            weights=str(weights), device=DEVICE,
        )
    except Exception as e:
        return image, f"_engine error: {e}_", "", "", render_autoannotate_diagram(), ""

    n = len(detections)
    classes = list(detections.data.get("class_name", []))
    confs = list(detections.confidence) if detections.confidence is not None else []
    by_class: dict[str, int] = {}
    for c in classes:
        by_class[c] = by_class.get(c, 0) + 1
    by_class_rows = "\n".join(f"| {c} | {n} |" for c, n in sorted(
        by_class.items(), key=lambda x: -x[1])) or "| — | 0 |"
    summary = f"""### Auto-annotation — {total_ms} ms on {DEVICE.upper()}

**Model:** `{weights}`  ·  **Prompts:** `{', '.join(prompts)}`
**Detections:** {n}  ·  **Mean confidence:** {(sum(confs)/len(confs)) if confs else 0:.2f}

#### Counts by class
| Class | Count |
|---|---|
{by_class_rows}

_Pipeline: `local_models/yolo_world@v1` → `bounding_box_visualization@v1` → `label_visualization@v1`_
"""
    coco_json = json.dumps(_detections_to_coco(detections, image.shape), indent=2)
    yolo_txt = _detections_to_yolo_txt(detections, image.shape, prompts)
    spec_json = json.dumps(rer.AUTOANNOTATE_WORKFLOW, indent=2)
    return annotated, summary, coco_json, yolo_txt, render_autoannotate_diagram(), spec_json


theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="cyan")

with gr.Blocks(title=DEMO_TITLE) as demo:
    gr.Markdown(f"""# {DEMO_TITLE}
**One engine. Different workflows. Different use-cases.**

Each tab below is a *different workflow JSON spec* running through the same pipeline engine (`run_workflow`).
Swap the spec — get a completely different AI capability. All self-hosted on this machine, YOLOv8 on **{DEVICE.upper()}**, no cloud required.

| Tab | Engine | Workflow | Use-case |
|---|---|---|---|
| 1 | Hand-rolled (`workflow.py`) | Single-image pipeline | Quick single-frame test |
| 2 | Hand-rolled (`workflow.py`) | Smart-camera workflow | Surveillance · counting · zone alerts |
| 3 | Hand-rolled (`workflow.py`) | Speed-estimation workflow | Per-object velocity from pixel displacement |
| **4** | **REAL Roboflow `ExecutionEngine`** | LocalYOLO + ByteTrack + Velocity | Speed estimation — backend swappable: **PyTorch** or **Triton** |
| **5** | **REAL Roboflow `ExecutionEngine`** | LocalYOLO + ByteTrack + TimeInZone | Smart camera — backend swappable: **PyTorch** or **Triton** |
| **6** | **REAL Roboflow `ExecutionEngine`** | RTSP → Tab-4 / Tab-5 workflow → webhook callback | Production pattern: RTSP URL + your alert API URL, backend swappable |
| **7** | **REAL Roboflow `ExecutionEngine`** | YOLO-World (open-vocab) | Auto-annotation: type any class names, get bboxes + COCO/YOLO export |

_Triton backend appears as an option only when `tritonclient` is installed AND the `optimize.triton.triton_yolo_plugin` is loaded. See `BACKENDS.md` and `optimize/triton/README.md`._
""")

    with gr.Tabs(selected=3):
        with gr.TabItem("1 · Single image", id=0):
            with gr.Row():
                with gr.Column(scale=1):
                    img_in = gr.Image(type="numpy", label="Input image")
                    run_btn = gr.Button("Run workflow", variant="primary", size="lg")
                    if SAMPLE_DIR.exists():
                        sample_paths = sorted(str(p) for p in SAMPLE_DIR.glob("*.jpg"))
                        if sample_paths:
                            gr.Examples(examples=sample_paths, inputs=img_in, label="Sample images")
                    keep_classes_1 = gr.CheckboxGroup(
                        choices=COCO_CLASSES_OF_INTEREST,
                        value=["person", "car", "bicycle", "motorbike", "bus", "truck"],
                        label="Classes to keep (filter stage)",
                    )
                    conf_1 = gr.Slider(0.05, 0.9, value=0.3, step=0.05, label="Detection confidence")
                    threshold_1 = gr.Slider(1, 20, value=3, step=1, label="Alert when in-zone count exceeds")
                    trails_1 = gr.Checkbox(value=True, label="Draw object trails")
                with gr.Column(scale=2):
                    img_out = gr.Image(label="Annotated output")
                    summary_1 = gr.Markdown()
            gr.Markdown("### Workflow pipeline")
            _initial_spec_1 = workflow_with_overrides(
                ["person", "car", "bicycle", "motorbike", "bus", "truck"], 0.3, 3, True
            )
            diagram_1 = gr.Image(
                label="Stages (left → right per frame)", show_label=False, height=240,
                value=render_workflow_diagram(_initial_spec_1),
            )
            with gr.Accordion("Raw JSON spec", open=False):
                spec_1 = gr.Code(language="json", label="workflow.json", lines=22)
            run_btn.click(
                run_single_frame,
                inputs=[img_in, keep_classes_1, conf_1, threshold_1, trails_1],
                outputs=[img_out, summary_1, diagram_1, spec_1],
            )

        with gr.TabItem("2 · Smart Camera (surveillance workflow)", id=1):
            with gr.Row():
                source_choice = gr.Radio(
                    choices=["Sample video (loops)", "Webcam (device 0)"],
                    value="Sample video (loops)",
                    label="Source",
                )
                stream_btn = gr.Button("Start stream", variant="primary", size="lg")
            with gr.Row():
                with gr.Column(scale=2):
                    stream_img = gr.Image(label="Annotated stream", streaming=True)
                with gr.Column(scale=1):
                    stream_summary = gr.Markdown()
            count_chart = gr.Image(label="Counts over time", streaming=True, height=220)
            with gr.Accordion("Workflow controls", open=True):
                keep_classes_2 = gr.CheckboxGroup(
                    choices=COCO_CLASSES_OF_INTEREST,
                    value=["person"],
                    label="Classes to keep",
                )
                with gr.Row():
                    conf_2 = gr.Slider(0.05, 0.9, value=0.3, step=0.05, label="Detection confidence")
                    threshold_2 = gr.Slider(1, 20, value=3, step=1, label="Alert threshold (in-zone count)")
                    trails_2 = gr.Checkbox(value=True, label="Draw object trails")
                stages_enabled = gr.CheckboxGroup(
                    choices=[s["name"] for s in wf.DEFAULT_WORKFLOW["stages"]],
                    value=[s["name"] for s in wf.DEFAULT_WORKFLOW["stages"]],
                    label="Enabled stages (uncheck to ablate)",
                )
            gr.Markdown("### Workflow pipeline")
            _initial_spec_2 = workflow_with_overrides(["person"], 0.3, 3, True)
            diagram_2 = gr.Image(
                label="Stages (left → right per frame)", show_label=False, height=240,
                value=render_workflow_diagram(_initial_spec_2),
            )
            with gr.Accordion("Raw JSON spec", open=False):
                spec_2 = gr.Code(language="json", label="workflow.json", lines=22)
            stream_btn.click(
                stream_camera,
                inputs=[source_choice, keep_classes_2, conf_2, threshold_2, trails_2, stages_enabled],
                outputs=[stream_img, stream_summary, diagram_2, spec_2, count_chart],
            )

        with gr.TabItem("3 · Speed Estimation (speed workflow)", id=2):
            gr.Markdown("""### Use-case: Object Speed Estimation
**Different workflow spec, same engine.** This workflow drops the zone/count/alert stages and adds
a `speed_estimator` stage that reads IoU-tracker history to compute per-object velocity (km/h).
Calibrate **meters-per-pixel** for your scene (default ~4 m camera height, wide-angle view).
""")
            with gr.Row():
                source_spd = gr.Radio(
                    choices=["Traffic CCTV sample", "Webcam (device 0)", "Sample video (loops)"],
                    value="Traffic CCTV sample",
                    label="Source",
                )
                speed_btn = gr.Button("Start speed estimation", variant="primary", size="lg")
            with gr.Row():
                with gr.Column(scale=2):
                    speed_img = gr.Image(label="Speed-annotated stream", streaming=True)
                with gr.Column(scale=1):
                    speed_summary = gr.Markdown()
            speed_chart = gr.Image(label="Speed bar chart (per active track)", streaming=True, height=220)
            with gr.Accordion("Speed & detection controls", open=True):
                keep_classes_spd = gr.CheckboxGroup(
                    choices=COCO_CLASSES_OF_INTEREST,
                    value=["person", "car", "bicycle", "motorbike"],
                    label="Classes to track",
                )
                with gr.Row():
                    conf_spd = gr.Slider(0.05, 0.9, value=0.35, step=0.05, label="Detection confidence")
                with gr.Row():
                    fps_hint = gr.Slider(5, 60, value=15, step=1,
                        label="Effective processing FPS (video fps ÷ 2, since we skip every other frame)")
                    pixel_to_meter = gr.Slider(0.001, 0.5, value=0.08, step=0.001,
                        label="Meters per pixel — tune for your scene: 0.05-0.1 = highway overhead cam, 0.01 = indoor")
                    smooth_n = gr.Slider(4, 30, value=10, step=1,
                        label="Smoothing frames (more = smoother speed reading, more latency)")
            gr.Markdown("### Speed estimation pipeline")
            _spd_diagram_initial = render_workflow_diagram(wf.SPEED_WORKFLOW)
            speed_diagram = gr.Image(
                label="Stages", show_label=False, height=240,
                value=_spd_diagram_initial,
            )
            with gr.Accordion("Raw JSON spec", open=False):
                speed_spec = gr.Code(language="json", label="speed_workflow.json", lines=22)
            speed_btn.click(
                stream_speed,
                inputs=[source_spd, keep_classes_spd, conf_spd, fps_hint, pixel_to_meter, smooth_n],
                outputs=[speed_img, speed_summary, speed_diagram, speed_spec, speed_chart],
            )

        with gr.TabItem("4 · Real Roboflow Engine (Speed)", id=3):
            gr.Markdown("""### Use-case: Speed Estimation — running on the **REAL** Roboflow workflow engine
This tab does NOT use our hand-rolled `workflow.py`. Every step here is a genuine Roboflow workflow block:
- `local_models/ultralytics_yolo@v1` — our custom plugin (`local_yolo_plugin.py`), wraps `yolov8n.pt`
- `roboflow_core/trackers_bytetrack@v1` — real ByteTrack
- `roboflow_core/velocity@v1` — real EMA-smoothed velocity (m/s with `pixels_per_meter` calibration)
- `roboflow_core/bounding_box_visualization@v1` and `roboflow_core/label_visualization@v1`

Plugged in via `WORKFLOWS_PLUGINS=local_yolo_plugin`. **No API key, no cloud, no `model_id`.**
""")
            with gr.Row():
                source_real = gr.Radio(
                    choices=["Traffic CCTV sample", "Sample video (loops)", "Webcam (device 0)"],
                    value="Traffic CCTV sample",
                    label="Source",
                )
                real_btn = gr.Button("Start (real engine)", variant="primary", size="lg")
            with gr.Row():
                with gr.Column(scale=2):
                    real_img = gr.Image(label="Real-engine annotated stream", streaming=True)
                with gr.Column(scale=1):
                    real_summary = gr.Markdown()
            with gr.Accordion("Engine controls", open=True):
                with gr.Row():
                    conf_real = gr.Slider(0.05, 0.9, value=0.30, step=0.05, label="Detection confidence")
                    ppm_real = gr.Slider(1.0, 100.0, value=12.5, step=0.5,
                        label="pixels_per_meter (Velocity block calibration — higher = slower km/h)")
                with gr.Row():
                    backend_real = gr.Radio(
                        choices=BACKEND_CHOICES, value=BACKEND_CHOICES[0],
                        label="Detector backend (swap the YOLO step — everything downstream stays the same)",
                    )
                    triton_url_real = gr.Textbox(
                        value="localhost:18001", label="Triton gRPC URL", scale=1,
                        info="ignored when backend is PyTorch",
                    )
                    triton_model_real = gr.Textbox(
                        value="yolov8n_onnx", label="Triton model name", scale=1,
                        info="must match a model in your Triton model_repository",
                    )
            gr.Markdown("### Real Roboflow workflow pipeline")
            real_diagram = gr.Image(
                label="Stages", show_label=False, height=240,
                value=render_real_workflow_diagram("pytorch"),
            )
            with gr.Accordion("Roboflow workflow JSON spec", open=False):
                real_spec = gr.Code(language="json", label="real_workflow.json", lines=30,
                                    value=json.dumps(rer.make_speed_workflow("pytorch"), indent=2))
            real_btn.click(
                stream_real_engine,
                inputs=[source_real, conf_real, ppm_real,
                        backend_real, triton_url_real, triton_model_real],
                outputs=[real_img, real_summary, real_diagram, real_spec],
            )

        with gr.TabItem("5 · Real Roboflow Engine (Smart Camera)", id=4):
            gr.Markdown("""### Use-case: Smart-camera surveillance — running on the **REAL** Roboflow workflow engine
Same surveillance use-case as Tab 2 (detect → track → zone analytics → annotate), but every step is a genuine Roboflow workflow block:
- `local_models/ultralytics_yolo@v1` — our plugin (with class-keep filter built in)
- `roboflow_core/trackers_bytetrack@v1` — real ByteTrack
- `roboflow_core/time_in_zone@v2` — real polygon-zone analytics, emits `time_in_zone` (seconds) per track
- `roboflow_core/polygon_zone_visualization@v1` — draws the zone overlay
- `roboflow_core/bounding_box_visualization@v1` + `roboflow_core/label_visualization@v1` — boxes + labels

No API key, no cloud. Plugged in via the same `WORKFLOWS_PLUGINS=local_yolo_plugin` mechanism as Tab 4.
""")
            with gr.Row():
                source_smart = gr.Radio(
                    choices=["Sample video (loops)", "Traffic CCTV sample", "Webcam (device 0)"],
                    value="Sample video (loops)",
                    label="Source",
                )
                smart_btn = gr.Button("Start (real smart-camera engine)", variant="primary", size="lg")
            with gr.Row():
                with gr.Column(scale=2):
                    smart_img = gr.Image(label="Real-engine annotated stream (zone + boxes + labels)", streaming=True)
                with gr.Column(scale=1):
                    smart_summary = gr.Markdown()
            with gr.Accordion("Engine controls", open=True):
                with gr.Row():
                    conf_smart = gr.Slider(0.05, 0.9, value=0.30, step=0.05, label="Detection confidence")
                    keep_smart = gr.CheckboxGroup(
                        choices=COCO_CLASSES_OF_INTEREST,
                        value=["person", "car", "bicycle", "motorbike", "bus", "truck"],
                        label="Classes to keep (filtered inside the YOLO block)",
                    )
                with gr.Row():
                    backend_smart = gr.Radio(
                        choices=BACKEND_CHOICES, value=BACKEND_CHOICES[0],
                        label="Detector backend",
                    )
                    triton_url_smart = gr.Textbox(value="localhost:18001", label="Triton gRPC URL", scale=1,
                                                  info="ignored when backend is PyTorch")
                    triton_model_smart = gr.Textbox(value="yolov8n_onnx", label="Triton model name", scale=1)
            gr.Markdown("### Real Roboflow smart-camera workflow pipeline")
            smart_diagram = gr.Image(
                label="Stages", show_label=False, height=240,
                value=render_smart_workflow_diagram("pytorch"),
            )
            with gr.Accordion("Roboflow workflow JSON spec", open=False):
                smart_spec = gr.Code(language="json", label="real_smart_workflow.json", lines=30,
                                     value=json.dumps(rer.make_smart_workflow("pytorch"), indent=2))
            smart_btn.click(
                stream_real_smart,
                inputs=[source_smart, conf_smart, keep_smart,
                        backend_smart, triton_url_smart, triton_model_smart],
                outputs=[smart_img, smart_summary, smart_diagram, smart_spec],
            )

        with gr.TabItem("6 · RTSP → Workflow → Webhook", id=5):
            gr.Markdown("""### Production pattern: RTSP camera → real workflow → callback your API

Paste an RTSP URL, pick a workflow, hit Start. The selected real-engine workflow runs against
the live stream and POSTs an alert to your webhook URL whenever an alert condition fires:

| Workflow | Alert condition | Payload |
|---|---|---|
| **Speed Estimation** | any tracked vehicle's smoothed speed > threshold (km/h) | `{track_id, class, speed_kmh, threshold_kmh, ts}` |
| **Smart Camera** | any tracked object's `time_in_zone` > threshold (sec) | `{in_zone, max_dwell_s, threshold_s, ts}` |

Speed alerts are deduplicated 5s per track to avoid spam. Webhook POSTs run with a 1.5s timeout
(production should set `fire_and_forget=True` on the `webhook_sink@v1` block instead).

**For headless production** (no UI): see `rtsp_runner.py` — uses `InferencePipeline.init_with_workflow(video_reference="rtsp://...", ...)`.
Buffer auto-set to `ADAPTIVE_DROP_OLDEST` + `EAGER` so you always process the freshest frame, with watchdog auto-reconnect.
""")
            with gr.Row():
                rtsp_url_in = gr.Textbox(
                    label="RTSP URL",
                    placeholder="rtsp://user:pass@192.168.1.10:554/Streaming/Channels/101",
                    value="rtsp://localhost:8554/traffic",
                    scale=3,
                )
                alert_url_in = gr.Textbox(
                    label="Your alert webhook URL (optional)",
                    placeholder="https://your-api.example.com/alerts",
                    scale=2,
                )
            with gr.Row():
                use_sim_btn = gr.Button("Use simulated traffic RTSP (localhost:8554)", size="sm")
                rtsp_btn = gr.Button("Start RTSP", variant="primary", size="lg")
            gr.Markdown("""
> **No real camera? Use the built-in simulator:**
> ```bash
> bash /home/dxv2k/inference-demo/rtsp_sim/start_rtsp_sim.sh
> ```
> This loops `traffic_cctv.mp4` into `rtsp://localhost:8554/traffic` via MediaMTX + ffmpeg.
""")
            use_sim_btn.click(lambda: "rtsp://localhost:8554/traffic", outputs=[rtsp_url_in])
            with gr.Row():
                with gr.Column(scale=2):
                    rtsp_img = gr.Image(label="RTSP annotated stream", streaming=True)
                with gr.Column(scale=1):
                    rtsp_summary = gr.Markdown()
                    rtsp_alerts = gr.Markdown(label="Alerts log")
            with gr.Row():
                rtsp_workflow = gr.Radio(
                    choices=["Speed Estimation", "Smart Camera (zone dwell)"],
                    value="Speed Estimation",
                    label="Workflow to run on the RTSP stream",
                )
            with gr.Accordion("Detection controls", open=True):
                with gr.Row():
                    rtsp_conf = gr.Slider(0.05, 0.9, value=0.30, step=0.05, label="Detection confidence")
                    rtsp_keep = gr.CheckboxGroup(
                        choices=COCO_CLASSES_OF_INTEREST,
                        value=["car", "truck", "bus", "motorbike"],
                        label="Classes (smart-camera filter only)",
                    )
                with gr.Row():
                    rtsp_speed_thresh = gr.Slider(5, 200, value=30, step=1,
                        label="Speed alert threshold (km/h) — fires when any track exceeds this")
                    rtsp_ppm = gr.Slider(1.0, 100.0, value=12.5, step=0.5,
                        label="pixels_per_meter (Velocity calibration)")
                    rtsp_dwell_thresh = gr.Slider(0.5, 30.0, value=2.0, step=0.5,
                        label="Dwell alert threshold (sec) — smart-camera only")
                with gr.Row():
                    rtsp_backend = gr.Radio(
                        choices=BACKEND_CHOICES, value=BACKEND_CHOICES[0],
                        label="Detector backend",
                    )
                    rtsp_triton_url = gr.Textbox(value="localhost:8001",
                        label="Triton gRPC URL", scale=1)
                    rtsp_triton_model = gr.Textbox(value="yolov8n_onnx",
                        label="Triton model name", scale=1)
            with gr.Accordion("Production pattern (rtsp_runner.py)", open=False):
                gr.Code(value="""# This UI uses cv2.VideoCapture(rtsp_url) for the live preview.
# For production (no UI), use rtsp_runner.py:

from rtsp_runner import make_rtsp_workflow, run_pipeline, default_alert_printer

spec = make_rtsp_workflow(
    alert_url="https://your-api.example.com/alerts",
    keep_classes=["person", "car"],
    zone=[[100,100],[540,100],[540,260],[100,260]],
    dwell_threshold_s=3.0,
)
pipeline = run_pipeline(
    rtsp_url="rtsp://user:pass@192.168.1.10:554/...",
    workflow_spec=spec,
    on_alert=default_alert_printer,   # optional: log alerts in Python too
    max_fps=6.0,
)
pipeline.join()  # blocks; auto-reconnects on stream drop via watchdog

# Multi-camera: run_multi_camera({"front": rtsp1, "back": rtsp2}, alert_url=...)
""", language="python", lines=18)
            rtsp_btn.click(
                stream_rtsp,
                inputs=[rtsp_url_in, rtsp_workflow, rtsp_conf, rtsp_keep,
                        alert_url_in, rtsp_dwell_thresh, rtsp_speed_thresh, rtsp_ppm,
                        rtsp_backend, rtsp_triton_url, rtsp_triton_model],
                outputs=[rtsp_img, rtsp_summary, rtsp_alerts],
            )

        with gr.TabItem("7 · Auto-annotation (YOLO-World)", id=6):
            gr.Markdown("""### Use-case: open-vocabulary auto-annotation
**No fine-tuning required.** Drop in an image, type a comma-separated list of class names you want bounding boxes for,
and get them — plus exportable labels in COCO JSON and YOLO TXT format. Useful for bootstrapping a dataset before training.

Powered by `local_models/yolo_world@v1` — our `yolo_world_plugin.py` wrapping Ultralytics' YOLO-World. Same plugin pattern as
the YOLOv8n block, but the model takes class prompts at inference time via CLIP text encoding.

| Workflow | Steps |
|---|---|
| Auto-annotate | `local_models/yolo_world@v1` → `bounding_box_visualization@v1` → `label_visualization@v1` |

**Tip:** YOLO-World scores are typically lower than COCO YOLO, so start with conf=0.10–0.15.
""")
            with gr.Row():
                with gr.Column(scale=1):
                    aa_img_in = gr.Image(type="numpy", label="Input image", height=320)
                    if SAMPLE_DIR.exists():
                        _aa_samples = sorted(str(p) for p in SAMPLE_DIR.glob("*.jpg"))
                        if _aa_samples:
                            gr.Examples(examples=_aa_samples, inputs=aa_img_in, label="Sample images")
                    aa_prompts = gr.Textbox(
                        value="person, bus, traffic light, backpack, dog, cat, bicycle",
                        label="Class prompts (comma-separated)",
                        info="Re-encoded by CLIP only when the list changes; cheap to swap.",
                    )
                    with gr.Row():
                        aa_conf = gr.Slider(0.01, 0.9, value=0.10, step=0.01,
                                            label="Confidence threshold")
                        aa_weights = gr.Dropdown(
                            choices=[
                                "yolov8s-world.pt",
                                "yolov8m-world.pt",
                                "yolov8l-world.pt",
                                "yolov8x-world.pt",
                            ],
                            value="yolov8s-world.pt",
                            label="YOLO-World checkpoint",
                            info="Larger = more accurate, slower; downloaded on first use.",
                        )
                    aa_btn = gr.Button("Auto-annotate", variant="primary", size="lg")
                with gr.Column(scale=2):
                    aa_img_out = gr.Image(label="Annotated output", height=480)
                    aa_summary = gr.Markdown()
            with gr.Tabs():
                with gr.TabItem("COCO JSON"):
                    aa_coco = gr.Code(language="json", label="coco.json", lines=18)
                with gr.TabItem("YOLO TXT"):
                    aa_yolo = gr.Code(language="markdown", label="image.txt (class_id cx cy w h, normalized)", lines=18)
                with gr.TabItem("Workflow JSON"):
                    aa_spec = gr.Code(language="json", label="autoannotate.json", lines=20,
                                      value=json.dumps(rer.AUTOANNOTATE_WORKFLOW, indent=2))
            gr.Markdown("### Workflow pipeline")
            aa_diagram = gr.Image(label="Stages", show_label=False, height=180,
                                  value=render_autoannotate_diagram())
            aa_btn.click(
                run_autoannotate,
                inputs=[aa_img_in, aa_prompts, aa_conf, aa_weights],
                outputs=[aa_img_out, aa_summary, aa_coco, aa_yolo, aa_diagram, aa_spec],
            )

    gr.Markdown("---")
    gr.Markdown(f"_{DEVICE.upper()} · "
                f"Tabs 1-3 use our hand-rolled engine (`workflow.py`); "
                f"Tabs 4-6 use the **real** Roboflow `ExecutionEngine` driven by `local_yolo_plugin` (PyTorch or Triton). "
                f"Tab 7 uses `yolo_world_plugin` for open-vocabulary auto-annotation. "
                f"No Roboflow API key required._")


if __name__ == "__main__":
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7872,
        show_error=True,
        share=False,
        theme=theme,
    )
