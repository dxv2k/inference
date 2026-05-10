"""
Multi-stage local CV workflow.

Stage graph (executed in order; each consumes outputs of previous stages):

    detect  →  filter  →  track  →  zone  →  count  →  alert
                                            │           │
                                            └───►  annotate (viz overlay)

Each stage is a pure function `(state, frame, params) → state`.
The workflow definition is data: change the dict, change the behavior.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Workflow definition — this dict drives everything else
# ---------------------------------------------------------------------------

DEFAULT_WORKFLOW = {
    "version": "1.0",
    "name": "smart-camera",
    "stages": [
        {"name": "detect",   "type": "object_detection", "params": {"model": "yolov8n", "conf": 0.30}},
        {"name": "filter",   "type": "class_filter",     "params": {"keep": ["person", "car", "bicycle", "motorbike", "bus", "truck", "dog", "cat"]}},
        {"name": "track",    "type": "iou_tracker",      "params": {"iou_thresh": 0.30, "max_age": 20}},
        {"name": "zone",     "type": "polygon_zone",     "params": {"name": "Watch Zone"}},  # polygon set at runtime
        {"name": "count",    "type": "counter",          "params": {}},
        {"name": "alert",    "type": "threshold_alert",  "params": {"metric": "in_zone", "threshold": 3, "cooldown_s": 5.0}},
        {"name": "annotate", "type": "annotate",         "params": {"draw_trails": True, "trail_len": 24}},
    ],
}


# ---------------------------------------------------------------------------
# Mutable workflow state — passed between stages
# ---------------------------------------------------------------------------

@dataclass
class WorkflowState:
    detections: list[dict] = field(default_factory=list)   # post-detect/filter
    tracks: dict[int, "Track"] = field(default_factory=dict)
    zone_polygon: np.ndarray | None = None                 # Nx2 int
    in_zone_ids: set[int] = field(default_factory=set)
    counts: dict[str, Any] = field(default_factory=dict)
    alerts: deque = field(default_factory=lambda: deque(maxlen=20))
    last_alert_at: float = 0.0
    annotated: np.ndarray | None = None
    history_count: deque = field(default_factory=lambda: deque(maxlen=120))   # (ts, total, in_zone)
    stage_latencies_ms: dict[str, int] = field(default_factory=dict)
    frame_idx: int = 0
    speeds: dict[int, float] = field(default_factory=dict)  # track_id → km/h


@dataclass
class Track:
    id: int
    bbox: np.ndarray            # xyxy
    cls: str
    conf: float
    age: int = 0                # frames since first seen
    missed: int = 0             # consecutive frames without a match
    history: deque = field(default_factory=lambda: deque(maxlen=64))  # centers


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------

def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """xyxy IoU."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    a_area = (a[2] - a[0]) * (a[3] - a[1])
    b_area = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (a_area + b_area - inter + 1e-9)


def stage_detect(state: WorkflowState, frame: np.ndarray, params: dict, ctx: dict) -> WorkflowState:
    model = ctx["model"]
    results = model.predict(frame, conf=params.get("conf", 0.3), device=ctx["device"], verbose=False)[0]
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        state.detections = []
        return state
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    names = ctx["class_names"]
    state.detections = [
        {"bbox": xyxy[i], "cls": names[int(cls[i])], "conf": float(conf[i])}
        for i in range(len(xyxy))
    ]
    return state


def stage_class_filter(state: WorkflowState, frame: np.ndarray, params: dict, ctx: dict) -> WorkflowState:
    keep = set(params.get("keep") or [])
    if not keep:
        return state
    state.detections = [d for d in state.detections if d["cls"] in keep]
    return state


_NEXT_TRACK_ID = [1]


def stage_iou_tracker(state: WorkflowState, frame: np.ndarray, params: dict, ctx: dict) -> WorkflowState:
    iou_thresh = params.get("iou_thresh", 0.3)
    max_age = params.get("max_age", 20)

    # 1) Match existing tracks to new detections via greedy IoU
    detections = state.detections
    matched_det_idx: set[int] = set()
    matched_track_id: set[int] = set()

    # Precompute IoU pairs sorted descending
    pairs: list[tuple[float, int, int]] = []
    track_items = list(state.tracks.items())
    for i, det in enumerate(detections):
        for tid, tr in track_items:
            if tr.cls != det["cls"]:
                continue
            iou = _bbox_iou(det["bbox"], tr.bbox)
            if iou >= iou_thresh:
                pairs.append((iou, i, tid))
    pairs.sort(reverse=True)

    for iou, i, tid in pairs:
        if i in matched_det_idx or tid in matched_track_id:
            continue
        matched_det_idx.add(i)
        matched_track_id.add(tid)
        det = detections[i]
        tr = state.tracks[tid]
        tr.bbox = det["bbox"]
        tr.conf = det["conf"]
        tr.age += 1
        tr.missed = 0
        cx = (det["bbox"][0] + det["bbox"][2]) / 2
        cy = (det["bbox"][1] + det["bbox"][3]) / 2
        tr.history.append((float(cx), float(cy)))

    # 2) Unmatched detections become new tracks
    for i, det in enumerate(detections):
        if i in matched_det_idx:
            continue
        new_id = _NEXT_TRACK_ID[0]
        _NEXT_TRACK_ID[0] += 1
        cx = (det["bbox"][0] + det["bbox"][2]) / 2
        cy = (det["bbox"][1] + det["bbox"][3]) / 2
        tr = Track(id=new_id, bbox=det["bbox"], cls=det["cls"], conf=det["conf"], age=1)
        tr.history.append((float(cx), float(cy)))
        state.tracks[new_id] = tr

    # 3) Bump missed counter on unmatched tracks; drop stale
    drop_ids = []
    for tid, tr in state.tracks.items():
        if tid not in matched_track_id:
            tr.missed += 1
            if tr.missed > max_age:
                drop_ids.append(tid)
    for tid in drop_ids:
        del state.tracks[tid]

    return state


def stage_polygon_zone(state: WorkflowState, frame: np.ndarray, params: dict, ctx: dict) -> WorkflowState:
    if state.zone_polygon is None:
        # Default zone: centered rectangle covering 60% width, 70% height
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        dx, dy = int(w * 0.30), int(h * 0.35)
        state.zone_polygon = np.array([
            [cx - dx, cy - dy], [cx + dx, cy - dy],
            [cx + dx, cy + dy], [cx - dx, cy + dy],
        ], dtype=np.int32)

    poly = state.zone_polygon
    in_zone = set()
    for tid, tr in state.tracks.items():
        if tr.missed > 0:
            continue
        cx = (tr.bbox[0] + tr.bbox[2]) / 2
        cy = tr.bbox[3]  # foot of bbox
        if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
            in_zone.add(tid)
    state.in_zone_ids = in_zone
    return state


def stage_counter(state: WorkflowState, frame: np.ndarray, params: dict, ctx: dict) -> WorkflowState:
    active = [tr for tr in state.tracks.values() if tr.missed == 0]
    by_class = Counter(tr.cls for tr in active)
    state.counts = {
        "total": len(active),
        "by_class": dict(by_class),
        "in_zone": len(state.in_zone_ids),
        "ever_seen": _NEXT_TRACK_ID[0] - 1,  # cumulative unique IDs
    }
    state.history_count.append((time.time(), state.counts["total"], state.counts["in_zone"]))
    return state


def stage_threshold_alert(state: WorkflowState, frame: np.ndarray, params: dict, ctx: dict) -> WorkflowState:
    metric = params.get("metric", "total")
    threshold = params.get("threshold", 5)
    cooldown = params.get("cooldown_s", 5.0)
    value = state.counts.get(metric, 0)
    now = time.time()
    if value > threshold and (now - state.last_alert_at) > cooldown:
        state.alerts.append({
            "ts": now,
            "metric": metric,
            "value": value,
            "threshold": threshold,
            "msg": f"{metric.replace('_', ' ').title()} = {value} (>{threshold})",
        })
        state.last_alert_at = now
    return state


def _hsv_color(idx: int) -> tuple[int, int, int]:
    h = (idx * 47) % 180
    bgr = cv2.cvtColor(np.uint8([[[h, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def stage_speed_estimator(state: WorkflowState, frame: np.ndarray, params: dict, ctx: dict) -> WorkflowState:
    """Estimate per-track speed from IoU-tracker history (pixel displacement → km/h)."""
    fps = params.get("fps", 15)                  # effective processing fps
    m_per_px = params.get("pixel_to_meter", 0.015)
    smooth_n = int(params.get("smoothing_frames", 12))

    speeds: dict[int, float] = {}
    for tid, tr in state.tracks.items():
        if tr.missed > 0 or len(tr.history) < 2:
            speeds[tid] = 0.0
            continue
        pts = list(tr.history)[-smooth_n:]
        if len(pts) < 2:
            speeds[tid] = 0.0
            continue
        total_px = sum(
            ((pts[i + 1][0] - pts[i][0]) ** 2 + (pts[i + 1][1] - pts[i][1]) ** 2) ** 0.5
            for i in range(len(pts) - 1)
        )
        avg_px_per_frame = total_px / (len(pts) - 1)
        speed_kmh = avg_px_per_frame * fps * m_per_px * 3.6
        speeds[tid] = round(speed_kmh, 1)
    state.speeds = speeds
    return state


def stage_annotate(state: WorkflowState, frame: np.ndarray, params: dict, ctx: dict) -> WorkflowState:
    img = frame.copy()  # frame is RGB already
    draw_trails = params.get("draw_trails", True)
    trail_len = int(params.get("trail_len", 24))

    # 1) zone overlay (translucent fill + border)
    if state.zone_polygon is not None:
        overlay = img.copy()
        cv2.fillPoly(overlay, [state.zone_polygon], (255, 60, 60))
        img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
        cv2.polylines(img, [state.zone_polygon], True, (255, 60, 60), 2, cv2.LINE_AA)

    # 2) tracks: bbox + label + trail
    for tid, tr in state.tracks.items():
        if tr.missed > 0:
            continue
        x1, y1, x2, y2 = map(int, tr.bbox)
        in_zone = tid in state.in_zone_ids
        color = (255, 60, 60) if in_zone else _hsv_color(tid)
        thickness = 3 if in_zone else 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        label = f"#{tid} {tr.cls} {tr.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        if draw_trails and len(tr.history) >= 2:
            pts = np.array(list(tr.history)[-trail_len:], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)

        if params.get("show_speed", False) and state.speeds:
            spd = state.speeds.get(tid, 0.0)
            if spd > 0.3:
                spd_lbl = f"{spd:.1f} km/h"
                cv2.putText(img, spd_lbl, (x1, y2 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # 3) HUD: count badges + alert flag
    h, w = img.shape[:2]
    pad = 12
    box_h = 92
    cv2.rectangle(img, (pad, pad), (pad + 320, pad + box_h), (24, 24, 32), -1)
    cv2.rectangle(img, (pad, pad), (pad + 320, pad + box_h), (90, 90, 110), 1)
    total = state.counts.get("total", 0)
    in_zone = state.counts.get("in_zone", 0)
    ever = state.counts.get("ever_seen", 0)
    cv2.putText(img, f"Total active : {total}",   (pad + 10, pad + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"In zone      : {in_zone}", (pad + 10, pad + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (90, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"Unique seen  : {ever}",    (pad + 10, pad + 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2, cv2.LINE_AA)

    if state.alerts and (time.time() - state.alerts[-1]["ts"] < 2.0):
        msg = state.alerts[-1]["msg"]
        cv2.rectangle(img, (w - 480, pad), (w - pad, pad + 44), (40, 40, 220), -1)
        cv2.putText(img, "ALERT  " + msg, (w - 470, pad + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    state.annotated = img
    return state


# Stage registry
STAGE_REGISTRY = {
    "object_detection": stage_detect,
    "class_filter":     stage_class_filter,
    "iou_tracker":      stage_iou_tracker,
    "polygon_zone":     stage_polygon_zone,
    "counter":          stage_counter,
    "threshold_alert":  stage_threshold_alert,
    "speed_estimator":  stage_speed_estimator,
    "annotate":         stage_annotate,
}


SPEED_WORKFLOW = {
    "version": "1.0",
    "name": "speed-estimator",
    "stages": [
        {"name": "detect",   "type": "object_detection", "params": {"model": "yolov8n", "conf": 0.35}},
        {"name": "filter",   "type": "class_filter",     "params": {"keep": ["person", "car", "bicycle", "motorbike", "bus", "truck"]}},
        {"name": "track",    "type": "iou_tracker",      "params": {"iou_thresh": 0.30, "max_age": 30}},
        {"name": "speed",    "type": "speed_estimator",  "params": {"fps": 15, "pixel_to_meter": 0.08, "smoothing_frames": 10}},
        {"name": "annotate", "type": "annotate",         "params": {"draw_trails": True, "trail_len": 30, "show_speed": True}},
    ],
}


# ---------------------------------------------------------------------------
# Workflow runner
# ---------------------------------------------------------------------------

def run_workflow(state: WorkflowState, frame_rgb: np.ndarray, workflow: dict, ctx: dict, enabled: set[str] | None = None) -> WorkflowState:
    """Execute one frame through the workflow."""
    state.frame_idx += 1
    state.stage_latencies_ms = {}
    for stage in workflow.get("stages", []):
        name = stage["name"]
        if enabled is not None and name not in enabled:
            continue
        impl = STAGE_REGISTRY.get(stage["type"])
        if impl is None:
            continue
        t0 = time.perf_counter()
        state = impl(state, frame_rgb, stage.get("params", {}), ctx)
        state.stage_latencies_ms[name] = int((time.perf_counter() - t0) * 1000)
    return state


def reset_state() -> WorkflowState:
    """Fresh state — used when restarting a stream."""
    global _NEXT_TRACK_ID
    _NEXT_TRACK_ID[0] = 1
    return WorkflowState()
