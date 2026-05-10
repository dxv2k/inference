"""
RTSP → real Roboflow workflow → callback alerts.

Two driver patterns:

1. `run_pipeline(rtsp_url, workflow_spec, on_alert)`
   Long-running InferencePipeline with our local plugin.
   Calls `on_alert(prediction, video_frame)` per processed frame.

2. `run_pipeline_with_webhook(rtsp_url, alert_url)`
   Same, but the workflow JSON itself contains a `webhook_sink@v1`
   step — alerts go directly from inside the engine to your URL,
   no Python callback needed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# Plugin registration must happen before inference imports.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("WORKFLOWS_PLUGINS", "local_yolo_plugin")

from inference.core.interfaces.camera.entities import (  # noqa: E402
    VideoFrame,
)
from inference.core.interfaces.camera.video_source import (  # noqa: E402
    BufferConsumptionStrategy,
    BufferFillingStrategy,
)
from inference.core.interfaces.stream.inference_pipeline import (  # noqa: E402
    InferencePipeline,
)


# A minimal "tripwire" workflow for RTSP: detect → track → time_in_zone →
# webhook_sink (only when an object has been in the zone > N seconds).
def make_rtsp_workflow(
    alert_url: str,
    keep_classes: list[str] | None = None,
    zone: list[list[int]] | None = None,
    dwell_threshold_s: float = 3.0,
) -> dict:
    keep_classes = keep_classes or ["person", "car", "truck", "bus", "motorbike", "bicycle"]
    zone = zone or [[100, 100], [540, 100], [540, 260], [100, 260]]
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "camera_id", "default_value": "cam-001"},
        ],
        "steps": [
            {
                "type": "local_models/ultralytics_yolo@v1",
                "name": "detect",
                "image": "$inputs.image",
                "weights": "yolov8n.pt",
                "device": "cuda",
                "confidence": 0.30,
                "keep_classes": keep_classes,
            },
            {
                "type": "roboflow_core/trackers_bytetrack@v1",
                "name": "track",
                "image": "$inputs.image",
                "detections": "$steps.detect.predictions",
                "minimum_consecutive_frames": 1,
                "lost_track_buffer": 30,
            },
            {
                "type": "roboflow_core/time_in_zone@v2",
                "name": "zone",
                "image": "$inputs.image",
                "detections": "$steps.track.tracked_detections",
                "zone": zone,
                "triggering_anchor": "BOTTOM_CENTER",
                "remove_out_of_zone_detections": True,
                "reset_out_of_zone_detections": True,
            },
            # Gate: only fire alert when at least one track exceeds the dwell threshold.
            # webhook_sink itself can do per-frame; we lean on time_in_zone giving us
            # the dwell metric and let the receiver dedupe.
            {
                "type": "roboflow_core/webhook_sink@v1",
                "name": "alert",
                "url": alert_url,
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "json_payload": {
                    "camera_id": "$inputs.camera_id",
                    "in_zone_count": "$steps.zone.timed_detections",
                    "dwell_threshold_s": dwell_threshold_s,
                },
                "fire_and_forget": True,
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": "detections", "selector": "$steps.zone.timed_detections"},
        ],
    }


def run_pipeline(
    rtsp_url: str,
    workflow_spec: dict,
    on_alert: Optional[Callable[[Any, VideoFrame], None]] = None,
    max_fps: Optional[float] = 6.0,
    workflows_parameters: Optional[dict] = None,
) -> InferencePipeline:
    """Start a pipeline against an RTSP URL. Returns the pipeline (call .join() to block)."""
    pipeline = InferencePipeline.init_with_workflow(
        video_reference=rtsp_url,
        workflow_specification=workflow_spec,
        api_key=None,
        max_fps=max_fps,
        workflows_parameters=workflows_parameters or {},
        on_prediction=on_alert,
        # Real-time tuning for live RTSP: never let the buffer back up.
        source_buffer_filling_strategy=BufferFillingStrategy.ADAPTIVE_DROP_OLDEST,
        source_buffer_consumption_strategy=BufferConsumptionStrategy.EAGER,
    )
    pipeline.start()
    return pipeline


def default_alert_printer(prediction: dict, video_frame: VideoFrame) -> None:
    """Example on_prediction callback — fire your own HTTP from here, log to DB, etc."""
    if not prediction:
        return
    dets = prediction.get("detections")
    n = len(dets) if dets is not None else 0
    if n == 0:
        return
    times = list(dets.data.get("time_in_zone", [])) if hasattr(dets, "data") else []
    in_zone = sum(1 for t in times if float(t) > 0.0)
    if in_zone:
        print(
            f"[alert] frame={video_frame.frame_id} ts={video_frame.frame_timestamp} "
            f"in_zone={in_zone} max_dwell_s={max(times, default=0):.2f}"
        )


# Convenience: spawn N pipelines, one per camera. Each pipeline is independent.
def run_multi_camera(
    cameras: dict[str, str],   # {camera_id: rtsp_url}
    alert_url: str,
    keep_classes: list[str] | None = None,
    zone: list[list[int]] | None = None,
    max_fps: float = 6.0,
) -> dict[str, InferencePipeline]:
    pipelines = {}
    for cam_id, url in cameras.items():
        spec = make_rtsp_workflow(alert_url=alert_url, keep_classes=keep_classes, zone=zone)
        pipelines[cam_id] = run_pipeline(
            rtsp_url=url,
            workflow_spec=spec,
            on_alert=default_alert_printer,
            max_fps=max_fps,
            workflows_parameters={"camera_id": cam_id},
        )
    return pipelines
