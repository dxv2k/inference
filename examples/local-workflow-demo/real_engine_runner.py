"""
Driver for the *actual* Roboflow Inference workflow engine — runs locally with
no API key by injecting our own LocalYoloBlock via the WORKFLOWS_PLUGINS hook.

Pipeline:
    LocalYolo → ByteTracker → Velocity → BoundingBoxVisualization → LabelVisualization
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Make the plugin discoverable by name and switch on plugin loading BEFORE
# any workflow code is imported.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
os.environ.setdefault("WORKFLOWS_PLUGINS", "local_yolo_plugin")

# Note: Some Roboflow model-manager defaults try to ping the cloud at import.
# Our workflow doesn't use a model_manager but we still pass one in init.
from inference.core.cache import cache  # noqa: E402
from inference.core.managers.base import ModelManager  # noqa: E402
from inference.core.registries.roboflow import RoboflowModelRegistry  # noqa: E402
from inference.core.workflows.core_steps.common.entities import StepExecutionMode  # noqa: E402
from inference.core.workflows.execution_engine.core import ExecutionEngine  # noqa: E402
from inference.core.workflows.execution_engine.entities.base import (  # noqa: E402
    ImageParentMetadata,
    OriginCoordinatesSystem,
    VideoMetadata,
    WorkflowImageData,
)


REAL_SPEED_WORKFLOW: dict = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "weights", "default_value": "yolov8n.pt"},
        {"type": "WorkflowParameter", "name": "device", "default_value": "cuda"},
        {"type": "WorkflowParameter", "name": "conf", "default_value": 0.30},
        {"type": "WorkflowParameter", "name": "pixels_per_meter", "default_value": 12.5},
    ],
    "steps": [
        {
            "type": "local_models/ultralytics_yolo@v1",
            "name": "detect",
            "image": "$inputs.image",
            "weights": "$inputs.weights",
            "device": "$inputs.device",
            "confidence": "$inputs.conf",
        },
        {
            "type": "roboflow_core/trackers_bytetrack@v1",
            "name": "track",
            "image": "$inputs.image",
            "detections": "$steps.detect.predictions",
            "minimum_iou_threshold": 0.30,
            "minimum_consecutive_frames": 1,
            "lost_track_buffer": 30,
            "track_activation_threshold": 0.25,
        },
        {
            "type": "roboflow_core/velocity@v1",
            "name": "speed",
            "image": "$inputs.image",
            "detections": "$steps.track.tracked_detections",
            "smoothing_alpha": 0.4,
            "pixels_per_meter": "$inputs.pixels_per_meter",
        },
        {
            "type": "roboflow_core/bounding_box_visualization@v1",
            "name": "boxes",
            "image": "$inputs.image",
            "predictions": "$steps.speed.velocity_detections",
            "thickness": 2,
            "copy_image": True,
        },
        {
            "type": "roboflow_core/label_visualization@v1",
            "name": "labels",
            "image": "$steps.boxes.image",
            "predictions": "$steps.speed.velocity_detections",
            "text": "Tracker Id",
            "text_position": "TOP_LEFT",
            "copy_image": False,
        },
    ],
    "outputs": [
        {"type": "JsonField", "name": "annotated", "selector": "$steps.labels.image"},
        {"type": "JsonField", "name": "detections", "selector": "$steps.speed.velocity_detections"},
    ],
}


SMART_CAMERA_WORKFLOW: dict = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "weights", "default_value": "yolov8n.pt"},
        {"type": "WorkflowParameter", "name": "device", "default_value": "cuda"},
        {"type": "WorkflowParameter", "name": "conf", "default_value": 0.30},
        {"type": "WorkflowParameter", "name": "keep_classes",
         "default_value": ["person", "car", "bicycle", "motorbike", "bus", "truck"]},
        {"type": "WorkflowParameter", "name": "zone",
         "default_value": [[100, 100], [540, 100], [540, 260], [100, 260]]},
    ],
    "steps": [
        {
            "type": "local_models/ultralytics_yolo@v1",
            "name": "detect",
            "image": "$inputs.image",
            "weights": "$inputs.weights",
            "device": "$inputs.device",
            "confidence": "$inputs.conf",
            "keep_classes": "$inputs.keep_classes",
        },
        {
            "type": "roboflow_core/trackers_bytetrack@v1",
            "name": "track",
            "image": "$inputs.image",
            "detections": "$steps.detect.predictions",
            "minimum_iou_threshold": 0.30,
            "minimum_consecutive_frames": 1,
            "lost_track_buffer": 30,
            "track_activation_threshold": 0.25,
        },
        {
            "type": "roboflow_core/time_in_zone@v2",
            "name": "zone",
            "image": "$inputs.image",
            "detections": "$steps.track.tracked_detections",
            "zone": "$inputs.zone",
            "triggering_anchor": "BOTTOM_CENTER",
            "remove_out_of_zone_detections": False,
            "reset_out_of_zone_detections": True,
        },
        {
            "type": "roboflow_core/polygon_zone_visualization@v1",
            "name": "zone_viz",
            "image": "$inputs.image",
            "zone": "$inputs.zone",
            "color": "#3b82f6",
            "opacity": 0.20,
            "copy_image": True,
        },
        {
            "type": "roboflow_core/bounding_box_visualization@v1",
            "name": "boxes",
            "image": "$steps.zone_viz.image",
            "predictions": "$steps.zone.timed_detections",
            "thickness": 2,
            "copy_image": False,
        },
        {
            "type": "roboflow_core/label_visualization@v1",
            "name": "labels",
            "image": "$steps.boxes.image",
            "predictions": "$steps.zone.timed_detections",
            "text": "Time In Zone",
            "text_position": "TOP_LEFT",
            "copy_image": False,
        },
    ],
    "outputs": [
        {"type": "JsonField", "name": "annotated", "selector": "$steps.labels.image"},
        {"type": "JsonField", "name": "detections", "selector": "$steps.zone.timed_detections"},
    ],
}


def _build_model_manager() -> ModelManager:
    """A model manager is required by the engine even if no Roboflow model is used."""
    return ModelManager(model_registry=RoboflowModelRegistry({}))


def _make_init_parameters():
    return {
        "workflows_core.model_manager": _build_model_manager(),
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }


# A persistent ThreadPoolExecutor avoids recreating worker threads each frame,
# which would otherwise reset cuDNN per-thread heuristics and slow conv2d ~6×.
# One executor is shared by all engines in this process — they each get their
# own pool of workers from it via max_concurrent_steps.
_PERSISTENT_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="wf-engine")


def init_engine() -> ExecutionEngine:
    return ExecutionEngine.init(
        workflow_definition=REAL_SPEED_WORKFLOW,
        init_parameters=_make_init_parameters(),
        max_concurrent_steps=1,
        executor=_PERSISTENT_EXECUTOR,
    )


def init_smart_engine() -> ExecutionEngine:
    return ExecutionEngine.init(
        workflow_definition=SMART_CAMERA_WORKFLOW,
        init_parameters=_make_init_parameters(),
        max_concurrent_steps=1,
        executor=_PERSISTENT_EXECUTOR,
    )


def wrap_frame(
    frame_rgb: np.ndarray, video_id: str, frame_number: int, fps: float
) -> WorkflowImageData:
    h, w = frame_rgb.shape[:2]
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(
            parent_id="image",
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=0, left_top_y=0, origin_width=w, origin_height=h
            ),
        ),
        numpy_image=frame_rgb,
        video_metadata=VideoMetadata(
            video_identifier=video_id,
            frame_number=frame_number,
            frame_timestamp=datetime.now(),
            fps=fps,
            comes_from_video_file=True,
        ),
    )


def _unwrap_image(image_or_data):
    return image_or_data.numpy_image if hasattr(image_or_data, "numpy_image") else image_or_data


def run_engine_on_frame(
    engine: ExecutionEngine,
    frame_rgb: np.ndarray,
    video_id: str,
    frame_number: int,
    fps: float,
    pixels_per_meter: float,
    confidence: float = 0.30,
    weights: str = "yolov8n.pt",
    device: str = "cuda",
) -> tuple[np.ndarray, Any, int]:
    img = wrap_frame(frame_rgb, video_id, frame_number, fps)
    t0 = time.perf_counter()
    result = engine.run(
        runtime_parameters={
            "image": [img],
            "weights": weights,
            "device": device,
            "conf": float(confidence),
            "pixels_per_meter": float(pixels_per_meter),
        }
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    out = result[0]
    return _unwrap_image(out["annotated"]), out["detections"], elapsed_ms


def run_smart_on_frame(
    engine: ExecutionEngine,
    frame_rgb: np.ndarray,
    video_id: str,
    frame_number: int,
    fps: float,
    zone: list[list[int]],
    confidence: float = 0.30,
    keep_classes: list[str] | None = None,
    weights: str = "yolov8n.pt",
    device: str = "cuda",
) -> tuple[np.ndarray, Any, int]:
    img = wrap_frame(frame_rgb, video_id, frame_number, fps)
    t0 = time.perf_counter()
    result = engine.run(
        runtime_parameters={
            "image": [img],
            "weights": weights,
            "device": device,
            "conf": float(confidence),
            "keep_classes": list(keep_classes) if keep_classes else None,
            "zone": zone,
        }
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    out = result[0]
    return _unwrap_image(out["annotated"]), out["detections"], elapsed_ms
