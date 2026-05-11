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

# Make plugins discoverable and configure WORKFLOWS_PLUGINS BEFORE importing
# inference. The Triton plugin is included only if tritonclient is importable;
# otherwise the engine init would fail trying to load it.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _triton_available() -> bool:
    try:
        import tritonclient.grpc  # noqa: F401
        return True
    except ImportError:
        return False


TRITON_AVAILABLE = _triton_available()
_PLUGINS = ["local_yolo_plugin", "yolo_world_plugin"]
if TRITON_AVAILABLE:
    _PLUGINS.append("optimize.triton.triton_yolo_plugin")
os.environ.setdefault("WORKFLOWS_PLUGINS", ",".join(_PLUGINS))

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


def _detect_step(backend: str, *, with_keep_classes: bool = False) -> dict:
    """Return the detector step appropriate for the chosen backend.
    Both backends emit the same `predictions` output kind, so downstream
    steps don't need to know which one ran.
    """
    if backend == "triton":
        step = {
            "type": "triton/yolo@v1",
            "name": "detect",
            "images": "$inputs.image",
            "triton_url": "$inputs.triton_url",
            "model_name": "$inputs.triton_model",
            "confidence": "$inputs.conf",
        }
    else:  # pytorch (default)
        step = {
            "type": "local_models/ultralytics_yolo@v1",
            "name": "detect",
            "images": "$inputs.image",
            "weights": "$inputs.weights",
            "device": "$inputs.device",
            "confidence": "$inputs.conf",
        }
    if with_keep_classes:
        step["keep_classes"] = "$inputs.keep_classes"
    return step


def _backend_inputs(backend: str) -> list[dict]:
    """Inputs that vary by backend (everything else is shared across workflows)."""
    if backend == "triton":
        return [
            {"type": "WorkflowParameter", "name": "triton_url", "default_value": "localhost:8001"},
            {"type": "WorkflowParameter", "name": "triton_model", "default_value": "yolov8n_onnx"},
        ]
    return [
        {"type": "WorkflowParameter", "name": "weights", "default_value": "yolov8n.pt"},
        {"type": "WorkflowParameter", "name": "device", "default_value": "cuda"},
    ]


def make_speed_workflow(backend: str = "pytorch") -> dict:
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            *_backend_inputs(backend),
            {"type": "WorkflowParameter", "name": "conf", "default_value": 0.30},
            {"type": "WorkflowParameter", "name": "pixels_per_meter", "default_value": 12.5},
        ],
        "steps": [
            _detect_step(backend),
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


def make_smart_workflow(backend: str = "pytorch") -> dict:
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            *_backend_inputs(backend),
            {"type": "WorkflowParameter", "name": "conf", "default_value": 0.30},
            {"type": "WorkflowParameter", "name": "keep_classes",
             "default_value": ["person", "car", "bicycle", "motorbike", "bus", "truck"]},
            {"type": "WorkflowParameter", "name": "zone",
             "default_value": [[100, 100], [540, 100], [540, 260], [100, 260]]},
        ],
        "steps": [
            _detect_step(backend, with_keep_classes=True),
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


def make_autoannotate_workflow() -> dict:
    """Open-vocabulary auto-annotation: YOLO-World detector + box/label viz.
    No tracking or temporal blocks — designed for single-image batches.
    """
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "prompts",
             "default_value": ["person", "car", "dog"]},
            {"type": "WorkflowParameter", "name": "weights",
             "default_value": "yolov8s-world.pt"},
            {"type": "WorkflowParameter", "name": "device", "default_value": "cuda"},
            {"type": "WorkflowParameter", "name": "conf", "default_value": 0.15},
        ],
        "steps": [
            {
                "type": "local_models/yolo_world@v1",
                "name": "detect",
                "images": "$inputs.image",
                "weights": "$inputs.weights",
                "device": "$inputs.device",
                "confidence": "$inputs.conf",
                "prompts": "$inputs.prompts",
            },
            {
                "type": "roboflow_core/bounding_box_visualization@v1",
                "name": "boxes",
                "image": "$inputs.image",
                "predictions": "$steps.detect.predictions",
                "thickness": 2,
                "copy_image": True,
            },
            {
                "type": "roboflow_core/label_visualization@v1",
                "name": "labels",
                "image": "$steps.boxes.image",
                "predictions": "$steps.detect.predictions",
                "text": "Class and Confidence",
                "text_position": "TOP_LEFT",
                "copy_image": False,
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": "annotated", "selector": "$steps.labels.image"},
            {"type": "JsonField", "name": "detections", "selector": "$steps.detect.predictions"},
        ],
    }


# Backwards compatibility: existing call sites that import the static specs.
REAL_SPEED_WORKFLOW: dict = make_speed_workflow("pytorch")
SMART_CAMERA_WORKFLOW: dict = make_smart_workflow("pytorch")
AUTOANNOTATE_WORKFLOW: dict = make_autoannotate_workflow()


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


# Cache one engine per (workflow, backend) pair. ByteTracker / Velocity /
# TimeInZone keep per-instance state (track IDs, dwell timers), so we want
# stable engine instances rather than rebuilding per frame.
_ENGINE_CACHE: dict[tuple[str, str], ExecutionEngine] = {}


_WORKFLOW_FACTORIES = {
    "speed": make_speed_workflow,
    "smart": make_smart_workflow,
    "autoannotate": lambda _backend: make_autoannotate_workflow(),  # backend-agnostic
}


def _engine_for(workflow: str, backend: str = "pytorch") -> ExecutionEngine:
    key = (workflow, backend)
    if key not in _ENGINE_CACHE:
        if backend == "triton" and not TRITON_AVAILABLE:
            raise RuntimeError(
                "Triton backend selected but tritonclient is not installed. "
                "Install with: uv pip install 'tritonclient[grpc]'"
            )
        factory = _WORKFLOW_FACTORIES.get(workflow)
        if factory is None:
            raise ValueError(f"unknown workflow type: {workflow!r}")
        spec = factory(backend)
        _ENGINE_CACHE[key] = ExecutionEngine.init(
            workflow_definition=spec,
            init_parameters=_make_init_parameters(),
            max_concurrent_steps=1,
            executor=_PERSISTENT_EXECUTOR,
        )
    return _ENGINE_CACHE[key]


def init_engine(backend: str = "pytorch") -> ExecutionEngine:
    return _engine_for("speed", backend)


def init_smart_engine(backend: str = "pytorch") -> ExecutionEngine:
    return _engine_for("smart", backend)


def init_autoannotate_engine() -> ExecutionEngine:
    return _engine_for("autoannotate", "pytorch")


def reset_engine_cache() -> None:
    """Clear cached engines — useful when a workflow/backend changes at runtime."""
    _ENGINE_CACHE.clear()


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


def _backend_runtime_params(
    backend: str, *, weights: str, device: str, triton_url: str, triton_model: str,
) -> dict:
    """Return only the runtime params relevant to the given backend.
    Passing extra params for the wrong backend would fail input validation."""
    if backend == "triton":
        return {"triton_url": triton_url, "triton_model": triton_model}
    return {"weights": weights, "device": device}


def run_engine_on_frame(
    engine: ExecutionEngine,
    frame_rgb: np.ndarray,
    video_id: str,
    frame_number: int,
    fps: float,
    pixels_per_meter: float,
    confidence: float = 0.30,
    *,
    backend: str = "pytorch",
    weights: str = "yolov8n.pt",
    device: str = "cuda",
    triton_url: str = "localhost:8001",
    triton_model: str = "yolov8n_onnx",
) -> tuple[np.ndarray, Any, int]:
    img = wrap_frame(frame_rgb, video_id, frame_number, fps)
    runtime = {
        "image": [img],
        "conf": float(confidence),
        "pixels_per_meter": float(pixels_per_meter),
        **_backend_runtime_params(
            backend, weights=weights, device=device,
            triton_url=triton_url, triton_model=triton_model,
        ),
    }
    t0 = time.perf_counter()
    result = engine.run(runtime_parameters=runtime)
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
    *,
    backend: str = "pytorch",
    weights: str = "yolov8n.pt",
    device: str = "cuda",
    triton_url: str = "localhost:8001",
    triton_model: str = "yolov8n_onnx",
) -> tuple[np.ndarray, Any, int]:
    img = wrap_frame(frame_rgb, video_id, frame_number, fps)
    runtime = {
        "image": [img],
        "conf": float(confidence),
        "keep_classes": list(keep_classes) if keep_classes else None,
        "zone": zone,
        **_backend_runtime_params(
            backend, weights=weights, device=device,
            triton_url=triton_url, triton_model=triton_model,
        ),
    }
    t0 = time.perf_counter()
    result = engine.run(runtime_parameters=runtime)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    out = result[0]
    return _unwrap_image(out["annotated"]), out["detections"], elapsed_ms



def run_autoannotate_on_image(
    engine: ExecutionEngine,
    image_rgb: np.ndarray,
    prompts: list[str],
    confidence: float = 0.15,
    weights: str = "yolov8s-world.pt",
    device: str = "cuda",
) -> tuple[np.ndarray, Any, int]:
    """Single-image auto-annotation. No video metadata needed — the workflow
    has no temporal blocks (no tracker, no velocity, no zone)."""
    img = wrap_frame(image_rgb, "autoannotate", 1, 1.0)
    t0 = time.perf_counter()
    result = engine.run(runtime_parameters={
        "image": [img],
        "prompts": list(prompts),
        "weights": weights,
        "device": device,
        "conf": float(confidence),
    })
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    out = result[0]
    return _unwrap_image(out["annotated"]), out["detections"], elapsed_ms


def run_autoannotate_batch(
    engine: ExecutionEngine,
    images_rgb: list[np.ndarray],
    prompts: list[str],
    confidence: float = 0.15,
    weights: str = "yolov8s-world.pt",
    device: str = "cuda",
    batch_size: int = 8,
) -> list[tuple[np.ndarray, Any]]:
    """Auto-annotate N images. Internally chunked into batches of `batch_size`
    so each engine.run() exercises the plugin's Batch[WorkflowImageData] path
    (one YOLO-World forward pass per batch). Returns N tuples of
    (annotated_rgb, sv.Detections) in the same order as the input list."""
    if not images_rgb:
        return []
    all_results: list[tuple[np.ndarray, Any]] = []
    for chunk_start in range(0, len(images_rgb), batch_size):
        chunk = images_rgb[chunk_start:chunk_start + batch_size]
        wrapped = [
            wrap_frame(img, f"autoannotate-{chunk_start + i}", chunk_start + i, 1.0)
            for i, img in enumerate(chunk)
        ]
        result = engine.run(runtime_parameters={
            "image": wrapped,
            "prompts": list(prompts),
            "weights": weights,
            "device": device,
            "conf": float(confidence),
        })
        for r in result:
            all_results.append((_unwrap_image(r["annotated"]), r["detections"]))
    return all_results
