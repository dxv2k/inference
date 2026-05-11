"""
Driver for the *actual* Roboflow Inference workflow engine — runs locally with
no API key by injecting our own LocalYoloBlock via the WORKFLOWS_PLUGINS hook.

Pipeline:
    LocalYolo → ByteTracker → Velocity → BoundingBoxVisualization → LabelVisualization
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Load .env if present so OPENROUTER_API_KEY is available without exporting
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass

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


def _sam3_available() -> bool:
    try:
        import sam3  # noqa: F401
        return True
    except ImportError:
        return False


TRITON_AVAILABLE = _triton_available()
SAM3_AVAILABLE = _sam3_available()
_PLUGINS = ["local_yolo_plugin", "yolo_world_plugin"]
if SAM3_AVAILABLE:
    _PLUGINS.append("sam3_plugin")
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
             "default_value": "yolov8x-worldv2.pt"},
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


def make_sam3_autoannotate_workflow() -> dict:
    """Single-step detection via SAM3 + box/label viz. Drop-in replacement
    for the YOLO-World autoannotate workflow when you want segmentation-grade
    bboxes from text prompts."""
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "prompts",
             "default_value": ["person", "car"]},
            {"type": "WorkflowParameter", "name": "conf", "default_value": 0.35},
        ],
        "steps": [
            {
                "type": "local_models/sam3@v1",
                "name": "detect",
                "images": "$inputs.image",
                "prompts": "$inputs.prompts",
                "confidence": "$inputs.conf",
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


_VLM_SYSTEM_PROMPT = """You are an expert vision-data annotator. Look at the
sample image and propose a concise list of class names that should be
auto-annotated across a larger dataset of similar images.

Rules:
- Reply with ONLY a JSON array of strings. No prose, no markdown fences, no
  trailing commas. Example: ["person", "car", "traffic light"]
- Use short, lowercase class names suitable for an object detector.
- Prefer specific concrete classes (e.g. "forklift" not "vehicle"; "hard
  hat" not "ppe"). 5-15 classes is a good range; more is OK if the scene
  warrants it.
- If extra user context is given, weight class choices toward that context.
- JUST the JSON array — nothing else."""


def make_vlm_only_workflow() -> dict:
    """Single-step workflow: VLM emits a class list as plain text.

    Uses the upstream `roboflow_core/openai_compatible@v1` block — no custom
    plugin. The block produces a STRING output; the caller parses the JSON
    array out of it (gemini-flash-lite reliably returns clean JSON given
    the system prompt, but we have a tolerant parser regardless).
    """
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "context", "default_value": ""},
            {"type": "WorkflowParameter", "name": "model",
             "default_value": "google/gemini-3.1-flash-lite"},
            {"type": "WorkflowParameter", "name": "api_key", "default_value": ""},
            {"type": "WorkflowParameter", "name": "base_url",
             "default_value": "https://openrouter.ai/api/v1"},
        ],
        "steps": [
            {
                "type": "roboflow_core/openai_compatible@v1",
                "name": "vlm",
                "base_url": "$inputs.base_url",
                "model_name": "$inputs.model",
                "api_key": "$inputs.api_key",
                "system_prompt": _VLM_SYSTEM_PROMPT,
                "prompt": (
                    "Context (may be empty): {{ $parameters.context }}\n\n"
                    "Examine this image and propose the class list. "
                    "Respond with ONLY a JSON array of class names.\n"
                    "{{ $parameters.image }}"
                ),
                "prompt_parameters": {
                    "image": "$inputs.image",
                    "context": "$inputs.context",
                },
                "max_tokens": 400,
                "temperature": 0.2,
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": "raw_text", "selector": "$steps.vlm.output"},
            {"type": "JsonField", "name": "error", "selector": "$steps.vlm.error_status"},
        ],
    }


# Backwards compatibility: existing call sites that import the static specs.
REAL_SPEED_WORKFLOW: dict = make_speed_workflow("pytorch")
SMART_CAMERA_WORKFLOW: dict = make_smart_workflow("pytorch")
AUTOANNOTATE_WORKFLOW: dict = make_autoannotate_workflow()
VLM_PROMPT_WORKFLOW: dict = make_vlm_only_workflow()


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
    "autoannotate": lambda _backend: make_autoannotate_workflow(),       # backend-agnostic
    "vlm": lambda _backend: make_vlm_only_workflow(),                     # backend-agnostic
    "sam3_autoannotate": lambda _backend: make_sam3_autoannotate_workflow(),
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


def init_vlm_engine() -> ExecutionEngine:
    return _engine_for("vlm", "pytorch")


def init_sam3_engine() -> ExecutionEngine:
    if not SAM3_AVAILABLE:
        raise RuntimeError(
            "sam3 package not installed. uv pip install sam3==0.1.3"
        )
    return _engine_for("sam3_autoannotate", "pytorch")


_JSON_ARRAY_RE = re.compile(r"\[[^\[\]]*\]", re.DOTALL)


def _parse_class_list(text: str) -> list[str]:
    """Tolerant parser for VLM responses. gemini-flash-lite usually returns a
    clean JSON array but occasionally wraps it in ```json fences or trailing
    commentary."""
    if not text:
        return []
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        pass
    m = _JSON_ARRAY_RE.search(text)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass
    cleaned = text.strip("[]").replace('"', "").replace("'", "")
    return [c.strip() for c in cleaned.split(",") if c.strip()]


def vlm_is_configured() -> bool:
    """True iff the OpenRouter API key is set in env (loaded from .env if present)."""
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def run_vlm_prompt_suggest(
    engine: ExecutionEngine,
    sample_images_rgb: list[np.ndarray],
    user_context: str = "",
    model: str = "google/gemini-3.1-flash-lite",
) -> list[str]:
    """Call the VLM workflow engine on 1 sample image, return suggested classes.
    All inference happens via engine.run() — the upstream
    `roboflow_core/openai_compatible@v1` block makes the OpenRouter call.
    """
    if not sample_images_rgb:
        return []
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Copy .env.example -> .env and put your key in."
        )
    # The upstream block isn't batch-aware; one image per call. Use the first sample.
    wrapped = [wrap_frame(sample_images_rgb[0], "vlm-sample-0", 1, 1.0)]
    result = engine.run(runtime_parameters={
        "image": wrapped,
        "context": user_context or "",
        "model": model,
        "api_key": api_key,
        "base_url": os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    })
    if not result:
        return []
    raw_text = result[0].get("raw_text") or ""
    err = result[0].get("error")
    if err:
        raise RuntimeError(f"VLM block returned error: {err}")
    return _parse_class_list(str(raw_text))


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
    weights: str = "yolov8x-worldv2.pt",
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
    weights: str = "yolov8x-worldv2.pt",
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


def run_sam3_batch(
    engine: ExecutionEngine,
    images_rgb: list[np.ndarray],
    prompts: list[str],
    confidence: float = 0.35,
    batch_size: int = 1,
) -> list[tuple[np.ndarray, Any]]:
    """SAM3 batched auto-annotation. SAM3 is heavy — default batch_size=1 to
    keep GPU memory under control. Same shape as run_autoannotate_batch so
    the Tab 8 handler can swap between detectors freely."""
    if not images_rgb:
        return []
    all_results: list[tuple[np.ndarray, Any]] = []
    for chunk_start in range(0, len(images_rgb), batch_size):
        chunk = images_rgb[chunk_start:chunk_start + batch_size]
        wrapped = [
            wrap_frame(img, f"sam3-{chunk_start + i}", chunk_start + i, 1.0)
            for i, img in enumerate(chunk)
        ]
        result = engine.run(runtime_parameters={
            "image": wrapped,
            "prompts": list(prompts),
            "conf": float(confidence),
        })
        for r in result:
            all_results.append((_unwrap_image(r["annotated"]), r["detections"]))
    return all_results
