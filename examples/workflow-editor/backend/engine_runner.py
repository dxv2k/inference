"""Thin wrapper around ``real_engine_runner`` (sibling demo dir).

Responsibilities:

* Cache ad-hoc ``ExecutionEngine`` instances by a hash of the workflow spec.
* Wrap a raw RGB numpy image into a :class:`WorkflowImageData`.
* Serialize a workflow output dict (returned by ``engine.run``) into JSON-able
  primitives (base64 PNG for images, plain dicts for ``sv.Detections``, etc.).
* Load a builtin workflow by id by calling the factory named in
  ``prebuilt_workflows.json``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# IMPORTANT: backend/__init__.py imports real_engine_runner first to lock in
# WORKFLOWS_PLUGINS. Re-importing here is fine — Python caches the module.
import real_engine_runner as rer  # type: ignore  # noqa: E402

from inference.core.workflows.execution_engine.core import ExecutionEngine  # noqa: E402
from inference.core.workflows.execution_engine.entities.base import (  # noqa: E402
    ImageParentMetadata,
    OriginCoordinatesSystem,
    VideoMetadata,
    WorkflowImageData,
)


_HERE = Path(__file__).resolve().parent
_PREBUILT_PATH = _HERE.parent / "prebuilt_workflows.json"


# ----- engine cache --------------------------------------------------------


_AD_HOC_ENGINES: dict[str, ExecutionEngine] = {}


def _spec_hash(spec: dict) -> str:
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def get_engine_for_spec(spec: dict) -> ExecutionEngine:
    """Return a cached :class:`ExecutionEngine` for ``spec`` (hashed)."""
    key = _spec_hash(spec)
    if key not in _AD_HOC_ENGINES:
        _AD_HOC_ENGINES[key] = ExecutionEngine.init(
            workflow_definition=spec,
            init_parameters=rer._make_init_parameters(),
            max_concurrent_steps=1,
            executor=rer._PERSISTENT_EXECUTOR,
        )
    return _AD_HOC_ENGINES[key]


def make_init_parameters() -> dict:
    """Expose the demo's init-parameters helper to other backend modules."""
    return rer._make_init_parameters()


# ----- image wrapping ------------------------------------------------------


def wrap_image_for_engine(
    rgb: np.ndarray,
    video_id: str = "editor-upload",
    frame_number: int = 1,
    fps: float = 1.0,
) -> WorkflowImageData:
    """Wrap an RGB numpy image as a :class:`WorkflowImageData`.

    We could just reuse ``rer.wrap_frame``, but inlining keeps this module's
    public API self-documenting.
    """
    h, w = rgb.shape[:2]
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(
            parent_id="image",
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=0, left_top_y=0, origin_width=w, origin_height=h
            ),
        ),
        numpy_image=rgb,
        video_metadata=VideoMetadata(
            video_identifier=video_id,
            frame_number=frame_number,
            frame_timestamp=datetime.now(),
            fps=fps,
            comes_from_video_file=True,
        ),
    )


# ----- output serialization ------------------------------------------------


def _encode_image_b64(arr: np.ndarray) -> str:
    """Encode an RGB numpy array as base64 PNG (BGR-converted, per brief)."""
    if arr.ndim == 3 and arr.shape[2] == 3:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        bgr = arr
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _serialize_detections(det: Any) -> dict[str, Any]:
    """Convert an ``sv.Detections`` (duck-typed) into a JSON-able dict."""
    out: dict[str, Any] = {}
    xyxy = getattr(det, "xyxy", None)
    if xyxy is not None:
        out["xyxy"] = np.asarray(xyxy).tolist()
    cls = getattr(det, "class_id", None)
    if cls is not None:
        out["class_id"] = np.asarray(cls).tolist()
    conf = getattr(det, "confidence", None)
    if conf is not None:
        out["confidence"] = np.asarray(conf).tolist()
    tid = getattr(det, "tracker_id", None)
    if tid is not None:
        out["tracker_id"] = np.asarray(tid).tolist()
    data = getattr(det, "data", None) or {}
    if "class_name" in data:
        out["class_name"] = list(np.asarray(data["class_name"]).tolist())
    # Surface tracker_id from data if not on the top-level attr.
    if "tracker_id" not in out and "tracker_id" in data:
        out["tracker_id"] = list(np.asarray(data["tracker_id"]).tolist())
    return out


def _is_detections(value: Any) -> bool:
    # supervision is an optional dep, so duck-type instead of isinstance.
    return (
        hasattr(value, "xyxy")
        and hasattr(value, "class_id")
        and hasattr(value, "confidence")
    )


def serialize_output(value: Any) -> Any:
    """Convert a single workflow output value into something jsonable."""
    if value is None:
        return None
    # WorkflowImageData / anything with a .numpy_image -> base64 PNG
    if hasattr(value, "numpy_image"):
        try:
            return _encode_image_b64(np.asarray(value.numpy_image))
        except Exception as exc:  # pragma: no cover - defensive
            return {"_error": f"image encode failed: {exc}"}
    if isinstance(value, np.ndarray):
        # Treat HxWx{3,4} arrays as images; everything else as a list.
        if value.ndim == 3 and value.shape[2] in (3, 4):
            try:
                return _encode_image_b64(value)
            except Exception:
                return value.tolist()
        return value.tolist()
    if _is_detections(value):
        return _serialize_detections(value)
    if isinstance(value, (list, tuple)):
        return [serialize_output(v) for v in value]
    if isinstance(value, dict):
        return {k: serialize_output(v) for k, v in value.items()}
    # primitives
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    # Fallback: try json.dumps to test serializability, else stringify.
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def serialize_outputs_list(raw_outputs: list[dict]) -> list[dict[str, Any]]:
    """Serialize ``engine.run(...)``'s list of result dicts."""
    return [
        {k: serialize_output(v) for k, v in (item or {}).items()}
        for item in raw_outputs
    ]


# ----- builtin workflow loader --------------------------------------------


class BuiltinNotFound(KeyError):
    """Raised when a builtin workflow id is not in prebuilt_workflows.json."""


class BuiltinUnavailable(RuntimeError):
    """Raised when a builtin workflow's optional dep is missing (e.g. sam3)."""


def _load_prebuilt_index() -> dict[str, dict]:
    with _PREBUILT_PATH.open() as f:
        data = json.load(f)
    return {entry["id"]: entry for entry in data.get("workflows", [])}


def list_prebuilt_raw() -> dict:
    """Return the full ``prebuilt_workflows.json`` contents."""
    with _PREBUILT_PATH.open() as f:
        return json.load(f)


def load_prebuilt(prebuilt_id: str) -> dict:
    """Return the full builtin entry (with workflow spec materialized).

    Returns a dict like::

        {"id": "speed", "name": "Speed Estimation",
         "workflow": {...spec...}, "default_runtime_params": {...}}
    """
    index = _load_prebuilt_index()
    if prebuilt_id not in index:
        raise BuiltinNotFound(prebuilt_id)
    entry = index[prebuilt_id]

    if entry.get("requires") == "sam3" and not getattr(rer, "SAM3_AVAILABLE", False):
        raise BuiltinUnavailable(
            f"builtin workflow '{prebuilt_id}' requires the 'sam3' package"
        )

    factory_name = entry.get("factory")
    if not factory_name:
        raise BuiltinNotFound(f"no factory for {prebuilt_id}")
    factory = getattr(rer, factory_name, None)
    if factory is None:
        raise BuiltinNotFound(f"factory '{factory_name}' not found on real_engine_runner")

    kwargs = entry.get("factory_kwargs") or {}
    # Map "backend" kwarg, which is the only kwarg used today.
    spec = factory(**kwargs) if kwargs else factory()

    return {
        "id": prebuilt_id,
        "name": entry.get("name", prebuilt_id),
        "workflow": spec,
        "default_runtime_params": entry.get("default_runtime_params") or {},
    }


# ----- plugins listing for /health ----------------------------------------


def loaded_plugins() -> list[str]:
    raw = os.environ.get("WORKFLOWS_PLUGINS", "")
    return [p.strip() for p in raw.split(",") if p.strip()]
