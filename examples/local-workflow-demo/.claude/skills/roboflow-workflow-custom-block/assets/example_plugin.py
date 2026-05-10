"""
Minimal Roboflow Workflows custom block — copy this and edit the inference call.

Register via env var BEFORE importing inference.*:
    os.environ["WORKFLOWS_PLUGINS"] = "example_plugin"

Then in a workflow JSON spec, add a step:
    {"type": "my_namespace/my_detector@v1",
     "name": "detect",
     "images": "$inputs.image",
     "confidence": 0.30}

The block emits sv.Detections with all metadata downstream blocks
(ByteTracker, Velocity, BoundingBoxVisualization, etc.) need.

Replace the body of `MyDetectorBlockV1.run()` with your own inference call —
the rest of the file (manifest, metadata population, parent attachment,
load_blocks) is the boilerplate you almost always want unchanged.
"""

from __future__ import annotations

import uuid
from typing import List, Literal, Optional, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    PREDICTION_TYPE_KEY,
)
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_sv_detections,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    Selector,
    WorkflowImageSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

CLASS_NAME_DATA_KEY = "class_name"

# --- Module-level model cache ---------------------------------------------
# Block instances are constructed often. Load weights once per (path, device).
_MODEL_CACHE: dict[str, object] = {}


def _get_model(weights: str, device: str):
    key = f"{weights}@{device}"
    if key not in _MODEL_CACHE:
        # Replace this with your own model loader:
        #   - torch.load(weights).to(device)
        #   - onnxruntime.InferenceSession(weights, providers=[...])
        #   - YOLO(weights).to(device)
        #   - your_sdk.load(weights, device=device)
        from ultralytics import YOLO   # example
        m = YOLO(weights); m.to(device)
        _MODEL_CACHE[key] = m
    return _MODEL_CACHE[key]


# --- Manifest -------------------------------------------------------------
class MyDetectorManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "My Detector",
            "version": "v1",
            "short_description": "Custom detector emitting Roboflow-compatible sv.Detections.",
            "license": "Apache-2.0",
            "block_type": "model",
        },
        protected_namespaces=(),
    )
    type: Literal["my_namespace/my_detector@v1"]
    images: WorkflowImageSelector = Field(description="Input image(s). Batch-aware.")
    weights: Union[str, Selector()] = Field(default="model.pt", description="Path to weights.")
    device: Union[str, Selector()] = Field(default="cuda", description="Torch device.")
    confidence: Union[float, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        default=0.30, description="Detection confidence threshold.",
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        # CRITICAL: enables batched run(). Without this the engine iterates
        # frame-by-frame and your batch dim never gets used.
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


# --- Block ----------------------------------------------------------------
class MyDetectorBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return MyDetectorManifest

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return []

    def run(
        self,
        images: Batch[WorkflowImageData],
        weights: str,
        device: str,
        confidence: float,
    ) -> BlockResult:
        model = _get_model(weights, device)

        # ---- 1. Run inference on the whole batch in one call ----
        # Replace this block with your own model call. Whatever you do, end
        # up with: per-image arrays of (xyxy, conf, cls_idx) and a list of
        # class names indexed by class id.
        frame_list = [img.numpy_image for img in images]
        results_list = model.predict(
            frame_list, conf=float(confidence), device=device, verbose=False,
        )
        names_map = model.names

        # ---- 2. Build sv.Detections per image with full metadata ----
        outputs: List[dict] = []
        for img, results in zip(images, results_list):
            h, w = img.numpy_image.shape[:2]
            boxes = results.boxes
            if boxes is None or len(boxes) == 0:
                xyxy = np.empty((0, 4), dtype=np.float32)
                cls_idx = np.empty((0,), dtype=int)
                conf_arr = np.empty((0,), dtype=np.float32)
            else:
                xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
                cls_idx = boxes.cls.cpu().numpy().astype(int)
                conf_arr = boxes.conf.cpu().numpy().astype(np.float32)

            n = len(xyxy)
            detections = sv.Detections(xyxy=xyxy, confidence=conf_arr, class_id=cls_idx)

            # These four data keys are the contract downstream blocks rely on.
            detections.data[CLASS_NAME_DATA_KEY] = np.array(
                [names_map[int(c)] for c in cls_idx], dtype=object,
            )
            detections.data[DETECTION_ID_KEY] = np.array(
                [str(uuid.uuid4()) for _ in range(n)], dtype=object,
            )
            detections.data[IMAGE_DIMENSIONS_KEY] = (
                np.tile(np.array([h, w], dtype=int), (n, 1))
                if n else np.empty((0, 2), dtype=int)
            )
            detections.data[PREDICTION_TYPE_KEY] = np.array(
                ["object-detection"] * n, dtype=object,
            )

            # Attach parent-coordinate metadata so Crop/Zone/Visualization
            # blocks can map detections back to the source frame.
            detections = attach_parents_coordinates_to_sv_detections(
                detections=detections, image=img,
            )
            outputs.append({"predictions": detections})
        return outputs


# --- Plugin entry point ---------------------------------------------------
def load_blocks() -> List[Type[WorkflowBlock]]:
    return [MyDetectorBlockV1]
