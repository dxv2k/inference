"""
Plugin block: local YOLO detector that produces sv.Detections compatible with
Roboflow Inference's real workflow engine (ByteTracker, Velocity, visualizations).

Registered via `WORKFLOWS_PLUGINS=local_yolo_plugin` env var.
"""

from __future__ import annotations

import uuid
from typing import List, Literal, Optional, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field
from ultralytics import YOLO

from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    PREDICTION_TYPE_KEY,
)
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_sv_detections,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    LIST_OF_VALUES_KIND,
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


_MODEL_CACHE: dict[str, YOLO] = {}


def _get_model(weights: str, device: str) -> YOLO:
    key = f"{weights}@{device}"
    if key not in _MODEL_CACHE:
        m = YOLO(weights)
        m.to(device)
        _MODEL_CACHE[key] = m
    return _MODEL_CACHE[key]


SHORT_DESCRIPTION = "Run a locally-loaded Ultralytics YOLO model (no Roboflow API)."
LONG_DESCRIPTION = """
Drop-in detector block that loads an Ultralytics YOLO checkpoint from disk
(e.g. `yolov8n.pt`) and emits Roboflow-compatible `sv.Detections` so that
downstream blocks like ByteTracker, Velocity, BoundingBoxVisualization, and
LabelVisualization work without any API key.
"""


class LocalYoloManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Local YOLO (Ultralytics)",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
        },
        protected_namespaces=(),
    )
    type: Literal["local_models/ultralytics_yolo@v1"]
    image: WorkflowImageSelector = Field(description="Input image (with video metadata).")
    weights: Union[str, Selector()] = Field(
        default="yolov8n.pt",
        description="Path or name of Ultralytics YOLO weights.",
    )
    device: Union[str, Selector()] = Field(
        default="cuda",
        description="Torch device: cuda or cpu.",
    )
    confidence: Union[float, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        default=0.30,
        description="Detection confidence threshold.",
    )
    keep_classes: Optional[Union[List[str], Selector(kind=[LIST_OF_VALUES_KIND])]] = Field(
        default=None,
        description="If set, only detections whose class_name is in this list are emitted.",
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class LocalYoloBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LocalYoloManifest

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return []

    def run(
        self,
        image: WorkflowImageData,
        weights: str,
        device: str,
        confidence: float,
        keep_classes: Optional[List[str]] = None,
    ) -> BlockResult:
        model = _get_model(weights, device)
        np_img = image.numpy_image
        h, w = np_img.shape[:2]

        results = model.predict(np_img, conf=float(confidence), device=device, verbose=False)[0]
        boxes = results.boxes

        if boxes is None or len(boxes) == 0:
            xyxy = np.empty((0, 4), dtype=np.float32)
            cls_idx = np.empty((0,), dtype=int)
            conf = np.empty((0,), dtype=np.float32)
        else:
            xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
            cls_idx = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy().astype(np.float32)
        names_map = model.names

        if keep_classes:
            keep = {str(c) for c in keep_classes}
            class_names = [names_map[int(c)] for c in cls_idx]
            mask = np.array([n in keep for n in class_names], dtype=bool) if len(class_names) else np.array([], dtype=bool)
            xyxy = xyxy[mask]
            cls_idx = cls_idx[mask]
            conf = conf[mask]
        n = len(xyxy)

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=cls_idx,
        )
        detections.data[CLASS_NAME_DATA_KEY] = np.array(
            [names_map[int(c)] for c in cls_idx], dtype=object
        )
        detections.data[DETECTION_ID_KEY] = np.array(
            [str(uuid.uuid4()) for _ in range(n)], dtype=object
        )
        detections.data[IMAGE_DIMENSIONS_KEY] = (
            np.tile(np.array([h, w], dtype=int), (n, 1)) if n else np.empty((0, 2), dtype=int)
        )
        detections.data[PREDICTION_TYPE_KEY] = np.array(
            ["object-detection"] * n, dtype=object
        )
        detections = attach_parents_coordinates_to_sv_detections(
            detections=detections, image=image,
        )
        return {"predictions": detections}


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [LocalYoloBlockV1]
