"""
Plugin block: Ultralytics YOLO-World (open-vocabulary detector) as a
Roboflow workflow block.

Unlike local_yolo_plugin (fixed COCO classes), YOLO-World takes a list of
text prompts at inference time — useful for auto-annotation workflows
where the class set is data-driven, not model-baked.

Register via WORKFLOWS_PLUGINS env var (or add to the env var the demo
sets in real_engine_runner.py).
"""

from __future__ import annotations

import uuid
from typing import List, Literal, Optional, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field
from ultralytics import YOLOWorld

from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_sv_detections,
)
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    PREDICTION_TYPE_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
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


# Cache one model instance per (weights, device). Re-encoding CLIP prompts
# is expensive (~200ms), so we also remember the last prompts list per model
# instance and only re-encode when it changes.
_MODEL_CACHE: dict[str, YOLOWorld] = {}
_LAST_PROMPTS: dict[int, tuple[str, ...]] = {}


def _get_model(weights: str, device: str) -> YOLOWorld:
    key = f"{weights}@{device}"
    if key not in _MODEL_CACHE:
        m = YOLOWorld(weights)
        m.to(device)
        _MODEL_CACHE[key] = m
    return _MODEL_CACHE[key]


def _ensure_prompts(model: YOLOWorld, prompts: list[str]) -> None:
    """Only re-encode CLIP text features when the prompt list actually changes."""
    key = id(model)
    new = tuple(prompts)
    if _LAST_PROMPTS.get(key) != new:
        model.set_classes(list(prompts))
        _LAST_PROMPTS[key] = new


SHORT_DESCRIPTION = "Open-vocabulary YOLO-World detector — bounding boxes for arbitrary class names."
LONG_DESCRIPTION = """
Plugs an Ultralytics YOLO-World checkpoint into a Roboflow workflow.
Takes a list of text prompts (`prompts`) and emits sv.Detections compatible
with downstream blocks. Primary use case: auto-annotation, where the class
set is data-driven and you want bboxes for novel classes without re-training.
"""


class YoloWorldManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "YOLO-World (open-vocabulary)",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
        },
        protected_namespaces=(),
    )
    type: Literal["local_models/yolo_world@v1"]
    images: WorkflowImageSelector = Field(description="Input image(s). Batch-aware.")
    weights: Union[str, Selector()] = Field(
        default="yolov8s-world.pt",
        description="YOLO-World checkpoint. Options: yolov8s-world.pt / yolov8m-world.pt / "
                    "yolov8l-world.pt / yolov8x-world.pt (slower & more accurate as size grows).",
    )
    device: Union[str, Selector()] = Field(
        default="cuda",
        description="Torch device: cuda or cpu.",
    )
    confidence: Union[float, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        default=0.15,
        description="Detection confidence threshold. YOLO-World runs at lower confidence "
                    "than vanilla YOLO because open-vocabulary scores tend to be lower.",
    )
    prompts: Union[List[str], Selector(kind=[LIST_OF_VALUES_KIND])] = Field(
        default=["person", "car"],
        description="List of class-name prompts. Each becomes a detectable class via "
                    "CLIP text encoding. Examples: ['hard hat', 'safety vest', 'forklift'].",
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class YoloWorldBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return YoloWorldManifest

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return []

    def run(
        self,
        images: Batch[WorkflowImageData],
        weights: str,
        device: str,
        confidence: float,
        prompts: List[str],
    ) -> BlockResult:
        model = _get_model(weights, device)
        _ensure_prompts(model, prompts)

        frame_list = [img.numpy_image for img in images]
        results_list = model.predict(
            frame_list, conf=float(confidence), device=device, verbose=False
        )
        # After set_classes(), model.names maps idx → prompt verbatim.
        names_map = model.names

        outputs: List[dict] = []
        for img, results in zip(images, results_list):
            np_img = img.numpy_image
            h, w = np_img.shape[:2]
            boxes = results.boxes
            if boxes is None or len(boxes) == 0:
                xyxy = np.empty((0, 4), dtype=np.float32)
                cls_idx = np.empty((0,), dtype=int)
                conf = np.empty((0,), dtype=np.float32)
            else:
                xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
                cls_idx = boxes.cls.cpu().numpy().astype(int)
                conf = boxes.conf.cpu().numpy().astype(np.float32)
            n = len(xyxy)

            detections = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls_idx)
            detections.data[CLASS_NAME_DATA_KEY] = np.array(
                [names_map[int(c)] for c in cls_idx], dtype=object
            )
            detections.data[DETECTION_ID_KEY] = np.array(
                [str(uuid.uuid4()) for _ in range(n)], dtype=object
            )
            detections.data[IMAGE_DIMENSIONS_KEY] = (
                np.tile(np.array([h, w], dtype=int), (n, 1))
                if n else np.empty((0, 2), dtype=int)
            )
            detections.data[PREDICTION_TYPE_KEY] = np.array(
                ["object-detection"] * n, dtype=object
            )
            detections = attach_parents_coordinates_to_sv_detections(
                detections=detections, image=img,
            )
            outputs.append({"predictions": detections})
        return outputs


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [YoloWorldBlockV1]
