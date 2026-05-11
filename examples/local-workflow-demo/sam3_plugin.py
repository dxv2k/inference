"""
Plugin block: SAM3 (Meta's text-prompted Segment Anything 3) wrapped as a
Roboflow WorkflowBlock. Uses Meta's `sam3` PyPI package + weights from
HuggingFace `facebook/sam3` — no Roboflow model bucket, no API key.

Pattern matches local_yolo_plugin / yolo_world_plugin. Emits sv.Detections
with `class_name` populated from the input prompt text, so downstream
visualization / tracker blocks work unchanged. Mask info is dropped — for
YOLO export we only need bboxes. The block computes the tight xyxy from
each mask via cv2.boundingRect.
"""

from __future__ import annotations

import os
import uuid
from typing import List, Literal, Optional, Type, Union

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image as PILImage
from pydantic import ConfigDict, Field

import sam3
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image as Sam3ImageDP,
    InferenceMetadata,
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)

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

# CLIP-style BPE vocabulary required by sam3's text encoder. Ships with the
# inference repo's perception_encoder package, so we re-use it instead of
# downloading a duplicate.
_DEFAULT_BPE_PATH = (
    "/media/ubuntu_data/viAct/inference/inference/models/perception_encoder/"
    "vision_encoder/bpe_simple_vocab_16e6.txt.gz"
)

# Lazy global cache — SAM3 is ~1.5 GB on first download and ~600 MB once on
# GPU, so we only ever want one instance.
_MODEL_LOCK = None  # set on first build
_MODEL = None
_TRANSFORM = None
_IMAGE_SIZE = int(os.environ.get("SAM3_IMAGE_SIZE", 1008))


def _build_model(bpe_path: str, device: str):
    global _MODEL, _TRANSFORM
    if _MODEL is None:
        _MODEL = sam3.build_sam3_image_model(
            bpe_path=bpe_path,
            device=device,
            load_from_HF=True,
            enable_segmentation=True,
        )
        _TRANSFORM = ComposeAPI(transforms=[
            RandomResizeAPI(
                sizes=_IMAGE_SIZE, max_size=_IMAGE_SIZE,
                square=True, consistent_transform=False,
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    return _MODEL, _TRANSFORM


def _build_text_query(coco_id: int, h: int, w: int, text: str) -> FindQueryLoaded:
    return FindQueryLoaded(
        query_text=text,
        image_id=0,
        object_ids_output=[],
        is_exhaustive=True,
        query_processing_order=coco_id,
        input_bbox=None,
        input_bbox_label=None,
        input_points=None,
        semantic_target=None,
        is_pixel_exhaustive=None,
        inference_metadata=InferenceMetadata(
            coco_image_id=coco_id,
            original_image_id=coco_id,
            original_category_id=1,
            original_size=(h, w),
            object_id=0,
            frame_index=0,
        ),
    )


def _mask_to_bbox(mask: np.ndarray) -> Optional[tuple[float, float, float, float]]:
    """Return xyxy tight bounding box from a binary mask, or None if empty."""
    if mask is None or mask.size == 0:
        return None
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


SHORT_DESCRIPTION = "Meta SAM3 with text prompts — self-hosted from HuggingFace facebook/sam3."
LONG_DESCRIPTION = """
Wraps Meta's SAM3 (Segment Anything 3) as a Roboflow workflow block. Takes
images + a list of text class prompts; emits sv.Detections where each
detection's bbox is the tight xyxy derived from its segmentation mask, and
`class_name` is set to the matching input prompt. Drop-in compatible with
downstream visualization and label_visualization blocks.

Weights download once from HuggingFace facebook/sam3 (~1.5 GB). No
Roboflow API key. No Roboflow model bucket. Single-digit FPS on a 3090 —
appropriate for offline auto-annotation, not real-time video.
"""


class Sam3Manifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "SAM3 (Meta, self-hosted)",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
        },
        protected_namespaces=(),
    )
    type: Literal["local_models/sam3@v1"]
    images: WorkflowImageSelector = Field(description="Input image(s). Batch-aware.")
    prompts: Union[List[str], Selector(kind=[LIST_OF_VALUES_KIND])] = Field(
        default=["person", "car"],
        description="List of text prompts (class names) to segment / detect.",
    )
    confidence: Union[float, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        default=0.35,
        description="Per-mask probability threshold. SAM3 confidence runs higher than YOLO-World.",
    )
    bpe_path: Union[str, Selector()] = Field(
        default=_DEFAULT_BPE_PATH,
        description="Path to CLIP-style BPE vocab (`bpe_simple_vocab_16e6.txt.gz`). "
                    "Bundled with the inference repo's perception_encoder.",
    )
    device: Union[str, Selector()] = Field(
        default="cuda",
        description="Torch device for SAM3.",
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


class Sam3BlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return Sam3Manifest

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return []

    def run(
        self,
        images: Batch[WorkflowImageData],
        prompts: List[str],
        confidence: float,
        bpe_path: str,
        device: str,
    ) -> BlockResult:
        model, transform = _build_model(bpe_path, device)
        post = PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=float(confidence),
            to_cpu=True,
        )

        outputs: List[dict] = []
        for image_idx, img in enumerate(images):
            np_img = img.numpy_image
            h, w = np_img.shape[:2]
            pil = PILImage.fromarray(np_img)

            # SAM3's image model asserts `num_frames == 1` (sam3_image.py:535):
            # each forward pass accepts exactly one text query. Loop the
            # prompts, merge results.
            xyxy_list: list[list[float]] = []
            conf_list: list[float] = []
            cls_list: list[int] = []
            name_list: list[str] = []

            for prompt_idx, text in enumerate(prompts):
                dp = Datapoint(
                    find_queries=[_build_text_query(coco_id=0, h=h, w=w, text=str(text))],
                    images=[Sam3ImageDP(data=pil, objects=[], size=(h, w))],
                )
                dp = transform(dp)
                batch = collate_fn_api(batch=[dp], dict_key="dummy")["dummy"]
                batch = copy_data_to_device(
                    batch, torch.device(device), non_blocking=True,
                )

                with torch.inference_mode():
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        raw = model(batch)
                        # PostProcessImage returns a single dict (one image)
                        # with tensors: scores (N,), labels (N,), boxes (N, 4),
                        # masks (N, 1, H, W). boxes are already in original
                        # image coords (use_original_sizes_box=True).
                        processed = post.process_results(raw, batch.find_metadatas)

                # processed = {coco_image_id (=0 in our queries):
                #               {scores, labels, boxes, masks}}
                if not isinstance(processed, dict) or not processed:
                    continue
                per_image = next(iter(processed.values()))
                if not isinstance(per_image, dict):
                    continue
                scores = per_image.get("scores")
                boxes = per_image.get("boxes")
                if scores is None or boxes is None:
                    continue
                scores_np = scores.float().cpu().numpy() if hasattr(scores, "cpu") else np.asarray(scores)
                boxes_np = boxes.cpu().numpy() if hasattr(boxes, "cpu") else np.asarray(boxes)
                keep_mask = scores_np >= float(confidence)
                kept_boxes = boxes_np[keep_mask]
                kept_scores = scores_np[keep_mask]
                cls_name = str(text)
                for box, s in zip(kept_boxes, kept_scores):
                    xyxy_list.append([float(box[0]), float(box[1]),
                                       float(box[2]), float(box[3])])
                    conf_list.append(float(s))
                    cls_list.append(prompt_idx)
                    name_list.append(cls_name)

            n = len(xyxy_list)
            if n:
                xyxy_arr = np.asarray(xyxy_list, dtype=np.float32)
                conf_arr = np.asarray(conf_list, dtype=np.float32)
                cls_arr = np.asarray(cls_list, dtype=int)
                name_arr = np.array(name_list, dtype=object)
            else:
                xyxy_arr = np.empty((0, 4), dtype=np.float32)
                conf_arr = np.empty((0,), dtype=np.float32)
                cls_arr = np.empty((0,), dtype=int)
                name_arr = np.array([], dtype=object)

            detections = sv.Detections(xyxy=xyxy_arr, confidence=conf_arr, class_id=cls_arr)
            detections.data[CLASS_NAME_DATA_KEY] = name_arr
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
            detections = attach_parents_coordinates_to_sv_detections(
                detections=detections, image=img,
            )
            outputs.append({"predictions": detections})
        return outputs


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [Sam3BlockV1]
