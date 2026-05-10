"""
Triton client plugin block — replaces local_yolo_plugin.py when you want
inference to come from a Triton Inference Server instead of a locally-loaded
Ultralytics model.

Workflow JSON change is one step type:
    "type": "local_models/ultralytics_yolo@v1"   →   "type": "triton/yolo@v1"

Plus two new params: triton_url, model_name.

Register via env var (mutually exclusive with the local plugin):
    WORKFLOWS_PLUGINS=triton_yolo_plugin

Requires:
    pip install tritonclient[grpc]   # or tritonclient[http]
"""

from __future__ import annotations

import uuid
from typing import List, Literal, Optional, Type, Union

import cv2
import numpy as np
import supervision as sv
import tritonclient.grpc as grpcclient
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

# Cache one client per Triton URL — gRPC connection setup is non-trivial.
_CLIENT_CACHE: dict[str, grpcclient.InferenceServerClient] = {}


def _get_client(triton_url: str) -> grpcclient.InferenceServerClient:
    if triton_url not in _CLIENT_CACHE:
        _CLIENT_CACHE[triton_url] = grpcclient.InferenceServerClient(
            url=triton_url, verbose=False,
        )
    return _CLIENT_CACHE[triton_url]


# COCO class names — bundled as a fallback. If your Triton model has different
# classes, override via the `class_names` parameter on the block.
COCO_CLASSES = (
    "person bicycle car motorbike aeroplane bus train truck boat traffic_light "
    "fire_hydrant stop_sign parking_meter bench bird cat dog horse sheep cow "
    "elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
    "skis snowboard sports_ball kite baseball_bat baseball_glove skateboard "
    "surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana "
    "apple sandwich orange broccoli carrot hot_dog pizza donut cake chair sofa "
    "pottedplant bed diningtable toilet tvmonitor laptop mouse remote keyboard "
    "cell_phone microwave oven toaster sink refrigerator book clock vase scissors "
    "teddy_bear hair_drier toothbrush"
).split()


def _letterbox(im: np.ndarray, new_shape: tuple[int, int] = (640, 640),
               color: tuple[int, int, int] = (114, 114, 114)) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize + pad image to new_shape (Ultralytics-compatible). Returns (im, scale, (pad_w, pad_h))."""
    h, w = im.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    if (w, h) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)


def _nms_yolov8(prediction: np.ndarray, conf_thres: float, iou_thres: float,
                max_det: int = 300) -> np.ndarray:
    """Lightweight NMS for YOLOv8 raw output. Input shape: (84, 8400) for COCO.
    Returns array of [x1, y1, x2, y2, conf, cls] per kept detection.
    """
    # YOLOv8 output is (4 + nc, num_anchors); transpose to (num_anchors, 4+nc).
    pred = prediction.T  # (8400, 84)
    boxes_xywh = pred[:, :4]                # cx, cy, w, h
    class_scores = pred[:, 4:]              # (8400, nc)
    cls = class_scores.argmax(axis=1)
    conf = class_scores.max(axis=1)
    keep = conf >= conf_thres
    boxes_xywh, conf, cls = boxes_xywh[keep], conf[keep], cls[keep]
    if len(boxes_xywh) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    # xywh → xyxy
    xyxy = np.empty_like(boxes_xywh)
    xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    # OpenCV NMS (per-class via class-offset trick)
    offset = cls.astype(np.float32) * 4096.0
    boxes_for_nms = xyxy.copy()
    boxes_for_nms[:, [0, 2]] += offset[:, None]
    boxes_for_nms[:, [1, 3]] += offset[:, None]
    idx = cv2.dnn.NMSBoxes(
        bboxes=boxes_for_nms.tolist(),
        scores=conf.tolist(),
        score_threshold=conf_thres,
        nms_threshold=iou_thres,
    )
    if len(idx) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    idx = np.asarray(idx).reshape(-1)[:max_det]
    out = np.concatenate([xyxy[idx], conf[idx, None], cls[idx, None].astype(np.float32)], axis=1)
    return out.astype(np.float32)


SHORT_DESCRIPTION = "Run a Triton-served YOLO model (no local weights, no Roboflow API)."
LONG_DESCRIPTION = """
Drop-in replacement for `local_models/ultralytics_yolo@v1` that gets predictions
from a Triton Inference Server. Pre-processes the image (letterbox to 640×640,
normalize, NCHW), gRPC-infers, runs NMS on the raw output, and emits
Roboflow-compatible `sv.Detections` so downstream blocks (ByteTracker, Velocity,
Visualizations, webhook_sink) work unchanged.
"""


class TritonYoloManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Triton YOLO",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
        },
        protected_namespaces=(),
    )
    type: Literal["triton/yolo@v1"]
    image: WorkflowImageSelector = Field(description="Input image.")
    triton_url: Union[str, Selector()] = Field(
        default="localhost:8001",
        description="Triton gRPC endpoint (host:port).",
    )
    model_name: Union[str, Selector()] = Field(
        default="yolov8n_onnx",
        description="Model name as registered in Triton's model_repository.",
    )
    input_name: Union[str, Selector()] = Field(
        default="images",
        description="Triton input tensor name (typically 'images' for YOLO ONNX exports).",
    )
    output_name: Union[str, Selector()] = Field(
        default="output0",
        description="Triton output tensor name (typically 'output0' for YOLO ONNX exports).",
    )
    imgsz: Union[int, Selector()] = Field(
        default=640,
        description="Square input size the model expects.",
    )
    confidence: Union[float, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        default=0.30,
        description="Score threshold applied during NMS post-processing.",
    )
    iou: Union[float, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        default=0.45,
        description="IoU threshold used by NMS.",
    )
    keep_classes: Optional[Union[List[str], Selector(kind=[LIST_OF_VALUES_KIND])]] = Field(
        default=None,
        description="If set, only emit detections whose class_name is in this list.",
    )
    class_names: Optional[Union[List[str], Selector(kind=[LIST_OF_VALUES_KIND])]] = Field(
        default=None,
        description="Custom class names (defaults to COCO 80 classes if omitted).",
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class TritonYoloBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return TritonYoloManifest

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return []

    def run(
        self,
        image: WorkflowImageData,
        triton_url: str,
        model_name: str,
        input_name: str,
        output_name: str,
        imgsz: int,
        confidence: float,
        iou: float,
        keep_classes: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
    ) -> BlockResult:
        np_img = image.numpy_image
        h, w = np_img.shape[:2]

        # 1. Preprocess: letterbox → normalize → NCHW float32
        lb, scale, (pad_w, pad_h) = _letterbox(np_img, (int(imgsz), int(imgsz)))
        x = lb.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None]   # HWC → 1×C×H×W
        x = np.ascontiguousarray(x)

        # 2. gRPC infer
        client = _get_client(triton_url)
        inp = grpcclient.InferInput(input_name, x.shape, "FP32")
        inp.set_data_from_numpy(x)
        out = grpcclient.InferRequestedOutput(output_name)
        result = client.infer(model_name=model_name, inputs=[inp], outputs=[out])
        raw = result.as_numpy(output_name)   # shape (1, 84, 8400) for COCO YOLOv8
        if raw is None:
            raise RuntimeError(f"Triton returned no output named {output_name!r}")

        # 3. Post-process: NMS in letterbox coords → un-letterbox to original image
        dets = _nms_yolov8(raw[0], conf_thres=float(confidence), iou_thres=float(iou))
        if len(dets):
            dets[:, [0, 2]] -= pad_w
            dets[:, [1, 3]] -= pad_h
            dets[:, :4] /= scale
            dets[:, [0, 2]] = dets[:, [0, 2]].clip(0, w - 1)
            dets[:, [1, 3]] = dets[:, [1, 3]].clip(0, h - 1)

        names = list(class_names) if class_names else list(COCO_CLASSES)

        # 4. Optional class filter
        if keep_classes:
            keep = {str(c) for c in keep_classes}
            cls_idx = dets[:, 5].astype(int) if len(dets) else np.empty((0,), dtype=int)
            mask = np.array([(names[c] in keep) if c < len(names) else False for c in cls_idx], dtype=bool)
            dets = dets[mask] if len(dets) else dets

        # 5. Build sv.Detections with the metadata Roboflow blocks expect
        n = len(dets)
        if n:
            xyxy = dets[:, :4].astype(np.float32)
            conf_arr = dets[:, 4].astype(np.float32)
            cls_arr = dets[:, 5].astype(int)
            class_name_arr = np.array(
                [names[c] if c < len(names) else f"cls_{c}" for c in cls_arr],
                dtype=object,
            )
        else:
            xyxy = np.empty((0, 4), dtype=np.float32)
            conf_arr = np.empty((0,), dtype=np.float32)
            cls_arr = np.empty((0,), dtype=int)
            class_name_arr = np.array([], dtype=object)

        detections = sv.Detections(xyxy=xyxy, confidence=conf_arr, class_id=cls_arr)
        detections.data[CLASS_NAME_DATA_KEY] = class_name_arr
        detections.data[DETECTION_ID_KEY] = np.array(
            [str(uuid.uuid4()) for _ in range(n)], dtype=object,
        )
        detections.data[IMAGE_DIMENSIONS_KEY] = (
            np.tile(np.array([h, w], dtype=int), (n, 1)) if n else np.empty((0, 2), dtype=int)
        )
        detections.data[PREDICTION_TYPE_KEY] = np.array(
            ["object-detection"] * n, dtype=object,
        )
        detections = attach_parents_coordinates_to_sv_detections(
            detections=detections, image=image,
        )
        return {"predictions": detections}


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [TritonYoloBlockV1]
