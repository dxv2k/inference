# Block Contract Reference

Every custom block is two classes (Manifest + Block) plus a `load_blocks()`
function. This file documents what the engine actually requires.

## Table of contents

- [Imports](#imports)
- [Manifest class](#manifest-class)
- [Block class](#block-class)
- [Single-frame vs batched run() signatures](#single-frame-vs-batched-run-signatures)
- [The sv.Detections contract](#the-svdetections-contract)
- [Parent-coordinates attachment](#parent-coordinates-attachment)
- [Returning the BlockResult](#returning-the-blockresult)
- [Plugin entry point](#plugin-entry-point)

## Imports

The full set of imports a typical detector block uses:

```python
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
```

## Manifest class

The manifest is a Pydantic model that declares the block's identity, inputs,
and outputs. It is **what the workflow JSON validates against**.

```python
class MyBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Human-readable name (UI builder)",
            "version": "v1",
            "short_description": "One line.",
            "long_description": "Multi-line markdown.",
            "license": "Apache-2.0",
            "block_type": "model",   # or "transformation", "sink", "formatter"
        },
        protected_namespaces=(),     # silence pydantic warning on `model_*` fields
    )

    # The unique identifier referenced as `"type": "..."` in workflow JSON.
    # Convention: "namespace/block_name@version".
    type: Literal["my_namespace/my_block@v1"]

    # Image input(s). Use plural + WorkflowImageSelector for batch-aware blocks.
    images: WorkflowImageSelector = Field(description="Input image(s). Batch-aware.")

    # Scalar params can be hard-coded or wired from $inputs / $steps via Selector.
    confidence: Union[float, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(
        default=0.30, description="Confidence threshold.",
    )
    keep_classes: Optional[Union[List[str], Selector(kind=[LIST_OF_VALUES_KIND])]] = Field(
        default=None, description="Filter to these class names if set.",
    )
    weights: Union[str, Selector()] = Field(
        default="model.pt", description="Path to weights.",
    )

    # === The four classmethods you almost always need ===

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        # CRITICAL: without this, batching silently disables.
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"
```

### Common kind types (for `Selector(kind=[...])`)

| Kind | Used for |
|---|---|
| `OBJECT_DETECTION_PREDICTION_KIND` | block emits / consumes `sv.Detections` for object detection |
| `INSTANCE_SEGMENTATION_PREDICTION_KIND` | masks present on `sv.Detections` |
| `KEYPOINT_DETECTION_PREDICTION_KIND` | keypoints attached |
| `CLASSIFICATION_PREDICTION_KIND` | dict with `top` / `confidence` / `predictions` |
| `FLOAT_ZERO_TO_ONE_KIND` | scalar in [0, 1] (thresholds) |
| `INTEGER_KIND`, `FLOAT_KIND`, `STRING_KIND`, `BOOLEAN_KIND` | scalars |
| `LIST_OF_VALUES_KIND` | list-of-strings or list-of-numbers |
| `IMAGE_KIND` | `WorkflowImageData` |
| `RGB_COLOR_KIND` | `"#ff8800"` |

There are more (`POINT_KIND`, `ZONE_KIND`, `BOUNDING_BOX_KIND`, etc.); browse
`inference/core/workflows/execution_engine/entities/types.py` if a downstream
block expects something specific.

### Selector vs scalar

- `weights: Union[str, Selector()]` — accepts a literal string OR
  `"$inputs.weights"` / `"$steps.foo.bar"`. Always use this `Union` form for
  user-tunable params; lets the workflow author wire from anywhere.
- `imgsz: int = 640` — accepts a literal only. Use when the value is a
  block-level constant the workflow author shouldn't override.

## Block class

The block class wraps the `run()` method and tells the engine which manifest
it belongs to.

```python
class MyBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return MyBlockManifest

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        # Names of init_parameters from the engine's init_parameters dict
        # that should be passed to __init__. Most custom blocks: [].
        # Use ["workflows_core.api_key", "workflows_core.model_manager"] only
        # if you actually need them.
        return []

    def run(self, images: Batch[WorkflowImageData], confidence: float, ...) -> BlockResult:
        ...
```

## Single-frame vs batched run() signatures

This is the most common bug. Pick one model, declare it consistently in
both the manifest and `run()`.

### Batched (recommended for any model that supports it)

```python
class Manifest(...):
    images: WorkflowImageSelector = Field(...)        # plural

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]                              # MUST list "images"

class Block(WorkflowBlock):
    def run(self, images: Batch[WorkflowImageData], ...) -> BlockResult:
        # images is a Batch — iterable of N WorkflowImageData
        # Return List[dict] with N elements (one per image).
        ...
```

### Single-frame

```python
class Manifest(...):
    image: WorkflowImageSelector = Field(...)         # singular
    # NO get_parameters_accepting_batches override

class Block(WorkflowBlock):
    def run(self, image: WorkflowImageData, ...) -> BlockResult:
        return {"predictions": ...}                    # single dict
```

### The footgun

Declaring `images: WorkflowImageSelector` (plural) **without** also overriding
`get_parameters_accepting_batches()` makes the engine fall back to iterating
frame-by-frame. Your `run()` will be called with a `Batch` of length 1 every
time, and the model's batch dim never gets used. **Both signals are required.**

## The sv.Detections contract

Downstream blocks (ByteTracker, Velocity, TimeInZone, BoundingBoxVisualization,
LabelVisualization, webhook_sink) read specific fields off `sv.Detections`.
Producing detections that don't have these fields will either crash the
downstream block or produce wrong output silently.

Required for object-detection outputs:

| Field | Type | Why |
|---|---|---|
| `xyxy` | `np.float32 [n, 4]` | bounding boxes in source-image pixel coords |
| `confidence` | `np.float32 [n]` | scores |
| `class_id` | `np.int64 [n]` | integer class index |
| `data["class_name"]` | `np.array dtype=object [n]` | string class labels — visualizations need this |
| `data["detection_id"]` | `np.array dtype=object [n]` | UUIDs, one per detection |
| `data["image_dimensions"]` | `np.int [n, 2]` | `[h, w]` repeated; used by zone blocks |
| `data["prediction_type"]` | `np.array dtype=object [n]` | usually `"object-detection"` |

```python
import uuid
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY, IMAGE_DIMENSIONS_KEY, PREDICTION_TYPE_KEY,
)

CLASS_NAME_DATA_KEY = "class_name"

n = len(xyxy)
detections = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls_idx)
detections.data[CLASS_NAME_DATA_KEY] = np.array(
    [class_names[int(c)] for c in cls_idx], dtype=object,
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
```

The empty-detection paths (`n == 0`) matter. Several downstream blocks call
`np.tile` / `np.array` on these fields; passing the wrong empty shape raises.

## Parent-coordinates attachment

ByteTracker, TimeInZone, and visualizations need to know how a detection maps
to the original frame's coordinate system (in case an upstream Crop or
DynamicCrop block has shifted things). One helper handles all of it:

```python
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_sv_detections,
)

detections = attach_parents_coordinates_to_sv_detections(
    detections=detections, image=img,    # img is the WorkflowImageData
)
```

Always call this last, after populating all the data keys. Forgetting it is
the silent-failure that bites when someone adds a Crop block upstream a year
later.

## Returning the BlockResult

For a batched detector:

```python
outputs: List[dict] = []
for img, raw in zip(images, model_results):
    detections = build_sv_detections(raw, img)
    outputs.append({"predictions": detections})
return outputs   # length == len(images)
```

For a single-frame block: `return {"predictions": detections}`.

The output keys must match `describe_outputs()` exactly.

## Plugin entry point

The module-level `load_blocks()` function is what the engine calls when
`WORKFLOWS_PLUGINS` lists this module:

```python
def load_blocks() -> List[Type[WorkflowBlock]]:
    return [MyBlockV1]                   # one or many — all registered together
```

Optional siblings (rarely needed for custom blocks):

```python
def load_kinds() -> List[Kind]: ...      # if you define new kind types
def load_initializers() -> dict: ...     # if your block needs init_parameters
```
