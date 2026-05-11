"""
WorkflowBlock that calls a VLM (Gemini via OpenRouter) on 1-2 sample images
and outputs a list of class names suitable for an open-vocabulary detector.

Used by Tab 8's auto-annotate-v2 workflow: this block runs once, its
`classes` output is wired into the YOLO-World block's `prompts` parameter
inside a single engine.run() — so the whole "VLM → detector → viz" chain
is one workflow execution, no Python orchestration.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Type, Union

import numpy as np
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    LIST_OF_VALUES_KIND,
    Selector,
    WorkflowImageSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

import vlm_prompt_generator as vlm


SHORT_DESCRIPTION = "Ask a VLM to suggest class names for auto-annotation."
LONG_DESCRIPTION = """
Sends 1-2 sample images to a VLM (default: Gemini 3.1 Flash Lite via
OpenRouter) and returns the JSON-array of class names it proposes. The
output is a LIST_OF_VALUES_KIND that can be wired into any
open-vocabulary detector block's `prompts` parameter (e.g.
local_models/yolo_world@v1).

Reads OPENROUTER_API_KEY from env (or .env via python-dotenv).
"""


class VlmPromptManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "VLM class-prompt suggester",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
        },
        protected_namespaces=(),
    )
    type: Literal["local_models/vlm_prompt@v1"]
    images: WorkflowImageSelector = Field(
        description="Sample image(s) the VLM examines. Pass 1-2; more becomes expensive."
    )
    context: Union[str, Selector()] = Field(
        default="",
        description="Optional plain-English context to bias class suggestions "
                    "(e.g. 'warehouse PPE compliance').",
    )
    model: Union[str, Selector()] = Field(
        default="google/gemini-3.1-flash-lite",
        description="OpenRouter model id. The default Gemini Flash Lite is cheap (~$0.0006/call) "
                    "and fast (~1-2 s) but any multimodal OpenRouter model id works.",
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="classes", kind=[LIST_OF_VALUES_KIND])]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class VlmPromptBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return VlmPromptManifest

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return []

    def run(
        self,
        images: Batch[WorkflowImageData],
        context: str,
        model: str,
    ) -> BlockResult:
        # All sample images in the batch go into ONE Gemini call.
        np_imgs: list[np.ndarray] = [img.numpy_image for img in images]
        try:
            classes, _debug = vlm.generate_class_prompts(
                sample_images=np_imgs,
                user_context=context or "",
                model=model,
                max_samples=len(np_imgs),
            )
        except RuntimeError as e:
            # Surface a runtime error so the engine wraps it sensibly,
            # but keep the output shape consistent so downstream blocks don't crash.
            raise

        # The engine expects one output entry per input batch element.
        # All entries get the same `classes` list — downstream blocks
        # (yolo-world) will read the same prompts whichever element they see.
        return [{"classes": list(classes)} for _ in images]


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [VlmPromptBlockV1]
