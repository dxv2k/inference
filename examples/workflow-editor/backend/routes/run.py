"""Single-image workflow execution endpoint."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .. import engine_runner

router = APIRouter()

_HERE = Path(__file__).resolve().parent
# Image paths are resolved relative to the sibling demo's sample_images dir.
_SAMPLE_DIR = (_HERE.parent.parent.parent / "local-workflow-demo" / "sample_images").resolve()

_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


def _decode_image_bytes(buf: bytes) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image (not jpg/png?)")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_image_from_path(rel: str) -> np.ndarray:
    candidate = (_SAMPLE_DIR / rel).resolve()
    try:
        candidate.relative_to(_SAMPLE_DIR)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"image_path must stay inside sample_images/: got {rel!r}",
        )
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"no such image: {rel}")
    bgr = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail=f"could not read image: {rel}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@router.post("/workflows/run")
async def run_workflow(
    workflow: str = Form(...),
    runtime_params: str = Form("{}"),
    image: Optional[UploadFile] = File(None),
    image_path: Optional[str] = Form(None),
) -> JSONResponse:
    start = time.perf_counter()

    # --- parse forms ------------------------------------------------------
    try:
        spec = json.loads(workflow)
        if not isinstance(spec, dict):
            raise ValueError("workflow must decode to a JSON object")
    except (ValueError, json.JSONDecodeError) as exc:
        return JSONResponse(
            {
                "ok": False,
                "error": f"invalid workflow JSON: {exc}",
                "elapsed_ms": 0,
            }
        )
    try:
        params = json.loads(runtime_params) if runtime_params else {}
        if not isinstance(params, dict):
            raise ValueError("runtime_params must decode to a JSON object")
    except (ValueError, json.JSONDecodeError) as exc:
        return JSONResponse(
            {
                "ok": False,
                "error": f"invalid runtime_params JSON: {exc}",
                "elapsed_ms": 0,
            }
        )

    # --- load image -------------------------------------------------------
    if image is None and not image_path:
        return JSONResponse(
            {
                "ok": False,
                "error": "must provide either 'image' upload or 'image_path' form field",
                "elapsed_ms": 0,
            }
        )

    try:
        if image is not None:
            buf = await image.read()
            if len(buf) > _MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"upload exceeds {_MAX_UPLOAD_BYTES} bytes",
                )
            rgb = _decode_image_bytes(buf)
        else:
            assert image_path is not None
            rgb = _load_image_from_path(image_path)
    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(
            {
                "ok": False,
                "error": f"failed to load image: {exc}",
                "elapsed_ms": int((time.perf_counter() - start) * 1000),
            }
        )

    # --- run engine -------------------------------------------------------
    try:
        engine = engine_runner.get_engine_for_spec(spec)
        wrapped = engine_runner.wrap_image_for_engine(rgb)
        result = engine.run(runtime_parameters={"image": [wrapped], **params})
    except Exception as exc:
        return JSONResponse(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_ms": int((time.perf_counter() - start) * 1000),
            }
        )

    outputs = engine_runner.serialize_outputs_list(result or [])
    return JSONResponse(
        {
            "ok": True,
            "elapsed_ms": int((time.perf_counter() - start) * 1000),
            "outputs": outputs,
        }
    )
