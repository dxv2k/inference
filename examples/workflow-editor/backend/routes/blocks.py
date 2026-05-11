"""Block catalog + liveness routes."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from .. import engine_runner
from ..schemas import HealthResponse

router = APIRouter()

_HERE = Path(__file__).resolve().parent
_MANIFEST = _HERE.parent.parent / "blocks_manifest.json"


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", plugins_loaded=engine_runner.loaded_plugins())


@router.get("/blocks")
def get_blocks() -> JSONResponse:
    if not _MANIFEST.exists():
        raise HTTPException(
            status_code=503,
            detail="blocks_manifest.json missing; run _introspect_blocks.py",
        )
    with _MANIFEST.open() as f:
        data = json.load(f)
    return JSONResponse(data)
