"""Builtin + saved workflow routes (no /run; see run.py)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from .. import engine_runner, workflows_store
from ..schemas import (
    SaveRequest,
    SaveResponse,
    SavedListResponse,
    SavedWorkflowResponse,
    SavedWorkflowSummary,
    ValidateRequest,
)

router = APIRouter()


# ----- builtin -------------------------------------------------------------


@router.get("/workflows/builtin")
def list_builtin() -> JSONResponse:
    return JSONResponse(engine_runner.list_prebuilt_raw())


@router.get("/workflows/builtin/{prebuilt_id}")
def get_builtin(prebuilt_id: str) -> JSONResponse:
    try:
        payload = engine_runner.load_prebuilt(prebuilt_id)
    except engine_runner.BuiltinNotFound:
        raise HTTPException(status_code=404, detail=f"unknown builtin: {prebuilt_id}")
    except engine_runner.BuiltinUnavailable as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return JSONResponse(payload)


# ----- saved ---------------------------------------------------------------


@router.get("/workflows/saved", response_model=SavedListResponse)
def list_saved() -> SavedListResponse:
    return SavedListResponse(
        workflows=[
            SavedWorkflowSummary(**entry) for entry in workflows_store.list_saved()
        ]
    )


@router.get("/workflows/saved/{name}", response_model=SavedWorkflowResponse)
def get_saved(name: str) -> SavedWorkflowResponse:
    try:
        spec = workflows_store.load_saved(name)
    except workflows_store.InvalidName as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except workflows_store.NotFound:
        raise HTTPException(status_code=404, detail=f"no saved workflow {name!r}")
    return SavedWorkflowResponse(name=name, workflow=spec)


@router.post("/workflows/save", response_model=SaveResponse)
def save_workflow(req: SaveRequest) -> SaveResponse:
    try:
        path = workflows_store.save(req.name, req.workflow)
    except workflows_store.InvalidName as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return SaveResponse(name=req.name, path=workflows_store.relative_save_path(path))


# ----- validate ------------------------------------------------------------


@router.post("/workflows/validate")
def validate_workflow(req: ValidateRequest) -> JSONResponse:
    # Import here so that any inference-module import order issues surface in
    # /api/health rather than during process startup.
    from inference.core.workflows.execution_engine.core import ExecutionEngine
    from inference.core.workflows.errors import WorkflowError

    try:
        ExecutionEngine.init(
            workflow_definition=req.workflow,
            init_parameters=engine_runner.make_init_parameters(),
            max_concurrent_steps=1,
        )
    except WorkflowError as exc:
        return JSONResponse(
            {
                "ok": False,
                "error": str(exc),
                "context": "workflow_compilation",
            }
        )
    except Exception as exc:  # pragma: no cover - catch-all for safety
        return JSONResponse(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "context": "workflow_compilation",
            }
        )
    return JSONResponse({"ok": True})
