"""Pydantic models for request/response shapes.

Canonical source for any conflict is PLAN.md section 3.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ----- health / blocks -----------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    plugins_loaded: list[str]


# ----- builtin / saved workflows -------------------------------------------


class BuiltinWorkflowSummary(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    factory: Optional[str] = None
    factory_kwargs: Optional[dict[str, Any]] = None
    default_runtime_params: Optional[dict[str, Any]] = None
    requires: Optional[str] = None
    requires_env: Optional[list[str]] = None


class BuiltinListResponse(BaseModel):
    workflows: list[BuiltinWorkflowSummary]


class BuiltinWorkflowResponse(BaseModel):
    id: str
    name: str
    workflow: dict[str, Any]
    default_runtime_params: dict[str, Any] = Field(default_factory=dict)


class SavedWorkflowSummary(BaseModel):
    name: str
    modified_at: str


class SavedListResponse(BaseModel):
    workflows: list[SavedWorkflowSummary]


class SavedWorkflowResponse(BaseModel):
    name: str
    workflow: dict[str, Any]


class SaveRequest(BaseModel):
    name: str
    workflow: dict[str, Any]


class SaveResponse(BaseModel):
    name: str
    path: str


# ----- validate ------------------------------------------------------------


class ValidateRequest(BaseModel):
    workflow: dict[str, Any]


class ValidateOk(BaseModel):
    ok: bool = True


class ValidateError(BaseModel):
    ok: bool = False
    error: str
    context: str = "workflow_compilation"


# ----- run -----------------------------------------------------------------
# The /run endpoint uses multipart/form-data so its request shape is not a
# pydantic model. Response shape is dynamic (one outputs dict per element).


class RunErrorResponse(BaseModel):
    ok: bool = False
    error: str
    elapsed_ms: int = 0
