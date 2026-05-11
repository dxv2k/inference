"""FastAPI app for the Workflow Editor backend.

Single-port architecture:

* ``/api/*`` — JSON endpoints (blocks, workflows, run, health).
* ``/`` — the built React app from ``frontend/dist/`` (if present).

On startup we regenerate ``blocks_manifest.json`` if missing or older than the
``_introspect_blocks.py`` script.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .routes import blocks, run, workflows

logger = logging.getLogger("workflow-editor")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_MANIFEST = _ROOT / "blocks_manifest.json"
_INTROSPECT = _ROOT / "_introspect_blocks.py"
_FRONTEND_DIST = _ROOT / "frontend" / "dist"


def _maybe_regen_manifest() -> None:
    """Run ``_introspect_blocks.py`` if the manifest is stale or missing."""
    if not _INTROSPECT.exists():
        logger.warning("introspect script missing at %s; skipping regen", _INTROSPECT)
        return
    needs_regen = (
        not _MANIFEST.exists()
        or _MANIFEST.stat().st_mtime < _INTROSPECT.stat().st_mtime
    )
    if not needs_regen:
        return
    logger.info("blocks_manifest.json is stale; regenerating...")
    try:
        subprocess.run(
            [sys.executable, str(_INTROSPECT)],
            check=True,
            cwd=str(_ROOT),
        )
    except subprocess.CalledProcessError as exc:
        logger.error("manifest regen failed: %s — continuing with old file", exc)


_maybe_regen_manifest()


app = FastAPI(title="Workflow Editor", version="0.1.0")
app.include_router(blocks.router, prefix="/api")
app.include_router(workflows.router, prefix="/api")
app.include_router(run.router, prefix="/api")


# Static frontend (optional — Builder B may not have built it yet).
if _FRONTEND_DIST.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(_FRONTEND_DIST), html=True),
        name="frontend",
    )
    logger.info("serving frontend from %s", _FRONTEND_DIST)
else:
    logger.warning(
        "frontend/dist/ not found at %s; serving API only", _FRONTEND_DIST
    )

    @app.get("/")
    def _no_frontend() -> dict:
        return {
            "status": "ok",
            "message": (
                "Workflow Editor backend running. The React frontend is not built; "
                "run `cd frontend && npm install && npm run build` or use the dev "
                "server on :5173. The API is available under /api/*."
            ),
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=7873, reload=False)
