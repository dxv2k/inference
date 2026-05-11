"""On-disk store for user-saved workflows.

Files live in ``examples/workflow-editor/saved_workflows/{name}.json``. Names
that contain path-traversal characters are rejected.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
SAVED_DIR = _HERE.parent / "saved_workflows"
SAVED_DIR.mkdir(parents=True, exist_ok=True)

# Pretty narrow: a-z, A-Z, 0-9, dash, underscore, dot. No spaces, no slashes,
# no leading dot (avoids hidden files).
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,127}$")
_TRAVERSAL_TOKENS = ("/", "\\", "..")


class InvalidName(ValueError):
    pass


class NotFound(KeyError):
    pass


def _check_name(name: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise InvalidName("workflow name must be a non-empty string")
    stripped = name.strip()
    for token in _TRAVERSAL_TOKENS:
        if token in stripped:
            raise InvalidName(
                f"workflow name may not contain {token!r}: {stripped!r}"
            )
    if not _NAME_RE.match(stripped):
        raise InvalidName(
            "workflow name must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}"
        )


def _path_for(name: str) -> Path:
    _check_name(name)
    candidate = (SAVED_DIR / f"{name}.json").resolve()
    # Belt-and-braces: ensure the resolved path is still inside SAVED_DIR.
    saved_resolved = SAVED_DIR.resolve()
    try:
        candidate.relative_to(saved_resolved)
    except ValueError as exc:
        raise InvalidName(f"resolved path escapes saved_workflows: {candidate}") from exc
    return candidate


def list_saved() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for p in sorted(SAVED_DIR.glob("*.json")):
        out.append(
            {
                "name": p.stem,
                "modified_at": datetime.fromtimestamp(
                    p.stat().st_mtime, tz=timezone.utc
                ).isoformat().replace("+00:00", "Z"),
            }
        )
    return out


def load_saved(name: str) -> dict[str, Any]:
    path = _path_for(name)
    if not path.exists():
        raise NotFound(name)
    with path.open() as f:
        return json.load(f)


def save(name: str, spec: dict[str, Any]) -> Path:
    path = _path_for(name)
    with path.open("w") as f:
        json.dump(spec, f, indent=2, sort_keys=False)
    return path


def relative_save_path(path: Path) -> str:
    """Return the path as a project-relative string, e.g. ``saved_workflows/foo.json``."""
    try:
        return str(path.relative_to(_HERE.parent))
    except ValueError:
        return str(path)
