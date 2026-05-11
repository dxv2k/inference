"""Workflow Editor backend package.

We import :mod:`real_engine_runner` from the sibling
``examples/local-workflow-demo/`` directory FIRST, because that module sets
``os.environ["WORKFLOWS_PLUGINS"]`` BEFORE any ``inference.*`` import. If any
other submodule of this package transitively imports ``inference`` before
``real_engine_runner`` runs, the custom plugins (local_yolo_plugin,
yolo_world_plugin, sam3_plugin, optional triton) will not be discovered.

Keep this import at the very top of the package so it runs first regardless of
which backend submodule is loaded first.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DEMO_DIR = _HERE.parent.parent / "local-workflow-demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

# This import has the side effect of configuring WORKFLOWS_PLUGINS and importing
# inference once. Subsequent imports of inference.* will reuse the same modules.
import real_engine_runner as _rer  # noqa: F401,E402
