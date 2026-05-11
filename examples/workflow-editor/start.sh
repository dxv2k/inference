#!/usr/bin/env bash
# One-shot launcher for the workflow editor.
#
#   1. Build the React frontend into frontend/dist/  (skips if up-to-date).
#   2. Regenerate blocks_manifest.json if the introspect script is newer.
#   3. Start the FastAPI backend on :7873  (also serves frontend/dist/).
#
# Prereqs:  uv (https://docs.astral.sh/uv/) and either pnpm or npm on PATH.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# ---------- helpers ----------------------------------------------------------
need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[start.sh] missing required command: $1" >&2
    exit 1
  fi
}

need uv

# ---------- 1. Frontend build -------------------------------------------------
if [ -d frontend ]; then
  needs_build=0
  if [ ! -d frontend/dist ] || [ ! -f frontend/dist/index.html ]; then
    needs_build=1
  elif [ -n "$(find frontend/src frontend/index.html frontend/package.json -newer frontend/dist/index.html 2>/dev/null | head -1)" ]; then
    needs_build=1
  fi

  if [ "$needs_build" = "1" ]; then
    echo "[start.sh] building frontend..."
    (
      cd frontend
      if command -v pnpm >/dev/null 2>&1; then
        pnpm install --frozen-lockfile 2>/dev/null || pnpm install
        pnpm build
      elif command -v npm >/dev/null 2>&1; then
        npm install
        npm run build
      else
        echo "[start.sh] neither pnpm nor npm found; install Node.js >= 18" >&2
        exit 1
      fi
    )
  else
    echo "[start.sh] frontend/dist is up to date, skipping build."
  fi
else
  echo "[start.sh] WARNING: frontend/ directory missing — backend will start but / will 404." >&2
fi

# ---------- 2. Block manifest -------------------------------------------------
if [ ! -f blocks_manifest.json ] || [ _introspect_blocks.py -nt blocks_manifest.json ]; then
  echo "[start.sh] regenerating blocks_manifest.json..."
  uv run python _introspect_blocks.py
else
  echo "[start.sh] blocks_manifest.json is up to date."
fi

# ---------- 3. Backend --------------------------------------------------------
echo "[start.sh] starting backend on http://localhost:7873 ..."
exec uv run python -m backend.main
