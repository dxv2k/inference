#!/usr/bin/env bash
# Export yolov8n.pt → ONNX, drop the result into Triton's model_repository.
# Run from the demo root: bash optimize/triton/export_to_onnx.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DEMO_DIR="$(cd "$HERE/../.." && pwd)"
WEIGHTS="${1:-yolov8n.pt}"
IMGSZ="${2:-640}"

cd "$DEMO_DIR"
echo "Exporting $WEIGHTS at imgsz=$IMGSZ → ONNX (dynamic batch) ..."
uv run yolo export model="$WEIGHTS" format=onnx imgsz="$IMGSZ" simplify=True dynamic=True

base="${WEIGHTS%.pt}"
mkdir -p "$HERE/model_repository/yolov8n_onnx/1"
mv "${base}.onnx" "$HERE/model_repository/yolov8n_onnx/1/model.onnx"
echo "wrote: $HERE/model_repository/yolov8n_onnx/1/model.onnx"
