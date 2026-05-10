#!/usr/bin/env bash
# Start a Triton Inference Server pointing at the local model_repository.
# Requires: docker + nvidia-container-toolkit (for --gpus all).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:24.10-py3}"
NAME="${NAME:-triton-yolo}"

if [[ ! -d "$HERE/model_repository/yolov8n_onnx/1" ]]; then
  echo "model_repository/yolov8n_onnx/1/ is empty." >&2
  echo "Run: bash $HERE/export_to_onnx.sh first." >&2
  exit 1
fi

# Stop existing container with same name (idempotent restart)
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker rm -f "$NAME" >/dev/null
fi

docker run -d \
  --name "$NAME" \
  --gpus all \
  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$HERE/model_repository:/models" \
  "$TRITON_IMAGE" \
  tritonserver --model-repository=/models --strict-model-config=false

echo "Waiting for Triton to be ready..."
for i in {1..30}; do
  if curl -sf http://localhost:8000/v2/health/ready >/dev/null 2>&1; then
    echo "Triton is ready."
    echo "  HTTP   : http://localhost:8000"
    echo "  gRPC   : localhost:8001"
    echo "  Metrics: http://localhost:8002/metrics"
    echo "  Models : $(curl -s http://localhost:8000/v2/models | head -c 200)"
    exit 0
  fi
  sleep 1
done

echo "Triton did not become ready within 30s. Logs:" >&2
docker logs "$NAME" --tail 40
exit 1
