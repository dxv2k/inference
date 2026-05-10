#!/usr/bin/env bash
# Start a Triton Inference Server pointing at the local model_repository.
# Requires: docker + nvidia-container-toolkit (for --gpus all).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:24.10-py3}"
NAME="${NAME:-triton-yolo}"
# Host port mappings — override if you have conflicts.
HTTP_PORT="${HTTP_PORT:-18000}"
GRPC_PORT="${GRPC_PORT:-18001}"
METRICS_PORT="${METRICS_PORT:-18002}"

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
  -p ${HTTP_PORT}:8000 -p ${GRPC_PORT}:8001 -p ${METRICS_PORT}:8002 \
  -v "$HERE/model_repository:/models" \
  "$TRITON_IMAGE" \
  tritonserver --model-repository=/models --strict-model-config=false

echo "Waiting for Triton to be ready..."
for i in {1..60}; do
  if curl -sf http://localhost:${HTTP_PORT}/v2/health/ready >/dev/null 2>&1; then
    echo "Triton is ready."
    echo "  HTTP   : http://localhost:${HTTP_PORT}"
    echo "  gRPC   : localhost:${GRPC_PORT}     <-- use this URL in the demo's UI"
    echo "  Metrics: http://localhost:${METRICS_PORT}/metrics"
    echo "  Models : $(curl -s http://localhost:${HTTP_PORT}/v2/models | head -c 200)"
    exit 0
  fi
  sleep 1
done

echo "Triton did not become ready within 60s. Logs:" >&2
docker logs "$NAME" --tail 40
exit 1
