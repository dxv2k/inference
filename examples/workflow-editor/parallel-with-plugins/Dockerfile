# Derived image: Roboflow parallel server + our editor's custom plugins.
#
# Layers on top of the official parallel server with:
#   - ultralytics + sam3 (so local_yolo_plugin and sam3_plugin work)
#   - PYTHONPATH and WORKFLOWS_PLUGINS pre-set
#   - A mount point at /plugins for the editor's plugin .py files
#
# Build:   docker build -t inference-parallel-plus -f Dockerfile .
# Run:     docker run -d --name inference-parallel-plus \
#             --gpus all --shm-size=8g -p 9011:9011 \
#             -v <repo>/examples/local-workflow-demo:/plugins:ro \
#             -v parallel-cache-plus:/tmp/cache \
#             inference-parallel-plus

FROM roboflow/roboflow-inference-server-gpu-parallel:latest

# Editor plugins need these at import time. The base image already has
# the rest of the inference deps; we only add what its missing.
# (--no-deps to avoid pulling a fresh torch and blowing up the image.)
RUN pip install --no-cache-dir --no-deps ultralytics==8.4.48 ultralytics-thop && \
    pip install --no-cache-dir --no-deps sam3==0.1.3 decord

# Plugin discovery: WORKFLOWS_PLUGINS_ENV makes blocks_loader find these
# names on startup. /plugins is where the editor's plugin source mounts.
ENV PYTHONPATH=/plugins
ENV WORKFLOWS_PLUGINS=local_yolo_plugin,yolo_world_plugin,sam3_plugin,openai_compatible_plugin
