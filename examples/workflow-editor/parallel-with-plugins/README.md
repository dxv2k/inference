# Parallel server + editor plugins

A derived Docker image that layers our custom workflow plugins onto
Roboflow's enterprise parallel inference server, so the editor's
workflows (`speed`, `smart`, `autoannotate`, `sam3_autoannotate`,
`vlm_only`) all run end-to-end against the Celery-backed `/infer/workflows`
endpoint.

The base image
`roboflow/roboflow-inference-server-gpu-parallel:latest` ships with:
- redis + 2× celery worker pools (`pre` + `post`) + GPU `infer.py` + gunicorn/uvicorn
- The full `roboflow_core/*` block library
- An older release than the demo branch — missing `openai_compatible@v1`

This image adds the gap-fillers:
- `pip install ultralytics sam3 decord` (`--no-deps`)
- `PYTHONPATH=/plugins`
- `WORKFLOWS_PLUGINS=local_yolo_plugin,yolo_world_plugin,sam3_plugin,openai_compatible_plugin`

The editor's plugin source is mounted at `/plugins:ro` at run time so we
don't bake source into the image.

## Build + run

```bash
# from this directory
docker build -t inference-parallel-plus .

docker run -d \
  --name inference-parallel-plus \
  --gpus=all --shm-size=8g \
  -p 9011:9011 \
  -e PORT=9011 -e HOST=0.0.0.0 -e NUM_CELERY_WORKERS=4 \
  -e REDIS_HOST=localhost -e REDIS_PORT=6379 \
  -e HF_HOME=/hfcache \
  -v <repo>/examples/local-workflow-demo:/plugins:ro \
  -v ~/.cache/huggingface:/hfcache:ro \
  -v parallel-cache-plus:/tmp/cache \
  inference-parallel-plus
```

The `HF_HOME` + `:/hfcache:ro` mount lets the container reuse SAM3 weights
the host already downloaded (`facebook/sam3` is a gated HF repo otherwise).

## Smoke test (all 5 builtins)

```bash
uv run --project ../../local-workflow-demo python -c "
import base64, requests
img = base64.b64encode(open('../../local-workflow-demo/sample_images/bus.jpg','rb').read()).decode()
for wid in ['speed','smart','autoannotate','sam3_autoannotate','vlm_only']:
    spec = requests.get(f'http://localhost:7873/api/workflows/builtin/{wid}').json()['workflow']
    r = requests.post('http://localhost:9011/infer/workflows', json={
        'inputs': {'image': {'type':'base64','value':img}},
        'specification': spec,
        'api_key': None,
    }, timeout=300)
    print(wid, r.status_code, len(r.json().get('outputs',[])))
"
```

Measured on RTX 3090 (first call includes engine compile + weight download):
- speed                2.4 s
- smart                0.1 s  (engine cached from previous run)
- autoannotate        19.7 s  (yolov8x-worldv2 first-time download)
- sam3_autoannotate   25.2 s  (sam3 model load — gated on HF, mount needed)
- vlm_only             1.2 s  (one Gemini API call)

## Where Celery actually accelerates things

Only the `roboflow_core/roboflow_*_model@vN` blocks go through Celery
dispatch — they call `model_manager.infer_from_request_sync()` which the
parallel server's `DispatchModelManager` reroutes to the Celery queue.

Our custom blocks (`local_models/*`) and `roboflow_core/sam3@vN`,
`yolo_world_model@v1`, `openai_compatible@v1` all call their backing
libraries directly. They run in the gunicorn worker process under the
engine's `max_concurrent_steps`, NOT under Celery.

So: bringing our workflows into the parallel server gets us a
benchmarking peer, not a speedup. To actually benefit from Celery
dispatch, run a workflow that uses `roboflow_core/roboflow_object_detection_model@v2`
with a Roboflow API key + a hosted model — that's the README's "76%
speedup" case.

## Plugin gap-filler notes

- `bpe_simple_vocab_16e6.txt.gz` is copied into the plugins dir so
  `sam3_plugin.py` can resolve it relative to its own file (the absolute
  host path it used before isn't valid inside the container).
- `openai_compatible_plugin.py` is a verbatim copy of
  `inference/core/workflows/core_steps/models/foundation/openai_compatible/v1.py`
  plus a `load_blocks()` entry point. The base image's older inference
  release predates this block.
