"""
VLM-driven class prompt generation for auto-annotation.

Given a small sample of images (and optional user context like "warehouse
PPE compliance"), ask a vision-language model what classes should be
auto-annotated. The output is a `list[str]` of class names that can be
fed directly into YOLO-World v2 (or any other open-vocabulary detector).

Network calls go to OpenRouter only. The local detector that runs against
the suggested prompts is fully self-hosted (Ultralytics YOLO-World v2).
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import requests

# Best-effort .env loader so users running `python app.py` from this dir get
# the OPENROUTER_API_KEY without exporting it manually. Loading is idempotent;
# real-environment values take precedence over .env contents.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def is_configured() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def _image_to_data_url(image_rgb: np.ndarray, max_side: int = 1024,
                      jpeg_quality: int = 85) -> str:
    """Resize-down + JPEG-encode an image to a base64 data URL.
    Keeps the request payload manageable — Gemini Flash Lite handles ~3 MB
    multimodal payloads fine, larger ones get throttled."""
    h, w = image_rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image_rgb = cv2.resize(image_rgb, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


SYSTEM_PROMPT = """You are an expert vision-data annotator. Look at the
sample images you are shown and propose a concise list of class names that
should be auto-annotated across the rest of the dataset.

Rules:
- Reply with ONLY a JSON array of strings. No prose, no markdown fences, no
  trailing commas. Example: ["person", "car", "traffic light"]
- Use short, lowercase class names suitable for an object detector.
- Prefer specific concrete classes (e.g. "forklift" not "vehicle"; "hard
  hat" not "ppe"). 5-15 classes is a good range; more is OK if the scene
  warrants it.
- If the user supplies extra context, weight class choices toward that
  context.
- Do NOT include the prompt back, do NOT include a JSON object, do NOT
  wrap in ```json fences. JUST the array."""


_JSON_ARRAY_RE = re.compile(r"\[[^\[\]]*\]", re.DOTALL)


def _parse_class_list(text: str) -> list[str]:
    """Tolerant parser — Gemini Flash Lite usually returns clean JSON, but
    occasionally wraps the array in ```json fences or trailing commentary.
    Strategy: find the first JSON array substring and json.loads it."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Direct attempt
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        pass
    # Fallback: regex the first [...] block
    m = _JSON_ARRAY_RE.search(text)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass
    # Last resort: split on commas
    cleaned = text.strip("[]").replace('"', "").replace("'", "")
    return [c.strip() for c in cleaned.split(",") if c.strip()]


def generate_class_prompts(
    sample_images: Iterable[np.ndarray],
    user_context: str = "",
    *,
    api_key: str | None = None,
    model: str | None = None,
    timeout: float = 30.0,
    max_samples: int = 2,
) -> tuple[list[str], dict]:
    """Call Gemini (or whichever OpenRouter model is configured) on 1-`max_samples`
    images and return (class_list, debug_info).

    debug_info contains: model name, total tokens, latency_ms, raw response text.
    Raises RuntimeError on auth / network errors with a clear message.
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Copy .env.example -> .env and put your key in."
        )
    model = model or os.environ.get("OPENROUTER_MODEL", "google/gemini-3.1-flash-lite")

    images = list(sample_images)[:max_samples]
    if not images:
        raise RuntimeError("generate_class_prompts called with zero sample images")

    user_text_parts = []
    if user_context.strip():
        user_text_parts.append(f"Context: {user_context.strip()}")
    user_text_parts.append(
        f"Here {'is one sample image' if len(images) == 1 else f'are {len(images)} sample images'} "
        "from the dataset. Propose the class list."
    )
    user_text = "\n\n".join(user_text_parts)

    content = [{"type": "text", "text": user_text}]
    for img in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": _image_to_data_url(img)},
        })

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0.2,
        "max_tokens": 400,
    }

    t0 = time.perf_counter()
    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/dxv2k/inference",
            "X-Title": "local-workflow-demo: auto-annotate v2",
        },
        json=body,
        timeout=timeout,
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    if not resp.ok:
        # OpenRouter returns useful JSON errors — surface them
        msg = resp.text[:500]
        raise RuntimeError(f"OpenRouter {resp.status_code}: {msg}")

    payload = resp.json()
    try:
        raw_text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"unexpected OpenRouter response shape: {payload}") from e

    classes = _parse_class_list(raw_text)
    debug = {
        "model": payload.get("model", model),
        "latency_ms": latency_ms,
        "usage": payload.get("usage", {}),
        "raw_response": raw_text,
        "n_sample_images": len(images),
    }
    return classes, debug
