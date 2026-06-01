from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .prompting import build_grounding_prompt


def _pil_to_data_url(image: Image.Image, *, mime_type: str = "image/png") -> str:
    from io import BytesIO

    buf = BytesIO()
    im = image
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    if mime_type == "image/jpeg":
        im.save(buf, format="JPEG", quality=92)
    else:
        im.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def run_openai_vg(
    images: Sequence[Image.Image],
    task_spec: str,
    model_name: str,
    num_candidates: int = 3,
    api_key: str | None = None,
    openai_image_mime_types: Sequence[str] | None = None,
    prompt: str | None = None,
) -> str:
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required when provider is openai")

    if len(images) != 1:
        raise ValueError("run_openai_vg expects exactly one RGB image")

    w, h = images[0].size
    if prompt is None:
        prompt = build_grounding_prompt(task_spec, w, h, num_candidates=num_candidates)

    if openai_image_mime_types is None:
        mimes = ["image/png"]
    else:
        mimes = list(openai_image_mime_types)
        if len(mimes) != len(images):
            raise ValueError("openai_image_mime_types length must match images")
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": _pil_to_data_url(im, mime_type=m)}}
        for im, m in zip(images, mimes, strict=True)
    ]
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "temperature": 0,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI API request failed: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI API connection failed: {exc}") from exc

    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected OpenAI response: {body}") from exc


def _pil_to_gemini_base64(image: Image.Image, mime_type: str) -> str:
    from io import BytesIO

    buf = BytesIO()
    im = image
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    if mime_type == "image/jpeg":
        im.save(buf, format="JPEG", quality=92)
    else:
        im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def run_gemini_vg(
    images: Sequence[Image.Image],
    task_spec: str,
    model_name: str,
    num_candidates: int = 3,
    api_key: str | None = None,
    code_execution: bool = False,
    gemini_image_mime_types: Sequence[str] | None = None,
    prompt: str | None = None,
) -> str:
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is required when provider is gemini")

    if len(images) != 1:
        raise ValueError("run_gemini_vg expects exactly one RGB image")

    w, h = images[0].size
    if prompt is None:
        prompt = build_grounding_prompt(task_spec, w, h, num_candidates=num_candidates)

    if gemini_image_mime_types is None:
        mimes = ["image/png"]
    else:
        mimes = list(gemini_image_mime_types)
        if len(mimes) != len(images):
            raise ValueError("gemini_image_mime_types length must match images")

    parts: list[dict[str, Any]] = []
    for im, mime in zip(images, mimes, strict=True):
        b64 = _pil_to_gemini_base64(im, mime)
        parts.append({"inline_data": {"mime_type": mime, "data": b64}})
    parts.append({"text": prompt})

    payload = {
        "contents": [
            {
                "parts": parts,
            }
        ],
        "generationConfig": {"temperature": 0},
    }
    if code_execution:
        payload["tools"] = [{"code_execution": {}}]
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        f"?key={key}"
    )
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Gemini API request failed: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini API connection failed: {exc}") from exc

    try:
        parts = body["candidates"][0]["content"]["parts"]
        return "\n".join(part.get("text", "") for part in parts if "text" in part).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Gemini response: {body}") from exc


def run_vg_inference(
    *,
    provider: str,
    images: Sequence[Image.Image],
    task_spec: str,
    model_path: str | Path,
    num_candidates: int = 3,
    api_key: str | None = None,
    code_execution: bool = False,
    openai_image_mime_types: Sequence[str] | None = None,
    gemini_image_mime_types: Sequence[str] | None = None,
    prompt: str | None = None,
) -> str:
    if len(images) != 1:
        raise ValueError("Visual grounding requires exactly one RGB image")
    if provider == "openai":
        return run_openai_vg(
            images=images,
            task_spec=task_spec,
            model_name=str(model_path),
            num_candidates=num_candidates,
            api_key=api_key,
            openai_image_mime_types=openai_image_mime_types,
            prompt=prompt,
        )
    if provider == "gemini":
        return run_gemini_vg(
            images=images,
            task_spec=task_spec,
            model_name=str(model_path),
            num_candidates=num_candidates,
            api_key=api_key,
            code_execution=code_execution,
            gemini_image_mime_types=gemini_image_mime_types,
            prompt=prompt,
        )
    raise ValueError(f"Unsupported provider: {provider}")
