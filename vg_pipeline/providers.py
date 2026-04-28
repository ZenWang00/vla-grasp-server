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

from .prompting import build_grounding_messages, build_grounding_prompt


def run_qwen_vg(
    model_path: str | Path,
    images: Sequence[Image.Image],
    task_spec: str,
    num_candidates: int = 3,
    max_new_tokens: int = 512,
) -> str:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    raw = Path(model_path).expanduser()
    resolved = raw.resolve()
    if resolved.is_dir():
        load_root = str(resolved)
    elif raw.is_absolute():
        raise FileNotFoundError(
            f"Local Qwen model directory does not exist: {resolved}\n"
            "Pass a real folder (e.g. --model models/qwen2.5-vl-7b from the repo root), "
            "not a placeholder path. Or use a Hugging Face id such as Qwen/Qwen2.5-VL-7B-Instruct."
        )
    else:
        load_root = str(raw)

    processor = AutoProcessor.from_pretrained(
        load_root,
        trust_remote_code=True,
    )
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        load_root,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    image_list = list(images)
    w, h = image_list[0].size
    messages = build_grounding_messages(task_spec, w, h, image_list, num_candidates=num_candidates)
    text_prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(
        text=text_prompt,
        images=image_list,
        return_tensors="pt",
    ).to(model.device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
    ]
    out = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return out


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
) -> str:
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required when provider is openai")

    if len(images) != 1:
        raise ValueError("run_openai_vg expects exactly one RGB image")

    w, h = images[0].size
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
) -> str:
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is required when provider is gemini")

    if len(images) != 1:
        raise ValueError("run_gemini_vg expects exactly one RGB image")

    w, h = images[0].size
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
    max_new_tokens: int = 512,
    openai_image_mime_types: Sequence[str] | None = None,
    gemini_image_mime_types: Sequence[str] | None = None,
) -> str:
    if len(images) != 1:
        raise ValueError("Visual grounding requires exactly one RGB image")
    if provider == "qwen_local":
        return run_qwen_vg(
            model_path=model_path,
            images=images,
            task_spec=task_spec,
            num_candidates=num_candidates,
            max_new_tokens=max_new_tokens,
        )
    if provider == "openai":
        return run_openai_vg(
            images=images,
            task_spec=task_spec,
            model_name=str(model_path),
            num_candidates=num_candidates,
            api_key=api_key,
            openai_image_mime_types=openai_image_mime_types,
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
        )
    raise ValueError(f"Unsupported provider: {provider}")
