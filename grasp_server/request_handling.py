"""Parse and materialize incoming multipart grasp requests.

The existing :func:`vg_pipeline.run_pipeline` consumes a "capture directory"
with ``camera_data.npy`` + ``color_preview.jpg``. To reuse it without changes,
each API request is decoded and written into a fresh capture-style directory
under ``output_vg/api_<run_id>/``.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


SUPPORTED_RGB_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ParsedGraspRequest:
    capture_dir: Path
    task_spec: str
    top_k: int
    num_candidates: int
    provider: str
    model: str
    frame_id: str
    rgb_shape: tuple[int, int, int]
    depth_shape: tuple[int, int]


def _decode_rgb(rgb_bytes: bytes, *, source_name: str | None) -> np.ndarray:
    if not rgb_bytes:
        raise ValueError("rgb file is empty")
    try:
        image = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"failed to decode rgb image: {exc}") from exc
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"rgb decoded to unexpected shape {arr.shape}; expected (H, W, 3)."
        )
    if source_name is not None:
        suf = Path(source_name).suffix.lower()
        if suf and suf not in SUPPORTED_RGB_SUFFIXES:
            # Pillow may still have decoded it; warn-by-shape only when shape was bad.
            pass
    return arr


def _decode_depth(depth_bytes: bytes) -> np.ndarray:
    if not depth_bytes:
        raise ValueError("depth file is empty")
    try:
        depth = np.load(io.BytesIO(depth_bytes), allow_pickle=False)
    except Exception as exc:
        raise ValueError(
            "failed to load depth .npy. Expected a float32 H x W array saved with numpy.save: "
            f"{exc}"
        ) from exc
    if depth.ndim != 2:
        raise ValueError(f"depth has shape {depth.shape}; expected (H, W).")
    depth = np.ascontiguousarray(depth, dtype=np.float32)
    return depth


def _decode_K(k_json: str) -> np.ndarray:
    if not k_json:
        raise ValueError("K is required (3x3 intrinsic matrix, JSON encoded)")
    try:
        payload = json.loads(k_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"K is not valid JSON: {exc}") from exc
    K = np.asarray(payload, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got shape {K.shape}")
    if not np.all(np.isfinite(K)):
        raise ValueError("K contains non-finite values")
    return K


def materialize_request(
    *,
    rgb_bytes: bytes,
    rgb_filename: str | None,
    depth_bytes: bytes,
    K_json: str,
    task_spec: str,
    top_k: int,
    num_candidates: int,
    provider: str,
    model: str,
    frame_id: str,
    capture_dir: Path,
) -> ParsedGraspRequest:
    """Validate inputs and write camera_data.npy + color_preview.jpg into ``capture_dir``."""
    if not task_spec or not task_spec.strip():
        raise ValueError("task_spec must be a non-empty string")
    if not frame_id or not frame_id.strip():
        raise ValueError("frame_id must be a non-empty string (e.g. 'camera_color_optical_frame')")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if num_candidates < 1:
        raise ValueError("num_candidates must be >= 1")

    rgb = _decode_rgb(rgb_bytes, source_name=rgb_filename)
    depth = _decode_depth(depth_bytes)
    K = _decode_K(K_json)

    if rgb.shape[:2] != depth.shape:
        raise ValueError(
            f"rgb shape {rgb.shape[:2]} does not match depth shape {depth.shape}; "
            "both must share the same (H, W)."
        )

    capture_dir.mkdir(parents=True, exist_ok=True)
    npy_path = capture_dir / "camera_data.npy"
    payload: dict[str, np.ndarray] = {
        "depth": depth,
        "K": K,
    }
    np.save(npy_path, payload, allow_pickle=True)

    color_preview = capture_dir / "color_preview.jpg"
    Image.fromarray(rgb, mode="RGB").save(color_preview, format="JPEG", quality=95)

    return ParsedGraspRequest(
        capture_dir=capture_dir,
        task_spec=task_spec.strip(),
        top_k=top_k,
        num_candidates=num_candidates,
        provider=provider,
        model=model,
        frame_id=frame_id.strip(),
        rgb_shape=(int(rgb.shape[0]), int(rgb.shape[1]), 3),
        depth_shape=(int(depth.shape[0]), int(depth.shape[1])),
    )
