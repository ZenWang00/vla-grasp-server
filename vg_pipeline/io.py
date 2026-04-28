from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def safe_npy_stem(npy_path: Path) -> str:
    """Filesystem-safe fragment from npy filename (no extension)."""
    stem = npy_path.stem
    stem = re.sub(r"[^\w\-.]+", "_", stem, flags=re.UNICODE)
    return stem or "frame"


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def resolve_output_dir(base: Path, npy_path: Path, *, run_id: str, no_subdir: bool) -> Path:
    """Each run defaults to base / {stem}_{run_id} / to preserve history."""
    if no_subdir:
        return base.resolve()
    stem = safe_npy_stem(npy_path.resolve())
    return (base / f"{stem}_{run_id}").resolve()


def load_observation_npy(path: str | Path) -> dict[str, Any]:
    """Load 0-dim object npy; .item() must be a dict with at least ``depth`` and ``K`` (camera matrix)."""
    path = Path(path)
    arr = np.load(path, allow_pickle=True)
    frame = arr.item()
    if not isinstance(frame, dict):
        raise TypeError(f"Expected dict in npy .item(), got {type(frame)}")
    return frame


def rgb_to_pil(rgb: np.ndarray) -> Image.Image:
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def resolve_rgbd_pair(stem_path: str | Path) -> tuple[Path, Path]:
    """Resolve ``{dir}/{id}_scene.{jpg,png}`` and ``{dir}/{id}_depth.{jpg,png}`` from ``stem_path`` = ``dir/id``."""
    stem_path = Path(stem_path).expanduser().resolve()
    parent = stem_path.parent
    name = stem_path.name
    scene: Path | None = None
    depth_img: Path | None = None
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        s = parent / f"{name}_scene{ext}"
        if s.is_file():
            scene = s
            break
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        d = parent / f"{name}_depth{ext}"
        if d.is_file():
            depth_img = d
            break
    if scene is None or depth_img is None:
        raise FileNotFoundError(
            f"Could not find both scene and depth images for stem {stem_path}. "
            f"Expected like {parent / (name + '_scene.jpg')} and {parent / (name + '_depth.jpg')}."
        )
    return scene, depth_img


def resolve_scene_image(stem_path: str | Path) -> Path:
    """Resolve ``{dir}/{id}_scene.{jpg,png}`` from ``stem_path`` = ``dir/id``."""
    stem_path = Path(stem_path).expanduser().resolve()
    parent = stem_path.parent
    name = stem_path.name
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        scene = parent / f"{name}_scene{ext}"
        if scene.is_file():
            return scene
    raise FileNotFoundError(
        f"Could not find scene image for stem {stem_path}. "
        f"Expected like {parent / (name + '_scene.jpg')}."
    )


def resolve_capture_dir(capture_dir: str | Path) -> tuple[Path, Path, Path | None]:
    """Resolve `camera_data.npy`, `color_preview.jpg`, and optional `depth_preview.jpg` from a capture folder."""
    capture_dir = Path(capture_dir).expanduser().resolve()
    if not capture_dir.is_dir():
        raise FileNotFoundError(f"Capture directory not found: {capture_dir}")

    npy_path = capture_dir / "camera_data.npy"
    scene_path = capture_dir / "color_preview.jpg"
    depth_preview_path = capture_dir / "depth_preview.jpg"

    if not npy_path.is_file():
        raise FileNotFoundError(f"Capture npy not found: {npy_path}")
    if not scene_path.is_file():
        raise FileNotFoundError(f"Capture RGB preview not found: {scene_path}")

    return npy_path, scene_path, depth_preview_path if depth_preview_path.is_file() else None


def mime_type_for_image_path(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suf == ".png":
        return "image/png"
    if suf == ".webp":
        return "image/webp"
    return "image/png"
