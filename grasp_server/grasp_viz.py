"""Project top-K grasp poses onto the captured RGB image and save an annotated copy.

Draws a directed arrow for each grasp: tail at the gripper center, head pointing
along the approach direction (z-column of the rotation matrix, same convention as
the HTML report generator in scripts/generate_grasp_report.py).
"""
from __future__ import annotations

import colorsys
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("grasp_server")

# Arrow length in metres projected onto the image.
_APPROACH_METRES = 0.08


def _rank_color(rank_index: int, total: int) -> tuple[int, int, int]:
    hue = 0.62 - (0.55 * (rank_index / max(total, 1)))
    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, 0.85, 1.0)
    return int(255 * r), int(255 * g), int(255 * b)


def _project(xyz: list[float] | np.ndarray, K: np.ndarray) -> tuple[float, float] | None:
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    if z <= 0.0:
        return None
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return float(u), float(v)


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    color: tuple[int, int, int],
    line_width: int,
) -> None:
    draw.line([start_xy, end_xy], fill=color, width=line_width)
    dx = end_xy[0] - start_xy[0]
    dy = end_xy[1] - start_xy[1]
    norm = float(np.hypot(dx, dy))
    if norm < 1.0:
        return
    ux, uy = dx / norm, dy / norm
    left = (-uy, ux)
    arrow_len = min(18.0, max(10.0, norm * 0.22))
    head_w = arrow_len * 0.45
    p1 = end_xy
    p2 = (end_xy[0] - arrow_len * ux + head_w * left[0], end_xy[1] - arrow_len * uy + head_w * left[1])
    p3 = (end_xy[0] - arrow_len * ux - head_w * left[0], end_xy[1] - arrow_len * uy - head_w * left[1])
    draw.polygon([p1, p2, p3], fill=color)


def save_grasp_viz(
    capture_dir: Path,
    run_dir: Path,
    grasps_json: list[dict[str, Any]],
    *,
    out_filename: str = "grasp_viz.jpg",
) -> Path | None:
    """Draw projected grasp arrows on the captured image; save to *run_dir/out_filename*.

    Returns the output path, or None if anything goes wrong.
    """
    preview_path = capture_dir / "color_preview.jpg"
    npy_path = capture_dir / "camera_data.npy"

    if not preview_path.is_file():
        logger.warning("grasp_viz: color_preview.jpg not found at %s", preview_path)
        return None
    if not npy_path.is_file():
        logger.warning("grasp_viz: camera_data.npy not found at %s", npy_path)
        return None
    if not grasps_json:
        logger.warning("grasp_viz: no grasps to visualize")
        return None

    try:
        cam_data = np.load(npy_path, allow_pickle=True).item()
        K = np.asarray(cam_data["K"], dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"K has wrong shape {K.shape}")
    except Exception as exc:
        logger.warning("grasp_viz: failed to load K from %s: %s", npy_path, exc)
        return None

    try:
        img = Image.open(preview_path).convert("RGB")
    except Exception as exc:
        logger.warning("grasp_viz: failed to open preview image: %s", exc)
        return None

    draw = ImageDraw.Draw(img)
    W, H = img.size
    line_width = max(3, min(W, H) // 150)
    dot_radius = max(6, min(W, H) // 80)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            size=max(14, dot_radius),
        )
    except Exception:
        font = ImageFont.load_default()

    total = len(grasps_json)
    for rank, grasp in enumerate(grasps_json):
        xyz = grasp.get("position_xyz")
        approach = grasp.get("approach_dir_xyz")
        if xyz is None or len(xyz) != 3:
            continue

        start_px = _project(xyz, K)
        if start_px is None:
            continue

        color = _rank_color(rank, total)

        # Arrow tail → head along approach direction if available.
        if approach is not None and len(approach) == 3:
            end_xyz = [
                float(xyz[0]) + _APPROACH_METRES * float(approach[0]),
                float(xyz[1]) + _APPROACH_METRES * float(approach[1]),
                float(xyz[2]) + _APPROACH_METRES * float(approach[2]),
            ]
            end_px = _project(end_xyz, K)
            if end_px is not None:
                _draw_arrow(draw, start_px, end_px, color, line_width)

        # Filled circle at the grasp center.
        u, v = start_px
        r = dot_radius
        draw.ellipse(
            [u - r, v - r, u + r, v + r],
            fill=color,
            outline=(0, 0, 0),
            width=max(2, r // 5),
        )

        # Label: rank and score.
        score = grasp.get("score", 0.0)
        label = f"#{rank + 1}  {score:.3f}"
        tx, ty = u + r + 4, v - r
        draw.text((tx + 1, ty + 1), label, fill=(0, 0, 0), font=font)
        draw.text((tx, ty), label, fill=color, font=font)

    out_path = run_dir / out_filename
    try:
        img.save(out_path, format="JPEG", quality=92)
        logger.info("grasp_viz: saved annotated image to %s", out_path)
    except Exception as exc:
        logger.warning("grasp_viz: failed to save %s: %s", out_path, exc)
        return None

    return out_path
