from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import backproject_depth_with_mask, render_pointcloud_3d_png

__all__ = [
    "build_pure_target_pointcloud",
    "project_points_to_rgb_full",
    "render_pointcloud_3d_png",
]


def build_pure_target_pointcloud(
    depth: np.ndarray,
    K: np.ndarray,
    tight_mask: np.ndarray,
) -> np.ndarray:
    """Return camera-frame points where ``tight_mask`` is positive and depth is valid.

    Output shape is ``(N, 3)`` float32. Validity additionally requires
    ``isfinite(depth) & (depth > 0)``.
    """
    return backproject_depth_with_mask(depth, K, mask=tight_mask)


def project_points_to_rgb_full(
    points: np.ndarray,
    K: np.ndarray,
    rgb: np.ndarray,
    path: Path,
    max_points: int = 5000,
) -> None:
    """Project a full-frame point cloud back onto the full RGB image as red dots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = np.asarray(rgb, dtype=np.uint8).copy()
    if points.shape[0] == 0:
        Image.fromarray(overlay, mode="RGB").save(str(path), format="PNG")
        return

    K = np.asarray(K, dtype=np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    pts = points.astype(np.float64, copy=False)
    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
    valid = np.isfinite(Z) & (Z > 0.0)
    if not np.any(valid):
        Image.fromarray(overlay, mode="RGB").save(str(path), format="PNG")
        return

    X, Y, Z = X[valid], Y[valid], Z[valid]
    u = np.rint((fx * X / Z) + cx).astype(np.int32)
    v = np.rint((fy * Y / Z) + cy).astype(np.int32)

    h, w = overlay.shape[:2]
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[in_bounds]
    v = v[in_bounds]

    if u.size > max_points:
        step = max(1, u.size // max_points)
        u = u[::step]
        v = v[::step]

    overlay[v, u] = np.array([255, 0, 0], dtype=np.uint8)
    Image.fromarray(overlay, mode="RGB").save(str(path), format="PNG")
