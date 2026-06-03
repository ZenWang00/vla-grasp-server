"""Lightweight 2D alignment-point flow: parse a single VLM point + gripper angle,
then recover its 3D camera-frame position from the depth map and intrinsics.

This is the geometry counterpart of :mod:`vg_pipeline.roi` for the alignment flow.
It deliberately avoids SAM2 / Contact-GraspNet: the VLM returns where to cut in
(``align_point``) and how to orient the jaws (``gripper_angle_deg``); the depth map
supplies the missing range so we can back-project to a metric 3D point.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .roi import _load_vlm_json, _scale_norm_xy_to_rgb


@dataclass(frozen=True)
class AlignResult:
    """Parsed alignment proposal in full-resolution pixel coordinates."""

    point_yx: tuple[int, int]
    angle_deg: float
    width_m: float | None


def parse_align_result(
    raw_text: str, *, canvas_h: int, canvas_w: int, rgb_h: int
) -> AlignResult:
    """Parse ``align_point`` (normalized 0-1000), ``gripper_angle_deg`` and optional ``width_mm``."""
    data = _load_vlm_json(raw_text)

    point_raw = data.get("align_point")
    if not isinstance(point_raw, list) or len(point_raw) != 2:
        raise ValueError("'align_point' must be a list [y, x]:\n" + str(data))
    y_val, x_val = point_raw
    point_yx = _scale_norm_xy_to_rgb(
        y_val, x_val, canvas_h=canvas_h, canvas_w=canvas_w, rgb_h=rgb_h
    )

    angle_raw = data.get("gripper_angle_deg")
    if not isinstance(angle_raw, (int, float)):
        raise ValueError("'gripper_angle_deg' must be a number:\n" + str(data))
    angle_deg = float(angle_raw)

    width_m: float | None = None
    width_raw = data.get("width_mm")
    if isinstance(width_raw, (int, float)) and float(width_raw) > 0.0:
        width_m = float(width_raw) / 1000.0

    return AlignResult(point_yx=point_yx, angle_deg=angle_deg, width_m=width_m)


def parse_align_results_multi(
    raw_text: str, *, canvas_h: int, canvas_w: int, rgb_h: int
) -> list[AlignResult]:
    """Parse VLM JSON with a ``candidates`` array into a list of :class:`AlignResult`.

    Raises ``ValueError`` if no candidates are found in the response.
    """
    data = _load_vlm_json(raw_text)
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or len(candidates) == 0:
        raise ValueError("VLM returned no candidates in align multi response:\n" + str(data))

    results: list[AlignResult] = []
    for item in candidates:
        point_raw = item.get("align_point")
        if not isinstance(point_raw, list) or len(point_raw) != 2:
            raise ValueError("'align_point' must be a list [y, x] in each candidate:\n" + str(item))
        y_val, x_val = point_raw
        point_yx = _scale_norm_xy_to_rgb(
            y_val, x_val, canvas_h=canvas_h, canvas_w=canvas_w, rgb_h=rgb_h
        )

        angle_raw = item.get("gripper_angle_deg")
        if not isinstance(angle_raw, (int, float)):
            raise ValueError("'gripper_angle_deg' must be a number in each candidate:\n" + str(item))
        angle_deg = float(angle_raw)

        width_m: float | None = None
        width_raw = item.get("width_mm")
        if isinstance(width_raw, (int, float)) and float(width_raw) > 0.0:
            width_m = float(width_raw) / 1000.0

        results.append(AlignResult(point_yx=point_yx, angle_deg=angle_deg, width_m=width_m))

    return results


def sample_depth_median(
    depth: np.ndarray, v: int, u: int, *, window: int = 5
) -> float:
    """Median of valid (``isfinite & > 0``) depth in a ``window``x``window`` patch around ``(v, u)``.

    Raises ``ValueError`` if no valid depth exists in the neighbourhood.
    """
    h, w = depth.shape
    if not (0 <= v < h and 0 <= u < w):
        raise ValueError(f"align_point ({v}, {u}) is outside the {h}x{w} depth image")
    r = max(0, window // 2)
    v0, v1 = max(0, v - r), min(h, v + r + 1)
    u0, u1 = max(0, u - r), min(w, u + r + 1)
    patch = np.asarray(depth[v0:v1, u0:u1], dtype=np.float64)
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        raise ValueError(
            f"no valid depth near align_point ({v}, {u}); the selected point has no depth"
        )
    return float(np.median(valid))


def deproject_pixel(u: float, v: float, z: float, K: np.ndarray) -> np.ndarray:
    """Back-project a single pixel ``(u, v)`` at depth ``z`` to a camera-frame ``(X, Y, Z)``.

    Uses the same pinhole model as :func:`geometry.backproject_depth_with_mask`.
    """
    K = np.asarray(K, dtype=np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (float(u) - cx) * z / fx
    y = (float(v) - cy) * z / fy
    return np.array([x, y, z], dtype=np.float64)
