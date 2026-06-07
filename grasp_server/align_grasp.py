"""Assemble a single 6-DoF grasp dict from a VLA 2D alignment point + gripper angle.

The output matches :func:`grasp_selection._serialize_grasp` field-for-field so the
ROS2 client (and the ``/select_and_execute`` -> ``/poll_publish`` path) consumes it
identically to a Contact-GraspNet grasp.

Pose convention (camera frame, same as Contact-GraspNet):

- column 0 (x): gripper "base" / closing direction
- column 1 (y): lateral
- column 2 (z): approach (direction the gripper moves into contact)

The VLA returns a single in-image gripper angle, interpreted as rotation about the
camera optical / approach axis (camera +Z). The approach is therefore fixed to +Z
and the closing direction lies in the image plane at ``angle_deg``.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from vg_pipeline.align import deproject_pixel, sample_depth_median

from .grasp_selection import _rotation_to_quaternion_xyzw

_DEFAULT_WIDTH_M = 0.05  # 50 mm — used when LLM does not estimate gripper width


def _pose_from_point_and_angle(position_xyz: np.ndarray, angle_deg: float) -> np.ndarray:
    """4x4 camera-frame pose: approach=+Z, closing direction rotated by ``angle_deg`` in-plane."""
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    x_axis = np.array([c, s, 0.0], dtype=np.float64)        # closing direction (image plane)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)    # approach (camera forward)
    y_axis = np.cross(z_axis, x_axis)                        # = [-s, c, 0]
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = np.asarray(position_xyz, dtype=np.float64)
    return pose


def build_align_grasp(
    depth: np.ndarray,
    K: np.ndarray,
    *,
    point_yx: tuple[int, int],
    angle_deg: float,
    width_m: float | None,
    depth_window: int = 5,
) -> list[dict[str, Any]]:
    """Return a single-element grasps list (same schema as ``select_top_k_grasps``).

    Raises ``ValueError`` when the alignment point has no valid depth nearby.
    """
    v, u = int(point_yx[0]), int(point_yx[1])
    z = sample_depth_median(np.asarray(depth), v, u, window=depth_window)
    position = deproject_pixel(u, v, z, K)
    pose = _pose_from_point_and_angle(position, angle_deg)
    qx, qy, qz, qw = _rotation_to_quaternion_xyzw(pose[:3, :3])

    grasp = {
        # TODO (tech debt): score is not comparable across pipelines for top-K selection
        # (A=approach dot-product, B=CGN confidence, C=1.0). To be unified later.
        "score": 1.0,
        "model_confidence": None,  # Option C has no model confidence; VLA-align only
        "pose_4x4": pose.tolist(),
        "position_xyz": pose[:3, 3].astype(float).tolist(),
        "quaternion_xyzw": [qx, qy, qz, qw],
        "width_m": float(width_m) if width_m is not None else _DEFAULT_WIDTH_M,
        "approach_dir_xyz": pose[:3, 2].astype(float).tolist(),
        "source": {
            "candidate_index": 0,
            "segment_id": 0,
            "grasp_index": 0,
            "predictions_npz": None,
        },
    }
    return [grasp]
