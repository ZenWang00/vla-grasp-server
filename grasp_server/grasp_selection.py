"""Load Contact-GraspNet predictions, pick top-K, convert to a ROS2-friendly schema.

Pose convention matches Contact-GraspNet's camera-frame 4x4:

- column 0 (x): gripper "base" / closing direction
- column 1 (y): lateral
- column 2 (z): approach (the direction the gripper moves into the contact)

The quaternion returned is xyzw so the future ROS2 node can drop it into a
``geometry_msgs/Pose`` without further conversion.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from vg_pipeline.grasp_results import (
    NormalizedGrasp,
    normalize_predictions_multi,
)


def _predictions_path_for_input(input_npz: Path) -> Path:
    return input_npz.parent / f"predictions_{input_npz.stem}.npz"


def predicted_npz_paths(input_npzs: list[Path]) -> list[Path]:
    """Companion ``predictions_<stem>.npz`` paths the CGN worker is expected to write."""
    return [_predictions_path_for_input(p) for p in input_npzs]


def _rotation_to_quaternion_xyzw(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to (qx, qy, qz, qw). Uses Shepperd's method."""
    m = np.asarray(R, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {m.shape}")
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    # Normalize to guard against numerical drift in the source rotation.
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm > 0.0:
        q = q / norm
    # Canonicalize sign so qw >= 0 (purely cosmetic, friendlier for diffs).
    if q[3] < 0.0:
        q = -q
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def _candidate_index_from_npz(predictions_npz: Path) -> int | None:
    match = re.search(r"(\d+)(?=\.npz$)", predictions_npz.name)
    if match is None:
        return None
    return int(match.group(1))


def _serialize_grasp(grasp: NormalizedGrasp) -> dict[str, Any]:
    pose = np.asarray(grasp.transform, dtype=np.float64)
    qx, qy, qz, qw = _rotation_to_quaternion_xyzw(pose[:3, :3])
    candidate_index = (
        _candidate_index_from_npz(grasp.source_npz) if grasp.source_npz is not None else None
    )
    return {
        # TODO (tech debt): score is not comparable across pipelines for top-K selection
        # (A=approach dot-product, B=CGN confidence, C=1.0). To be unified later.
        "score": float(grasp.score),
        # model_confidence: CGN output, Option B only. Same source as score (score field
        # does not yet represent geometric quality — tech debt). Not used in any
        # computation currently; do not conflate with score if score's source changes.
        "model_confidence": float(grasp.score),
        "pose_4x4": pose.tolist(),
        "position_xyz": pose[:3, 3].astype(float).tolist(),
        "quaternion_xyzw": [qx, qy, qz, qw],
        "width_m": None if grasp.width_m is None else float(grasp.width_m),
        "approach_dir_xyz": pose[:3, 2].astype(float).tolist(),
        "source": {
            "candidate_index": candidate_index,
            "segment_id": int(grasp.segment_id),
            "grasp_index": int(grasp.grasp_index),
            "predictions_npz": (
                None if grasp.source_npz is None else grasp.source_npz.name
            ),
        },
    }


def select_top_k_grasps(
    predictions_npzs: list[Path],
    top_k: int,
) -> list[dict[str, Any]]:
    """Merge predictions across candidates, sort by score, return JSON-ready top-K grasps in camera frame."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if not predictions_npzs:
        raise ValueError("predictions_npzs is empty")

    grasps = normalize_predictions_multi(predictions_npzs)
    selected = grasps[:top_k]
    return [_serialize_grasp(g) for g in selected]
