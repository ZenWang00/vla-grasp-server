"""Option A: approach-aware geometry-based grasp synthesis.

Generates parallel-jaw grasp candidates purely from point cloud PCA and an
optional user approach direction — no VLM, SAM2, or Contact-GraspNet required.

Algorithm
---------
1. Back-project the depth map to a point cloud (central-crop + depth-range filter).
2. PCA via SVD → 3 principal axes + per-axis extents.
3. For each of 3 axes × 2 signs × n_rot in-plane rotations = 18 raw candidates:
   - approach_vec  = ±pca_axis_i
   - base_closing  = min-extent axis in the plane ⊥ to approach (stable contact)
   - closing_vec   = base_closing rotated k*(π/n_rot) around approach_vec
   - position      = centroid − approach_vec * (half-extent + 20 mm clearance)
   - score         = max(0, cos(approach_vec, user_approach_dir))
   - width_m       = clamp(min-extent in ⊥ plane, 0.02, 0.08)
4. De-duplicate: two poses are "too similar" only when position, approach AND
   closing direction are all close — the closing check is critical so that in-plane
   rotation variants are not collapsed back to 6.
5. Sort by score desc, return top_k.

Output schema is identical to Options B (Contact-GraspNet) and C (VLM align),
so the ROS2 client and viz pipeline consume it without modification.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from vg_pipeline.geometry import backproject_depth_with_mask
from .grasp_selection import _rotation_to_quaternion_xyzw

_MIN_POINTS = 50


def _pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (centroid, axes, extents).

    axes: (3,3) float64, columns are principal axes in descending-variance order.
    extents: (3,) float64, span along each principal axis.
    """
    centroid = points.mean(axis=0).astype(np.float64)
    centered = (points - centroid).astype(np.float64)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    axes = Vt.T  # columns = principal axes, desc variance
    projected = centered @ axes
    extents = projected.max(axis=0) - projected.min(axis=0)
    return centroid, axes, extents


def _build_pose(position: np.ndarray, approach: np.ndarray, closing: np.ndarray) -> np.ndarray:
    """Return a 4×4 camera-frame pose matrix.

    Convention (same as align_grasp.py and Contact-GraspNet):
      col-0 (x): closing direction
      col-1 (y): lateral = cross(approach, closing_ortho)
      col-2 (z): approach direction
    """
    z = approach / np.linalg.norm(approach)
    x = closing - np.dot(closing, z) * z  # orthogonalise vs approach
    x_norm = np.linalg.norm(x)
    if x_norm < 1e-8:
        # closing is parallel to approach — pick an arbitrary perpendicular
        perp = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = perp - np.dot(perp, z) * z
        x = x / np.linalg.norm(x)
    else:
        x = x / x_norm
    y = np.cross(z, x)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = z
    pose[:3, 3] = position
    return pose


def _too_similar(pa: np.ndarray, pb: np.ndarray) -> bool:
    """True when position, approach AND closing direction are all close.

    The closing-direction check is mandatory: in-plane rotation variants share
    the same position and approach but differ only in closing direction, so
    omitting it would collapse all n_rot variants to 1 candidate.
    Parallel-jaw symmetry → use |cos| for closing (θ ≡ θ+180°).
    """
    if np.linalg.norm(pa[:3, 3] - pb[:3, 3]) >= 0.01:
        return False
    app_cos = np.clip(np.dot(pa[:3, 2], pb[:3, 2]), -1.0, 1.0)
    if math.degrees(math.acos(app_cos)) >= 15.0:
        return False
    close_cos = np.clip(abs(np.dot(pa[:3, 0], pb[:3, 0])), 0.0, 1.0)
    return math.degrees(math.acos(close_cos)) < 20.0


def _generate_raw_candidates(
    centroid: np.ndarray,
    axes: np.ndarray,
    extents: np.ndarray,
    approach_dir: np.ndarray,
    n_rot: int,
    n_pos: int,
) -> list[tuple[float, float, np.ndarray]]:
    """Return list of (score, width_m, pose_4x4) for all raw candidates."""
    candidates: list[tuple[float, float, np.ndarray]] = []
    long_axis = axes[:, 0]  # highest-variance axis (used for n_pos translation)
    long_extent = float(extents[0])

    for i in range(3):
        other = [j for j in range(3) if j != i]
        # Closing = min-extent axis in the plane perpendicular to approach axis i
        if extents[other[0]] <= extents[other[1]]:
            close_idx, width_extent = other[0], float(extents[other[0]])
        else:
            close_idx, width_extent = other[1], float(extents[other[1]])
        base_closing = axes[:, close_idx]
        width_m = float(np.clip(width_extent, 0.02, 0.08))
        standoff = float(extents[i]) / 2.0 + 0.02

        for sign in (1, -1):
            approach_vec = sign * axes[:, i]
            base_pos = centroid - approach_vec * standoff
            score = float(max(0.0, np.dot(approach_vec, approach_dir)))

            # n_pos: positions along the long axis (centroid ± fractions)
            pos_offsets: list[np.ndarray]
            if n_pos <= 1:
                pos_offsets = [base_pos]
            else:
                pos_offsets = []
                for p in range(n_pos):
                    frac = (p / (n_pos - 1) - 0.5) * long_extent * 0.5
                    pos_offsets.append(base_pos + long_axis * frac)

            # n_rot: in-plane rotations of closing direction around approach axis
            for pos in pos_offsets:
                for k in range(n_rot):
                    theta = math.pi * k / n_rot
                    c, s = math.cos(theta), math.sin(theta)
                    closing = c * base_closing + s * np.cross(approach_vec, base_closing)
                    pose = _build_pose(pos.copy(), approach_vec.copy(), closing)
                    candidates.append((score, width_m, pose))

    return candidates


def generate_geometry_grasps(
    depth: np.ndarray,
    K: np.ndarray,
    *,
    approach_dir_cam: list[float] | None = None,
    top_k: int = 5,
    n_rot: int = 3,
    n_pos: int = 1,
    center_crop_frac: float = 0.6,
    depth_min_m: float = 0.05,
    depth_max_m: float = 1.5,
) -> list[dict[str, Any]]:
    """Generate up to top_k geometry-based grasp candidates.

    Parameters
    ----------
    depth:
        (H, W) depth map in metres.
    K:
        (3, 3) camera intrinsic matrix.
    approach_dir_cam:
        Preferred approach direction in camera frame [x, y, z].  Defaults to
        camera +Z (straight forward) when omitted.
    top_k:
        Maximum number of grasps to return.
    n_rot:
        Number of in-plane closing-direction rotations per seed axis (default 3
        → 0°, 60°, 120°).  Total raw candidates = 3 × 2 × n_rot × n_pos.
    n_pos:
        Number of grip positions along the long axis (1 = centroid only).
    center_crop_frac:
        Fraction of image (centred) used for PCA; 1.0 = full image.
    depth_min_m / depth_max_m:
        Depth range filter applied before back-projection.

    Raises
    ------
    ValueError
        If fewer than _MIN_POINTS valid points remain after filtering.
    """
    depth = np.asarray(depth, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    H, W = depth.shape

    # Build pixel mask: central crop + depth range
    range_mask = (np.isfinite(depth)) & (depth > depth_min_m) & (depth < depth_max_m)
    if center_crop_frac < 1.0:
        cy, cx = H // 2, W // 2
        rh = max(1, int(H * center_crop_frac / 2))
        rw = max(1, int(W * center_crop_frac / 2))
        crop_mask = np.zeros((H, W), dtype=bool)
        crop_mask[cy - rh:cy + rh, cx - rw:cx + rw] = True
        mask = range_mask & crop_mask
    else:
        mask = range_mask

    points = backproject_depth_with_mask(depth.astype(np.float32), K, mask=mask)
    if len(points) < _MIN_POINTS:
        raise ValueError(
            f"only {len(points)} valid depth points after filtering "
            f"(need >= {_MIN_POINTS}); check depth_max_m or center_crop_frac"
        )

    approach_dir = np.array(approach_dir_cam, dtype=np.float64) if approach_dir_cam else np.array([0.0, 0.0, 1.0])
    norm = np.linalg.norm(approach_dir)
    if norm < 1e-8:
        approach_dir = np.array([0.0, 0.0, 1.0])
    else:
        approach_dir = approach_dir / norm

    centroid, axes, extents = _pca(points)
    raw = _generate_raw_candidates(centroid, axes, extents, approach_dir, n_rot, n_pos)

    # De-duplicate: keep first occurrence when position + approach + closing all close
    kept: list[tuple[float, float, np.ndarray]] = []
    for cand in raw:
        if not any(_too_similar(cand[2], k[2]) for k in kept):
            kept.append(cand)

    # Sort by score descending, take top_k
    kept.sort(key=lambda c: -c[0])
    kept = kept[:top_k]

    grasps: list[dict[str, Any]] = []
    for idx, (score, width_m, pose) in enumerate(kept):
        qx, qy, qz, qw = _rotation_to_quaternion_xyzw(pose[:3, :3])
        grasps.append({
            "score": score,
            "pose_4x4": pose.tolist(),
            "position_xyz": pose[:3, 3].tolist(),
            "quaternion_xyzw": [qx, qy, qz, qw],
            "width_m": width_m,
            "approach_dir_xyz": pose[:3, 2].tolist(),
            "source": {
                "candidate_index": idx,
                "segment_id": 0,
                "grasp_index": idx,
                "predictions_npz": None,
            },
        })
    return grasps
