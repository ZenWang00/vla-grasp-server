"""Pre-IK server-side filter: width (hard) + clearance/collision/contact (soft scores).

Called from /trigger_ik_check before GEN output is sent to the ROS2 IK checker.

Hard constraints (boolean, physically impossible → drop):
  - width: object wider than max gripper opening

Soft metrics (continuous [0,1], stored in GraspDict for scorer to read):
  - clearance_score:       approach path depth clearance
  - collision_score:       inverse gripper-pointcloud penetration depth
  - contact_quality_score: contact-point surface-normal alignment

Collision also has a secondary hard threshold (max_penetration_m): grasps with
excessive penetration are rejected even if clearance/contact are fine.

Filter order (cheapest-first): width → collision (hard+score) → clearance (score)
  → contact_quality (score).

All operations are CPU-only numpy; never touches the GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vg_pipeline.grasp_results import FINGER_LENGTH_METERS, GRIPPER_DEPTH_METERS


@dataclass(frozen=True)
class GraspFilterConfig:
    min_width_m: float
    max_width_m: float
    min_clearance_m: float        # reference scale for clearance score (score=1 at this value)
    depth_trunc_m: float          # depth values above this treated as missing
    collision_voxel_size_m: float # voxel resolution for collision check
    max_penetration_m: float      # hard-reject if gripper penetrates point cloud by more than this


@dataclass(frozen=True)
class GraspFilterReport:
    total: int                   # number of grasps passed in
    passed: list[dict]           # grasps surviving all checks, annotated with soft scores
    rejected_width: list[int]    # indices (into original list) rejected by width check
    rejected_clearance: list[int] # always empty (clearance is soft-only); kept for API compat
    rejected_collision: list[int] # indices rejected by collision hard threshold
    num_passed: int              # len(passed) — convenience field


_DEFAULT_CONFIG = GraspFilterConfig(
    min_width_m=0.02,
    max_width_m=0.08,
    min_clearance_m=0.01,
    depth_trunc_m=1.5,
    collision_voxel_size_m=0.005,
    max_penetration_m=0.005,
)


def check_width(grasp: dict[str, Any], cfg: GraspFilterConfig) -> bool:
    """Return True if width_m is within [cfg.min_width_m, cfg.max_width_m].

    width_m=None is treated as unknown and rejected (conservative default).
    """
    w = grasp.get("width_m")
    if w is None:
        return False
    return cfg.min_width_m <= float(w) <= cfg.max_width_m


def _backproject_depth(
    depth: np.ndarray, K: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    """Return (N, 3) float64 point cloud from depth image, optionally masked."""
    H, W = depth.shape
    u_grid, v_grid = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    if mask is not None:
        u_grid = u_grid[mask]
        v_grid = v_grid[mask]
        d = depth[mask].astype(np.float64)
    else:
        d = depth.flatten().astype(np.float64)
        u_grid = u_grid.flatten()
        v_grid = v_grid.flatten()

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u_grid - cx) / fx * d
    y = (v_grid - cy) / fy * d
    return np.stack([x, y, d], axis=1)


def _project_to_pixel(pt_3d: np.ndarray, K: np.ndarray) -> tuple[int, int]:
    """Project a 3D camera-frame point to (u, v) pixel coordinates."""
    if pt_3d[2] <= 0:
        return (-1, -1)
    u = float(K[0, 0]) * pt_3d[0] / pt_3d[2] + float(K[0, 2])
    v = float(K[1, 1]) * pt_3d[1] / pt_3d[2] + float(K[1, 2])
    return (int(round(u)), int(round(v)))


def _score_clearance(
    grasp: dict[str, Any],
    depth: np.ndarray,
    K: np.ndarray,
    cfg: GraspFilterConfig,
) -> float:
    """Continuous [0,1] score for approach-path depth clearance.

    Samples depth pixels in a cylinder of radius width_m/2 along the approach
    ray from the gripper back to the scene.  Score = 1 when the nearest scene
    point is at least cfg.min_clearance_m ahead of the gripper face; 0 when
    there is no clearance.  Returns 0.0 when depth data is invalid/empty.
    """
    try:
        pos = np.array(grasp["position_xyz"], dtype=np.float64)
        approach = np.array(grasp["approach_dir_xyz"], dtype=np.float64)
        width_m = float(grasp.get("width_m") or 0.04)
    except (KeyError, TypeError, ValueError):
        return 0.0

    H, W = depth.shape
    grasp_depth = float(pos[2])  # depth of gripper face in camera frame
    if grasp_depth <= 0:
        return 0.0

    # Project gripper position to pixel
    u0, v0 = _project_to_pixel(pos, K)

    # Radius in pixels: project a point offset by width_m/2 perpendicular to approach
    closing = np.array(grasp.get("closing_dir_xyz", [1.0, 0.0, 0.0]), dtype=np.float64)
    closing -= np.dot(closing, approach) * approach
    closing_norm = np.linalg.norm(closing)
    if closing_norm > 1e-8:
        closing = closing / closing_norm
    else:
        # Fallback: use pose col-0 as closing direction
        pose = np.array(grasp["pose_4x4"], dtype=np.float64)
        closing = pose[:3, 0]

    offset_pt = pos + closing * (width_m / 2.0)
    u1, v1 = _project_to_pixel(offset_pt, K)
    radius_px = max(1, int(np.sqrt((u1 - u0) ** 2 + (v1 - v0) ** 2)))

    # Sample depth in a circle around the approach pixel
    ys, xs = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    circle = xs * xs + ys * ys <= radius_px * radius_px
    yy = v0 + np.where(circle)[0] - radius_px
    xx = u0 + np.where(circle)[1] - radius_px
    valid = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    yy, xx = yy[valid], xx[valid]
    if len(yy) == 0:
        return 0.0

    patch_depths = depth[yy, xx].astype(np.float64)
    patch_depths = patch_depths[(patch_depths > 0) & (patch_depths < cfg.depth_trunc_m)]
    if len(patch_depths) == 0:
        return 0.0

    min_scene_depth = float(np.min(patch_depths))
    clearance = min_scene_depth - grasp_depth
    return float(np.clip(clearance / cfg.min_clearance_m, 0.0, 1.0))


def _score_collision(
    grasp: dict[str, Any],
    depth: np.ndarray,
    K: np.ndarray,
    cfg: GraspFilterConfig,
) -> tuple[bool, float]:
    """Hard filter + continuous [0,1] collision score.

    Voxelises the back-projected point cloud and checks whether the gripper
    finger volumes (two rectangular boxes) intersect occupied voxels.

    Returns:
        (passes, score) where:
          passes = True if max penetration depth <= cfg.max_penetration_m
          score  = 1.0 - clamp(max_penetration / cfg.max_penetration_m, 0, 1)

    Boundary: penetration == max_penetration_m → passes=True, score=0.0.
    Depth invalid → (True, 0.0): allow but score lowest (cannot check).
    """
    try:
        pose = np.array(grasp["pose_4x4"], dtype=np.float64)
        width_m = float(grasp.get("width_m") or 0.04)
    except (KeyError, TypeError, ValueError):
        return (True, 0.0)

    closing_dir = pose[:3, 0]  # col-0: closing direction
    approach_dir = pose[:3, 2]  # col-2: approach direction
    position = pose[:3, 3]

    H, W = depth.shape
    valid_mask = (depth > 0) & (depth < cfg.depth_trunc_m)
    if not np.any(valid_mask):
        return (True, 0.0)

    pts = _backproject_depth(depth, K, valid_mask)
    if len(pts) == 0:
        return (True, 0.0)

    # Transform points into gripper frame: origin=position, axes=(closing, lateral, approach)
    lateral_dir = pose[:3, 1]
    pts_rel = pts - position  # (N, 3)
    pts_cl = pts_rel @ closing_dir   # along closing axis
    pts_la = pts_rel @ lateral_dir   # along lateral axis
    pts_ap = pts_rel @ approach_dir  # along approach axis

    half_width = width_m / 2.0
    finger_half_thick = 0.008  # 8 mm half-thickness of each finger (approximate)

    # Left finger box: closing > +half_width, |lateral| < FINGER_LENGTH/2, |approach| < GRIPPER_DEPTH
    # Right finger box: closing < -half_width, same lateral/approach bounds
    ap_min, ap_max = -GRIPPER_DEPTH_METERS, 0.0
    la_half = FINGER_LENGTH_METERS / 2.0

    max_penetration = 0.0
    for sign in (1.0, -1.0):
        in_box = (
            (np.abs(pts_la) < la_half) &
            (pts_ap > ap_min) & (pts_ap < ap_max) &
            (sign * pts_cl > half_width) &
            (sign * pts_cl < half_width + 2 * finger_half_thick)
        )
        if np.any(in_box):
            # Penetration depth = how far inside the finger
            pen = float(np.max(sign * pts_cl[in_box] - half_width))
            max_penetration = max(max_penetration, pen)

    passes = max_penetration <= cfg.max_penetration_m
    score = 1.0 - float(np.clip(max_penetration / cfg.max_penetration_m, 0.0, 1.0))
    return (passes, score)


def _score_contact_quality(
    grasp: dict[str, Any],
    depth: np.ndarray,
    K: np.ndarray,
    cfg: GraspFilterConfig,
) -> float:
    """Continuous [0,1] score for contact-point surface-normal alignment.

    Projects the two contact points (position ± width_m/2 * closing_dir) onto the
    depth image, samples a local 3×3 depth patch, back-projects to 3D, and estimates
    the local surface normal via PCA.  Score = mean |dot(normal, closing_dir)|
    across both contact points.

    Returns 0.0 if either contact point has no valid depth (conservative).
    """
    try:
        pose = np.array(grasp["pose_4x4"], dtype=np.float64)
        width_m = float(grasp.get("width_m") or 0.04)
    except (KeyError, TypeError, ValueError):
        return 0.0

    position = pose[:3, 3]
    closing_dir = pose[:3, 0]

    H, W = depth.shape
    scores: list[float] = []

    for sign in (1.0, -1.0):
        contact_pt = position + sign * (width_m / 2.0) * closing_dir
        u, v = _project_to_pixel(contact_pt, K)
        if u < 1 or u >= W - 1 or v < 1 or v >= H - 1:
            return 0.0

        # 3×3 depth patch
        patch_v = np.arange(v - 1, v + 2)
        patch_u = np.arange(u - 1, u + 2)
        uu, vv = np.meshgrid(patch_u, patch_v)
        patch_depth = depth[vv, uu].astype(np.float64)
        patch_valid = (patch_depth > 0) & (patch_depth < cfg.depth_trunc_m)

        n_valid = int(np.sum(patch_valid))
        if n_valid < 3:
            return 0.0  # not enough points for PCA

        # Back-project valid patch pixels to 3D
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        pu = uu[patch_valid].astype(np.float64)
        pv = vv[patch_valid].astype(np.float64)
        pd = patch_depth[patch_valid]
        px = (pu - cx) / fx * pd
        py = (pv - cy) / fy * pd
        patch_pts = np.stack([px, py, pd], axis=1)  # (n_valid, 3)

        # Surface normal = smallest singular vector (PCA)
        centered = patch_pts - patch_pts.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return 0.0
        normal = Vt[-1]  # smallest singular vector

        scores.append(float(abs(np.dot(normal, closing_dir))))

    if len(scores) < 2:
        return 0.0
    return float(np.mean(scores))


def filter_grasps(
    grasps: list[dict[str, Any]],
    depth: np.ndarray,
    K: np.ndarray,
    cfg: GraspFilterConfig | None = None,
) -> GraspFilterReport:
    """Apply hard width filter + collision hard threshold; compute soft scores for passing grasps.

    Order: width (hard) → collision (hard + score) → clearance (score) → contact_quality (score).
    A grasp only advances to the next check if it passed all previous hard filters.

    Soft scores are stored back into the GraspDict copy:
      grasp["clearance_score"]       float [0, 1]
      grasp["collision_score"]       float [0, 1]
      grasp["contact_quality_score"] float [0, 1]

    cfg=None uses _DEFAULT_CONFIG.
    """
    if cfg is None:
        cfg = _DEFAULT_CONFIG

    depth = np.asarray(depth, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    rejected_width: list[int] = []
    rejected_collision: list[int] = []
    passed: list[dict] = []

    for i, grasp in enumerate(grasps):
        # --- Hard: width ---
        if not check_width(grasp, cfg):
            rejected_width.append(i)
            continue

        # --- Hard threshold + soft score: collision ---
        col_passes, col_score = _score_collision(grasp, depth, K, cfg)
        if not col_passes:
            rejected_collision.append(i)
            continue

        # --- Soft scores: clearance and contact quality ---
        clr_score = _score_clearance(grasp, depth, K, cfg)
        ctq_score = _score_contact_quality(grasp, depth, K, cfg)

        # Annotate a shallow copy of the grasp with soft scores
        annotated = dict(grasp)
        annotated["clearance_score"] = clr_score
        annotated["collision_score"] = col_score
        annotated["contact_quality_score"] = ctq_score
        passed.append(annotated)

    return GraspFilterReport(
        total=len(grasps),
        passed=passed,
        rejected_width=rejected_width,
        rejected_clearance=[],
        rejected_collision=rejected_collision,
        num_passed=len(passed),
    )
