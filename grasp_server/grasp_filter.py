"""Pre-IK server-side filter: width, clearance, collision.

Called from /trigger_ik_check before GEN output is sent to the ROS2 IK checker.
Checks are applied cheapest-first: width → clearance → collision.
All operations are CPU-only numpy; never touches the GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GraspFilterConfig:
    min_width_m: float
    max_width_m: float
    min_clearance_m: float       # min depth-based clearance along approach axis
    depth_trunc_m: float         # depth values above this treated as missing
    collision_voxel_size_m: float  # voxel resolution for collision check


@dataclass(frozen=True)
class GraspFilterReport:
    total: int                   # number of grasps passed in
    passed: list[dict]           # grasps surviving all checks (GraspDict elements)
    rejected_width: list[int]    # indices (into original list) rejected by width
    rejected_clearance: list[int]
    rejected_collision: list[int]
    num_passed: int              # len(passed) — convenience field


_DEFAULT_CONFIG = GraspFilterConfig(
    min_width_m=0.02,
    max_width_m=0.08,
    min_clearance_m=0.01,
    depth_trunc_m=1.5,
    collision_voxel_size_m=0.005,
)


def check_width(grasp: dict[str, Any], cfg: GraspFilterConfig) -> bool:
    """Return True if width_m is within [cfg.min_width_m, cfg.max_width_m].

    width_m=None is treated as unknown and rejected (conservative default).
    """
    raise NotImplementedError


def check_clearance(
    grasp: dict[str, Any],
    depth: np.ndarray,
    K: np.ndarray,
    cfg: GraspFilterConfig,
) -> bool:
    """Return True if the approach path has sufficient depth clearance.

    Projects the approach vector into the depth image and checks that no
    depth sample within the gripper footprint is closer than cfg.min_clearance_m
    to the gripper body along the approach direction.
    """
    raise NotImplementedError


def check_collision(
    grasp: dict[str, Any],
    depth: np.ndarray,
    K: np.ndarray,
    cfg: GraspFilterConfig,
) -> bool:
    """Return True if the gripper body does not collide with the observed scene.

    Voxelises the back-projected point cloud and checks gripper finger volumes
    for occupancy within cfg.collision_voxel_size_m resolution.
    """
    raise NotImplementedError


def filter_grasps(
    grasps: list[dict[str, Any]],
    depth: np.ndarray,
    K: np.ndarray,
    cfg: GraspFilterConfig | None = None,
) -> GraspFilterReport:
    """Apply all three filters in sequence; return a GraspFilterReport.

    Order: width → clearance → collision (cheapest first).
    A grasp only advances to the next check if it passed the previous one.
    cfg=None uses _DEFAULT_CONFIG.
    """
    raise NotImplementedError
