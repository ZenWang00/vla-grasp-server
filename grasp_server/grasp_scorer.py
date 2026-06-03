"""Post-IK scorer: selects the single best grasp from IK-passing candidates.

Called from /select_and_execute. Operates entirely on GraspDict fields;
no depth map or camera data required.

Score source: grasp["score"] from ik_results — ROS2 passes GraspDict back
unchanged, so all original GEN fields are intact.

weight_ik_margin is a RESERVED extension slot, always 0.0 for now.
Activate when: ROS2 _pin_solve_ik returns (q, J), submit_ik_result carries
a per-grasp "ik_margin" dict, and both joint_margin and manipulability are
normalised within-batch via min-max (NOT absolute thresholds — values are
robot-geometry-dependent).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GraspScorerConfig:
    weight_gen_score: float             # geometric: original model confidence
    weight_approach: float              # geometric: approach direction alignment
    weight_width: float                 # geometric: width preference
    weight_ik_margin: float             # RESERVED — always 0.0; see module docstring
    preferred_approach_xyz: list[float] # camera-frame preferred approach direction
    preferred_width_m: float            # target gripper opening in metres


@dataclass(frozen=True)
class GraspScoringReport:
    scores: list[float]        # composite score per input grasp (same order as input)
    ranked_indices: list[int]  # input indices sorted best-first
    best_index: int            # index of the single best grasp in the input list
    best_grasp: dict           # copy of grasps[best_index]


_DEFAULT_CONFIG = GraspScorerConfig(
    weight_gen_score=0.5,
    weight_approach=0.3,
    weight_width=0.2,
    weight_ik_margin=0.0,
    preferred_approach_xyz=[0.0, 0.0, 1.0],
    preferred_width_m=0.05,
)


def _score_gen(grasp: dict) -> float:
    """Sub-score from the original GEN confidence (grasp["score"]).

    Normalises to [0, 1]; returns 0.0 for missing or invalid values.
    """
    raise NotImplementedError


def _score_approach(grasp: dict, cfg: GraspScorerConfig) -> float:
    """Sub-score based on alignment of approach with cfg.preferred_approach_xyz.

    dot(grasp["approach_dir_xyz"], normalised preferred) mapped from [-1, 1] → [0, 1].
    """
    raise NotImplementedError


def _score_width(grasp: dict, cfg: GraspScorerConfig) -> float:
    """Sub-score based on proximity of width_m to cfg.preferred_width_m.

    Returns 1.0 when width_m == preferred_width_m, decays toward 0 for
    large deviations. Returns a conservative default when width_m is None.
    """
    raise NotImplementedError


def _score_ik_margin(grasp: dict, cfg: GraspScorerConfig) -> float:
    """Sub-score from IK quality metrics. RESERVED — always returns 0.0.

    When activated: expects grasp["ik_margin"] = {"joint_margin": float,
    "manipulability": float}, both normalised within-batch via min-max.
    """
    return 0.0


def compute_composite_score(grasp: dict, cfg: GraspScorerConfig) -> float:
    """Weighted sum of four sub-scores, normalised so active weights sum to 1."""
    raise NotImplementedError


def rank_grasps(
    grasps: list[dict],
    cfg: GraspScorerConfig | None = None,
) -> GraspScoringReport:
    """Score all grasps and return a full GraspScoringReport.

    cfg=None uses _DEFAULT_CONFIG.
    Raises ValueError if grasps is empty.
    """
    raise NotImplementedError


def select_best(
    grasps: list[dict],
    cfg: GraspScorerConfig | None = None,
) -> dict:
    """Return the single highest-scoring grasp dict.

    Thin wrapper around rank_grasps.
    Raises ValueError if grasps is empty.
    """
    raise NotImplementedError
