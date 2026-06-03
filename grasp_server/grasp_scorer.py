"""Post-IK scorer: selects the single best grasp from IK-passing candidates.

Called from /select_and_execute. Operates entirely on GraspDict fields;
no depth map or camera data required.

Soft geometric-quality metrics (four active, two RESERVED):
  weight_clearance:    approach path depth clearance     (pre-computed by filter)
  weight_collision:    inverse gripper-pointcloud pen.   (pre-computed by filter)
  weight_width_margin: room left in gripper opening      (computed here from width_m)
  weight_contact:      contact-point surface normal aln  (pre-computed by filter)
  weight_approach:     RESERVED (0.0) — approach axis of TCP frame unresolved (Q1);
                       input (gripper approach in base frame) not yet plumbed.
                       When activating: query TF lio_gripper_interface_link for TCP
                       approach axis in base frame; transform grasp approach_dir_xyz
                       from camera frame to base frame before dot-product.
  weight_ik_margin:    RESERVED (0.0) — activate when ROS2 _pin_solve_ik returns (q, J)
                       and submit_ik_result carries per-grasp "ik_margin" dict with
                       joint_margin and manipulability normalised within-batch via min-max.

CGN model confidence (model_confidence field, Option B only) is intentionally
excluded from scoring for now — it is pipeline-specific and not a geometric metric.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GraspScorerConfig:
    weight_clearance: float             # approach path depth clearance
    weight_collision: float             # inverse penetration depth
    weight_width_margin: float          # room left in gripper opening
    weight_contact: float               # contact-point surface-normal alignment
    weight_approach: float              # RESERVED — always 0.0; see module docstring
    weight_ik_margin: float             # RESERVED — always 0.0; see module docstring
    max_width_m: float                  # gripper max opening, for margin normalisation
    preferred_approach_xyz: list[float] # RESERVED: gripper TCP approach axis in base frame
                                        # (NOT camera +Z; see module docstring)


@dataclass(frozen=True)
class GraspScoringReport:
    scores: list[float]        # composite score per input grasp (same order as input)
    ranked_indices: list[int]  # input indices sorted best-first
    best_index: int            # index of the single best grasp in the input list
    best_grasp: dict           # copy of grasps[best_index]


_DEFAULT_CONFIG = GraspScorerConfig(
    weight_clearance=0.30,
    weight_collision=0.25,
    weight_width_margin=0.25,
    weight_contact=0.20,
    weight_approach=0.0,   # RESERVED
    weight_ik_margin=0.0,  # RESERVED
    max_width_m=0.08,
    preferred_approach_xyz=[0.0, 0.0, 1.0],
)


def _score_width_margin(grasp: dict, cfg: GraspScorerConfig) -> float:
    """Sub-score for width margin: how much of the gripper opening is unused.

    score = (max_width_m - width_m) / max_width_m, clamped to [0, 1].
    Larger margin → more stable grasp.

    width_m is guaranteed non-None here: check_width already rejected None grasps.
    """
    width_m = float(grasp["width_m"])
    margin = cfg.max_width_m - width_m
    return float(max(0.0, min(1.0, margin / cfg.max_width_m)))


def _score_clearance(grasp: dict, cfg: GraspScorerConfig) -> float:
    """Sub-score from pre-computed approach clearance (stored by filter_grasps).

    Falls back to 0.0 (conservative/worst case) when not yet computed.
    """
    return float(grasp.get("clearance_score", 0.0))


def _score_collision(grasp: dict, cfg: GraspScorerConfig) -> float:
    """Sub-score from pre-computed collision check (stored by filter_grasps).

    Falls back to 0.0 (conservative/worst case) when not yet computed.
    """
    return float(grasp.get("collision_score", 0.0))


def _score_contact(grasp: dict, cfg: GraspScorerConfig) -> float:
    """Sub-score from pre-computed contact-point normal alignment (stored by filter_grasps).

    Falls back to 0.0 (conservative/worst case) when not yet computed.
    """
    return float(grasp.get("contact_quality_score", 0.0))


def _score_approach(grasp: dict, cfg: GraspScorerConfig) -> float:
    """Sub-score for approach direction alignment. RESERVED — always returns 0.0.

    When activated:
      1. Obtain gripper TCP approach axis in base frame from TF (Q1 unresolved —
         confirm which axis of lio_gripper_interface_link is the approach axis).
      2. Store as cfg.preferred_approach_xyz (set at request time, not config time).
      3. Transform grasp["approach_dir_xyz"] from camera/optical frame to base frame
         via the camera→base extrinsic (available after IK).
      4. score = (dot(approach_base, preferred_base) + 1) / 2  (maps [-1,1] → [0,1])
    """
    return 0.0


def _score_ik_margin(grasp: dict, cfg: GraspScorerConfig) -> float:
    """Sub-score from IK quality metrics. RESERVED — always returns 0.0.

    When activated: expects grasp["ik_margin"] = {"joint_margin": float,
    "manipulability": float}, both normalised within-batch via min-max.
    """
    return 0.0


def compute_composite_score(grasp: dict, cfg: GraspScorerConfig) -> float:
    """Weighted sum of sub-scores, normalised so active (non-zero) weights sum to 1."""
    weights = [
        cfg.weight_clearance,
        cfg.weight_collision,
        cfg.weight_width_margin,
        cfg.weight_contact,
        cfg.weight_approach,
        cfg.weight_ik_margin,
    ]
    sub_scores = [
        _score_clearance(grasp, cfg),
        _score_collision(grasp, cfg),
        _score_width_margin(grasp, cfg),
        _score_contact(grasp, cfg),
        _score_approach(grasp, cfg),
        _score_ik_margin(grasp, cfg),
    ]
    active_sum = sum(w for w in weights if w > 0.0)
    if active_sum == 0.0:
        return 0.0
    return float(sum(w * s for w, s in zip(weights, sub_scores)) / active_sum)


def rank_grasps(
    grasps: list[dict],
    cfg: GraspScorerConfig | None = None,
) -> GraspScoringReport:
    """Score all grasps and return a full GraspScoringReport.

    cfg=None uses _DEFAULT_CONFIG.
    Raises ValueError if grasps is empty.
    """
    if not grasps:
        raise ValueError("rank_grasps: grasps list is empty")
    if cfg is None:
        cfg = _DEFAULT_CONFIG

    scores = [compute_composite_score(g, cfg) for g in grasps]
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    best_index = ranked_indices[0]
    return GraspScoringReport(
        scores=scores,
        ranked_indices=ranked_indices,
        best_index=best_index,
        best_grasp=dict(grasps[best_index]),
    )


def select_best(
    grasps: list[dict],
    cfg: GraspScorerConfig | None = None,
) -> dict:
    """Return the single highest-scoring grasp dict.

    Thin wrapper around rank_grasps.
    Raises ValueError if grasps is empty.
    """
    return rank_grasps(grasps, cfg).best_grasp
