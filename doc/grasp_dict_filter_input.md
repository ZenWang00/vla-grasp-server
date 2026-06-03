# GraspDict Schema at `filter_grasps` Input

`filter_grasps(grasps, depth, K, cfg)` in `grasp_server/grasp_filter.py` receives a
`list[dict]` produced by one of the three grasp-generation pipelines. This document
describes the exact schema entering the filter and how each pipeline populates every field.

---

## Common schema (all three options)

| Field | Type | Description |
|---|---|---|
| `score` | `float` | Pipeline-specific quality proxy; used for initial top-K sorting **only**. **Not** a geometric quality score — meaning differs across pipelines (tech debt; see §4). |
| `model_confidence` | `float \| None` | CGN model confidence — Option B only. `None` for A and C. |
| `pose_4x4` | `list[list[float]]` (4×4) | Camera-frame SE(3) pose. Convention: col-0 = closing dir, col-1 = lateral, col-2 = approach dir, col-3 = position. |
| `position_xyz` | `list[float]` (3,) | Gripper face centre in camera frame (metres). Equal to `pose_4x4[:3, 3]` (numpy) — the 4th column, rows 0–2. |
| `quaternion_xyzw` | `list[float]` (4,) | Rotation as `[qx, qy, qz, qw]`. Derived from `pose_4x4[:3,:3]` via Shepperd's method; canonical sign (`qw ≥ 0`). |
| `width_m` | `float \| None` | Gripper opening width in metres. `None` when unavailable (Option C without VLM-provided width). `check_width` rejects `None` — such grasps never reach the scorer. |
| `approach_dir_xyz` | `list[float]` (3,) | Unit approach direction in camera frame. Equal to column 2 of `pose_4x4`. |
| `source` | `dict` | Provenance metadata. Sub-fields: `candidate_index`, `segment_id`, `grasp_index`, `predictions_npz`. |

### `source` sub-fields

| Sub-field | Type | A | B | C |
|---|---|---|---|---|
| `candidate_index` | `int \| None` | Rank index (0 = best) | Parsed from NPZ filename (`\d+` before `.npz`) | `0` |
| `segment_id` | `int` | `0` | SAM2 segment ID | `0` |
| `grasp_index` | `int` | Rank index (same as candidate_index) | CGN prediction index within segment | `0` |
| `predictions_npz` | `str \| None` | `null` | NPZ filename (e.g. `predictions_input_0.npz`) | `null` |

### Fields NOT present before filtering

These three fields are `None`/absent on entry and are added by `filter_grasps` to each
grasp that survives the hard filters:

| Field | Added by filter | Range | Meaning |
|---|---|---|---|
| `clearance_score` | yes | [0, 1] | Approach-path depth clearance score |
| `collision_score` | yes | [0, 1] | Inverse gripper–point-cloud penetration score |
| `contact_quality_score` | yes | [0, 1] | Contact-point surface-normal alignment score |

---

## Option A — Geometry (`geometry_grasp.py`)

**Source:** PCA of the back-projected depth point cloud. No VLM, SAM2, or CGN.

### How each field is populated

| Field | Value |
|---|---|
| `score` | `max(0, dot(approach_vec, user_approach_dir))` — cosine similarity with the caller-supplied preferred approach direction, clipped to [0, 1]. |
| `model_confidence` | `null` |
| `pose_4x4` | Constructed from `_build_pose(position, approach_vec, closing_vec)`. Approach = ±PCA axis i; closing = min-extent PCA axis in the perpendicular plane, optionally rotated by k·π/n_rot. |
| `position_xyz` | `centroid − approach_vec × (half_extent_i + 0.02 m standoff)` |
| `width_m` | `clamp(min_extent_in_perpendicular_plane, 0.02, 0.08)` — always finite, always within the hard filter bounds. |
| `source.candidate_index` | Rank index after de-duplication and sort-by-score (0 = highest score). |
| `source.segment_id` | Always `0` (no segmentation). |
| `source.predictions_npz` | Always `null`. |

**Cardinality:** 1 to `top_k` grasps (default `top_k=5`).  Raw candidate pool is
`3 axes × 2 signs × n_rot × n_pos` before de-duplication.

### Example

```json
{
  "score": 0.847,
  "model_confidence": null,
  "pose_4x4": [
    [ 0.000,  0.000,  1.000,  0.000],
    [ 0.000,  1.000,  0.000,  0.000],
    [-1.000,  0.000,  0.000,  0.520],
    [ 0.000,  0.000,  0.000,  1.000]
  ],
  "position_xyz": [0.000, 0.000, 0.520],
  "quaternion_xyzw": [0.000, 0.707, 0.000, 0.707],
  "width_m": 0.045,
  "approach_dir_xyz": [1.000, 0.000, 0.000],
  "source": {
    "candidate_index": 0,
    "segment_id": 0,
    "grasp_index": 0,
    "predictions_npz": null
  }
}
```

---

## Option B — VLM + SAM2 + Contact-GraspNet (`grasp_selection.py`)

**Source:** `NormalizedGrasp` objects produced by `normalize_predictions_multi()` from
one or more `predictions_*.npz` files written by the CGN worker.

### How each field is populated

| Field | Value |
|---|---|
| `score` | CGN prediction confidence directly from `scores[segment_id][grasp_index]` in the NPZ. Sorted descending before top-K selection. Same numeric value as `model_confidence` (tech debt — see §4). |
| `model_confidence` | Same CGN confidence as `score`. Semantic field: marks this as B-pipeline CGN output. Currently not used in any computation. |
| `pose_4x4` | `pred_grasps_cam[segment_id][grasp_index]` from the NPZ (4×4 float32, CGN camera-frame convention). |
| `position_xyz` | `pose_4x4[:3, 3]`. |
| `width_m` | Derived from CGN contact points via `_derive_width()`: `2 × dot(center − contact_pt + GRIPPER_DEPTH × approach, base_dir)`. `null` when `contact_pts` key is absent from NPZ or result is non-finite. |
| `source.candidate_index` | Integer parsed from the NPZ filename (`predictions_input_<N>.npz` → N), corresponding to the VLM candidate index. |
| `source.segment_id` | SAM2 segment ID (integer key in the NPZ dict). |
| `source.predictions_npz` | NPZ filename string, e.g. `"predictions_input_0.npz"`. |

**Cardinality:** Up to `top_k` grasps merged and re-sorted across all VLM candidates.
Each candidate contributes N_grasps per segment; total pool can be large.

### Example

```json
{
  "score": 0.923,
  "model_confidence": 0.923,
  "pose_4x4": [
    [ 0.812,  0.341, -0.473,  0.018],
    [-0.274,  0.939,  0.206, -0.042],
    [ 0.516, -0.033,  0.856,  0.487],
    [ 0.000,  0.000,  0.000,  1.000]
  ],
  "position_xyz": [0.018, -0.042, 0.487],
  "quaternion_xyzw": [0.091, -0.236, 0.143, 0.957],
  "width_m": 0.062,
  "approach_dir_xyz": [-0.473, 0.206, 0.856],
  "source": {
    "candidate_index": 0,
    "segment_id": 1,
    "grasp_index": 3,
    "predictions_npz": "predictions_input_0.npz"
  }
}
```

---

## Option C — VLA Alignment (`align_grasp.py`)

**Source:** Single 2D alignment point + gripper angle from a VLA model.
No SAM2 or CGN. One grasp per request.

### How each field is populated

| Field | Value |
|---|---|
| `score` | Always `1.0` (fixed). Represents "user/VLA confirmed alignment". Not a geometric metric. |
| `model_confidence` | Always `null`. The VLA produces a 2D point + angle, not a grasp confidence score. |
| `pose_4x4` | Built by `_pose_from_point_and_angle`: approach fixed to camera `+Z`, closing direction = `[cos(angle_deg), sin(angle_deg), 0]` in the image plane. |
| `position_xyz` | Back-projected from depth at the alignment pixel using the median of a `depth_window × depth_window` patch (default 5×5). |
| `width_m` | VLM-provided value if the model outputs one; otherwise `null`. **If `null`, this grasp is rejected by `check_width` and never reaches the scorer.** |
| `approach_dir_xyz` | Always `[0.0, 0.0, 1.0]` (camera optical axis / `+Z`). **Simplification assumption**: the VLA only provides a 2D point + in-plane angle, so the depth direction is forced to the camera optical axis. This is not the same as the gripper approach axis (see R3 — the two axes differ by the eye-in-hand calibration rotation). Consequence: all Option C grasps share the same `approach_dir_xyz`, so the `weight_approach` sub-score (once activated) will assign the same value to every Option C candidate — zero discriminative power within the Option C pool. |
| `source.candidate_index` | Always `0`. |
| `source.segment_id` | Always `0`. |
| `source.predictions_npz` | Always `null`. |

**Cardinality:** Always exactly 1 grasp.

### Example — VLM provided width

```json
{
  "score": 1.0,
  "model_confidence": null,
  "pose_4x4": [
    [ 0.766,  0.643,  0.000,  0.031],
    [-0.643,  0.766,  0.000, -0.018],
    [ 0.000,  0.000,  1.000,  0.412],
    [ 0.000,  0.000,  0.000,  1.000]
  ],
  "position_xyz": [0.031, -0.018, 0.412],
  "quaternion_xyzw": [0.000, 0.000, 0.321, 0.947],
  "width_m": 0.050,
  "approach_dir_xyz": [0.000, 0.000, 1.000],
  "source": {
    "candidate_index": 0,
    "segment_id": 0,
    "grasp_index": 0,
    "predictions_npz": null
  }
}
```

Pose corresponds to: approach = camera `+Z`, closing direction at `angle_deg = 40°` in
the image plane (`cos 40° ≈ 0.766`, `sin 40° ≈ 0.643`), position at depth 0.412 m.

### Example — VLM did not provide width (`width_m: null`)

```json
{
  "score": 1.0,
  "model_confidence": null,
  "pose_4x4": [
    [ 1.000,  0.000,  0.000,  0.005],
    [ 0.000,  1.000,  0.000, -0.010],
    [ 0.000,  0.000,  1.000,  0.395],
    [ 0.000,  0.000,  0.000,  1.000]
  ],
  "position_xyz": [0.005, -0.010, 0.395],
  "quaternion_xyzw": [0.000, 0.000, 0.000, 1.000],
  "width_m": null,
  "approach_dir_xyz": [0.000, 0.000, 1.000],
  "source": {
    "candidate_index": 0,
    "segment_id": 0,
    "grasp_index": 0,
    "predictions_npz": null
  }
}
```

This grasp is **rejected by `check_width`** (conservative: `None` → reject) and does
not appear in `GraspFilterReport.passed`.

---

## Post-filter schema (grasps in `GraspFilterReport.passed`)

Grasps that survive all hard filters are shallow-copied and annotated with three new
float fields before being placed in `passed`:

```json
{
  "score": 0.923,
  "model_confidence": 0.923,
  "pose_4x4": ["...same as input..."],
  "position_xyz": [0.018, -0.042, 0.487],
  "quaternion_xyzw": [0.091, -0.236, 0.143, 0.957],
  "width_m": 0.062,
  "approach_dir_xyz": [-0.473, 0.206, 0.856],
  "source": {"...same as input..."},
  "clearance_score": 0.84,
  "collision_score": 1.00,
  "contact_quality_score": 0.71
}
```

These three scores are the inputs read by `grasp_scorer.rank_grasps()` during
`/select_and_execute`. Grasps that bypass `filter_grasps` (e.g. in tests) fall back
to `0.0` for all three (conservative/worst-case).

---

## §4 Known cross-pipeline inconsistency (`score` field)

**Tech debt:** The `score` field is not semantically comparable across the three pipelines:

| Pipeline | `score` meaning |
|---|---|
| A | `max(0, dot(approach, preferred_approach))` ∈ [0, 1] |
| B | CGN prediction confidence (same value as `model_confidence`) ∈ [0, 1] |
| C | Fixed `1.0` |

The field is only used for initial top-K candidate selection within each pipeline, not
for cross-pipeline comparison. However, if A/B/C outputs are ever merged and jointly
ranked by `score`, the result is biased (Option C always scores 1.0, Option A never
exceeds its approach alignment). **To be unified under a common pre-filter metric in a
future refactor.**

The `model_confidence` field was introduced specifically to preserve the Option B CGN
value under a clearly-scoped name, decoupled from the ambiguous `score` field.
