# Payload Sent to the ROS2 Client for IK Analysis

After `filter_grasps` completes, the server queues a payload on `pending_publish`.
The ROS2 client (`grasp_pose_client_node`) polls `GET /poll_publish` at 2 Hz, picks up
this payload, runs the IK feasibility check, and posts the passing grasps back via
`POST /submit_ik_result`.

---

## 1. Top-level envelope (`GET /poll_publish` response)

```json
{
  "run_id":        "20240603_152301_abc123",
  "run_dir":       "/tmp/grasp_runs/20240603_152301_abc123",
  "frame_id":      "camera_color_optical_frame",
  "elapsed_ms":    847,
  "num_candidates": 5,
  "top_k":          5,
  "grasps":        [ ...see §2... ],
  "grasp_viz":     "/tmp/grasp_runs/20240603_152301_abc123/viz.png",
  "trace_id":      "20240603_152301_xyz789",
  "status":        "pending",
  "mode":          "ik_check"
}
```

| Field | Type | Notes |
|---|---|---|
| `run_id` | `str` | Unique ID for the generation run. |
| `run_dir` | `str` | Server-side directory holding NPZs and viz files. |
| `frame_id` | `str` | Camera frame in which all grasp poses are expressed. Always `"camera_color_optical_frame"`. |
| `elapsed_ms` | `int` | Wall-clock time for the generation pipeline (ms). |
| `num_candidates` | `int` | Number of grasps in `grasps` (= number that survived `filter_grasps`). |
| `top_k` | `int` | Requested top-K from the original pipeline call. |
| `grasps` | `list[dict]` | Filter-annotated GraspDicts — see §2. |
| `grasp_viz` | `str \| null` | Path to a 2D overlay visualisation, if generated. |
| `trace_id` | `str` | Logical IK-round-trip ID. Returned to the server in `submit_ik_result` and in `/select_and_execute`. |
| `status` | `str` | Always `"pending"` at the time of polling (server has not yet received IK results). |
| `mode` | `str` | `"ik_check"` for this phase; `"execute"` for the final single-grasp execution payload. |

**Fields consumed by `GraspResult.from_json`** (ROS2 client dataclass):
`run_id`, `run_dir`, `frame_id`, `elapsed_ms`, `grasps`, `mode`, `trace_id`.
The remaining envelope fields (`num_candidates`, `top_k`, `grasp_viz`, `status`) are
ignored by the client but present for the Web UI.

---

## 2. Per-grasp schema inside `grasps[]`

These are the GraspDicts that survived `filter_grasps` (see
[grasp_dict_filter_input.md](grasp_dict_filter_input.md) for the input schema).
Each dict is a shallow copy of the input with three new score fields appended.

### Full field listing

| Field | Type | Camera frame? | ROS2 reads? | Notes |
|---|---|---|---|---|
| `score` | `float` | — | yes (execute phase) | Pipeline-specific proxy. See §4 of filter input doc. |
| `model_confidence` | `float \| None` | — | no | B-only CGN confidence. Passed through unchanged. |
| `pose_4x4` | `list[list[float]]` (4×4) | yes | **yes (IK + execute)** | Primary input to `_transform_to_base()`. |
| `position_xyz` | `list[float]` (3,) | yes | yes (fallback) | Used only when `pose_4x4` is absent (should never happen). |
| `quaternion_xyzw` | `list[float]` (4,) | yes | **yes (IK + execute)** | Used to build the Pinocchio SE3 target after base-frame transform. Format: `[qx, qy, qz, qw]`. |
| `width_m` | `float` | — | yes (execute phase) | Always a `float` at this point — `None` was rejected by `check_width`. |
| `approach_dir_xyz` | `list[float]` (3,) | yes | no (currently) | Not used by the ROS2 client. Will be used by `weight_approach` scorer when activated. |
| `source` | `dict` | — | no | Provenance metadata. Passed through for logging/debug. |
| **`clearance_score`** | `float` | — | no | Added by filter. Consumed by `grasp_scorer._score_clearance()` in `/select_and_execute`. |
| **`collision_score`** | `float` | — | no | Added by filter. Consumed by `grasp_scorer._score_collision()`. |
| **`contact_quality_score`** | `float` | — | no | Added by filter. Consumed by `grasp_scorer._score_contact()`. |

The three `*_score` fields are completely transparent to the ROS2 client: it reads the
grasp dicts, tests IK feasibility using `pose_4x4` / `quaternion_xyzw`, and returns
the passing grasp dicts **unchanged** (including all three score fields). The scorer
reads them only in the later `/select_and_execute` phase.

---

## 3. IK check flow inside the ROS2 client

```
GET /poll_publish
  → GraspResult.from_json(payload)
        .grasps = [GraspDict, ...]           # camera frame

_check_ik_feasibility(grasps):
  T = TF(ik_base_link ← camera_color_optical_frame)  # looked up from TF tree

  for each grasp:
    p_ik = _transform_to_base(grasp, T)
      # reads grasp["pose_4x4"] (preferred)
      # or fallback: grasp["position_xyz"] + grasp["quaternion_xyzw"]
      # → returns dict with base-frame position_xyz + quaternion_xyzw

    target = pin.SE3(
      Quaternion(p_ik["quaternion_xyzw"][3],   # w
                 p_ik["quaternion_xyzw"][0:3])  # x,y,z
      p_ik["position_xyz"]
    )
    q_solution = _pin_solve_ik(target, q0)
    if q_solution is not None:
      passed.append(grasp)  # ← original camera-frame dict, soft scores intact

POST /submit_ik_result  {run_id, trace_id, grasps: passed}
```

**Key point:** `passed` contains the original camera-frame GraspDicts (with soft scores
still attached). The base-frame conversion is done in-memory only; it is never written
back to the dict. The server receives camera-frame poses.

---

## 4. Example: single grasp as it appears in `grasps[]` at `/poll_publish`

This is a Contact-GraspNet (Option B) grasp after `filter_grasps` annotation:

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
  },
  "clearance_score": 0.84,
  "collision_score": 1.00,
  "contact_quality_score": 0.71
}
```

After the ROS2 client's TF transform (illustrative — actual values depend on robot state):

```json
{
  "position_xyz":    [-0.123,  0.341,  0.612],
  "quaternion_xyzw": [ 0.203, -0.114,  0.612,  0.754]
}
```

This base-frame pair is used to build the Pinocchio SE3 IK target. The original camera-frame
dict (with `clearance_score` etc.) is passed back to the server if IK succeeds.

---

## 5. `POST /submit_ik_result` — what the client sends back

```json
{
  "run_id":   "20240603_152301_abc123",
  "trace_id": "20240603_152301_xyz789",
  "grasps":   [ ...IK-passing camera-frame GraspDicts, unchanged... ]
}
```

The server stores these in `ik_results[trace_id]` with `status="complete"`.
`/select_and_execute` then calls `grasp_scorer.rank_grasps(ik_grasps)`, which reads
the pre-attached `clearance_score`, `collision_score`, and `contact_quality_score`
to compute composite geometric quality scores and pick the single best candidate.

---

## 6. `POST /select_and_execute` — request and response

**Request** (JSON body):
```json
{ "trace_id": "20240603_152301_xyz789" }
```

**Response** (on success, HTTP 200):
```json
{
  "status":         "ok",
  "trace_id":       "20240603_152301_xyz789",
  "num_ik_passing": 3,
  "num_selected":   1,
  "scored_grasps": [
    {
      "...all fields from §2...",
      "composite_score": 0.821
    }
  ],
  "ranked_indices": [2, 0, 1],
  "best_index":     2
}
```

| Field | Type | Notes |
|---|---|---|
| `scored_grasps` | `list[dict]` | IK-passing GraspDicts in original order (index matches `best_index`), each extended with `composite_score`. |
| `composite_score` | `float` | Weighted average of four active sub-scores: clearance×0.30 + collision×0.25 + width_margin×0.25 + contact×0.20, normalised so active weights sum to 1. |
| `ranked_indices` | `list[int]` | Indices into `scored_grasps`, sorted best-first. |
| `best_index` | `int` | Index of the winning grasp in `scored_grasps`. |

**Side effects:**
1. Queues `grasps=[best_grasp]` with `mode="execute"` on `pending_publish` — consumed by the next `GET /poll_publish` call from the ROS2 client.
2. Generates `grasp_viz_best.jpg` in the run directory (gold arrow overlay on the original capture image); served by `GET /grasp_viz_best_image`.
3. Pops the `ik_results[trace_id]` entry — calling `/select_and_execute` again with the same `trace_id` returns HTTP 404.

**Error cases:**
- HTTP 404 — `trace_id` unknown or already consumed.
- HTTP 422 — IK rejected every candidate (`grasps` list empty after IK check).

---

## 7. Schema alignment — execute payload (`GET /poll_publish`, `mode="execute"`)

After `/select_and_execute` the ROS2 client's next `GET /poll_publish` receives an
envelope **identical to §1** but with:

| Field | Value in execute phase |
|---|---|
| `mode` | `"execute"` |
| `grasps` | Exactly **1** GraspDict — the highest-scoring IK-passing candidate, camera frame. |
| `status` | `"complete"` (IK round-trip already finished). |
| `num_candidates` | `1` |

The ROS2 client's `_poll_publish` handler branches on `mode`:

```
if result.mode == "ik_check":
    _check_ik_feasibility(grasps)   # solve IK, POST /submit_ik_result, return
if result.mode == "execute":
    for each grasp:
        T = TF(ik_base_link ← camera_color_optical_frame)
        pose_base = _transform_to_base(grasp, T)
        publish → ~/best_grasp (PoseStamped), ~/grasps (PoseArray)
        broadcast TF frame for RViz
```

GraspDict fields consumed by the ROS2 client during the execute phase:

| Field | Used for |
|---|---|
| `pose_4x4` | Primary input to `_transform_to_base()` |
| `position_xyz` | Fallback if `pose_4x4` absent |
| `quaternion_xyzw` | Rotation for published PoseStamped |
| `width_m` | Published alongside pose (gripper command) |
| `frame_id` (envelope) | Header frame for PoseStamped / PoseArray |

The soft-score fields (`clearance_score`, `collision_score`, `contact_quality_score`,
`composite_score`) are carried through transparently and visible in ROS2 logs but are
not consumed for execution.
