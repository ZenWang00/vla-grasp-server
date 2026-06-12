# Coordinate Frames and Transformations

This document traces every coordinate-system choice and transformation in the pipeline,
from raw sensor data to the final robot command.

---

## 1. Coordinate Frames Overview

| Frame name | ROS2 frame_id | Origin | Axes convention | Who defines it |
|---|---|---|---|---|
| **Image (pixel)** | — | Top-left corner of image | u = right, v = down | Camera driver |
| **VLM canvas** | — | Top-left, normalized 0–1000 | same orientation | VLM prompt |
| **Camera optical** | `camera_color_optical_frame` | Color sensor optical center | +X right, +Y down, +Z into scene | RealSense driver |
| **Robot base** | `LIO_base_link` | Robot base (kinematic chain root + output frame) | robot-specific | URDF |

All grasp `pose_4x4` dictionaries live in the **camera optical frame** from the moment
they are created until they are handed to the robot controller.  The base-frame transform
is computed in memory during the IK check and again during execution; it is **never**
written back to the dict.

---

## 2. Grasp Pose Convention (camera optical frame)

Every `pose_4x4` (4×4 float, SE(3)) follows the same column convention, shared across
all three generation pipelines:

```
pose_4x4 = [col0 | col1 | col2 | col3]
             x      y      z      t
```

| Column | Axis name | Meaning |
|---|---|---|
| 0 | **closing / base** | Direction the gripper jaws slide when closing |
| 1 | **lateral** | Perpendicular to both closing and approach; completes the right-hand frame |
| 2 | **approach** | Direction the gripper travels to make contact with the object |
| 3 | **position** | 3D position of the gripper face centre (metres, camera frame) |

`position_xyz` in the dict is always `pose_4x4[:3, 3]`.
`approach_dir_xyz` is always `pose_4x4[:3, 2]`.
`quaternion_xyzw` is derived from `pose_4x4[:3, :3]` via Shepperd's method with canonical
sign (`qw ≥ 0`).

---

## 3. Stage-by-stage Transformation Log

### 3.1 Camera capture → pending_capture

`ROS2 GraspPoseClientNode` captures a time-synced triple from the RealSense driver:

| Data | ROS2 topic | Frame |
|---|---|---|
| Color image (PNG) | `/camera/camera/color/image_raw` | `camera_color_optical_frame` |
| Aligned depth (metres NPY) | `/camera/camera/aligned_depth_to_color/image_raw` | `camera_color_optical_frame` |
| Camera intrinsics K | `/camera/camera/color/camera_info` | — |

`frame_id` is taken directly from the color message header (default
`"camera_color_optical_frame"`).  No coordinate transformation happens at this stage.
The depth is aligned to the color sensor by the RealSense driver, so both images share
the same pixel grid and the same intrinsic matrix **K**.

---

### 3.2 VLM → 2D pixel (Option B / C only)

The VLM returns coordinates in a **normalized canvas space** (0–1000 in both axes,
y-first):

```json
{ "align_point": [y_norm, x_norm], "gripper_angle_deg": θ }
```

`parse_align_results_multi` (`vg_pipeline/align.py`) converts to full-resolution pixel
coordinates via `_scale_norm_xy_to_rgb`:

```
pixel_v = round(y_norm / 1000 * rgb_h)
pixel_u = round(x_norm / 1000 * rgb_w)   (adjusted for any letterbox padding)
```

The result is a row-column index `(v, u)` in the same coordinate system as the depth
image.

For **Option B** the VLM additionally produces bounding boxes that are scaled the same
way; SAM2 then segments within those boxes to produce a binary mask in pixel space.

---

### 3.3 Depth back-projection → camera-frame 3D point

Both Option B (via `geometry.backproject_depth_with_mask`) and Option C (via
`align.deproject_pixel` + `sample_depth_median`) use the standard pinhole model:

```
X = (u − cx) · Z / fx
Y = (v − cy) · Z / fy
Z = depth[v, u]   (metres)
```

with `(fx, fy, cx, cy)` from the intrinsic matrix **K** uploaded with the capture.
The output is a point `[X, Y, Z]` in the **camera optical frame**.

For Option C a `5×5` median window is taken around the alignment pixel to reduce
depth noise before back-projection (`sample_depth_median`).

---

### 3.4 Option C: 2D align point → 6-DoF pose

`build_align_grasp` (`grasp_server/align_grasp.py`) constructs the rotation from the
in-plane gripper angle returned by the VLM:

```
approach = [0, 0, 1]                        # fixed: camera +Z (optical axis)
closing  = [cos(θ), sin(θ), 0]             # in-image plane, angle from VLM
lateral  = cross(approach, closing)         # = [−sin(θ), cos(θ), 0]
```

where `θ = gripper_angle_deg` in radians.  The resulting pose has

- approach direction always equal to the camera optical axis (`+Z`),
- closing and lateral directions lying in the image plane.

**Limitation:** because the approach is fixed to camera `+Z`, all Option C grasps share
the same approach direction.  This is a deliberate simplification; the VLM only provides
a 2D point and an in-plane angle, not a full 3D orientation.

---

### 3.5 Option B: Contact-GraspNet → camera-frame 6-DoF poses

The SAM2 mask is back-projected to a point cloud (camera frame), then exported as an
NPZ file whose key `input_points` is an `(N, 3)` float32 array in the camera optical
frame.  The exported `depth` is the **full scene** with far background beyond
`cgn_depth_clip_m` (default 1.5 m) zeroed — the support plane (table) must be in the
point cloud so CGN suppresses bottom-up approach directions; the `segmap` focuses
`local_regions` / `filter_grasps` on the target.  Contact-GraspNet operates entirely
in the camera frame and writes predictions back in the same frame:

- `pred_grasps_cam[seg_id][grasp_idx]` — 4×4 float32, camera frame
- `scores[seg_id][grasp_idx]` — scalar confidence
- `contact_pts[seg_id][grasp_idx]` — contact point pair, camera frame

`normalize_predictions_multi` (`vg_pipeline/grasp_results.py`) deserialises these and
produces `GraspDict` entries with `pose_4x4` directly from CGN's output.  No additional
rotation is applied; the CGN convention already matches the column layout in §2.

Gripper width is derived from the contact points:

```
width_m = 2 × dot(center − contact_pt + GRIPPER_DEPTH × approach, base_dir)
```

where `center`, `contact_pt`, `approach`, and `base_dir` are all in the camera frame.

---

### 3.6 Server-side filter: scores computed in camera frame

`filter_grasps` (`grasp_server/grasp_filter.py`) computes three soft scores using the
camera-frame depth image and intrinsics **K** together with the camera-frame `pose_4x4`:

| Score | Geometry used |
|---|---|
| `clearance_score` | Project approach-path points via `K`; check depth clearance |
| `collision_score` | Project gripper body via `K`; count depth penetrations |
| `contact_quality_score` | Camera-frame surface normals vs approach direction |

All scores are computed in the camera frame.  No frame transformation occurs here.

In addition to the width hard filter, an **approach-direction hard filter** rejects
grasps whose approach axis exceeds `max_approach_angle_deg` (default 90°) from the
camera `+Z` axis — i.e. grasps that would reach the object from behind, toward the
camera.  The check is camera-frame only (no extrinsics needed); the value
`dot(approach, [0,0,1])` is stored on passing grasps as `approach_camera_cos`.
Option C grasps always pass (their approach is exactly `[0,0,1]`).

The collision hard-reject is **skipped** for Option C grasps (`source.predictions_npz
is None`) because their fixed `approach=[0,0,1]` produces systematic false positives in
the penetration check.

---

### 3.7 IK feasibility check: camera frame → IK base frame (in memory only)

`_check_ik_feasibility` (`grasp_pose_client_node.py`) runs on the ROS2 side:

```
T = TF(LIO_base_link ← camera_color_optical_frame)   # from ROS2 TF tree
p_base = _transform_to_base(grasp, T)
         R_base = (T @ pose_4x4_cam)[:3,:3] @ GRASP_TO_TCP_AXES   # axis remap, see below
         t_base = (T @ pose_4x4_cam)[:3,3]
```

`GRASP_TO_TCP_AXES` re-expresses the grasp axis convention (X=closing, Y=lateral,
Z=approach) in the LIO TCP convention of `lio_tcp_link` (X=approach, Y=closing,
Z=lateral), so the resulting quaternion is directly the IK target orientation.

`_transform_to_base` prefers `pose_4x4` (preferred path) and falls back to
`(position_xyz, quaternion_xyzw)` if the 4×4 is absent.  The result is a dict with
`position_xyz` and `quaternion_xyzw` in `LIO_base_link`.

This base-frame pair is used only to build the Pinocchio SE3 IK target:

```python
target_SE3 = pin.SE3(
    pin.Quaternion(qw, qx, qy, qz).toRotationMatrix(),
    pos_base
)
```

**The base-frame values are never written back to the GraspDict.**  The original
camera-frame dict (with all soft scores intact) is what the client posts to
`/submit_ik_result`.

---

### 3.8 Execution: camera frame → robot base frame + offset

When `mode == "execute"` the ROS2 client applies a second (independent) TF lookup:

```
T = TF(LIO_base_link ← camera_color_optical_frame)
R_base = (T @ pose_4x4_cam)[:3, :3] @ GRASP_TO_TCP_AXES   # grasp → LIO TCP axis remap
t_base = (T @ pose_4x4_cam)[:3, 3] + grasp_offset_base
```

`grasp_offset_base` is a constant 3-vector in `LIO_base_link` coordinates,
configurable via the ROS2 parameter `grasp_offset_base_xyz`.  It absorbs a systematic
hand-eye / extrinsic bias (e.g. a known mounting offset).  Default is `[0, 0, 0]`.

The final `PoseStamped` is published with `frame_id = "LIO_base_link"`.

---

## 4. Frame Transition Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  RealSense driver                                               │
│  color + depth → camera_color_optical_frame                     │
│  +X right, +Y down, +Z forward                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │  K (fx, fy, cx, cy) + depth (metres)
          ┌────────────────▼────────────────┐
          │  Depth back-projection           │
          │  X=(u−cx)Z/fx  Y=(v−cy)Z/fy     │
          │  → 3D point cloud (camera frame) │
          └──────────┬──────────────────────┘
                     │
          ┌──────────▼──────────────────────────────────────────┐
          │  Grasp generation (all options → camera frame)       │
          │                                                       │
          │  Option B (CGN):   pose_4x4 from CGN output          │
          │  Option C (Align): approach=+Z, closing=angle_deg    │
          │                                                       │
          │  GraspDict.pose_4x4  ← camera_color_optical_frame   │
          └──────────┬──────────────────────────────────────────┘
                     │  pose_4x4 (camera frame), depth, K
          ┌──────────▼──────────────────────────────────────────┐
          │  filter_grasps — scores in camera frame               │
          │  clearance_score, collision_score,                    │
          │  contact_quality_score                                │
          │  GraspDict unchanged (camera frame)                  │
          └──────────┬──────────────────────────────────────────┘
                     │
          ┌──────────▼──────────────────────────────────────────┐
          │  ROS2 IK check (in memory only)                       │
          │                                                       │
          │  TF: camera_color_optical_frame → LIO_base_link      │
          │  p_ik = T_ik_base_cam @ pose_4x4_cam                 │
          │  → Pinocchio SE3 target (LIO_base_link)               │
          │  → IK solve → pass/fail                               │
          │                                                       │
          │  Passed GraspDicts returned to server in camera frame │
          └──────────┬──────────────────────────────────────────┘
                     │  IK-passing GraspDicts (still camera frame)
          ┌──────────▼──────────────────────────────────────────┐
          │  rank_grasps / select_and_execute                     │
          │  composite_score from soft scores                     │
          │  → best_grasp (camera frame)                          │
          └──────────┬──────────────────────────────────────────┘
                     │
          ┌──────────▼──────────────────────────────────────────┐
          │  ROS2 execution                                       │
          │                                                       │
          │  TF: camera_color_optical_frame → LIO_base_link      │
          │  pose_base = T_base_cam @ pose_4x4_cam               │
          │  t_final   = pose_base[:3,3] + grasp_offset_base     │
          │                                                       │
          │  → PoseStamped  frame_id=LIO_base_link               │
          │  → PoseArray    frame_id=LIO_base_link               │
          │  → TF broadcast: grasp_best, grasp_best_cam          │
          └─────────────────────────────────────────────────────┘
```

---

## 5. TF Lookup Summary

Two separate TF lookups are performed by the ROS2 client; they use different target
frames on purpose:

| Phase | Source frame | Target frame | Parameter | Purpose |
|---|---|---|---|---|
| IK check | `camera_color_optical_frame` | `LIO_base_link` | `ik_base_link` | Kinematic chain root for Pinocchio URDF |
| Execution | `camera_color_optical_frame` | `LIO_base_link` | `robot_base_frame_id` | World anchor for PoseStamped output |

Both parameters are now unified to `LIO_base_link`, so the two lookups return the
same transform.  They are kept as separate parameters to allow configurations where
the IK chain root differs from the published output frame.  (`LIO_robot_base_link`,
266 mm above and Rz(90°) from `LIO_base_link`, still exists in the full-robot TF
tree but is no longer used as an output frame.)

---

## 6. Quaternion Sign Convention

Throughout the codebase quaternions are stored as `[qx, qy, qz, qw]`.

When building the Pinocchio SE3 target the order is reversed to `(qw, qx, qy, qz)`
as required by Pinocchio:

```python
pin.Quaternion(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])
```

Shepperd's method (used in both `grasp_selection._rotation_to_quaternion_xyzw` and
`_transform_to_base`) enforces `qw ≥ 0` (canonical sign) after normalisation:

```python
if q[3] < 0.0:
    q = -q
```

---

## 7. Known Limitations and Design Notes

| Item | Description |
|---|---|
| **Option C approach axis** | Fixed to camera `+Z`. Correct only when the gripper faces straight into the scene. A top-down camera mount aligned with the robot's workspace makes this reasonable; a tilted or side-mounted camera would need a different default. |
| **IK result not stored** | The base-frame transform computed during IK is discarded after the feasibility check. The execution phase recomputes it from a second TF lookup. A TF change between the two lookups (e.g. if the robot base is on a mobile platform) can introduce a small inconsistency. |
| **grasp_offset_base** | Applied only during execution, not during IK. If the offset is large it can cause an IK-passing grasp (checked without offset) to become unreachable after the offset is applied. Keep this value small (< 2 cm typical). |
| **Camera-frame scores** | All soft scores (`clearance_score`, `collision_score`, `contact_quality_score`) are evaluated in the camera frame using the depth image. They do not account for occlusions or objects behind the robot's reach boundary; that filtering is delegated entirely to the IK step. |
| **Approach hard filter is camera-frame** | The `max_approach_angle_deg` reject compares the approach axis against camera `+Z`, not gravity. It is valid for a camera mount facing the workspace; it does not by itself prefer top-down grasps in the robot base frame (the table in the CGN point cloud handles that). |
