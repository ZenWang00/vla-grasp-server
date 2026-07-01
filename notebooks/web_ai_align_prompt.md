# 6-DOF Grasp Pose Prompt for Web AI (RGB + Depth)

Copy the entire content below and send it to the web AI together with **two images**:
1. **`color_preview.jpg`** — the RGB scene image (1280×720)
2. **`depth_preview.jpg`** — the depth visualization (JET colormap, same 1280×720 resolution, aligned to RGB)

---

## System / Context

You are an expert in Embodied AI and robot vision. Your task is to output a **full 6-DOF grasp pose** (3D position + 3D orientation) in the camera coordinate frame. This is NOT a 2D bounding box task — you must reason about the 3D geometry of the scene and output coordinates in **meters and 3D vectors**.

You will receive two images of the same scene:
- **RGB image**: standard color image showing the scene appearance (1280×720).
- **Depth image (JET colormap)**: a false-color visualization of the depth map. Warmer colors (red/yellow) = closer to the camera; cooler colors (blue/purple) = farther away. This image is pixel-aligned with the RGB image — pixel (u, v) in both images corresponds to the same 3D point.

## Camera Intrinsics & Coordinate System

The images come from an **Intel RealSense D455** RGB-D camera. The depth is already aligned to the RGB frame. **All coordinates below are in the camera frame, NOT world frame.**

### Intrinsic Matrix K

The camera uses the **pinhole camera model**:

```
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

Typical values at 1280×720:
- **fx ≈ 640** px
- **fy ≈ 640** px
- **cx ≈ 640** px
- **cy ≈ 360** px

### Pinhole Projection (the math you need to use)

**2D pixel → 3D position (deprojection):**

```
Given: pixel (u, v) and depth Z (meters)
Compute:
  X = (u - cx) * Z / fx
  Y = (v - cy) * Z / fy
  Z = Z
```

**3D position → 2D pixel (projection):**

```
Given: 3D point (X, Y, Z) in camera frame
Compute:
  u = fx * (X / Z) + cx
  v = fy * (Y / Z) + cy
```

### Camera Coordinate Frame

```
       +Y (image down)
       |
       |
       +-------> +X (image right)
      /
     /
   +Z (into the scene, perpendicular to the image plane)
```

- **Origin**: camera optical center
- **+X**: points to the right in the image
- **+Y**: points downward in the image
- **+Z**: points forward, perpendicularly into the scene

The table/support surface is roughly parallel to the XZ plane. The camera is mounted above and looks slightly downward, so the table appears in the lower portion of the image, at negative Y values.

### How to Estimate Depth from the JET Visualization

The depth image uses the JET colormap. To estimate depth Z at a pixel:

1. Read the color at your target pixel in the depth image
2. Map it to an approximate depth:
   - **Dark red / maroon**: ~0.3–0.4 m (very close)
   - **Red / orange**: ~0.4–0.55 m
   - **Yellow**: ~0.55–0.7 m
   - **Green**: ~0.7–1.0 m
   - **Cyan**: ~1.0–1.3 m
   - **Blue**: ~1.3–1.8 m
   - **Dark blue / purple**: ~1.8–2.5+ m (far)
3. Interpolate between these bands — neighboring pixels give clues about the 3D surface orientation
4. Be conservative: if the depth color changes rapidly around your point (edges, occlusions), the depth is unreliable there — pick a different point

## Robot Gripper Specification

The robot uses a **parallel-jaw gripper**:

| Parameter | Value |
|---|---|
| Type | Parallel-jaw, two-finger |
| Max jaw opening | **0.08 m (80 mm)** |
| Min useful opening | **0.02 m (20 mm)** |
| Approach direction | **Fixed as camera +Z** = [0, 0, 1] (straight into the scene) |
| Closing direction rotation plane | **XY plane only** (image plane) |
| Gripper angle convention | **0° = jaws close along +X (image right); positive = rotate toward +Y (image down)** |

### Full 6-DOF Pose Construction

The 6-DOF grasp pose consists of:

1. **Position** (3 DOF): `[X, Y, Z]` in meters, camera frame — the 3D point where the gripper center aligns with the object.

2. **Orientation** (3 DOF), defined by two orthogonal axes:
   - **Approach axis**: fixed as `[0, 0, 1]` — the gripper always approaches perpendicular to the image (along camera +Z).
   - **Closing axis**: `[cos(θ), sin(θ), 0]` — the direction the two jaws close, rotating within the image plane. This is the ONLY orientation degree of freedom you control, parameterized by `gripper_angle_deg` = θ.
   - **Lateral axis**: automatically = approach × closing = `[-sin(θ), cos(θ), 0]`

In other words, the full 6-DOF pose is:

```
Position:       [X, Y, Z]           ← you compute from pixel + estimated depth
Closing dir:    [cos(θ), sin(θ), 0]  ← you choose θ = gripper_angle_deg
Approach dir:   [0, 0, 1]            ← fixed
```

### Scale Reference

At depth Z meters, one pixel ≈ **Z / fx** meters. Gripper opening converted to image pixels:

> pixels_needed = gripper_opening_mm / 1000 × fx / Z

Quick reference table:

| Depth color | Z (m) | 80 mm in px | 20 mm in px |
|---|---|---|---|
| Dark red | 0.35 | ~146 px | ~37 px |
| Red/Orange | 0.50 | ~102 px | ~26 px |
| Yellow | 0.65 | ~79 px | ~20 px |
| Green | 0.85 | ~60 px | ~15 px |
| Cyan | 1.15 | ~45 px | ~11 px |
| Blue | 1.50 | ~34 px | ~9 px |

The object's pixel width along the closing direction must be **between the 20 mm and 80 mm pixel values** for the object's depth.

## Your Task

Given the RGB and depth images, output the **full 6-DOF grasp pose** for `{TASK_SPEC}`. Propose `{NUM_CANDIDATES}` diverse candidates ranked from best to worst.

### Reasoning Steps

1. **Identify the target**: find `{TASK_SPEC}` in the RGB image. Cross-reference with the depth image to understand its full 3D shape — where it sits in 3D space, how thick it is, where it separates from the background.

2. **Estimate 3D position for each candidate**: for each grasp point you consider:
   - Locate the pixel (u, v) in the RGB image
   - Look up the same pixel in the depth image — what color is it?
   - Estimate Z (meters) from the depth color using the table above
   - Compute **X = (u − cx) × Z / fx** and **Y = (v − cy) × Z / fy**
   - This gives you the **3D position [X, Y, Z]** in camera frame — this is your grasp position

3. **Diversity planning**: mentally divide the target into distinct spatial zones (upper/middle/lower, left/right, near/far based on depth). Assign each candidate to a different zone. At least one candidate should have a gripper_angle_deg ≥ 30° different from the others.

4. **Choose gripper angle (θ)**: for each candidate, determine `gripper_angle_deg` such that:
   - The closing direction `[cos(θ), sin(θ), 0]` crosses the object's **narrow** dimension at that point
   - The object width along the closing direction fits in [20 mm, 80 mm]
   - Estimate the pixel width along the closing direction, then convert to meters using: width_m = width_px × Z / fx
   - Reject orientations where width_m > 0.08 m (won't fit) or width_m < 0.02 m (too thin, unstable)

5. **Table clearance**: the table is visible in the depth image as a large flat region, typically at the bottom. The grasp Z must be **less than** (in front of) the table depth at that image location. The grasp Y should be above the object-table contact line. Use the depth image to verify — if the depth at your point is the same as the table depth, you're on the table, not the object.

6. **Self-check**: verify:
   - No two candidates' 3D positions are within ~3 cm of each other (Euclidean distance in camera frame)
   - Every position has table clearance
   - Every closing-direction width fits [20 mm, 80 mm]
   - The pixel (u, v) maps to solid object surface in both RGB and depth images

### What to Avoid
- Handles, spouts, edges, tips, weak joints, high-curvature regions
- Transparent, reflective, or depth-less regions (if the depth image shows a dark hole or speckled noise, avoid it)
- The contact line between object and table (visible in depth as a sharp transition)
- Orientations where the closing span clearly exceeds 80 mm

## Output Format

Output ONLY a JSON object (no markdown fences, no extra text):

```json
{
  "target": "{TASK_SPEC}",
  "image_size": [720, 1280],
  "camera_intrinsics_used": {"fx": 640, "fy": 640, "cx": 640, "cy": 360},
  "candidates": [
    {
      "rank": 1,
      "position_3d": [X, Y, Z],
      "gripper_angle_deg": 45.0,
      "closing_direction_3d": [cx, cy, cz],
      "approach_direction_3d": [0.0, 0.0, 1.0],
      "pixel_uv": [u, v],
      "align_point_norm": [y_0_1000, x_0_1000],
      "estimated_depth_m": 0.55,
      "estimated_object_width_along_close_m": 0.045,
      "reasoning": "Zone: <name>. Depth color: <color> → Z≈<value>m. Position computed: X=(u-cx)*Z/fx=..., Y=(v-cy)*Z/fy=..., Z=... Why this angle, how this candidate differs from others, why width fits."
    }
  ]
}
```

### Field Descriptions

| Field | Type | Description |
|---|---|---|
| `position_3d` | `[X, Y, Z]` | **6-DOF position** in camera frame, meters. Computed via deprojection from pixel + estimated depth. |
| `gripper_angle_deg` | number | In-image closing rotation. 0° = horizontal (+X), 90° = vertical (+Y). |
| `closing_direction_3d` | `[cx, cy, cz]` | **6-DOF orientation**: unit vector of jaw closing direction. Must be `[cos(θ), sin(θ), 0]` with cz=0 (in image plane). |
| `approach_direction_3d` | `[0, 0, 1]` | **6-DOF orientation**: approach direction. Always camera +Z. |
| `pixel_uv` | `[u, v]` | The exact pixel coordinates (not normalized) where you placed the grasp. Used for server-side refinement with precise depth. |
| `align_point_norm` | `[y, x]` | Same point, normalized to 0–1000 range. 0 = top/left, 1000 = bottom/right. Fallback for the existing server pipeline. |
| `estimated_depth_m` | number | Your Z estimate in meters from reading the depth JET colormap. |
| `estimated_object_width_along_close_m` | number | Your estimate of the object's width along the closing direction, in meters. Must be in [0.02, 0.08]. |
| `reasoning` | string | Must include: zone name, depth color observed, Z estimate, deprojection calculation, how this differs from other candidates, and width feasibility confirmation. |

### The 6-DOF Pose in Summary

Each candidate fully defines a grasp in 3D space:

```
Frame: CAMERA (not world)

Position:    position_3d = [X, Y, Z]           ← meters, from pixel + estimated depth
Closing:     closing_direction_3d = [cos(θ), sin(θ), 0]
Lateral:     approach × closing = [-sin(θ), cos(θ), 0]   (computed automatically)
Approach:    approach_direction_3d = [0, 0, 1]  ← fixed, perpendicular to image

This is a complete right-handed 6-DOF grasp frame.
```

The robot will move the gripper so its tool-center-point is at `position_3d`, with the jaws closing along `closing_direction_3d` and approaching along `approach_direction_3d`. The server will refine `position_3d` by sampling the raw depth map (float32 meters) at `pixel_uv` with a 5×5 median window, giving millimeter-accurate Z — your estimated position provides the geometric reasoning, the server provides the precision.

### Important Notes

- **You are the one computing the 3D position.** Do NOT just output 2D pixel coordinates and let the server do the rest. Use the depth image to estimate Z, then apply the pinhole deprojection formula yourself. Show your work in the `reasoning` field.
- `closing_direction_3d` cz must be exactly 0 — the closing happens in the image plane only.
- `pixel_uv` must land on solid object surface — the server will validate by checking the raw depth map. If there's no valid depth there, the candidate is rejected.
- If you see the depth image has large black/noisy regions (missing depth, common with reflective or very dark surfaces), do NOT place grasp points there.
- The Z axis in camera frame points INTO the scene. So larger Z = farther from camera. The table should have larger Z than the object (since the object is in front of the table from the camera's perspective... actually, the table is below the object, and the camera looks downward, so the table may be at similar or larger Z depending on the setup. Use the depth image to verify.)
