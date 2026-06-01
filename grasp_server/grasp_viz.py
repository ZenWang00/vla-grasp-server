"""Project top-K grasp poses onto the captured RGB image and save an annotated copy.

Draws a directed arrow for each grasp: tail at the gripper center, head pointing
along the approach direction (z-column of the rotation matrix, same convention as
the HTML report generator in scripts/generate_grasp_report.py).

Also provides :func:`save_grasp_viz_3d` which generates a self-contained interactive
Plotly HTML file showing the scene point cloud together with gripper wireframes and
XYZ axis arrows for each grasp.
"""
from __future__ import annotations

import colorsys
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("grasp_server")

# Arrow length in metres projected onto the image.
_APPROACH_METRES = 0.08


def _rank_color(rank_index: int, total: int) -> tuple[int, int, int]:
    hue = 0.62 - (0.55 * (rank_index / max(total, 1)))
    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, 0.85, 1.0)
    return int(255 * r), int(255 * g), int(255 * b)


def _project(xyz: list[float] | np.ndarray, K: np.ndarray) -> tuple[float, float] | None:
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    if z <= 0.0:
        return None
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return float(u), float(v)


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    color: tuple[int, int, int],
    line_width: int,
) -> None:
    draw.line([start_xy, end_xy], fill=color, width=line_width)
    dx = end_xy[0] - start_xy[0]
    dy = end_xy[1] - start_xy[1]
    norm = float(np.hypot(dx, dy))
    if norm < 1.0:
        return
    ux, uy = dx / norm, dy / norm
    left = (-uy, ux)
    arrow_len = min(18.0, max(10.0, norm * 0.22))
    head_w = arrow_len * 0.45
    p1 = end_xy
    p2 = (end_xy[0] - arrow_len * ux + head_w * left[0], end_xy[1] - arrow_len * uy + head_w * left[1])
    p3 = (end_xy[0] - arrow_len * ux - head_w * left[0], end_xy[1] - arrow_len * uy - head_w * left[1])
    draw.polygon([p1, p2, p3], fill=color)


def save_grasp_viz(
    capture_dir: Path,
    run_dir: Path,
    grasps_json: list[dict[str, Any]],
    *,
    out_filename: str = "grasp_viz.jpg",
) -> Path | None:
    """Draw projected grasp arrows on the captured image; save to *run_dir/out_filename*.

    Returns the output path, or None if anything goes wrong.
    """
    preview_path = capture_dir / "color_preview.jpg"
    npy_path = capture_dir / "camera_data.npy"

    if not preview_path.is_file():
        logger.warning("grasp_viz: color_preview.jpg not found at %s", preview_path)
        return None
    if not npy_path.is_file():
        logger.warning("grasp_viz: camera_data.npy not found at %s", npy_path)
        return None
    if not grasps_json:
        logger.warning("grasp_viz: no grasps to visualize")
        return None

    try:
        cam_data = np.load(npy_path, allow_pickle=True).item()
        K = np.asarray(cam_data["K"], dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"K has wrong shape {K.shape}")
    except Exception as exc:
        logger.warning("grasp_viz: failed to load K from %s: %s", npy_path, exc)
        return None

    try:
        img = Image.open(preview_path).convert("RGB")
    except Exception as exc:
        logger.warning("grasp_viz: failed to open preview image: %s", exc)
        return None

    draw = ImageDraw.Draw(img)
    W, H = img.size
    line_width = max(3, min(W, H) // 150)
    dot_radius = max(6, min(W, H) // 80)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            size=max(14, dot_radius),
        )
    except Exception:
        font = ImageFont.load_default()

    total = len(grasps_json)
    for rank, grasp in enumerate(grasps_json):
        xyz = grasp.get("position_xyz")
        approach = grasp.get("approach_dir_xyz")
        if xyz is None or len(xyz) != 3:
            continue

        start_px = _project(xyz, K)
        if start_px is None:
            continue

        color = _rank_color(rank, total)

        # Arrow tail → head along approach direction if available.
        if approach is not None and len(approach) == 3:
            end_xyz = [
                float(xyz[0]) + _APPROACH_METRES * float(approach[0]),
                float(xyz[1]) + _APPROACH_METRES * float(approach[1]),
                float(xyz[2]) + _APPROACH_METRES * float(approach[2]),
            ]
            end_px = _project(end_xyz, K)
            if end_px is not None:
                _draw_arrow(draw, start_px, end_px, color, line_width)

        # Filled circle at the grasp center.
        u, v = start_px
        r = dot_radius
        draw.ellipse(
            [u - r, v - r, u + r, v + r],
            fill=color,
            outline=(0, 0, 0),
            width=max(2, r // 5),
        )

        # Label: rank and score.
        score = grasp.get("score", 0.0)
        label = f"#{rank + 1}  {score:.3f}"
        tx, ty = u + r + 4, v - r
        draw.text((tx + 1, ty + 1), label, fill=(0, 0, 0), font=font)
        draw.text((tx, ty), label, fill=color, font=font)

    out_path = run_dir / out_filename
    try:
        img.save(out_path, format="JPEG", quality=92)
        logger.info("grasp_viz: saved annotated image to %s", out_path)
    except Exception as exc:
        logger.warning("grasp_viz: failed to save %s: %s", out_path, exc)
        return None

    return out_path


# ── 3-D Plotly visualization ──────────────────────────────────────────────────

def _sample_indices(count: int, max_points: int) -> np.ndarray:
    if count <= max_points:
        return np.arange(count, dtype=np.int64)
    step = max(1, count // max_points)
    return np.arange(0, count, step, dtype=np.int64)[:max_points]


def _gripper_wireframe(
    pose: np.ndarray,
    width_m: float | None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return gripper skeleton world-frame points and segment index pairs."""
    from vg_pipeline.grasp_results import FINGER_LENGTH_METERS, GRIPPER_DEPTH_METERS

    width = width_m if (width_m is not None and width_m > 0) else 0.08
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    local_pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, GRIPPER_DEPTH_METERS],
            [-width / 2.0, 0.0, GRIPPER_DEPTH_METERS],
            [-width / 2.0, 0.0, GRIPPER_DEPTH_METERS + FINGER_LENGTH_METERS],
            [width / 2.0, 0.0, GRIPPER_DEPTH_METERS],
            [width / 2.0, 0.0, GRIPPER_DEPTH_METERS + FINGER_LENGTH_METERS],
        ],
        dtype=np.float32,
    )
    world_pts = (local_pts @ rotation.T) + translation.reshape(1, 3)
    connections = [(0, 1), (2, 3), (4, 5), (2, 4)]
    return world_pts, connections


def save_grasp_viz_3d(
    capture_dir: Path,
    run_dir: Path,
    grasps_json: list[dict[str, Any]],
    *,
    out_filename: str = "grasp_viz_3d.html",
    max_pointcloud_points: int = 15000,
) -> Path | None:
    """Build an interactive Plotly 3-D scene + gripper visualization.

    Saves a self-contained HTML file to *run_dir/out_filename* and returns
    its path, or None on failure.

    Point cloud source priority:
    1. ``pc_full`` / ``pc_colors`` arrays inside any ``predictions_*.npz`` in *run_dir*
       (same data Contact-GraspNet used internally).
    2. Depth back-projection from ``camera_data.npy`` + ``color_preview.jpg``.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("grasp_viz_3d: plotly not installed — skipping 3-D visualization")
        return None

    npy_path = capture_dir / "camera_data.npy"
    if not npy_path.is_file():
        logger.warning("grasp_viz_3d: camera_data.npy not found at %s", npy_path)
        return None
    if not grasps_json:
        logger.warning("grasp_viz_3d: no grasps to visualize")
        return None

    try:
        cam_data = np.load(npy_path, allow_pickle=True).item()
        K = np.asarray(cam_data["K"], dtype=np.float64)
        depth = np.asarray(cam_data["depth"], dtype=np.float32)
    except Exception as exc:
        logger.warning("grasp_viz_3d: failed to load camera data: %s", exc)
        return None

    # ── Scene point cloud ─────────────────────────────────────────────────────
    scene_points: np.ndarray | None = None
    scene_colors: np.ndarray | None = None

    for pred_npz in sorted(run_dir.glob("predictions_*.npz")):
        try:
            with np.load(pred_npz, allow_pickle=True) as pf:
                if "pc_full" in pf.files:
                    scene_points = np.asarray(pf["pc_full"], dtype=np.float32)
                    if "pc_colors" in pf.files:
                        scene_colors = np.asarray(pf["pc_colors"], dtype=np.uint8)
                    break
        except Exception:
            continue

    if scene_points is None or scene_points.shape[0] == 0:
        try:
            from vg_pipeline.geometry import backproject_depth_with_mask
            scene_points = backproject_depth_with_mask(depth, K)
            preview_path = capture_dir / "color_preview.jpg"
            if preview_path.is_file():
                rgb = np.asarray(Image.open(preview_path).convert("RGB"))
                valid_mask = np.isfinite(depth) & (depth > 0.0)
                scene_colors = rgb[valid_mask]
        except Exception as exc:
            logger.warning("grasp_viz_3d: depth back-projection failed: %s", exc)

    if scene_points is None or scene_points.shape[0] == 0:
        logger.warning("grasp_viz_3d: no scene points available")
        return None

    idx = _sample_indices(scene_points.shape[0], max_pointcloud_points)
    pts = scene_points[idx]
    cols = (
        scene_colors[idx]
        if scene_colors is not None and scene_colors.shape[0] == scene_points.shape[0]
        else None
    )

    # ── Build Plotly figure ───────────────────────────────────────────────────
    figure = go.Figure()

    if cols is not None:
        marker_color: Any = [f"rgb({r},{g},{b})" for r, g, b in cols.tolist()]
        marker_extra: dict[str, Any] = {"color": marker_color}
    else:
        marker_extra = {"color": pts[:, 2], "colorscale": "Viridis"}

    figure.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        name="scene point cloud",
        marker={"size": 2, "opacity": 0.6, **marker_extra},
        hoverinfo="skip",
    ))

    total = len(grasps_json)
    axis_scale = 0.05

    for rank, grasp in enumerate(grasps_json):
        pose_raw = grasp.get("pose_4x4")
        if pose_raw is None:
            continue
        try:
            pose = np.asarray(pose_raw, dtype=np.float64)
            if pose.shape != (4, 4):
                continue
        except Exception:
            continue

        center = pose[:3, 3]
        rotation = pose[:3, :3]
        score = grasp.get("score", 0.0)
        width_m = grasp.get("width_m")

        color = _rank_color(rank, total)
        rgb_str = f"rgb({color[0]},{color[1]},{color[2]})"

        # Grasp center marker.
        figure.add_trace(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]],
            mode="markers+text",
            name=f"#{rank + 1}  score={score:.3f}",
            text=[str(rank + 1)],
            textposition="top center",
            marker={"size": 6, "color": rgb_str},
            hovertemplate=f"rank={rank + 1}<br>score={score:.4f}<extra></extra>",
        ))

        # XYZ axis arrows (red=base, green=lateral, blue=approach).
        for axis_name, col_idx, axis_color in [
            ("x/base", 0, "rgb(255,90,90)"),
            ("y/lateral", 1, "rgb(90,220,100)"),
            ("z/approach", 2, "rgb(80,150,255)"),
        ]:
            tip = center + axis_scale * rotation[:, col_idx]
            figure.add_trace(go.Scatter3d(
                x=[center[0], tip[0]], y=[center[1], tip[1]], z=[center[2], tip[2]],
                mode="lines",
                name=axis_name,
                line={"color": axis_color, "width": 5},
                hoverinfo="skip",
                showlegend=(rank == 0),
            ))

        # Gripper wireframe.
        g_pts, connections = _gripper_wireframe(pose, width_m)
        for s_idx, e_idx in connections:
            seg = g_pts[[s_idx, e_idx]]
            figure.add_trace(go.Scatter3d(
                x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
                mode="lines",
                line={"color": rgb_str, "width": 6},
                hoverinfo="skip",
                showlegend=False,
            ))

    figure.update_layout(
        title="3D Grasp Visualization — frame colors: red=x/base  green=y/lateral  blue=z/approach",
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        scene={
            "xaxis_title": "X (m)",
            "yaxis_title": "Y (m)",
            "zaxis_title": "Z (m)",
            "aspectmode": "data",
        },
        legend={"orientation": "h"},
    )

    out_path = run_dir / out_filename
    try:
        figure.write_html(str(out_path), include_plotlyjs=True, full_html=True)
        logger.info("grasp_viz_3d: saved to %s", out_path)
    except Exception as exc:
        logger.warning("grasp_viz_3d: failed to save %s: %s", out_path, exc)
        return None

    return out_path
