from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def backproject_roi_points(
    cropped_depth: np.ndarray,
    K: np.ndarray,
    xmin: int,
    ymin: int,
) -> np.ndarray:
    """Back-project ROI depth to camera-frame points, returning float32 (N, 3)."""
    if cropped_depth.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    K = np.asarray(K, dtype=np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    h, w = cropped_depth.shape
    ys_local, xs_local = np.indices((h, w), dtype=np.float64)
    Z = np.asarray(cropped_depth, dtype=np.float64)

    valid = np.isfinite(Z) & (Z > 0.0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    u = xs_local + float(xmin)
    v = ys_local + float(ymin)

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    points = np.stack([X[valid], Y[valid], Z[valid]], axis=1).astype(np.float32, copy=False)
    return points


def render_pointcloud_3d_png(points: np.ndarray, path: Path, width: int = 960, height: int = 720) -> None:
    """Render point cloud PNG; prefers Open3D offscreen, falls back to Matplotlib if unavailable."""
    try:
        import open3d as o3d
    except Exception as exc:
        raise RuntimeError(
            "Open3D is required to generate pointcloud_3d.png. "
            "Install it in this environment first (e.g. `pip install open3d`)."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points shape (N,3), got {points.shape}")
    if points.shape[0] == 0:
        raise ValueError("Empty point cloud; cannot render pointcloud_3d.png")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))

    vis = o3d.visualization.Visualizer()
    created = vis.create_window(window_name="roi_pointcloud", width=width, height=height, visible=False)
    if created:
        try:
            vis.add_geometry(pcd)
            ctr = vis.get_view_control()
            ctr.set_zoom(0.6)
            vis.poll_events()
            vis.update_renderer()
            ok = vis.capture_screen_image(str(path), do_render=True)
            if ok:
                return
        finally:
            vis.destroy_window()

    # Fallback for headless servers where Open3D cannot create GLFW/OSMesa context.
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "Open3D could not render offscreen and Matplotlib fallback is unavailable. "
            "Install OSMesa/EGL support for Open3D or install Matplotlib."
        ) from exc

    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    xyz = points.astype(np.float64, copy=False)
    max_points = 30000
    if xyz.shape[0] > max_points:
        step = max(1, xyz.shape[0] // max_points)
        xyz = xyz[::step]
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.5, c=xyz[:, 2], cmap="viridis")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("ROI Point Cloud")
    fig.tight_layout()
    fig.savefig(str(path))
    plt.close(fig)


def project_points_to_rgb_overlay_png(
    points: np.ndarray,
    K: np.ndarray,
    cropped_rgb: np.ndarray,
    xmin: int,
    ymin: int,
    path: Path,
    max_points: int = 5000,
) -> None:
    """Project 3D points back to ROI RGB plane and render red dot overlay."""
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = np.asarray(cropped_rgb, dtype=np.uint8).copy()
    if points.shape[0] == 0:
        Image.fromarray(overlay, mode="RGB").save(str(path), format="PNG")
        return

    K = np.asarray(K, dtype=np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    pts = points.astype(np.float64, copy=False)
    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
    valid = np.isfinite(Z) & (Z > 0.0)
    if not np.any(valid):
        Image.fromarray(overlay, mode="RGB").save(str(path), format="PNG")
        return

    X, Y, Z = X[valid], Y[valid], Z[valid]
    u_global = (fx * X / Z) + cx
    v_global = (fy * Y / Z) + cy

    u_local = np.rint(u_global - float(xmin)).astype(np.int32)
    v_local = np.rint(v_global - float(ymin)).astype(np.int32)

    h, w = overlay.shape[:2]
    in_bounds = (u_local >= 0) & (u_local < w) & (v_local >= 0) & (v_local < h)
    u_local = u_local[in_bounds]
    v_local = v_local[in_bounds]

    if u_local.size > max_points:
        step = max(1, u_local.size // max_points)
        u_local = u_local[::step]
        v_local = v_local[::step]

    overlay[v_local, u_local] = np.array([255, 0, 0], dtype=np.uint8)
    Image.fromarray(overlay, mode="RGB").save(str(path), format="PNG")
