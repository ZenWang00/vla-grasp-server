#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import colorsys
import html
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vg_pipeline.geometry import backproject_roi_points


GRIPPER_DEPTH_METERS = 0.1034
FINGER_LENGTH_METERS = 0.06


@dataclass(frozen=True)
class NormalizedGrasp:
    segment_id: int
    grasp_index: int
    score: float
    transform: np.ndarray
    center_xyz: np.ndarray
    contact_point_xyz: np.ndarray | None
    width_m: float | None

    @property
    def rotation(self) -> np.ndarray:
        return self.transform[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        return self.transform[:3, 3]

    @property
    def base_dir(self) -> np.ndarray:
        return self.rotation[:, 0]

    @property
    def lateral_dir(self) -> np.ndarray:
        return self.rotation[:, 1]

    @property
    def approach_dir(self) -> np.ndarray:
        return self.rotation[:, 2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an interactive HTML grasp report.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run output directory containing manifest.json.")
    parser.add_argument(
        "--predictions-npz",
        type=Path,
        default=None,
        help="Contact-GraspNet predictions NPZ. Defaults to a single predictions*.npz inside --run-dir.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to <run-dir>/report.html.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="How many top-scoring grasps to visualize.")
    parser.add_argument(
        "--max-pointcloud-points",
        type=int,
        default=15000,
        help="Maximum number of 3D scene points kept in the Plotly view.",
    )
    return parser.parse_args()


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found under {run_dir}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _resolve_existing_path(run_dir: Path, path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    candidates = [path]
    if not path.is_absolute():
        candidates.insert(0, run_dir / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _guess_predictions_path(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("predictions*.npz"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            "No predictions NPZ found. Pass --predictions-npz explicitly or place one predictions*.npz under the run dir."
        )
    raise ValueError(
        f"Multiple prediction NPZ files found under {run_dir}: {[m.name for m in matches]}. "
        "Pass --predictions-npz explicitly."
    )


def _summarize_npz(npz: np.lib.npyio.NpzFile) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for key in npz.files:
        value = npz[key]
        item: dict[str, Any] = {
            "key": key,
            "dtype": str(getattr(value, "dtype", type(value).__name__)),
            "shape": list(getattr(value, "shape", ())),
        }
        if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
            obj = value.item()
            if isinstance(obj, dict):
                item["object_dict_keys"] = [int(k) for k in obj.keys()]
                item["object_dict_shapes"] = {
                    str(int(k)): list(np.asarray(v).shape) for k, v in obj.items()
                }
        summary.append(item)
    return summary


def _require_object_dict(npz: np.lib.npyio.NpzFile, key: str) -> dict[int, np.ndarray]:
    if key not in npz.files:
        raise KeyError(f"Prediction NPZ missing required key {key!r}")
    value = npz[key]
    if not (isinstance(value, np.ndarray) and value.dtype == object and value.shape == ()):
        raise ValueError(f"Expected {key!r} to be a scalar object array containing a dict, got shape={value.shape}")
    payload = value.item()
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {key!r} to hold a dict, got {type(payload).__name__}")
    normalized: dict[int, np.ndarray] = {}
    for raw_key, raw_value in payload.items():
        normalized[int(raw_key)] = np.asarray(raw_value)
    return normalized


def _derive_width(transform: np.ndarray, contact_point: np.ndarray | None) -> float | None:
    if contact_point is None:
        return None
    base_dir = transform[:3, 0]
    approach_dir = transform[:3, 2]
    center = transform[:3, 3]
    width = 2.0 * float(np.dot(center - contact_point + (GRIPPER_DEPTH_METERS * approach_dir), base_dir))
    if not np.isfinite(width):
        return None
    return abs(width)


def _normalize_predictions(npz_path: Path) -> tuple[list[NormalizedGrasp], list[dict[str, Any]]]:
    with np.load(npz_path, allow_pickle=True) as predictions_npz:
        schema_summary = _summarize_npz(predictions_npz)
        pred_grasps = _require_object_dict(predictions_npz, "pred_grasps_cam")
        scores = _require_object_dict(predictions_npz, "scores")
        contact_pts = _require_object_dict(predictions_npz, "contact_pts") if "contact_pts" in predictions_npz.files else {}

    grasps: list[NormalizedGrasp] = []
    for segment_id, transforms in pred_grasps.items():
        transforms = np.asarray(transforms, dtype=np.float32)
        segment_scores = np.asarray(scores.get(segment_id), dtype=np.float32)
        if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
            raise ValueError(
                f"Unexpected pred_grasps_cam[{segment_id}] shape {transforms.shape}; expected (N, 4, 4)."
            )
        if segment_scores.shape != (transforms.shape[0],):
            raise ValueError(
                f"Unexpected scores[{segment_id}] shape {segment_scores.shape}; expected ({transforms.shape[0]},)."
            )
        segment_contacts = None
        if segment_id in contact_pts:
            segment_contacts = np.asarray(contact_pts[segment_id], dtype=np.float32)
            if segment_contacts.shape != (transforms.shape[0], 3):
                raise ValueError(
                    f"Unexpected contact_pts[{segment_id}] shape {segment_contacts.shape}; "
                    f"expected ({transforms.shape[0]}, 3)."
                )

        for grasp_index in range(transforms.shape[0]):
            transform = transforms[grasp_index]
            contact_point = segment_contacts[grasp_index] if segment_contacts is not None else None
            grasps.append(
                NormalizedGrasp(
                    segment_id=segment_id,
                    grasp_index=grasp_index,
                    score=float(segment_scores[grasp_index]),
                    transform=transform,
                    center_xyz=transform[:3, 3].copy(),
                    contact_point_xyz=None if contact_point is None else contact_point.copy(),
                    width_m=_derive_width(transform, contact_point),
                )
            )

    if not grasps:
        raise ValueError(f"No grasps found in {npz_path}")

    grasps.sort(key=lambda item: item.score, reverse=True)
    return grasps, schema_summary


def _load_contact_input_npz(run_dir: Path, manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    export_info = manifest.get("contact_graspnet_export") or {}
    export_path = _resolve_existing_path(run_dir, export_info.get("path"))
    if export_path is None:
        raise FileNotFoundError(
            "Could not locate Contact-GraspNet input NPZ from manifest['contact_graspnet_export']['path']."
        )
    with np.load(export_path) as data:
        required_keys = {"depth", "K", "rgb"}
        missing = required_keys.difference(data.files)
        if missing:
            raise KeyError(f"Input NPZ {export_path} is missing keys: {sorted(missing)}")
        return {key: np.asarray(data[key]) for key in data.files}


def _load_rgb_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _load_scene_rgb(run_dir: Path, manifest: dict[str, Any], contact_input: dict[str, np.ndarray]) -> np.ndarray:
    scene_path = _resolve_existing_path(run_dir, manifest.get("scene_image_path"))
    if scene_path is not None:
        return _load_rgb_image(scene_path)
    rgb_bgr = np.asarray(contact_input["rgb"], dtype=np.uint8)
    if rgb_bgr.ndim != 3 or rgb_bgr.shape[2] != 3:
        raise ValueError(f"Expected input NPZ rgb to have shape (H, W, 3), got {rgb_bgr.shape}")
    return rgb_bgr[:, :, ::-1].copy()


def _load_optional_image(run_dir: Path, relative_or_absolute: str | None) -> np.ndarray | None:
    image_path = _resolve_existing_path(run_dir, relative_or_absolute)
    if image_path is None:
        return None
    try:
        return _load_rgb_image(image_path)
    except Exception:
        return None


def _png_data_uri(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _sample_indices(count: int, max_points: int) -> np.ndarray:
    if count <= max_points:
        return np.arange(count, dtype=np.int64)
    step = max(1, count // max_points)
    return np.arange(0, count, step, dtype=np.int64)[:max_points]


def _build_scene_point_cloud(contact_input: dict[str, np.ndarray], predictions_npz_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(predictions_npz_path, allow_pickle=True) as predictions_npz:
        if "pc_full" in predictions_npz.files:
            points = np.asarray(predictions_npz["pc_full"], dtype=np.float32)
            colors = None
            if "pc_colors" in predictions_npz.files:
                colors = np.asarray(predictions_npz["pc_colors"], dtype=np.uint8)
            return points, colors

    depth = np.asarray(contact_input["depth"], dtype=np.float32)
    K = np.asarray(contact_input["K"], dtype=np.float64)
    rgb = np.asarray(contact_input["rgb"], dtype=np.uint8)[:, :, ::-1].copy()
    points = backproject_roi_points(depth, K, xmin=0, ymin=0)
    valid = np.isfinite(depth) & (depth > 0.0)
    colors = rgb[valid]
    return points, colors


def _project_camera_points(points_xyz: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_xyz, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]
    valid = np.isfinite(Z) & (Z > 0.0)
    uv = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(valid):
        uv[valid, 0] = (K[0, 0] * X[valid] / Z[valid]) + K[0, 2]
        uv[valid, 1] = (K[1, 1] * Y[valid] / Z[valid]) + K[1, 2]
    return uv, valid


def _rank_color(rank_index: int, total: int) -> tuple[int, int, int]:
    hue = 0.62 - (0.55 * (rank_index / max(total, 1)))
    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, 0.85, 1.0)
    return int(255 * r), int(255 * g), int(255 * b)


def _draw_arrow(draw: ImageDraw.ImageDraw, start_xy: tuple[float, float], end_xy: tuple[float, float], color: tuple[int, int, int]) -> None:
    draw.line([start_xy, end_xy], fill=color, width=5)
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
    p2 = (end_xy[0] - (arrow_len * ux) + (head_w * left[0]), end_xy[1] - (arrow_len * uy) + (head_w * left[1]))
    p3 = (end_xy[0] - (arrow_len * ux) - (head_w * left[0]), end_xy[1] - (arrow_len * uy) - (head_w * left[1]))
    draw.polygon([p1, p2, p3], fill=color)


def _draw_box(draw: ImageDraw.ImageDraw, box: dict[str, int], color: tuple[int, int, int], width: int = 4) -> None:
    draw.rectangle([box["xmin"], box["ymin"], box["xmax"], box["ymax"]], outline=color, width=width)


def _make_overlay_image(
    scene_rgb: np.ndarray,
    K: np.ndarray,
    grasps: list[NormalizedGrasp],
    manifest: dict[str, Any],
    top_k: int,
) -> np.ndarray:
    image = Image.fromarray(scene_rgb, mode="RGB")
    draw = ImageDraw.Draw(image)

    object_box = (manifest.get("object_detection") or {}).get("parsed_box")
    if isinstance(object_box, dict):
        _draw_box(draw, object_box, color=(0, 220, 90), width=4)

    for roi in manifest.get("parsed_boxes", []):
        if isinstance(roi, dict):
            _draw_box(draw, roi, color=(255, 200, 0), width=2)

    selected = grasps[:top_k]
    if not selected:
        return np.asarray(image)

    start_points = np.stack([grasp.center_xyz for grasp in selected], axis=0)
    end_points = np.stack([grasp.center_xyz + (0.08 * grasp.approach_dir) for grasp in selected], axis=0)
    start_uv, start_valid = _project_camera_points(start_points, K)
    end_uv, end_valid = _project_camera_points(end_points, K)
    contact_uv = None
    contact_valid = None
    if any(grasp.contact_point_xyz is not None for grasp in selected):
        contact_points = np.stack(
            [grasp.contact_point_xyz if grasp.contact_point_xyz is not None else grasp.center_xyz for grasp in selected],
            axis=0,
        )
        contact_uv, contact_valid = _project_camera_points(contact_points, K)

    for rank_index, grasp in enumerate(selected):
        if not (start_valid[rank_index] and end_valid[rank_index]):
            continue
        color = _rank_color(rank_index, len(selected))
        start_xy = (float(start_uv[rank_index, 0]), float(start_uv[rank_index, 1]))
        end_xy = (float(end_uv[rank_index, 0]), float(end_uv[rank_index, 1]))
        _draw_arrow(draw, start_xy, end_xy, color)
        radius = 7
        draw.ellipse(
            [start_xy[0] - radius, start_xy[1] - radius, start_xy[0] + radius, start_xy[1] + radius],
            outline=(255, 255, 255),
            fill=color,
            width=2,
        )
        if contact_uv is not None and contact_valid is not None and contact_valid[rank_index]:
            cx = float(contact_uv[rank_index, 0])
            cy = float(contact_uv[rank_index, 1])
            draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 255, 255), outline=color, width=1)
        label = f"#{rank_index + 1} {grasp.score:.3f}"
        draw.text((start_xy[0] + 10, start_xy[1] - 18), label, fill=(255, 255, 255))

    return np.asarray(image)


def _colors_for_plotly(colors: np.ndarray | None, point_count: int) -> list[str] | np.ndarray:
    if colors is None or colors.shape[0] != point_count:
        return np.zeros(point_count, dtype=np.float32)
    clipped = np.asarray(colors, dtype=np.uint8)
    return [f"rgb({r},{g},{b})" for r, g, b in clipped.tolist()]


def _gripper_wireframe_points(grasp: NormalizedGrasp) -> tuple[np.ndarray, list[tuple[int, int]]]:
    width = grasp.width_m if grasp.width_m is not None else 0.08
    local_points = np.array(
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
    world_points = (local_points @ grasp.rotation.T) + grasp.translation.reshape(1, 3)
    connections = [(0, 1), (2, 3), (4, 5), (2, 4)]
    return world_points, connections


def _build_plotly_figure(
    scene_points: np.ndarray,
    scene_colors: np.ndarray | None,
    grasps: list[NormalizedGrasp],
    max_pointcloud_points: int,
) -> str:
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        raise RuntimeError(
            "Plotly is required to generate report.html. Install it first, e.g. `python -m pip install plotly`."
        ) from exc

    scene_sample = _sample_indices(scene_points.shape[0], max_pointcloud_points)
    sampled_points = scene_points[scene_sample]
    sampled_colors = scene_colors[scene_sample] if scene_colors is not None and scene_colors.shape[0] == scene_points.shape[0] else None

    figure = go.Figure()
    marker_color = _colors_for_plotly(sampled_colors, sampled_points.shape[0])
    marker_kwargs: dict[str, Any] = {"size": 2, "opacity": 0.7}
    if isinstance(marker_color, list):
        marker_kwargs["color"] = marker_color
    else:
        marker_kwargs["color"] = sampled_points[:, 2]
        marker_kwargs["colorscale"] = "Viridis"
    figure.add_trace(
        go.Scatter3d(
            x=sampled_points[:, 0],
            y=sampled_points[:, 1],
            z=sampled_points[:, 2],
            mode="markers",
            name="scene point cloud",
            marker=marker_kwargs,
            hoverinfo="skip",
        )
    )

    axis_scale = 0.05
    for rank_index, grasp in enumerate(grasps):
        color = _rank_color(rank_index, len(grasps))
        rgb = f"rgb({color[0]},{color[1]},{color[2]})"
        center = grasp.center_xyz
        figure.add_trace(
            go.Scatter3d(
                x=[center[0]],
                y=[center[1]],
                z=[center[2]],
                mode="markers+text",
                name=f"grasp #{rank_index + 1}",
                text=[str(rank_index + 1)],
                textposition="top center",
                marker={"size": 5, "color": rgb},
                hovertemplate=(
                    f"rank={rank_index + 1}<br>"
                    f"segment={grasp.segment_id}<br>"
                    f"score={grasp.score:.4f}<br>"
                    f"width={grasp.width_m:.4f} m<extra></extra>"
                    if grasp.width_m is not None
                    else f"rank={rank_index + 1}<br>segment={grasp.segment_id}<br>score={grasp.score:.4f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

        for axis_name, axis_dir, axis_color in [
            ("x/base", grasp.base_dir, "rgb(255,90,90)"),
            ("y/lateral", grasp.lateral_dir, "rgb(90,255,140)"),
            ("z/approach", grasp.approach_dir, "rgb(80,150,255)"),
        ]:
            axis_tip = center + (axis_scale * axis_dir)
            figure.add_trace(
                go.Scatter3d(
                    x=[center[0], axis_tip[0]],
                    y=[center[1], axis_tip[1]],
                    z=[center[2], axis_tip[2]],
                    mode="lines",
                    name=f"{axis_name} #{rank_index + 1}",
                    line={"color": axis_color, "width": 5},
                    hoverinfo="skip",
                    showlegend=(rank_index == 0),
                )
            )

        gripper_points, connections = _gripper_wireframe_points(grasp)
        for start_idx, end_idx in connections:
            segment = gripper_points[[start_idx, end_idx]]
            figure.add_trace(
                go.Scatter3d(
                    x=segment[:, 0],
                    y=segment[:, 1],
                    z=segment[:, 2],
                    mode="lines",
                    line={"color": rgb, "width": 6},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    figure.update_layout(
        title="3D Grasp Visualization",
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        scene={
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Z",
            "aspectmode": "data",
        },
        legend={"orientation": "h"},
    )
    return figure.to_html(full_html=False, include_plotlyjs=True)


def _artifact_cards(run_dir: Path, manifest: dict[str, Any], scene_rgb: np.ndarray, overlay_rgb: np.ndarray) -> list[tuple[str, str]]:
    cards: list[tuple[str, str]] = [("Scene RGB", _png_data_uri(scene_rgb)), ("Scene + projected grasps", _png_data_uri(overlay_rgb))]
    object_detection = manifest.get("object_detection") or {}
    sam2_segmentation = manifest.get("sam2_segmentation") or {}
    optional_specs = [
        ("Object box overlay", (object_detection.get("paths") or {}).get("object_box_overlay")),
        ("Object mask overlay", (sam2_segmentation.get("paths") or {}).get("object_mask_overlay")),
    ]
    for title, candidate_path in optional_specs:
        image = _load_optional_image(run_dir, candidate_path)
        if image is not None:
            cards.append((title, _png_data_uri(image)))
    return cards


def _build_summary_rows(
    run_dir: Path,
    predictions_npz: Path,
    manifest: dict[str, Any],
    schema_summary: list[dict[str, Any]],
    grasps: list[NormalizedGrasp],
) -> str:
    top_score = grasps[0].score if grasps else float("nan")
    rows = [
        ("run_dir", str(run_dir)),
        ("predictions_npz", str(predictions_npz)),
        ("run_id", str(manifest.get("run_id"))),
        ("task_spec", str(manifest.get("task_spec"))),
        ("provider", str(manifest.get("provider"))),
        ("model", str(manifest.get("model_path"))),
        ("total_grasps", str(len(grasps))),
        ("best_score", f"{top_score:.4f}" if np.isfinite(top_score) else "n/a"),
    ]
    rows_html = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(value)}</td></tr>"
        for key, value in rows
    )
    schema_html = html.escape(json.dumps(schema_summary, indent=2))
    return f"""
    <div class="card">
      <h2>Run Summary</h2>
      <table class="summary-table">{rows_html}</table>
    </div>
    <div class="card">
      <h2>Prediction Schema</h2>
      <pre>{schema_html}</pre>
    </div>
    """


def _build_grasp_table(grasps: list[NormalizedGrasp]) -> str:
    rows = []
    for rank_index, grasp in enumerate(grasps, start=1):
        width_text = f"{grasp.width_m:.4f}" if grasp.width_m is not None else "n/a"
        rows.append(
            "<tr>"
            f"<td>{rank_index}</td>"
            f"<td>{grasp.segment_id}</td>"
            f"<td>{grasp.grasp_index}</td>"
            f"<td>{grasp.score:.4f}</td>"
            f"<td>{width_text}</td>"
            f"<td>[{grasp.center_xyz[0]:.3f}, {grasp.center_xyz[1]:.3f}, {grasp.center_xyz[2]:.3f}]</td>"
            "</tr>"
        )
    body = "".join(rows)
    return f"""
    <div class="card">
      <h2>Top Grasps</h2>
      <table class="grasp-table">
        <thead>
          <tr>
            <th>rank</th>
            <th>segment</th>
            <th>index</th>
            <th>score</th>
            <th>width_m</th>
            <th>center_xyz</th>
          </tr>
        </thead>
        <tbody>{body}</tbody>
      </table>
    </div>
    """


def _build_html_document(
    run_dir: Path,
    predictions_npz: Path,
    manifest: dict[str, Any],
    schema_summary: list[dict[str, Any]],
    cards: list[tuple[str, str]],
    plotly_html: str,
    grasps: list[NormalizedGrasp],
) -> str:
    cards_html = "".join(
        f"""
        <div class="card">
          <h2>{html.escape(title)}</h2>
          <img src="{data_uri}" alt="{html.escape(title)}" />
        </div>
        """
        for title, data_uri in cards
    )
    summary_html = _build_summary_rows(run_dir, predictions_npz, manifest, schema_summary, grasps)
    grasp_table_html = _build_grasp_table(grasps)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Grasp Report - {html.escape(str(manifest.get("run_id", "unknown")))}</title>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      font-family: Arial, sans-serif;
      background: #101218;
      color: #f0f3f8;
    }}
    h1, h2 {{
      margin-top: 0;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 18px;
      margin-bottom: 18px;
    }}
    .card {{
      background: #171b24;
      border: 1px solid #293140;
      border-radius: 12px;
      padding: 16px;
      box-sizing: border-box;
    }}
    img {{
      width: 100%;
      height: auto;
      border-radius: 8px;
      background: #0d1117;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      text-align: left;
      padding: 8px 10px;
      border-bottom: 1px solid #293140;
      vertical-align: top;
    }}
    th {{
      color: #9fb2cc;
      font-weight: 600;
    }}
    pre {{
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      color: #d7e1f0;
    }}
    .plotly-card {{
      margin-bottom: 18px;
    }}
    .hint {{
      color: #9fb2cc;
    }}
  </style>
</head>
<body>
  <h1>Standalone HTML Grasp Report</h1>
  <p class="hint">
    3D frame colors: red = grasp x/base, green = grasp y/lateral, blue = grasp z/approach.
    The projected 2D arrows use the same blue approach direction.
  </p>

  <div class="grid">
    {summary_html}
  </div>

  <div class="card plotly-card">
    <h2>3D View</h2>
    {plotly_html}
  </div>

  <div class="grid">
    {cards_html}
  </div>

  {grasp_table_html}
</body>
</html>
"""


def main() -> None:
    args = _parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if args.max_pointcloud_points < 100:
        raise ValueError("--max-pointcloud-points must be >= 100")

    run_dir = args.run_dir.resolve()
    predictions_npz = (args.predictions_npz.resolve() if args.predictions_npz is not None else _guess_predictions_path(run_dir))
    output_html = args.output_html.resolve() if args.output_html is not None else run_dir / "report.html"

    manifest = _load_manifest(run_dir)
    grasps, schema_summary = _normalize_predictions(predictions_npz)
    selected_grasps = grasps[: args.top_k]
    contact_input = _load_contact_input_npz(run_dir, manifest)
    scene_rgb = _load_scene_rgb(run_dir, manifest, contact_input)
    scene_points, scene_colors = _build_scene_point_cloud(contact_input, predictions_npz)
    overlay_rgb = _make_overlay_image(
        scene_rgb=scene_rgb,
        K=np.asarray(contact_input["K"], dtype=np.float64),
        grasps=selected_grasps,
        manifest=manifest,
        top_k=args.top_k,
    )
    cards = _artifact_cards(run_dir, manifest, scene_rgb, overlay_rgb)
    plotly_html = _build_plotly_figure(
        scene_points=scene_points,
        scene_colors=scene_colors,
        grasps=selected_grasps,
        max_pointcloud_points=args.max_pointcloud_points,
    )
    document = _build_html_document(
        run_dir=run_dir,
        predictions_npz=predictions_npz,
        manifest=manifest,
        schema_summary=schema_summary,
        cards=cards,
        plotly_html=plotly_html,
        grasps=selected_grasps,
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(document, encoding="utf-8")

    print("Prediction schema:")
    print(json.dumps(schema_summary, indent=2))
    print(f"Normalized grasps: {len(grasps)} total, visualizing top {len(selected_grasps)}")
    print(f"Wrote HTML report to: {output_html}")


if __name__ == "__main__":
    main()
