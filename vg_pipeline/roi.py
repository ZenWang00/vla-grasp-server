from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import rgb_to_pil


@dataclass(frozen=True)
class ParsedVlmResult:
    object_box: tuple[int, int, int, int] | None
    object_point: tuple[int, int] | None
    grasp_boxes: list[tuple[int, int, int, int]]


def _normalize_box(ymin: int, xmin: int, ymax: int, xmax: int) -> tuple[int, int, int, int]:
    nymin, nymax = (ymin, ymax) if ymin <= ymax else (ymax, ymin)
    nxmin, nxmax = (xmin, xmax) if xmin <= xmax else (xmax, xmin)
    return nymin, nxmin, nymax, nxmax


def _scale_norm_xy_to_rgb(
    y_val: float,
    x_val: float,
    *,
    canvas_h: int,
    canvas_w: int,
    rgb_h: int,
) -> tuple[int, int]:
    pixel_y = (float(y_val) / 1000.0) * canvas_h
    pixel_x = (float(x_val) / 1000.0) * canvas_w
    if pixel_y > rgb_h:
        pixel_y -= rgb_h
    return int(round(pixel_y)), int(round(pixel_x))


def _parse_candidate_box(candidate: dict[str, object], *, candidate_name: str) -> list[float]:
    box_key = "grasp_region_box" if "grasp_region_box" in candidate else "box_2d"
    if box_key not in candidate:
        raise ValueError(f"No 'grasp_region_box' or legacy 'box_2d' found in {candidate_name}:\n" + str(candidate))
    vlm_box = candidate[box_key]
    if not isinstance(vlm_box, list) or len(vlm_box) != 4:
        raise ValueError(f"'{box_key}' in {candidate_name} must have 4 elements [ymin, xmin, ymax, xmax]")
    return vlm_box


def _parse_point(value: object, *, field_name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"'{field_name}' must have 2 elements [y, x]")
    return value


def _load_vlm_json(raw_text: str) -> dict[str, object]:
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in VLM output:\n" + raw_text)

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError("Failed to decode JSON from VLM output: " + str(exc))
    if not isinstance(data, dict):
        raise ValueError("VLM output JSON must decode to an object")
    return data


def parse_vlm_result(raw_text: str, canvas_h: int, canvas_w: int, rgb_h: int) -> ParsedVlmResult:
    """Parse object-level and grasp-level normalized coordinates from VLM JSON output."""
    data = _load_vlm_json(raw_text)

    candidates_raw = data.get("candidates")
    if candidates_raw is None:
        candidates = [data]
    elif isinstance(candidates_raw, list) and candidates_raw:
        candidates = [item for item in candidates_raw if isinstance(item, dict)]
        if not candidates:
            raise ValueError("JSON contains 'candidates' but none are valid objects:\n" + str(data))
    else:
        raise ValueError("'candidates' must be a non-empty list when present")

    object_box: tuple[int, int, int, int] | None = None
    object_box_raw = data.get("object_box")
    if object_box_raw is not None:
        ymin, xmin, ymax, xmax = _parse_candidate_box(
            {"grasp_region_box": object_box_raw},
            candidate_name="top-level object_box",
        )
        final_ymin, final_xmin = _scale_norm_xy_to_rgb(
            ymin, xmin, canvas_h=canvas_h, canvas_w=canvas_w, rgb_h=rgb_h
        )
        final_ymax, final_xmax = _scale_norm_xy_to_rgb(
            ymax, xmax, canvas_h=canvas_h, canvas_w=canvas_w, rgb_h=rgb_h
        )
        object_box = _normalize_box(final_ymin, final_xmin, final_ymax, final_xmax)

    object_point: tuple[int, int] | None = None
    object_point_raw = data.get("object_point")
    if object_point_raw is not None:
        y_val, x_val = _parse_point(object_point_raw, field_name="object_point")
        object_point = _scale_norm_xy_to_rgb(
            y_val, x_val, canvas_h=canvas_h, canvas_w=canvas_w, rgb_h=rgb_h
        )

    parsed_boxes: list[tuple[int, int, int, int]] = []
    for idx, candidate in enumerate(candidates):
        ymin, xmin, ymax, xmax = _parse_candidate_box(candidate, candidate_name=f"candidate[{idx}]")

        final_ymin, final_xmin = _scale_norm_xy_to_rgb(
            ymin, xmin, canvas_h=canvas_h, canvas_w=canvas_w, rgb_h=rgb_h
        )
        final_ymax, final_xmax = _scale_norm_xy_to_rgb(
            ymax, xmax, canvas_h=canvas_h, canvas_w=canvas_w, rgb_h=rgb_h
        )
        parsed_boxes.append(_normalize_box(final_ymin, final_xmin, final_ymax, final_xmax))

    return ParsedVlmResult(object_box=object_box, object_point=object_point, grasp_boxes=parsed_boxes)


def parse_box_coords(raw_text: str, canvas_h: int, canvas_w: int, rgb_h: int) -> list[tuple[int, int, int, int]]:
    """Backward-compatible wrapper returning only grasp-region boxes."""
    return parse_vlm_result(raw_text, canvas_h=canvas_h, canvas_w=canvas_w, rgb_h=rgb_h).grasp_boxes


def crop_depth_rgb(
    depth: np.ndarray,
    rgb: np.ndarray,
    ymin: int,
    xmin: int,
    ymax: int,
    xmax: int,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy slices [ymin:ymax, xmin:xmax]; same indices on aligned rgb/depth."""
    cropped_depth = depth[ymin:ymax, xmin:xmax].copy()
    cropped_rgb = rgb[ymin:ymax, xmin:xmax, :].copy()
    return cropped_depth, cropped_rgb


def save_box_overlay_png(
    full_rgb: np.ndarray,
    path: Path,
    *,
    ymin: int,
    xmin: int,
    ymax: int,
    xmax: int,
    color: tuple[int, int, int] = (255, 0, 0),
    point_yx: tuple[int, int] | None = None,
) -> None:
    """Save the full RGB scene with the selected box outline and optional point marker."""
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = np.asarray(full_rgb, dtype=np.uint8).copy()
    thickness = max(2, min(6, round(min(overlay.shape[0], overlay.shape[1]) * 0.004)))
    y0 = max(0, min(ymin, overlay.shape[0] - 1))
    y1 = max(0, min(ymax - 1, overlay.shape[0] - 1))
    x0 = max(0, min(xmin, overlay.shape[1] - 1))
    x1 = max(0, min(xmax - 1, overlay.shape[1] - 1))
    color_arr = np.array(color, dtype=np.uint8)
    overlay[y0 : y0 + thickness, x0 : x1 + 1] = color_arr
    overlay[max(0, y1 - thickness + 1) : y1 + 1, x0 : x1 + 1] = color_arr
    overlay[y0 : y1 + 1, x0 : x0 + thickness] = color_arr
    overlay[y0 : y1 + 1, max(0, x1 - thickness + 1) : x1 + 1] = color_arr
    if point_yx is not None:
        py = max(0, min(int(point_yx[0]), overlay.shape[0] - 1))
        px = max(0, min(int(point_yx[1]), overlay.shape[1] - 1))
        radius = max(2, thickness * 2)
        overlay[max(0, py - radius) : min(overlay.shape[0], py + radius + 1), max(0, px - thickness // 2) : min(overlay.shape[1], px + thickness // 2 + 1)] = color_arr
        overlay[max(0, py - thickness // 2) : min(overlay.shape[0], py + thickness // 2 + 1), max(0, px - radius) : min(overlay.shape[1], px + radius + 1)] = color_arr
    pil = rgb_to_pil(overlay)
    pil.save(str(path), format="PNG")


def save_cropped_rgb_png(
    full_rgb: np.ndarray,
    path: Path,
    *,
    ymin: int,
    xmin: int,
    ymax: int,
    xmax: int,
) -> None:
    """Save the full RGB scene with the selected grasp ROI outlined in red."""
    save_box_overlay_png(
        full_rgb,
        path,
        ymin=ymin,
        xmin=xmin,
        ymax=ymax,
        xmax=xmax,
        color=(255, 0, 0),
    )
