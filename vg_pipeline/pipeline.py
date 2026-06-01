from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .clean3d import (
    build_pure_target_pointcloud,
    project_points_to_rgb_full,
    render_pointcloud_3d_png,
)
from .io import load_observation_npy, new_run_id, resolve_capture_dir, resolve_scene_image
from .manifest import write_manifest
from .providers import run_vg_inference
from .roi import parse_vlm_result, save_box_overlay_png
from .sam2_segment import (
    Sam2SegmentationResult,
    run_sam2_global,
    run_sam2_local,
    save_mask_overlay_png,
    save_mask_png,
)


DEFAULT_SCHEMA_VERSION = "vla_dual_sam2_v1"


@dataclass(frozen=True)
class LoadedPipelineInputs:
    depth: np.ndarray
    K: np.ndarray
    rgb: np.ndarray
    scene_path: Path
    depth_aux_path: Path | None


def _load_pipeline_inputs(
    *,
    npy_path: Path,
    rgbd_stem: str | Path | None,
    scene_image_path: str | Path | None,
    depth_aux_image_path: str | Path | None,
) -> LoadedPipelineInputs:
    if rgbd_stem is not None:
        if scene_image_path is not None or depth_aux_image_path is not None:
            raise ValueError("Do not combine rgbd_stem with scene_image_path/depth_aux_image_path.")
        scene_p = resolve_scene_image(rgbd_stem)
        depth_p = None
    elif scene_image_path is not None:
        scene_p = Path(scene_image_path).resolve()
        if not scene_p.is_file():
            raise FileNotFoundError(f"Scene image not found: {scene_p}")
        depth_p = Path(depth_aux_image_path).resolve() if depth_aux_image_path is not None else None
        if depth_p is not None and not depth_p.is_file():
            raise FileNotFoundError(f"Depth visualization image not found: {depth_p}")
    else:
        raise ValueError("RGB scene image is required: pass rgbd_stem, or pass scene_image_path.")

    obs = load_observation_npy(npy_path)
    depth = np.asarray(obs["depth"])
    K = obs["K"]
    h, w = int(depth.shape[0]), int(depth.shape[1])

    scene_pil = Image.open(scene_p).convert("RGB")
    rgb = np.asarray(scene_pil, dtype=np.uint8)
    if rgb.shape[0] != h or rgb.shape[1] != w:
        raise ValueError(
            f"Scene image size {rgb.shape[:2]} does not match depth size ({h}, {w}) from npy."
        )

    return LoadedPipelineInputs(
        depth=depth,
        K=K,
        rgb=rgb,
        scene_path=scene_p,
        depth_aux_path=depth_p,
    )


def _clip_box(
    raw_box: tuple[int, int, int, int], *, h: int, w: int
) -> tuple[int, int, int, int] | None:
    ymin, xmin, ymax, xmax = raw_box
    ymin = max(0, min(ymin, h))
    ymax = max(0, min(ymax, h))
    xmin = max(0, min(xmin, w))
    xmax = max(0, min(xmax, w))
    if ymax <= ymin or xmax <= xmin:
        return None
    return ymin, xmin, ymax, xmax


def _clip_point(
    point: tuple[int, int] | None, *, h: int, w: int
) -> tuple[int, int] | None:
    if point is None:
        return None
    py, px = point
    return max(0, min(py, h - 1)), max(0, min(px, w - 1))


def _box_boolean_mask(
    h: int, w: int, ymin: int, xmin: int, ymax: int, xmax: int
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[ymin:ymax, xmin:xmax] = True
    return mask


def _clip_tight_mask_to_grasp_box(
    mask: np.ndarray,
    *,
    ymin: int,
    xmin: int,
    ymax: int,
    xmax: int,
) -> np.ndarray:
    """Intersect SAM2 local mask with the VLM grasp_region_box (removes spill outside ROI)."""
    h, w = mask.shape
    box_mask = _box_boolean_mask(h, w, ymin, xmin, ymax, xmax)
    return (mask.astype(bool) & box_mask).astype(np.uint8)


def _run_sam2_global_stage(
    *,
    out_dir: Path,
    rgb: np.ndarray,
    depth: np.ndarray,
    object_box: tuple[int, int, int, int] | None,
    object_point: tuple[int, int] | None,
    enable_sam2: bool,
    sam2_model: str,
    sam2_device: str | None,
) -> tuple[Sam2SegmentationResult | None, dict[str, Any]]:
    status_info: dict[str, Any] = {
        "enabled": bool(enable_sam2),
        "model_name": sam2_model if enable_sam2 else None,
        "device": sam2_device,
        "positive_pixels": None,
        "paths": {
            "global_mask": None,
            "global_mask_overlay": None,
            "global_segmap": None,
        },
        "status": "disabled" if not enable_sam2 else "pending",
    }
    if not enable_sam2:
        return None, status_info
    if object_box is None:
        status_info["status"] = "skipped_missing_object_box"
        return None, status_info

    result = run_sam2_global(
        rgb,
        object_box=object_box,
        object_point=object_point,
        model_name=sam2_model,
        device=sam2_device,
        depth=depth,
    )
    global_mask_rel = "global_mask.png"
    global_mask_overlay_rel = "global_mask_overlay.png"
    global_segmap_rel = "global_segmap.npy"
    save_mask_png(result.mask, out_dir / global_mask_rel)
    save_mask_overlay_png(rgb, result.mask, out_dir / global_mask_overlay_rel)
    np.save(out_dir / global_segmap_rel, result.segmap)
    status_info = {
        "enabled": True,
        "model_name": result.model_name,
        "device": result.device,
        "positive_pixels": result.positive_pixels,
        "paths": {
            "global_mask": global_mask_rel,
            "global_mask_overlay": global_mask_overlay_rel,
            "global_segmap": global_segmap_rel,
        },
        "status": "ok",
    }
    return result, status_info


def _run_per_candidate_stage(
    *,
    out_dir: Path,
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    h: int,
    w: int,
    grasp_boxes: list[tuple[int, int, int, int]],
    grasp_points: list[tuple[int, int] | None],
    global_mask: np.ndarray | None,
    enable_sam2: bool,
    sam2_model: str,
    sam2_device: str | None,
    render_pointcloud_3d: bool,
    export_contact_graspnet_input: bool,
    contact_graspnet_export_template: str,
) -> tuple[
    list[dict[str, int]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    parsed_boxes: list[dict[str, int]] = []
    sam2_local_entries: list[dict[str, Any]] = []
    clean_3d_entries: list[dict[str, Any]] = []
    cgn_export_entries: list[dict[str, Any]] = []

    for idx, raw_box in enumerate(grasp_boxes):
        clipped = _clip_box(raw_box, h=h, w=w)
        if clipped is None:
            continue
        ymin, xmin, ymax, xmax = clipped
        suffix = f"{idx:03d}"
        parsed_box = {"index": idx, "ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax}
        parsed_boxes.append(parsed_box)

        grasp_point = _clip_point(
            grasp_points[idx] if idx < len(grasp_points) else None,
            h=h,
            w=w,
        )
        grasp_point_manifest = (
            {"y": grasp_point[0], "x": grasp_point[1]} if grasp_point is not None else None
        )

        sam2_local_paths = {
            "tight_grasp_mask": None,
            "tight_grasp_mask_overlay": None,
            "tight_grasp_segmap": None,
        }
        sam2_local_entry: dict[str, Any] = {
            "index": idx,
            "grasp_region_box": parsed_box,
            "grasp_point": grasp_point_manifest,
            "positive_pixels": None,
            "paths": sam2_local_paths,
            "status": "disabled" if not enable_sam2 else "pending",
        }
        clean_3d_entry: dict[str, Any] = {
            "index": idx,
            "point_count": 0,
            "paths": {
                "pure_target_pointcloud": None,
                "pure_target_pointcloud_3d": None,
                "pure_target_pointcloud_overlay": None,
            },
            "status": "disabled" if not enable_sam2 else "pending",
        }
        cgn_entry: dict[str, Any] = {
            "index": idx,
            "enabled": bool(export_contact_graspnet_input),
            "status": "disabled" if not export_contact_graspnet_input else "pending",
            "path": None,
            "keys": None,
            "segmap_source": "tight_grasp_mask",
            "global_mask_included": False,
        }

        if not enable_sam2:
            sam2_local_entries.append(sam2_local_entry)
            clean_3d_entries.append(clean_3d_entry)
            cgn_export_entries.append(cgn_entry)
            continue

        local_result = run_sam2_local(
            rgb,
            grasp_region_box=(ymin, xmin, ymax, xmax),
            grasp_point=grasp_point,
            model_name=sam2_model,
            device=sam2_device,
            depth=depth,
        )
        tight_mask = _clip_tight_mask_to_grasp_box(
            local_result.mask,
            ymin=ymin,
            xmin=xmin,
            ymax=ymax,
            xmax=xmax,
        )
        tight_segmap = np.zeros(tight_mask.shape, dtype=np.uint8)
        tight_segmap[tight_mask.astype(bool)] = 1

        tight_mask_rel = f"tight_grasp_mask_{suffix}.png"
        tight_mask_overlay_rel = f"tight_grasp_mask_overlay_{suffix}.png"
        tight_segmap_rel = f"tight_grasp_segmap_{suffix}.npy"
        save_mask_png(tight_mask, out_dir / tight_mask_rel)
        save_mask_overlay_png(rgb, tight_mask, out_dir / tight_mask_overlay_rel)
        np.save(out_dir / tight_segmap_rel, tight_segmap)
        sam2_local_entry.update(
            {
                "model_name": local_result.model_name,
                "device": local_result.device,
                "positive_pixels": int(tight_mask.astype(bool).sum()),
                "sam2_positive_pixels_before_box_clip": local_result.positive_pixels,
                "tight_mask_clipped_to_grasp_box": True,
                "paths": {
                    "tight_grasp_mask": tight_mask_rel,
                    "tight_grasp_mask_overlay": tight_mask_overlay_rel,
                    "tight_grasp_segmap": tight_segmap_rel,
                },
                "status": "ok",
            }
        )

        points = build_pure_target_pointcloud(depth, K, tight_mask)
        pointcloud_rel = f"pure_target_pointcloud_{suffix}.npy"
        pointcloud_3d_rel = f"pure_target_pointcloud_3d_{suffix}.png"
        pointcloud_overlay_rel = f"pure_target_pointcloud_overlay_{suffix}.png"
        np.save(out_dir / pointcloud_rel, points.astype(np.float32, copy=False))
        rendered_3d_path: str | None = None
        if render_pointcloud_3d and points.shape[0] > 0:
            render_pointcloud_3d_png(points, out_dir / pointcloud_3d_rel)
            rendered_3d_path = pointcloud_3d_rel
        project_points_to_rgb_full(points, K, rgb, out_dir / pointcloud_overlay_rel)
        clean_3d_entry.update(
            {
                "point_count": int(points.shape[0]),
                "paths": {
                    "pure_target_pointcloud": pointcloud_rel,
                    "pure_target_pointcloud_3d": rendered_3d_path,
                    "pure_target_pointcloud_overlay": pointcloud_overlay_rel,
                },
                "status": "ok" if points.shape[0] > 0 else "empty_pointcloud",
            }
        )

        if export_contact_graspnet_input:
            export_rel = contact_graspnet_export_template.format(idx=idx)
            export_path = out_dir / export_rel
            # Contact-GraspNet's loader converts stored rgb from BGR to RGB, so export BGR here.
            rgb_bgr = rgb[:, :, ::-1].copy()
            depth_arr = np.asarray(depth)
            # Use global_mask (full object) for depth context when available so CGN can
            # reason about the complete object geometry (e.g. a hollow cup, not just a
            # curved wall fragment). tight_segmap still directs CGN to the grasp region.
            # Fallback to full unmasked depth when global_mask is unavailable.
            if global_mask is not None:
                depth_for_cgn = np.where(global_mask.astype(bool), depth_arr, 0.0).astype(
                    depth_arr.dtype, copy=False
                )
            else:
                depth_for_cgn = depth_arr.astype(depth_arr.dtype, copy=False)
            payload: dict[str, np.ndarray] = {
                "depth": depth_for_cgn,
                "K": np.asarray(K),
                "segmap": tight_segmap.astype(np.uint8, copy=False),
                "rgb": rgb_bgr,
            }
            keys = ["depth", "K", "segmap", "rgb"]
            global_included = False
            if global_mask is not None:
                payload["global_mask"] = global_mask.astype(np.uint8, copy=False)
                keys.append("global_mask")
                global_included = True
            np.savez(export_path, **payload)
            cgn_entry.update(
                {
                    "status": "ok",
                    "path": export_rel,
                    "keys": keys,
                    "segmap_source": "tight_grasp_mask",
                    "depth_masked_to_global_object": global_mask is not None,
                    "global_mask_included": global_included,
                }
            )

        sam2_local_entries.append(sam2_local_entry)
        clean_3d_entries.append(clean_3d_entry)
        cgn_export_entries.append(cgn_entry)

    return parsed_boxes, sam2_local_entries, clean_3d_entries, cgn_export_entries


def run_pipeline(
    npy_path: str | Path,
    task_spec: str,
    out_dir: str | Path,
    model_path: str | Path,
    provider: str = "gemini",
    api_key: str | None = None,
    code_execution: bool = False,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
    num_candidates: int = 3,
    *,
    render_pointcloud_3d: bool = True,
    enable_sam2: bool = False,
    sam2_model: str = "facebook/sam2.1-hiera-small",
    sam2_device: str | None = None,
    export_contact_graspnet_input: bool = False,
    contact_graspnet_export_template: str = "contact_graspnet_input_{idx:03d}.npz",
    run_id: str | None = None,
    scene_image_path: str | Path | None = None,
    depth_aux_image_path: str | Path | None = None,
    rgbd_stem: str | Path | None = None,
    capture_dir: str | Path | None = None,
) -> Path:
    if capture_dir is not None:
        if scene_image_path is not None or depth_aux_image_path is not None or rgbd_stem is not None:
            raise ValueError("Do not combine capture_dir with rgbd_stem/scene_image_path/depth_aux_image_path.")
        capture_npy_path, capture_scene_path, capture_depth_path = resolve_capture_dir(capture_dir)
        npy_path = capture_npy_path
        scene_image_path = capture_scene_path
        depth_aux_image_path = capture_depth_path

    npy_path = Path(npy_path).resolve()
    out_dir = Path(out_dir).resolve()
    if run_id is None:
        run_id = new_run_id()
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = _load_pipeline_inputs(
        npy_path=npy_path,
        rgbd_stem=rgbd_stem,
        scene_image_path=scene_image_path,
        depth_aux_image_path=depth_aux_image_path,
    )
    h, w = int(loaded.depth.shape[0]), int(loaded.depth.shape[1])

    raw_model_text = run_vg_inference(
        provider=provider,
        images=[Image.fromarray(loaded.rgb)],
        task_spec=task_spec,
        model_path=model_path,
        num_candidates=num_candidates,
        api_key=api_key,
        code_execution=code_execution,
        openai_image_mime_types=["image/png"],
        gemini_image_mime_types=["image/png"],
    )
    (out_dir / "raw_model_text.txt").write_text(raw_model_text, encoding="utf-8")

    parsed_vlm = parse_vlm_result(
        raw_model_text,
        canvas_h=h,
        canvas_w=w,
        rgb_h=h,
    )

    object_box_manifest: dict[str, int] | None = None
    object_point_manifest: dict[str, int] | None = None
    object_box_overlay_path: str | None = None
    object_box_tuple: tuple[int, int, int, int] | None = None
    object_point_tuple: tuple[int, int] | None = None
    if parsed_vlm.object_box is not None:
        object_box_tuple = _clip_box(parsed_vlm.object_box, h=h, w=w)
        if object_box_tuple is not None:
            oymin, oxmin, oymax, oxmax = object_box_tuple
            object_box_manifest = {"ymin": oymin, "xmin": oxmin, "ymax": oymax, "xmax": oxmax}
            object_point_tuple = _clip_point(parsed_vlm.object_point, h=h, w=w)
            if object_point_tuple is not None:
                object_point_manifest = {
                    "y": object_point_tuple[0],
                    "x": object_point_tuple[1],
                }
            object_box_overlay_path = "object_box_overlay.png"
            save_box_overlay_png(
                loaded.rgb,
                out_dir / object_box_overlay_path,
                ymin=oymin,
                xmin=oxmin,
                ymax=oymax,
                xmax=oxmax,
                color=(0, 255, 0),
                point_yx=object_point_tuple,
            )

    global_result, sam2_global_manifest = _run_sam2_global_stage(
        out_dir=out_dir,
        rgb=loaded.rgb,
        depth=loaded.depth,
        object_box=object_box_tuple,
        object_point=object_point_tuple,
        enable_sam2=enable_sam2,
        sam2_model=sam2_model,
        sam2_device=sam2_device,
    )
    global_mask = global_result.mask if global_result is not None else None

    parsed_boxes, sam2_local_entries, clean_3d_entries, cgn_export_entries = _run_per_candidate_stage(
        out_dir=out_dir,
        rgb=loaded.rgb,
        depth=loaded.depth,
        K=loaded.K,
        h=h,
        w=w,
        grasp_boxes=parsed_vlm.grasp_boxes,
        grasp_points=parsed_vlm.grasp_points,
        global_mask=global_mask,
        enable_sam2=enable_sam2,
        sam2_model=sam2_model,
        sam2_device=sam2_device,
        render_pointcloud_3d=render_pointcloud_3d,
        export_contact_graspnet_input=export_contact_graspnet_input,
        contact_graspnet_export_template=contact_graspnet_export_template,
    )

    if not parsed_boxes:
        raise ValueError("All parsed boxes are invalid or out of image bounds after clipping")

    write_manifest(
        out_dir,
        schema_version=schema_version,
        provider=provider,
        model_path=str(model_path),
        npy_path=str(npy_path),
        run_id=run_id,
        task_spec=task_spec,
        raw_model_text=raw_model_text,
        object_detection={
            "parsed_box": object_box_manifest,
            "parsed_point": object_point_manifest,
            "paths": {"object_box_overlay": object_box_overlay_path},
        },
        sam2_global=sam2_global_manifest,
        sam2_local=sam2_local_entries,
        clean_local_3d=clean_3d_entries,
        contact_graspnet_export=cgn_export_entries,
        parsed_boxes=parsed_boxes,
        scene_image_path=str(loaded.scene_path),
        depth_aux_image_path=str(loaded.depth_aux_path) if loaded.depth_aux_path is not None else None,
    )
    return out_dir
