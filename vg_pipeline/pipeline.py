from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .geometry import (
    backproject_roi_points,
    project_points_to_rgb_overlay_png,
    render_pointcloud_3d_png,
)
from .io import load_observation_npy, new_run_id, resolve_capture_dir, resolve_scene_image
from .manifest import write_manifest
from .providers import run_vg_inference
from .roi import crop_depth_rgb, parse_vlm_result, save_box_overlay_png, save_cropped_rgb_png
from .sam2_segment import run_sam2_segmentation, save_mask_overlay_png, save_mask_png


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


def _build_roi_outputs(
    *,
    out_dir: Path,
    grasp_boxes_tuples: list[tuple[int, int, int, int]],
    depth: np.ndarray,
    rgb: np.ndarray,
    K: np.ndarray,
    h: int,
    w: int,
    render_pointcloud_3d: bool,
) -> tuple[list[dict[str, int]], list[dict[str, Any]]]:
    parsed_boxes: list[dict[str, int]] = []
    roi_outputs: list[dict[str, Any]] = []

    for idx, raw_box in enumerate(grasp_boxes_tuples):
        ymin, xmin, ymax, xmax = raw_box
        ymin = max(0, min(ymin, h))
        ymax = max(0, min(ymax, h))
        xmin = max(0, min(xmin, w))
        xmax = max(0, min(xmax, w))
        if ymax <= ymin or xmax <= xmin:
            continue

        suffix = f"{idx:03d}"
        parsed_box = {"index": idx, "ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax}
        parsed_boxes.append(parsed_box)

        cropped_depth, cropped_rgb = crop_depth_rgb(depth, rgb, ymin, xmin, ymax, xmax)
        cropped_rgb_rel = f"cropped_rgb_{suffix}.png"
        cropped_depth_rel = f"cropped_depth_{suffix}.npy"
        roi_pointcloud_rel = f"roi_pointcloud_{suffix}.npy"
        pointcloud_3d_rel = f"pointcloud_3d_{suffix}.png"
        projected_points_rel = f"projected_points_on_rgb_{suffix}.png"

        save_cropped_rgb_png(
            rgb,
            out_dir / cropped_rgb_rel,
            ymin=ymin,
            xmin=xmin,
            ymax=ymax,
            xmax=xmax,
        )
        np.save(out_dir / cropped_depth_rel, cropped_depth)

        roi_points = backproject_roi_points(cropped_depth, K, xmin=xmin, ymin=ymin)
        np.save(out_dir / roi_pointcloud_rel, roi_points.astype(np.float32, copy=False))

        pointcloud_3d_manifest_path: str | None = None
        if render_pointcloud_3d and roi_points.shape[0] > 0:
            render_pointcloud_3d_png(roi_points, out_dir / pointcloud_3d_rel)
            pointcloud_3d_manifest_path = pointcloud_3d_rel

        project_points_to_rgb_overlay_png(
            roi_points,
            K=K,
            cropped_rgb=cropped_rgb,
            xmin=xmin,
            ymin=ymin,
            path=out_dir / projected_points_rel,
        )
        roi_outputs.append(
            {
                "index": idx,
                "parsed_box": parsed_box,
                "point_count": int(roi_points.shape[0]),
                "paths": {
                    "cropped_depth": cropped_depth_rel,
                    "cropped_rgb": cropped_rgb_rel,
                    "roi_pointcloud": roi_pointcloud_rel,
                    "pointcloud_3d": pointcloud_3d_manifest_path,
                    "projected_points_on_rgb": projected_points_rel,
                },
            }
        )

    return parsed_boxes, roi_outputs


def run_pipeline(
    npy_path: str | Path,
    task_spec: str,
    out_dir: str | Path,
    model_path: str | Path,
    provider: str = "qwen_local",
    api_key: str | None = None,
    code_execution: bool = False,
    schema_version: str = "qwen_vg_dual_box_v2",
    num_candidates: int = 3,
    *,
    render_pointcloud_3d: bool = True,
    enable_sam2: bool = False,
    sam2_model: str = "facebook/sam2.1-hiera-small",
    sam2_device: str | None = None,
    export_contact_graspnet_input: bool = False,
    contact_graspnet_export_name: str = "contact_graspnet_input.npz",
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
        oymin, oxmin, oymax, oxmax = parsed_vlm.object_box
        oymin = max(0, min(oymin, h))
        oymax = max(0, min(oymax, h))
        oxmin = max(0, min(oxmin, w))
        oxmax = max(0, min(oxmax, w))
        if oymax > oymin and oxmax > oxmin:
            object_box_tuple = (oymin, oxmin, oymax, oxmax)
            object_box_manifest = {"ymin": oymin, "xmin": oxmin, "ymax": oymax, "xmax": oxmax}
            if parsed_vlm.object_point is not None:
                py, px = parsed_vlm.object_point
                object_point_tuple = (max(0, min(py, h - 1)), max(0, min(px, w - 1)))
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

    sam2_segmentation: dict[str, Any] = {
        "enabled": bool(enable_sam2),
        "model_name": sam2_model if enable_sam2 else None,
        "device": sam2_device,
        "positive_pixels": None,
        "paths": {
            "object_mask": None,
            "object_mask_overlay": None,
            "object_segmap": None,
        },
        "status": "disabled" if not enable_sam2 else "pending",
    }
    contact_graspnet_export: dict[str, Any] = {
        "enabled": bool(export_contact_graspnet_input),
        "status": "disabled" if not export_contact_graspnet_input else "pending",
        "path": None,
        "keys": None,
    }
    if enable_sam2:
        if object_box_tuple is None:
            sam2_segmentation["status"] = "skipped_missing_object_box"
        else:
            sam2_result = run_sam2_segmentation(
                loaded.rgb,
                object_box=object_box_tuple,
                object_point=object_point_tuple,
                model_name=sam2_model,
                device=sam2_device,
                depth=loaded.depth,
            )
            object_mask_rel = "object_mask.png"
            object_mask_overlay_rel = "object_mask_overlay.png"
            object_segmap_rel = "object_segmap.npy"
            save_mask_png(sam2_result.mask, out_dir / object_mask_rel)
            save_mask_overlay_png(loaded.rgb, sam2_result.mask, out_dir / object_mask_overlay_rel)
            np.save(out_dir / object_segmap_rel, sam2_result.segmap)
            sam2_segmentation = {
                "enabled": True,
                "model_name": sam2_result.model_name,
                "device": sam2_result.device,
                "positive_pixels": sam2_result.positive_pixels,
                "paths": {
                    "object_mask": object_mask_rel,
                    "object_mask_overlay": object_mask_overlay_rel,
                    "object_segmap": object_segmap_rel,
                },
                "status": "ok",
            }
    if export_contact_graspnet_input:
        if sam2_segmentation["status"] != "ok":
            contact_graspnet_export["status"] = "skipped_missing_segmap"
        else:
            export_rel = contact_graspnet_export_name
            export_path = out_dir / export_rel
            # Contact-GraspNet's loader converts stored rgb from BGR to RGB, so export BGR here.
            rgb_bgr = loaded.rgb[:, :, ::-1].copy()
            object_segmap = sam2_result.segmap.astype(np.uint8, copy=False)
            np.savez(
                export_path,
                depth=np.asarray(loaded.depth),
                K=np.asarray(loaded.K),
                segmap=object_segmap,
                rgb=rgb_bgr,
            )
            contact_graspnet_export = {
                "enabled": True,
                "status": "ok",
                "path": export_rel,
                "keys": ["depth", "K", "segmap", "rgb"],
            }

    parsed_boxes, roi_outputs = _build_roi_outputs(
        out_dir=out_dir,
        grasp_boxes_tuples=parsed_vlm.grasp_boxes,
        depth=loaded.depth,
        rgb=loaded.rgb,
        K=loaded.K,
        h=h,
        w=w,
        render_pointcloud_3d=render_pointcloud_3d,
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
        sam2_segmentation=sam2_segmentation,
        contact_graspnet_export=contact_graspnet_export,
        parsed_boxes=parsed_boxes,
        roi_outputs=roi_outputs,
        scene_image_path=str(loaded.scene_path),
        depth_aux_image_path=str(loaded.depth_aux_path) if loaded.depth_aux_path is not None else None,
    )
    return out_dir
