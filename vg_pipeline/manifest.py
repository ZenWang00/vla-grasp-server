from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_manifest(
    out_dir: Path,
    *,
    schema_version: str,
    provider: str,
    model_path: str,
    npy_path: str,
    run_id: str,
    task_spec: str,
    raw_model_text: str,
    object_detection: dict[str, Any],
    sam2_segmentation: dict[str, Any],
    contact_graspnet_export: dict[str, Any],
    parsed_boxes: list[dict[str, int]],
    roi_outputs: list[dict[str, Any]],
    scene_image_path: str,
    depth_aux_image_path: str | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_p = Path(npy_path)
    manifest: dict[str, Any] = {
        "schema_version": schema_version,
        "run_id": run_id,
        "npy_filename": npy_p.name,
        "npy_stem": npy_p.stem,
        "provider": provider,
        "model_path": str(model_path),
        "npy_input_path": str(npy_path),
        "scene_image_path": scene_image_path,
        "depth_aux_image_path": depth_aux_image_path,
        "task_spec": task_spec,
        "raw_model_text": raw_model_text,
        "object_detection": object_detection,
        "sam2_segmentation": sam2_segmentation,
        "contact_graspnet_export": contact_graspnet_export,
        "box_count": len(parsed_boxes),
        "parsed_boxes": parsed_boxes,
        "rois": roi_outputs,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
