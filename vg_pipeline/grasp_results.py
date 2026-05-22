"""Shared Contact-GraspNet prediction loading + normalization.

Both the HTML report generator and the FastAPI grasp service consume Contact-GraspNet
prediction NPZs via this module so they agree on schema validation, width derivation,
and score ordering.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


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
    source_npz: Path | None = None

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


def summarize_npz(npz: np.lib.npyio.NpzFile) -> list[dict[str, Any]]:
    """Lightweight key/shape/dtype summary used by the HTML report."""
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
        raise ValueError(
            f"Expected {key!r} to be a scalar object array containing a dict, got shape={value.shape}"
        )
    payload = value.item()
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {key!r} to hold a dict, got {type(payload).__name__}")
    normalized: dict[int, np.ndarray] = {}
    for raw_key, raw_value in payload.items():
        normalized[int(raw_key)] = np.asarray(raw_value)
    return normalized


def _derive_width(transform: np.ndarray, contact_point: np.ndarray | None) -> float | None:
    """Recover the gripper width that Contact-GraspNet implicitly encodes via contact points."""
    if contact_point is None:
        return None
    base_dir = transform[:3, 0]
    approach_dir = transform[:3, 2]
    center = transform[:3, 3]
    width = 2.0 * float(
        np.dot(center - contact_point + (GRIPPER_DEPTH_METERS * approach_dir), base_dir)
    )
    if not np.isfinite(width):
        return None
    return abs(width)


def normalize_predictions(
    npz_path: str | Path,
) -> tuple[list[NormalizedGrasp], list[dict[str, Any]]]:
    """Flatten Contact-GraspNet ``pred_grasps_cam``/``scores`` dicts into a sorted grasp list."""
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=True) as predictions_npz:
        schema_summary = summarize_npz(predictions_npz)
        pred_grasps = _require_object_dict(predictions_npz, "pred_grasps_cam")
        scores = _require_object_dict(predictions_npz, "scores")
        contact_pts = (
            _require_object_dict(predictions_npz, "contact_pts")
            if "contact_pts" in predictions_npz.files
            else {}
        )

    grasps: list[NormalizedGrasp] = []
    for segment_id, transforms in pred_grasps.items():
        transforms = np.asarray(transforms, dtype=np.float32)
        if transforms.size == 0:
            continue
        if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
            raise ValueError(
                f"Unexpected pred_grasps_cam[{segment_id}] shape {transforms.shape}; expected (N, 4, 4)."
            )
        segment_scores = np.asarray(scores.get(segment_id), dtype=np.float32)
        if segment_scores.shape != (transforms.shape[0],):
            raise ValueError(
                f"Unexpected scores[{segment_id}] shape {segment_scores.shape}; "
                f"expected ({transforms.shape[0]},)."
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
            contact_point = (
                segment_contacts[grasp_index] if segment_contacts is not None else None
            )
            grasps.append(
                NormalizedGrasp(
                    segment_id=int(segment_id),
                    grasp_index=int(grasp_index),
                    score=float(segment_scores[grasp_index]),
                    transform=transform,
                    center_xyz=transform[:3, 3].copy(),
                    contact_point_xyz=None if contact_point is None else contact_point.copy(),
                    width_m=_derive_width(transform, contact_point),
                    source_npz=npz_path,
                )
            )

    if not grasps:
        raise ValueError(f"No grasps found in {npz_path}")

    grasps.sort(key=lambda item: item.score, reverse=True)
    return grasps, schema_summary


def normalize_predictions_multi(
    npz_paths: list[str | Path],
) -> list[NormalizedGrasp]:
    """Merge predictions from multiple NPZs (one per VLM candidate) into a single sorted list."""
    merged: list[NormalizedGrasp] = []
    for path in npz_paths:
        grasps, _ = normalize_predictions(path)
        merged.extend(grasps)
    if not merged:
        raise ValueError("No grasps found across any provided prediction NPZ.")
    merged.sort(key=lambda item: item.score, reverse=True)
    return merged
