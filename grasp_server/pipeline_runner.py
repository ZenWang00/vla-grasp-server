"""Thin wrapper around :func:`vg_pipeline.run_pipeline` for the HTTP service."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from vg_pipeline import run_pipeline


@dataclass(frozen=True)
class PipelineRunResult:
    run_dir: Path
    manifest_path: Path
    exported_npzs: list[Path]


def _read_cgn_exports_from_manifest(run_dir: Path) -> list[Path]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json missing under {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("contact_graspnet_export") or []
    if isinstance(entries, dict):
        entries = [entries]
    npzs: list[Path] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("status") != "ok":
            continue
        rel = entry.get("path")
        if not rel:
            continue
        candidate = (run_dir / rel).resolve()
        if candidate.is_file():
            npzs.append(candidate)
    return npzs


def run_vg_pipeline(
    *,
    capture_dir: Path,
    task_spec: str,
    out_base: Path,
    run_id: str,
    provider: str,
    model: str,
    num_candidates: int,
    sam2_model: str = "facebook/sam2.1-hiera-small",
    sam2_device: str | None = None,
) -> PipelineRunResult:
    """Run the VLM + dual SAM2 + Contact-GraspNet-export pipeline.

    Always exports per-candidate ``contact_graspnet_input_NNN.npz`` so the
    downstream CGN worker can score them.
    """
    out_base.mkdir(parents=True, exist_ok=True)
    out_dir = (out_base / f"api_{run_id}").resolve()

    run_dir = run_pipeline(
        capture_dir / "camera_data.npy",
        task_spec,
        out_dir,
        model,
        provider=provider,
        api_key=None,
        code_execution=False,
        num_candidates=num_candidates,
        render_pointcloud_3d=False,
        enable_sam2=True,
        sam2_model=sam2_model,
        sam2_device=sam2_device,
        export_contact_graspnet_input=True,
        contact_graspnet_export_template="contact_graspnet_input_{idx:03d}.npz",
        run_id=run_id,
        capture_dir=capture_dir,
    )
    run_dir = Path(run_dir)
    exports = _read_cgn_exports_from_manifest(run_dir)
    if not exports:
        raise RuntimeError(
            f"vg_pipeline produced no contact_graspnet_input_*.npz exports under {run_dir}; "
            "see manifest.json for details."
        )
    return PipelineRunResult(
        run_dir=run_dir,
        manifest_path=run_dir / "manifest.json",
        exported_npzs=exports,
    )
