"""
Visual grounding pipeline: RGB scene + .npy (depth, K) → VLM (single RGB image) → <box> → crop + manifest.

Boxes are always in the RGB image frame. Provide ``--capture-dir``, ``--rgbd-stem``, or ``--scene-image``.

By default each run writes under ``--out-dir / {npy_stem}_{run_id} /`` so repeated runs keep history.
Use ``--no-subdir`` to write directly into ``--out-dir`` (overwrites fixed filenames).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from vg_pipeline import new_run_id, resolve_output_dir, run_pipeline

__all__ = ["main", "new_run_id", "resolve_output_dir", "run_pipeline"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision grounding + object/grasp boxes + depth/RGB ROI crop")
    parser.add_argument("--npy", type=Path, default=None, help="Path to observation .npy")
    parser.add_argument("--task-spec", type=str, required=True, help="Natural language grounding task")
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=None,
        help="Capture folder like captures/20260417_120019 → auto-resolves camera_data.npy and color_preview.jpg.",
    )
    parser.add_argument(
        "--scene-image",
        type=Path,
        default=None,
        help="RGB scene aligned with npy depth. Required unless --capture-dir or --rgbd-stem is used.",
    )
    parser.add_argument(
        "--depth-aux-image",
        type=Path,
        default=None,
        help="Optional aligned depth visualization path. Recorded for provenance only; not sent to the VLM.",
    )
    parser.add_argument(
        "--rgbd-stem",
        type=Path,
        default=None,
        help="Stem like output_rgbd/1 → resolves 1_scene.* in the same directory. "
        "Mutually exclusive with --scene-image/--depth-aux-image.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output_vg"),
        help="Base directory; each run writes to a subdir {npy_stem}_{run_id}/ unless --no-subdir",
    )
    parser.add_argument(
        "--no-subdir",
        action="store_true",
        help="Write directly into --out-dir (overwrites on repeat); default is one subdir per run",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="qwen_local",
        choices=["qwen_local", "openai", "gemini"],
        help="Inference backend: local qwen model or remote OpenAI/Gemini API",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/qwen2.5-vl-7b",
        help="qwen_local: existing directory with weights (e.g. models/qwen2.5-vl-7b) or a HF model id "
        "(e.g. Qwen/Qwen2.5-VL-7B-Instruct). openai/gemini: API model name.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key override. Else use OPENAI_API_KEY / GEMINI_API_KEY / GOOGLE_API_KEY",
    )
    parser.add_argument(
        "--code-execution",
        action="store_true",
        help="Enable Gemini code_execution tool (only used when --provider gemini).",
    )
    parser.add_argument(
        "--schema-version",
        type=str,
        default="qwen_vg_dual_box_v2",
        help="Written to manifest.json",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=3,
        help="How many grasp candidates the VLM should propose, ordered best-first.",
    )
    parser.add_argument(
        "--no-render-pointcloud3d",
        action="store_true",
        help="Disable pointcloud_3d_XXX.png rendering (enabled by default).",
    )
    parser.add_argument(
        "--enable-sam2",
        action="store_true",
        help="Run local SAM2 segmentation from VLM object_box/object_point prompts.",
    )
    parser.add_argument(
        "--sam2-model",
        type=str,
        default="facebook/sam2.1-hiera-small",
        help="Hugging Face SAM2 model id used when --enable-sam2 is set.",
    )
    parser.add_argument(
        "--sam2-device",
        type=str,
        default=None,
        help="Optional SAM2 device override, e.g. cuda, cpu, mps. Default is auto-detect.",
    )
    parser.add_argument(
        "--export-contact-graspnet-input",
        action="store_true",
        help="Export a Contact-GraspNet-ready .npz with depth/K/segmap/rgb after SAM2 succeeds.",
    )
    parser.add_argument(
        "--contact-graspnet-export-name",
        type=str,
        default="contact_graspnet_input.npz",
        help="Filename for the optional Contact-GraspNet export under the run output directory.",
    )
    args = parser.parse_args()

    if args.num_candidates < 1:
        parser.error("--num-candidates must be >= 1")

    if args.capture_dir is not None:
        if args.npy is not None or args.rgbd_stem is not None or args.scene_image is not None or args.depth_aux_image is not None:
            parser.error("Do not combine --capture-dir with --npy/--rgbd-stem/--scene-image/--depth-aux-image.")
        npy_resolved = args.capture_dir.resolve() / "camera_data.npy"
    else:
        if args.npy is None:
            parser.error("Provide --capture-dir or --npy.")
        if args.rgbd_stem is not None:
            if args.scene_image is not None or args.depth_aux_image is not None:
                parser.error("Do not combine --rgbd-stem with --scene-image/--depth-aux-image.")
        else:
            if args.scene_image is None:
                parser.error("Provide --capture-dir, --rgbd-stem, or --scene-image.")
        npy_resolved = args.npy.resolve()

    run_id = new_run_id()
    out_dir = resolve_output_dir(
        args.out_dir,
        npy_resolved,
        run_id=run_id,
        no_subdir=args.no_subdir,
    )

    out = run_pipeline(
        npy_resolved,
        args.task_spec,
        out_dir,
        args.model,
        provider=args.provider,
        api_key=args.api_key,
        code_execution=args.code_execution,
        schema_version=args.schema_version,
        num_candidates=args.num_candidates,
        render_pointcloud_3d=not args.no_render_pointcloud3d,
        enable_sam2=args.enable_sam2,
        sam2_model=args.sam2_model,
        sam2_device=args.sam2_device,
        export_contact_graspnet_input=args.export_contact_graspnet_input,
        contact_graspnet_export_name=args.contact_graspnet_export_name,
        run_id=run_id,
        scene_image_path=args.scene_image.resolve() if args.scene_image else None,
        depth_aux_image_path=args.depth_aux_image.resolve() if args.depth_aux_image else None,
        rgbd_stem=args.rgbd_stem.resolve() if args.rgbd_stem else None,
        capture_dir=args.capture_dir.resolve() if args.capture_dir else None,
    )
    print(f"Done. run_id={run_id} Outputs in: {out}")


if __name__ == "__main__":
    main()
