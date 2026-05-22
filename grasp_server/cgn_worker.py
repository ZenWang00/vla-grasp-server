"""Long-running Contact-GraspNet worker.

This script is launched **inside the ``contact_graspnet`` conda env** by the
FastAPI server (`grasp_server.cgn_client`). It loads the Contact-GraspNet model
once, then waits for JSON-line requests on stdin and writes JSON-line replies
on stdout. Stderr is reserved for human-readable logs.

Protocol (newline-delimited JSON):

    request:  {"input_npz": "/abs/path/contact_graspnet_input_000.npz",
               "output_npz": "/abs/path/predictions_contact_graspnet_input_000.npz"}
    ready:    {"event": "ready", "pid": 1234}
    success:  {"status": "ok", "output_npz": "...", "stats": {...}}
    failure:  {"status": "error", "error": "...", "input_npz": "..."}

The implementation mirrors ``contact_graspnet_pytorch/inference.py`` but keeps
the model resident across requests.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterator


def _emit(payload: dict[str, Any]) -> None:
    """Write a single JSON line to stdout and flush immediately."""
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _log(message: str) -> None:
    sys.stderr.write(f"[cgn_worker] {message}\n")
    sys.stderr.flush()


@contextlib.contextmanager
def _stdout_to_stderr() -> Iterator[None]:
    """Contact-GraspNet prints progress to stdout; keep stdout JSON-only for IPC."""
    previous = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = previous


def _load_estimator(ckpt_dir: Path, forward_passes: int):
    # Imports are deferred so a stdin-EOF before the first request does not
    # pay the cost of loading torch / Contact-GraspNet.
    from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
    from contact_graspnet_pytorch import config_utils
    from contact_graspnet_pytorch.checkpoints import CheckpointIO

    global_config = config_utils.load_config(str(ckpt_dir), batch_size=forward_passes)
    estimator = GraspEstimator(global_config)

    model_checkpoint_dir = ckpt_dir / "checkpoints"
    checkpoint_io = CheckpointIO(
        checkpoint_dir=str(model_checkpoint_dir), model=estimator.model
    )
    try:
        checkpoint_io.load("model.pt")
    except FileExistsError:
        _log(
            "No model.pt in checkpoints/; continuing with random init "
            "(same fallback as contact_graspnet_pytorch/inference.py)."
        )
    return estimator


def _run_one(
    estimator,
    input_npz: Path,
    output_npz: Path,
    *,
    local_regions: bool,
    filter_grasps: bool,
    skip_border_objects: bool,
    z_range: tuple[float, float],
    forward_passes: int,
) -> dict[str, Any]:
    import numpy as np
    from data import load_available_input_data

    segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(str(input_npz), K=None)

    if segmap is None and (local_regions or filter_grasps):
        raise ValueError(
            "Need a segmap key in the input NPZ to use local_regions / filter_grasps."
        )

    pc_segments: dict[int, np.ndarray] = {}
    with _stdout_to_stderr():
        if pc_full is None:
            pc_full, pc_segments, pc_colors = estimator.extract_point_clouds(
                depth,
                cam_K,
                segmap=segmap,
                rgb=rgb,
                skip_border_objects=skip_border_objects,
                z_range=list(z_range),
            )

        pred_grasps_cam, scores, contact_pts, _ = estimator.predict_scene_grasps(
            pc_full,
            pc_segments=pc_segments,
            local_regions=local_regions,
            filter_grasps=filter_grasps,
            forward_passes=forward_passes,
        )

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(output_npz),
        pc_full=pc_full,
        pred_grasps_cam=pred_grasps_cam,
        scores=scores,
        contact_pts=contact_pts,
        pc_colors=pc_colors,
    )
    grasp_count = sum(int(np.asarray(v).shape[0]) for v in pred_grasps_cam.values())
    return {
        "output_npz": str(output_npz),
        "stats": {
            "segments": len(pred_grasps_cam),
            "grasps_total": grasp_count,
            "pc_full_points": int(pc_full.shape[0]),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent Contact-GraspNet worker.")
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        required=True,
        help="Path to a Contact-GraspNet checkpoint directory (containing config.yaml + checkpoints/model.pt).",
    )
    parser.add_argument(
        "--forward-passes",
        type=int,
        default=1,
        help="Forward passes per inference (also sets batch_size in the global config).",
    )
    parser.add_argument(
        "--z-min",
        type=float,
        default=0.2,
        help="Lower z-range used when constructing the scene point cloud from depth.",
    )
    parser.add_argument(
        "--z-max",
        type=float,
        default=1.8,
        help="Upper z-range used when constructing the scene point cloud from depth.",
    )
    parser.add_argument(
        "--no-local-regions",
        action="store_true",
        help="Disable Contact-GraspNet's local_regions cropping (default: enabled).",
    )
    parser.add_argument(
        "--no-filter-grasps",
        action="store_true",
        help="Disable Contact-GraspNet's filter_grasps stage (default: enabled).",
    )
    parser.add_argument(
        "--skip-border-objects",
        action="store_true",
        help="Skip segments touching the depth border when extracting local regions.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    ckpt_dir = args.ckpt_dir.expanduser().resolve()
    if not ckpt_dir.is_dir():
        _emit({"event": "fatal", "error": f"ckpt-dir not found: {ckpt_dir}"})
        return 2

    _log(f"loading Contact-GraspNet from {ckpt_dir}")
    started = time.perf_counter()
    try:
        estimator = _load_estimator(ckpt_dir, args.forward_passes)
    except Exception as exc:
        _log(f"FATAL: failed to load model: {exc}\n{traceback.format_exc()}")
        _emit({"event": "fatal", "error": str(exc)})
        return 3
    _log(f"model ready in {time.perf_counter() - started:.2f}s, pid={os.getpid()}")
    _emit({"event": "ready", "pid": os.getpid()})

    z_range = (float(args.z_min), float(args.z_max))
    while True:
        line = sys.stdin.readline()
        if line == "":
            _log("stdin closed, exiting")
            return 0
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit({"status": "error", "error": f"invalid json: {exc}", "raw": line[:200]})
            continue

        input_npz = request.get("input_npz")
        output_npz = request.get("output_npz")
        if not input_npz or not output_npz:
            _emit({
                "status": "error",
                "error": "request must contain both 'input_npz' and 'output_npz'",
            })
            continue
        input_path = Path(input_npz)
        output_path = Path(output_npz)
        if not input_path.is_file():
            _emit({
                "status": "error",
                "error": f"input_npz not found: {input_npz}",
                "input_npz": input_npz,
            })
            continue

        t0 = time.perf_counter()
        try:
            result = _run_one(
                estimator,
                input_path,
                output_path,
                local_regions=not args.no_local_regions,
                filter_grasps=not args.no_filter_grasps,
                skip_border_objects=args.skip_border_objects,
                z_range=z_range,
                forward_passes=args.forward_passes,
            )
        except Exception as exc:
            _log(f"inference failed for {input_npz}: {exc}\n{traceback.format_exc()}")
            _emit({
                "status": "error",
                "error": str(exc),
                "input_npz": input_npz,
            })
            continue
        elapsed = time.perf_counter() - t0
        result["status"] = "ok"
        result["input_npz"] = str(input_path)
        result["elapsed_s"] = round(elapsed, 3)
        _emit(result)


if __name__ == "__main__":
    raise SystemExit(main())
