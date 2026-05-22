#!/usr/bin/env python3
"""Smoke test the grasp HTTP server against an existing capture directory.

Loads ``camera_data.npy`` (containing depth + K) and ``color_preview.jpg`` from a
capture folder, encodes them as multipart, POSTs to ``/grasp``, and pretty-prints
the top-K poses returned by the server.

Use this to:
- Confirm the server is reachable and responding before bringing up ROS2.
- Mirror exactly what ``grasp_pose_client_node`` will eventually send, but
  without depending on ROS2 or a connected camera.

Example:

    .venv/bin/python scripts/smoke_test_grasp.py \\
        --capture-dir captures/20260417_120019 \\
        --task-spec "Target: the blue bottle"
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image


def load_capture(capture_dir: Path) -> tuple[bytes, bytes, str]:
    """Return (rgb_png_bytes, depth_npy_bytes, K_json) ready for multipart upload."""
    npy_path = capture_dir / "camera_data.npy"
    color_path = capture_dir / "color_preview.jpg"
    if not npy_path.is_file():
        raise FileNotFoundError(f"missing {npy_path}")
    if not color_path.is_file():
        raise FileNotFoundError(f"missing {color_path}")

    data = np.load(npy_path, allow_pickle=True).item()
    if not isinstance(data, dict):
        raise TypeError(f"{npy_path} did not contain a dict")
    if "depth" not in data or "K" not in data:
        raise KeyError(f"{npy_path} must contain at least 'depth' and 'K' keys")

    depth = np.ascontiguousarray(np.asarray(data["depth"], dtype=np.float32))
    K = np.asarray(data["K"], dtype=np.float64).reshape(3, 3)
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth.shape}")

    rgb_png_bytes = color_path.read_bytes()
    pil = Image.open(io.BytesIO(rgb_png_bytes))
    if pil.size[::-1] != depth.shape:
        raise ValueError(
            f"color_preview.jpg size {pil.size[::-1]} != depth shape {depth.shape}"
        )

    depth_buf = io.BytesIO()
    np.save(depth_buf, depth, allow_pickle=False)
    return rgb_png_bytes, depth_buf.getvalue(), json.dumps(K.tolist())


def post_grasp(
    *,
    server_url: str,
    rgb_png_bytes: bytes,
    depth_npy_bytes: bytes,
    K_json: str,
    task_spec: str,
    frame_id: str,
    top_k: int,
    num_candidates: int,
    timeout_s: float,
    T_base_gripper_json: str | None = None,
) -> dict:
    url = server_url.rstrip("/") + "/grasp"
    data: dict = {
        "K": K_json,
        "task_spec": task_spec,
        "frame_id": frame_id,
        "top_k": str(top_k),
        "num_candidates": str(num_candidates),
    }
    if T_base_gripper_json is not None:
        data["T_base_gripper"] = T_base_gripper_json
    response = requests.post(
        url,
        files={
            "rgb": ("rgb.png", rgb_png_bytes, "image/png"),
            "depth": ("depth.npy", depth_npy_bytes, "application/octet-stream"),
        },
        data=data,
        timeout=timeout_s,
    )
    if response.status_code != 200:
        try:
            body = response.json()
        except Exception:
            body = response.text[:1000]
        raise RuntimeError(f"HTTP {response.status_code}: {body}")
    return response.json()


def probe_health(server_url: str) -> None:
    url = server_url.rstrip("/") + "/health"
    try:
        r = requests.get(url, timeout=5)
    except requests.RequestException as exc:
        raise SystemExit(f"could not reach {url}: {exc}")
    if r.status_code != 200:
        raise SystemExit(f"{url} returned HTTP {r.status_code}: {r.text[:500]}")
    payload = r.json()
    print(f"health: status={payload.get('status')}, "
          f"worker_ready={payload.get('worker_ready')}, "
          f"worker_pid={payload.get('worker_pid')}")
    if not payload.get("worker_ready"):
        recent_stderr = payload.get("recent_worker_stderr") or []
        for line in recent_stderr[-10:]:
            print(f"  worker_stderr> {line}")
        raise SystemExit(
            "worker not ready; install Contact-GraspNet weights and restart server"
        )


def summarize_grasp(grasp: dict, rank: int) -> str:
    pos = grasp.get("position_xyz", [])
    quat = grasp.get("quaternion_xyzw", [])
    width = grasp.get("width_m")
    score = grasp.get("score", float("nan"))
    width_str = f"{width:.4f}m" if isinstance(width, (int, float)) else "n/a"
    pos_str = ", ".join(f"{v:.3f}" for v in pos) if pos else "?"
    quat_str = ", ".join(f"{v:.3f}" for v in quat) if quat else "?"
    lines = [
        f"  #{rank} score={score:.4f}  "
        f"pos=[{pos_str}]  quat_xyzw=[{quat_str}]  width={width_str}"
    ]
    if "base_frame" in grasp:
        bf = grasp["base_frame"]
        bf_pos = bf.get("position_xyz", [])
        bf_quat = bf.get("quaternion_xyzw", [])
        bf_pos_str = ", ".join(f"{v:.3f}" for v in bf_pos) if bf_pos else "?"
        bf_quat_str = ", ".join(f"{v:.3f}" for v in bf_quat) if bf_quat else "?"
        lines.append(
            f"       base_frame: pos=[{bf_pos_str}]  quat_xyzw=[{bf_quat_str}]"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-dir",
        type=Path,
        required=True,
        help="Capture folder containing camera_data.npy + color_preview.jpg.",
    )
    parser.add_argument(
        "--task-spec",
        type=str,
        default="Target: the cup",
        help="Natural-language task description sent to the VLM.",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8765",
        help="Base URL of the grasp HTTP server.",
    )
    parser.add_argument(
        "--frame-id",
        type=str,
        default="camera_color_optical_frame",
        help="Frame id to put on the returned PoseStamped[]; arbitrary for smoke testing.",
    )
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=180.0,
        help="HTTP request timeout. Set high since VLM + SAM2 + CGN can take >30s.",
    )
    parser.add_argument(
        "--no-health-probe",
        action="store_true",
        help="Skip the GET /health step (still useful when the server is intentionally degraded).",
    )
    parser.add_argument(
        "--T-base-gripper",
        type=str,
        default=None,
        metavar="JSON",
        help=(
            "Optional 4x4 gripper-to-base transform at capture time, JSON encoded. "
            "When provided (and GRASP_T_GRIPPER_CAMERA is configured on the server), "
            "each grasp will also include a 'base_frame' field. "
            "Example: '[[1,0,0,0.3],[0,1,0,0],[0,0,1,0.5],[0,0,0,1]]'"
        ),
    )
    args = parser.parse_args()

    capture_dir = args.capture_dir.resolve()
    if not capture_dir.is_dir():
        raise SystemExit(f"--capture-dir not found: {capture_dir}")

    print(f"server_url: {args.server_url}")
    print(f"capture_dir: {capture_dir}")
    print(f"task_spec: {args.task_spec!r}")

    if not args.no_health_probe:
        probe_health(args.server_url)

    rgb_png, depth_npy, K_json = load_capture(capture_dir)
    print(f"rgb_png: {len(rgb_png)} bytes, depth_npy: {len(depth_npy)} bytes, K: {K_json}")
    print("posting /grasp (this can take >30s the first time)...")

    if args.T_base_gripper:
        print(f"T_base_gripper: {args.T_base_gripper}")
    payload = post_grasp(
        server_url=args.server_url,
        rgb_png_bytes=rgb_png,
        depth_npy_bytes=depth_npy,
        K_json=K_json,
        task_spec=args.task_spec,
        frame_id=args.frame_id,
        top_k=args.top_k,
        num_candidates=args.num_candidates,
        timeout_s=args.timeout_s,
        T_base_gripper_json=args.T_base_gripper,
    )

    grasps = payload.get("grasps") or []
    print(
        f"\nrun_id={payload.get('run_id')}  "
        f"elapsed_ms={payload.get('elapsed_ms')}  "
        f"grasps={len(grasps)}  "
        f"frame_id={payload.get('frame_id')}"
    )
    print(f"artifacts: {payload.get('run_dir')}/report.html")
    if not grasps:
        print("WARNING: server returned no grasps")
        return 1
    print("\ntop grasps:")
    for rank, grasp in enumerate(grasps, start=1):
        print(summarize_grasp(grasp, rank))
    return 0


if __name__ == "__main__":
    sys.exit(main())
