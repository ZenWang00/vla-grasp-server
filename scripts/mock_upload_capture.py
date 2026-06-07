#!/usr/bin/env python3
"""Mock the ROS2 node's camera upload step.

One-shot mode (default): upload once so the server has a pending_capture.

Daemon mode (--daemon): keep running, poll /poll_capture_request, and
re-upload every time the web UI's "Run Grasp" / "Run Align" button is
clicked. This is the correct mode for frontend testing — the UI always
triggers a fresh capture request before running the pipeline.

Usage:
    # One-shot (just seed the server):
    .venv/bin/python scripts/mock_upload_capture.py \\
        --capture-dir captures/20260417_120019

    # Daemon (respond to every UI button click):
    .venv/bin/python scripts/mock_upload_capture.py \\
        --capture-dir captures/20260417_120019 --daemon
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image


def load_capture(capture_dir: Path) -> tuple[bytes, bytes, str]:
    """Return (rgb_jpeg_bytes, depth_npy_bytes, K_json)."""
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
        raise KeyError(f"{npy_path} must have 'depth' and 'K' keys")

    depth = np.ascontiguousarray(np.asarray(data["depth"], dtype=np.float32))
    K = np.asarray(data["K"], dtype=np.float64).reshape(3, 3)

    rgb_bytes = color_path.read_bytes()

    pil = Image.open(io.BytesIO(rgb_bytes))
    if pil.size[::-1] != depth.shape:
        raise ValueError(
            f"color_preview.jpg size {pil.size[::-1]} != depth shape {depth.shape}"
        )

    depth_buf = io.BytesIO()
    np.save(depth_buf, depth, allow_pickle=False)
    return rgb_bytes, depth_buf.getvalue(), json.dumps(K.tolist())


def upload_capture(
    server_url: str,
    rgb_bytes: bytes,
    depth_bytes: bytes,
    K_json: str,
    frame_id: str,
    timeout_s: float,
) -> dict:
    url = server_url.rstrip("/") + "/upload_capture"
    r = requests.post(
        url,
        files={
            "rgb": ("color_preview.jpg", rgb_bytes, "image/jpeg"),
            "depth": ("depth.npy", depth_bytes, "application/octet-stream"),
        },
        data={"K": K_json, "frame_id": frame_id},
        timeout=timeout_s,
    )
    if r.status_code != 200:
        try:
            body = r.json()
        except Exception:
            body = r.text[:500]
        raise RuntimeError(f"upload_capture HTTP {r.status_code}: {body}")
    return r.json()


def push_frame(server_url: str, rgb_bytes: bytes, timeout_s: float) -> None:
    url = server_url.rstrip("/") + "/push_frame"
    r = requests.post(
        url,
        files={"frame": ("frame.jpg", rgb_bytes, "image/jpeg")},
        timeout=timeout_s,
    )
    if r.status_code != 200:
        print(f"  warning: /push_frame returned HTTP {r.status_code} (non-fatal)")


def do_upload(
    server_url: str,
    rgb_bytes: bytes,
    depth_bytes: bytes,
    K_json: str,
    frame_id: str,
    timeout_s: float,
    push: bool,
) -> None:
    result = upload_capture(server_url, rgb_bytes, depth_bytes, K_json, frame_id, timeout_s)
    print(f"  uploaded: shape={result.get('shape')}  uploaded_at={result.get('uploaded_at'):.3f}")
    if push:
        push_frame(server_url, rgb_bytes, timeout_s)


def run_daemon(
    server_url: str,
    rgb_bytes: bytes,
    depth_bytes: bytes,
    K_json: str,
    frame_id: str,
    timeout_s: float,
    poll_interval_s: float,
    push: bool,
) -> None:
    print(f"daemon: polling {server_url}/poll_capture_request every {poll_interval_s}s")
    print("Press Ctrl-C to stop.\n")
    while True:
        try:
            r = requests.get(
                server_url.rstrip("/") + "/poll_capture_request",
                timeout=timeout_s,
            )
            if r.status_code == 200 and r.json().get("requested"):
                print(f"[{time.strftime('%H:%M:%S')}] capture requested → uploading ...")
                do_upload(server_url, rgb_bytes, depth_bytes, K_json, frame_id, timeout_s, push)
        except Exception as exc:
            print(f"  error: {type(exc).__name__}: {exc}")
        time.sleep(poll_interval_s)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=Path("captures/20260417_120019"),
        help="Capture folder with camera_data.npy + color_preview.jpg.",
    )
    parser.add_argument("--server-url", type=str, default="http://localhost:8765")
    parser.add_argument("--frame-id", type=str, default="camera_color_optical_frame")
    parser.add_argument(
        "--daemon", action="store_true",
        help="Keep running; re-upload whenever the UI requests a new capture.",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=0.3,
        help="Seconds between /poll_capture_request checks in daemon mode (default 0.3).",
    )
    parser.add_argument(
        "--no-push-frame", action="store_true",
        help="Skip POST /push_frame (live preview won't update).",
    )
    parser.add_argument("--timeout-s", type=float, default=10.0)
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir).resolve()
    if not capture_dir.is_dir():
        raise SystemExit(f"--capture-dir not found: {capture_dir}")

    print(f"server : {args.server_url}")
    print(f"capture: {capture_dir}")

    rgb_bytes, depth_bytes, K_json = load_capture(capture_dir)
    K = json.loads(K_json)
    print(f"rgb    : {len(rgb_bytes)} bytes")
    print(f"depth  : {len(depth_bytes)} bytes")
    print(f"K      : fx={K[0][0]:.1f}  fy={K[1][1]:.1f}  cx={K[0][2]:.1f}  cy={K[1][2]:.1f}")

    push = not args.no_push_frame

    # Always do an initial upload so the server has a pending_capture immediately.
    print("\ninitial upload → POST /upload_capture ...")
    do_upload(args.server_url, rgb_bytes, depth_bytes, K_json, args.frame_id, args.timeout_s, push)

    if args.daemon:
        run_daemon(
            server_url=args.server_url,
            rgb_bytes=rgb_bytes,
            depth_bytes=depth_bytes,
            K_json=K_json,
            frame_id=args.frame_id,
            timeout_s=args.timeout_s,
            poll_interval_s=args.poll_interval,
            push=push,
        )
    else:
        print(f"\ndone — open {args.server_url} to test via the web UI")
        print("tip: use --daemon so the UI's capture step doesn't time out on each click.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
