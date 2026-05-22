# vla-grasp-server

## Environment setup

This repository is intended to run in its own Python environment.

Recommended setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Dependency files:

- `requirements.in`: direct project dependencies
- `requirements.txt`: versions validated in the current project environment

Notes:

- the locked `torch` / `torchvision` versions in `requirements.txt` match the currently validated CUDA-enabled environment
- if your machine uses a different CUDA or CPU-only setup, you may need to adjust those two packages while keeping the rest of the dependency set the same
- `Contact-GraspNet` is recommended to stay in a separate environment from this repository
- `plotly` is used by the standalone HTML report generator

## VLM grounding step

This repository currently uses a single RGB scene image to query a vision-language model (VLM) for:

- an object-level detection result: `object_box` and optional `object_point`
- one or more grasp-level candidates: `grasp_region_box`, `grasp_point`, and `reasoning`

At this stage, only the RGB image is sent to the VLM. Depth and camera intrinsics from the `.npy` observation are loaded locally and used later for ROI cropping, point-cloud backprojection, and output visualization.

### Invocation notes

The current grounding step is intended to record the VLM call as a standalone stage before later segmentation or grasp generation steps.

- Backend used in the current workflow: Gemini
- Recommended model for this stage: `gemini-robotics-er-1.6-preview`
- Required credential: export either `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- Important: the VLM receives only one RGB image; `.npy` depth/K are not sent to the remote model

Example environment setup:

```bash
export GOOGLE_API_KEY=YOUR_KEY
```

or

```bash
export GEMINI_API_KEY=YOUR_KEY
```

### Example command

Run the grounding pipeline with Gemini. This will query the VLM, save the raw response, parse both object-level and grasp-level boxes, and write visualized outputs under `output_vg/`:

```bash
python vg_roi_pipeline.py \
  --npy test_data/1.npy \
  --scene-image output_rgbd/1_scene.jpg \
  --task-spec "Target: the green cup" \
  --provider gemini \
  --model gemini-robotics-er-1.6-preview
```

If the capture folder already contains `camera_data.npy` and `color_preview.jpg`, you can use:

```bash
python vg_roi_pipeline.py \
  --capture-dir captures/20260417_120218 \
  --task-spec "Target: the cup" \
  --provider gemini \
  --model gemini-robotics-er-1.6-preview
```

If you already have an aligned RGB image but want the `.npy` only for local post-processing, the current implementation still requires passing the `.npy` at this stage because the same pipeline immediately continues into ROI/depth artifacts after VLM inference.

### Inputs

- `scene-image` or `color_preview.jpg`: the only image sent to the VLM
- `.npy` observation: local depth map and camera intrinsics used after VLM inference
- `task-spec`: natural-language task description, for example `Target: the cup`

### Outputs

Each run writes a new folder under `output_vg/` and stores:

- `raw_model_text.txt`: raw VLM response
- `manifest.json`: parsed metadata and file paths (schema `vla_dual_sam2_v1`)
- `object_box_overlay.png`: full RGB image with the detected object box and point

When `--enable-sam2` is set the run also writes the dual-SAM2 / clean-3D artifacts described in the next two sections.

### Current data flow

1. Load aligned RGB + depth/K from disk.
2. Send only the RGB image to the VLM and parse `object_box` / `object_point` and per-candidate `grasp_region_box` / `grasp_point`.
3. Run SAM2 once on `object_box` (whole-object "global" mask) and once per candidate on `grasp_region_box` (graspable-part "tight" mask).
4. Build a clean per-candidate camera-frame point cloud from the tight mask combined with depth and `K`.
5. Optionally export one Contact-GraspNet-ready NPZ per candidate (`segmap = tight_grasp_mask`, plus an extra `global_mask` key for downstream collision/context).

## Dual SAM2 segmentation step

When `--enable-sam2` is set, the pipeline runs two SAM2 passes locally:

- **SAM2 Global** consumes `object_box` (and optional `object_point`) to produce the whole-object mask. Artifacts:
  - `global_mask.png`, `global_mask_overlay.png`, `global_segmap.npy`
- **SAM2 Local** runs once per VLM grasp candidate, consuming `grasp_region_box` (and optional `grasp_point`). Artifacts per candidate index `NNN`:
  - `tight_grasp_mask_NNN.png`, `tight_grasp_mask_overlay_NNN.png`, `tight_grasp_segmap_NNN.npy`

The global mask is intended for downstream scene / collision context. The tight mask drives the clean local 3D point cloud.

### Clean local 3D point cloud

For each candidate, the tight grasp mask is combined with the full-frame depth + `K` to back-project only pixels that lie inside the graspable region. Artifacts per candidate index `NNN`:

- `pure_target_pointcloud_NNN.npy`: float32 `(N, 3)` camera-frame points (only depth-valid + mask-positive pixels)
- `pure_target_pointcloud_3d_NNN.png`: offscreen 3D render (Open3D, Matplotlib fallback); skip with `--no-clean-local-3d-render`
- `pure_target_pointcloud_overlay_NNN.png`: red-dot projection of the cloud back onto the full RGB scene

### Contact-GraspNet export

When `--export-contact-graspnet-input` is set, the pipeline writes one NPZ per candidate (`contact_graspnet_input_NNN.npz` by default). Each NPZ contains:

- `depth`, `K`
- `rgb` (BGR, matches the existing Contact-GraspNet loader convention)
- `segmap = tight_grasp_mask` for that candidate — Contact-GraspNet's segment-only branch then isolates grasps to the graspable part
- `global_mask` (extra key) — the SAM2 Global whole-object mask, carried through for downstream collision / context use

Important:

- the export is optional and requires `--enable-sam2`
- file naming is controlled by `--contact-graspnet-export-template` (default `contact_graspnet_input_{idx:03d}.npz`); the template must contain `{idx`
- Contact-GraspNet itself is not invoked here; the NPZs are consumed by an external Contact-GraspNet environment
- `segmap` is the tight grasp mask (SAM2 local, clipped to the VLM `grasp_region_box`), not the whole-object `global_mask`
- exported `depth` is zeroed outside the tight mask so CGN's scene point cloud cannot pull in the full object / table context
- `global_mask` is stored only as extra metadata for downstream planners; CGN's loader ignores unknown keys

When running Contact-GraspNet `inference.py`, prefer predicting on the segmented point cloud only (not the default 0.2–0.6 m scene cube). In `contact_graspnet_pytorch/contact_graspnet_pytorch/inference.py`, call `predict_scene_grasps(..., local_regions=True, filter_grasps=True, use_cam_boxes=False)` or patch the script accordingly so `pc_regions` comes from `filter_pc_segments(pc_segments)` instead of `extract_3d_cam_boxes(full_pc, ...)`.

### Example command with dual SAM2

```bash
python vg_roi_pipeline.py \
  --capture-dir captures/20260417_120218 \
  --task-spec "Target: the cup" \
  --provider gemini \
  --model gemini-robotics-er-1.6-preview \
  --enable-sam2 \
  --sam2-model facebook/sam2.1-hiera-small
```

If needed, you can force the SAM2 device:

```bash
python vg_roi_pipeline.py \
  --capture-dir captures/20260417_120218 \
  --task-spec "Target: the cup" \
  --provider gemini \
  --model gemini-robotics-er-1.6-preview \
  --enable-sam2 \
  --sam2-model facebook/sam2.1-hiera-small \
  --sam2-device cuda
```

### Example command with Contact-GraspNet export

```bash
python vg_roi_pipeline.py \
  --capture-dir captures/20260417_120218 \
  --task-spec "Target: the cup" \
  --provider gemini \
  --model gemini-robotics-er-1.6-preview \
  --enable-sam2 \
  --sam2-model facebook/sam2.1-hiera-small \
  --export-contact-graspnet-input
```

### One-shot: pipeline + Contact-GraspNet inference

To run `vg_roi_pipeline.py` and then Contact-GraspNet `inference.py` on every exported `contact_graspnet_input_*.npz` in sequence (uses conda env `contact_graspnet` by default):

```bash
./scripts/run_vg_and_contact_graspnet.sh \
  --capture-dir captures/20260417_120019 \
  --task-spec "Target: the blue bottle" \
  --provider gemini \
  --model gemini-robotics-er-1.6-preview \
  --enable-sam2 \
  --sam2-device cpu \
  --export-contact-graspnet-input \
  --num-candidates 1
```

Optional environment variables:

- `CONTACT_GRASPNET_DIR` — path to `contact_graspnet_pytorch/contact_graspnet_pytorch` (default: `~/contact_graspnet_pytorch/contact_graspnet_pytorch`)
- `CONTACT_GRASPNET_ENV` — conda env name (default: `contact_graspnet`)
- `SKIP_CONTACT_GRASPNET=1` — only run the vision pipeline
- `RUN_GRASP_REPORT=1` — after CGN, also generate `report.html` per candidate

## Standalone HTML grasp report

After Contact-GraspNet inference writes a predictions NPZ back into the same run directory, you can generate a browser-openable HTML report without requiring a GUI session.

Expected inputs:

- one finished run directory containing `manifest.json`
- one or more `contact_graspnet_input_NNN.npz` exports
- one Contact-GraspNet predictions file such as `predictions_contact_graspnet_input_000.npz`

The report generator:

- loads the original RGB scene from `manifest.json` or falls back to the exported input NPZ
- infers the candidate index from the predictions filename (the trailing integer before `.npz`) and selects the matching Contact-GraspNet input NPZ from the manifest's per-candidate list, defaulting to candidate `000` when no match is found
- inspects the prediction NPZ schema and prints its keys and shapes
- flattens Contact-GraspNet's per-object dictionaries into a top-k grasp list
- projects grasp approach arrows back onto the 2D scene image
- renders an interactive Plotly 3D point-cloud view with grasp frames and a simple gripper wireframe
- shows per-candidate tight-mask + pure-target-pointcloud overlay cards alongside the global-mask card

Example command:

```bash
python scripts/generate_grasp_report.py \
  --run-dir output_vg/camera_data_20260428_125745_300936 \
  --predictions-npz output_vg/camera_data_20260428_125745_300936/predictions_contact_graspnet_input_000.npz \
  --top-k 12
```

Output:

- `report.html` under the run directory by default

Notes:

- the 3D frame colors are fixed as `x/base = red`, `y/lateral = green`, `z/approach = blue`
- the 2D arrows are projected from the same `z/approach` axis used in the 3D view
- if `--predictions-npz` is omitted, the script will auto-pick a single `predictions*.npz` file inside the run directory
- if the prediction schema is not recognized, the script raises an error after printing the discovered key/shape summary
- the current implementation is aligned to the Contact-GraspNet-style fields observed so far: `pred_grasps_cam`, `scores`, `contact_pts`, `pc_full`, and optional `pc_colors`
- the 3D Plotly view is rendered from the selected candidate only; multi-candidate aggregation is a planned follow-up

## Grasp HTTP API server

The end-to-end pipeline is also exposed as a single HTTP endpoint so the future ROS2 node only needs to upload an RGBD frame and parse a JSON response.

Architecture:

- A FastAPI process runs inside this repo's `.venv` and serves `POST /grasp` and `GET /health`.
- A long-lived `cgn_worker.py` subprocess is launched at startup inside the `contact_graspnet` conda env. It loads the Contact-GraspNet model once and reads inference requests over stdin/stdout as JSON lines.
- An `asyncio.Lock` serializes the full handler so the VLM, SAM2 and Contact-GraspNet stages never compete for the GPU.

### Launch

```bash
export GEMINI_API_KEY=YOUR_KEY   # or GOOGLE_API_KEY
./scripts/run_server.sh
```

Environment overrides (see `scripts/run_server.sh`):

- `GRASP_SERVER_HOST` / `GRASP_SERVER_PORT` (default `0.0.0.0` / `8765`)
- `GRASP_SERVER_OUTPUT_BASE` (default `output_vg/`; each request writes one `api_<run_id>/` subdir, same as the CLI flow)
- `GRASP_SERVER_PROVIDER` / `GRASP_SERVER_MODEL` (default `gemini` / `gemini-robotics-er-1.6-preview`)
- `CONTACT_GRASPNET_REPO` / `CONTACT_GRASPNET_ENV` / `CONTACT_GRASPNET_CKPT` — shared with `scripts/run_vg_and_contact_graspnet.sh`
- `CONTACT_GRASPNET_PYTHON` — optional explicit python interpreter to skip `conda run`

### `POST /grasp` (multipart/form-data)

| field | type | required | notes |
| --- | --- | --- | --- |
| `rgb` | file | yes | PNG/JPG, H×W×3 uint8 |
| `depth` | file | yes | `.npy` saved via `numpy.save`, float32 H×W in meters |
| `K` | form | yes | JSON-encoded 3×3 intrinsics, e.g. `"[[fx,0,cx],[0,fy,cy],[0,0,1]]"` |
| `task_spec` | form | yes | natural-language target, e.g. `"Target: the cup"` |
| `frame_id` | form | yes | TF frame the returned poses live in; ROS2 client copies this from the source `Image` message header (e.g. `camera_color_optical_frame` for `realsense2_camera`) |
| `top_k` | form | no | how many grasps to return, default `1` |
| `num_candidates` | form | no | how many `grasp_region_box` proposals the VLM emits, default `1` |
| `provider` / `model` | form | no | per-request overrides for the VLM backend |

`rgb` and `depth` must share the same `(H, W)`.

Response 200:

```json
{
  "run_id": "20260520_213000_123456",
  "run_dir": "/abs/path/output_vg/api_20260520_213000_123456",
  "frame_id": "camera_optical_frame",
  "elapsed_ms": 9123,
  "rgb_shape": [720, 1280, 3],
  "depth_shape": [720, 1280],
  "num_candidates": 1,
  "top_k": 1,
  "grasps": [
    {
      "score": 0.869,
      "pose_4x4": [[...],[...],[...],[...]],
      "position_xyz": [x, y, z],
      "quaternion_xyzw": [qx, qy, qz, qw],
      "width_m": 0.067,
      "approach_dir_xyz": [ax, ay, az],
      "source": {
        "candidate_index": 0,
        "segment_id": 1,
        "grasp_index": 167,
        "predictions_npz": "predictions_contact_graspnet_input_000.npz"
      }
    }
  ]
}
```

Pose convention (camera frame, same as Contact-GraspNet):

- `pose_4x4` column 0 = gripper **x / base** (closing direction)
- column 1 = **y / lateral**
- column 2 = **z / approach** (the direction the gripper enters the contact)
- `quaternion_xyzw` is derived from the 3×3 rotation; the future ROS2 node can drop it directly into `geometry_msgs/Pose`.

Errors:

- `400` — invalid request body (shape mismatch, malformed K, missing task_spec, …)
- `502` — Contact-GraspNet worker reported an error
- `503` — worker not yet ready (e.g. failed to load weights)

### `GET /health`

Returns `{status, worker_ready, worker_pid, last_request_ms, last_run_dir, config, recent_worker_stderr}`. Useful to confirm the persistent CGN worker is up before sending real frames.

### Smoke test (recommended)

`scripts/smoke_test_grasp.py` loads an existing capture folder, calls `/health`
to check the worker is up, then POSTs to `/grasp` exactly the way the future
ROS2 client will. Use it before bringing up ROS2 to isolate server-side issues:

```bash
.venv/bin/python scripts/smoke_test_grasp.py \
    --capture-dir captures/20260417_120019 \
    --task-spec "Target: the blue bottle"
```

A successful run prints `health: status=ok, worker_ready=True`, the run_id,
artifact path (`output_vg/api_<run_id>/report.html`), and a one-line summary of
each top-K grasp (score / position / quaternion / width).

### Smoke test with raw curl

```bash
.venv/bin/python -c "import numpy as np; d=np.load('captures/20260417_120019/camera_data.npy', allow_pickle=True).item(); np.save('/tmp/depth.npy', d['depth'])"
.venv/bin/python -c "import numpy as np, json; d=np.load('captures/20260417_120019/camera_data.npy', allow_pickle=True).item(); print(json.dumps(d['K'].tolist()))" > /tmp/K.json

curl -X POST http://localhost:8765/grasp \
  -F "rgb=@captures/20260417_120019/color_preview.jpg" \
  -F "depth=@/tmp/depth.npy" \
  -F "K=$(cat /tmp/K.json)" \
  -F "task_spec=Target: the blue bottle" \
  -F "frame_id=camera_color_optical_frame" \
  -F "top_k=1"
```

### Notes / scope

- ROS2 is out of scope here. The future ROS2 node is responsible for converting `quaternion_xyzw` + `position_xyz` into `geometry_msgs/PoseStamped`, attaching `frame_id`, and any further frame remap (e.g. flipping the approach axis for `tool0`).
- Each request still writes the same `output_vg/api_<run_id>/` artifacts the CLI produces, so the existing `generate_grasp_report.py` workflow can be used to debug an API call after the fact.

### What is sent to the VLM

Sent remotely:

- exactly one RGB image
- the text prompt (`task-spec`)

Not sent remotely:

- depth map from `.npy`
- camera intrinsics `K`
- local ROI point-cloud outputs
