# vla-grasp-server

## Project Overview

This is the **Semester Project Final Release (v1.0.0)** - the grasp server component of the VLA (Vision-Language-Action) robotic grasp generation system.

### Related Components
- **Grasp Pose Generation Engine**: https://github.com/ZenWang00/Grasp-Pose-Generation
- **Project Presentation**: https://docs.google.com/presentation/d/1xuitCtljjJ73u928OtHOnfCbR0izvSX1LKqYPRg62_A/edit?usp=sharing

For a complete understanding of the system architecture and results, please refer to the presentation and the accompanying Grasp-Pose-Generation repository.

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
- `ipykernel`: required for interactive notebook development and VS Code notebook kernel support

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

- `depth` (full scene, far background beyond `--cgn-depth-clip` — default 1.5 m — zeroed), `K`
- `rgb` (BGR, matches the existing Contact-GraspNet loader convention)
- `segmap = tight_grasp_mask` for that candidate — Contact-GraspNet's segment-only branch then isolates grasps to the graspable part
- `global_mask` (extra key) — the SAM2 Global whole-object mask, carried through for downstream collision / context use

Important:

- the export is optional and requires `--enable-sam2`
- file naming is controlled by `--contact-graspnet-export-template` (default `contact_graspnet_input_{idx:03d}.npz`); the template must contain `{idx`
- Contact-GraspNet itself is not invoked here; the NPZs are consumed by an external Contact-GraspNet environment
- `segmap` is the tight grasp mask (SAM2 local, clipped to the VLM `grasp_region_box`), not the whole-object `global_mask`
- exported `depth` is the **full scene** clipped at `--cgn-depth-clip` (default 1.5 m): CGN needs the support plane (table) in its point cloud to suppress bottom-up approach directions — its training only avoids grasps that pass through visible scene points. Target focusing is still guaranteed by `segmap` via `local_regions` / `filter_grasps`
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

### Architecture overview

A FastAPI process runs inside this repo's `.venv` and exposes a **Web UI** (served at `GET /`) plus a set of REST endpoints used by the Web UI and the ROS2 `GraspPoseClientNode` in parallel.

A long-lived `cgn_worker.py` subprocess is launched at startup inside the `contact_graspnet` conda env. It loads the Contact-GraspNet model once and receives inference requests over stdin/stdout as JSON lines. An `asyncio.Lock` serializes each generation request so the VLM, SAM2, and Contact-GraspNet stages never compete for the GPU.

### Launch

```bash
export GEMINI_API_KEY=YOUR_KEY   # or GOOGLE_API_KEY
./scripts/run_server.sh
```

Environment overrides (see `scripts/run_server.sh`):

- `GRASP_SERVER_HOST` / `GRASP_SERVER_PORT` (default `0.0.0.0` / `8765`)
- `GRASP_SERVER_OUTPUT_BASE` (default `output_vg/`; each run writes one subdir)
- `GRASP_SERVER_PROVIDER` / `GRASP_SERVER_MODEL` (default `gemini` / `gemini-robotics-er-1.6-preview`)
- `CONTACT_GRASPNET_REPO` / `CONTACT_GRASPNET_ENV` / `CONTACT_GRASPNET_CKPT` — shared with `scripts/run_vg_and_contact_graspnet.sh`
- `CONTACT_GRASPNET_PYTHON` — optional explicit python interpreter to skip `conda run`

### Multi-phase workflow

The normal end-to-end flow involves six phases coordinated between the Web UI, the server, and the ROS2 client node:

```
Phase 1 — Capture
  Web UI  →  POST /request_capture          set capture_requested flag
  ROS2    →  GET  /poll_capture_request      (2 Hz) consume flag
  ROS2    →  POST /upload_capture           upload rgb + depth + K + frame_id

Phase 2 — Grasp generation (choose one)
  Web UI  →  POST /run_grasp                Option A: VLM + SAM2 + Contact-GraspNet
  Web UI  →  POST /run_align               Option B: VLM align-point only (no SAM2/CGN)

Phase 3 — Server-side pre-IK filter
  Web UI  →  POST /trigger_ik_check
             server runs filter_grasps() (hard width/collision + soft scores)
             queues filtered grasps as pending_publish with mode="ik_check"

Phase 4 — ROS2 IK feasibility check
  ROS2    →  GET  /poll_publish             picks up mode="ik_check" payload
             runs Pinocchio two-stage IK for each candidate
  ROS2    →  POST /submit_ik_result         returns IK-passing grasps

Phase 5 — Best-grasp selection
  Web UI  →  GET  /ik_result_status         poll until ready=true
  Web UI  →  POST /select_and_execute       rank_grasps() → composite score → best
             queues best grasp as pending_publish with mode="execute"

Phase 6 — Execution
  ROS2    →  GET  /poll_publish             picks up mode="execute" payload
             transforms best grasp camera→base frame, publishes PoseStamped/PoseArray
```

### Capture endpoints

| Endpoint | Caller | Description |
| --- | --- | --- |
| `POST /request_capture` | Web UI | Sets server flag asking ROS2 to take a snapshot |
| `GET /poll_capture_request` | ROS2 (2 Hz) | Returns `{requested: true}` once, then clears flag |
| `POST /upload_capture` | ROS2 | Uploads `rgb` (PNG/JPG), `depth` (.npy float32 H×W m), `K` (JSON 3×3), `frame_id` |
| `GET /capture_status` | Web UI | Returns `{uploaded_at}` timestamp of last capture |
| `POST /push_frame` | ROS2 | Push a live JPEG for the Web UI preview stream |
| `GET /latest_frame` | Web UI | Returns the most recent JPEG from ROS2 |

### Generation endpoints

#### `POST /run_grasp` — Option A: VLM + SAM2 + Contact-GraspNet

| field | type | default | notes |
| --- | --- | --- | --- |
| `task_spec` | form | required | natural-language target, e.g. `"grasp the cup"` |
| `num_candidates` | form | `1` | VLM `grasp_region_box` proposals |
| `top_k` | form | `1` | top-K grasps to return |
| `provider` / `model` | form | server default | per-request VLM override |

Response 200: `{run_id, run_dir, frame_id, elapsed_ms, num_candidates, top_k, grasps[], grasp_viz}`

#### `POST /run_align` — Option B: VLM align-point only (lightweight)

Same form fields as `/run_grasp` except no `top_k`. The VLM returns a 2D alignment point + gripper angle; the server back-projects the depth to get a 6-DoF pose. Approach direction is fixed to camera `+Z`. Does not use SAM2 or Contact-GraspNet.

Response 200: same schema as `/run_grasp`.

#### GraspDict schema (items in `grasps[]`)

| field | type | notes |
| --- | --- | --- |
| `score` | float | pipeline-specific quality proxy |
| `model_confidence` | float \| null | CGN confidence (Option A only) |
| `pose_4x4` | 4×4 list | camera-frame SE(3); col-0=closing, col-1=lateral, col-2=approach, col-3=position |
| `position_xyz` | [x, y, z] | camera frame, metres |
| `quaternion_xyzw` | [qx, qy, qz, qw] | derived from pose_4x4 rotation |
| `width_m` | float | gripper opening width in metres |
| `approach_dir_xyz` | [ax, ay, az] | unit approach vector (camera frame) |
| `source` | dict | `candidate_index`, `segment_id`, `grasp_index`, `predictions_npz` |

### Filter / IK / scoring endpoints

| Endpoint | Caller | Description |
| --- | --- | --- |
| `POST /trigger_ik_check` | Web UI | Pre-filter grasps (width + collision hard filters, soft scores); returns `{trace_id, num_before_filter, num_after_filter}` |
| `GET /ik_result_status?trace_id=…` | Web UI (poll) | `{ready, count}` — ready when ROS2 has submitted IK results |
| `POST /submit_ik_result` | ROS2 | Body: `{run_id, trace_id, grasps: [IK-passing GraspDicts]}` |
| `POST /select_and_execute` | Web UI | Body: `{trace_id}` — scores IK-passing grasps, picks best, queues for execution; returns `{best_index, scored_grasps, ranked_indices}` |
| `GET /poll_publish` | ROS2 (2 Hz) | Returns `{mode, grasps[], trace_id, …}` once then clears; `mode` is `"ik_check"` or `"execute"` |

### Visualisation endpoints

| Endpoint | Description |
| --- | --- |
| `GET /latest_image` | Most recent capture RGB |
| `GET /grasp_viz_image` | 2-D grasp-arrow overlay on last capture |
| `GET /grasp_viz_best_image` | Gold-arrow best-grasp overlay (after select_and_execute) |
| `GET /grasp_viz_3d` | Interactive 3-D point-cloud + grasp frames (HTML) |

### `GET /health`

Returns `{status, worker_ready, worker_pid, last_request_ms, last_run_dir, config, recent_worker_stderr}`.

### Legacy single-shot endpoint

`POST /grasp` (multipart: `rgb`, `depth`, `K`, `task_spec`, `frame_id`, `top_k`, `num_candidates`) runs the full Option A pipeline in one request and returns grasps directly. This endpoint bypasses the capture / IK-check / execution phases and is kept for smoke-testing and backward compatibility.

### Smoke test (recommended)

`scripts/smoke_test_grasp.py` loads an existing capture folder, calls `/health`
to confirm the CGN worker is up, then POSTs to the legacy `/grasp` endpoint to
isolate server-side issues before bringing up ROS2:

```bash
.venv/bin/python scripts/smoke_test_grasp.py \
    --capture-dir captures/20260417_120019 \
    --task-spec "Target: the blue bottle"
```

### What is sent to the VLM

Sent remotely:

- exactly one RGB image
- the text prompt (`task-spec`)

Not sent remotely:

- depth map
- camera intrinsics `K`
- local ROI / point-cloud outputs
