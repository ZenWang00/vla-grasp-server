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
- `manifest.json`: parsed metadata and file paths
- `object_box_overlay.png`: full RGB image with the detected object box and point
- `object_mask.png`, `object_mask_overlay.png`, `object_segmap.npy` when `--enable-sam2` is used
- `cropped_rgb_XXX.png`: full RGB image with each grasp-region ROI outlined
- `cropped_depth_XXX.npy`, `roi_pointcloud_XXX.npy`, `pointcloud_3d_XXX.png`, `projected_points_on_rgb_XXX.png`

### Current data flow

1. Load aligned RGB + depth/K from disk.
2. Send only the RGB image to the VLM.
3. Parse `object_box` / `object_point` and `grasp_region_box` / `grasp_point`.
4. Use depth and intrinsics locally to crop ROIs and build point-cloud artifacts.

## SAM2 segmentation step

The pipeline can optionally run local `SAM 2` segmentation immediately after VLM grounding.

- Input image for SAM2: the original RGB scene image from disk
- Prompt source: parsed `object_box` and optional `object_point` from the VLM output
- Output artifacts:
  - `object_mask.png`: binary mask of the target object
  - `object_mask_overlay.png`: RGB image with the mask blended on top
  - `object_segmap.npy`: full-image segmap with background `0` and target object `1`

Important:

- `SAM 2` uses the original RGB image, not `object_box_overlay.png`
- `SAM 2` runs locally; no extra remote API call is introduced
- the first version keeps depth refinement as a future extension hook and writes a minimal full-image segmap

### Contact-GraspNet export

After `SAM 2` succeeds, the pipeline can optionally export a Contact-GraspNet-ready `.npz` bundle.

- Output file name by default: `contact_graspnet_input.npz`
- Exported keys:
  - `depth`
  - `K`
  - `segmap`
  - `rgb`
- The export is written from the current in-memory pipeline data, so it stays aligned with the exact RGB/depth/segmap used in that run.

Important:

- this export is optional and disabled by default
- it requires a successful `SAM 2` segmentation step
- the stored `rgb` is written in the channel order expected by the current Contact-GraspNet loader

### Example command with SAM2

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

### What is sent to the VLM

Sent remotely:

- exactly one RGB image
- the text prompt (`task-spec`)

Not sent remotely:

- depth map from `.npy`
- camera intrinsics `K`
- local ROI point-cloud outputs
