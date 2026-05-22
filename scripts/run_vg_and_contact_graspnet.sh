#!/usr/bin/env bash
# Run vg_roi_pipeline, then Contact-GraspNet inference on every exported contact_graspnet_input_*.npz.
#
# Usage (same flags as vg_roi_pipeline.py):
#   ./scripts/run_vg_and_contact_graspnet.sh \
#     --capture-dir captures/20260417_120019 \
#     --task-spec "Target: the blue bottle" \
#     --provider gemini \
#     --model gemini-robotics-er-1.6-preview \
#     --enable-sam2 \
#     --sam2-device cpu \
#     --export-contact-graspnet-input \
#     --num-candidates 1
#
# Environment overrides:
#   CONTACT_GRASPNET_REPO  default: ~/contact_graspnet_pytorch (repo root with setup.py)
#   CONTACT_GRASPNET_DIR   default: $CONTACT_GRASPNET_REPO/contact_graspnet_pytorch
#   CONTACT_GRASPNET_ENV   default: contact_graspnet  (conda env name)
#   SKIP_CONTACT_GRASPNET  set to 1 to only run vg_roi_pipeline
#   RUN_GRASP_REPORT       set to 1 to also run scripts/generate_grasp_report.py per candidate

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CGN_REPO="${CONTACT_GRASPNET_REPO:-$HOME/contact_graspnet_pytorch}"
CGN_DIR="${CONTACT_GRASPNET_DIR:-$CGN_REPO/contact_graspnet_pytorch}"
CGN_ENV="${CONTACT_GRASPNET_ENV:-contact_graspnet}"
VG_PYTHON="${VG_PYTHON:-$REPO_ROOT/.venv/bin/python}"

if [[ ! -x "$VG_PYTHON" ]]; then
  echo "VG_PYTHON not found: $VG_PYTHON (create .venv or set VG_PYTHON)" >&2
  exit 1
fi

cd "$REPO_ROOT"

echo "==> [1/2] vg_roi_pipeline"
VG_LOG="$(mktemp)"
trap 'rm -f "$VG_LOG"' EXIT

set +e
"$VG_PYTHON" vg_roi_pipeline.py "$@" 2>&1 | tee "$VG_LOG"
VG_STATUS=${PIPESTATUS[0]}
set -e
if [[ "$VG_STATUS" -ne 0 ]]; then
  exit "$VG_STATUS"
fi

OUT_DIR="$(grep -E 'Outputs in:' "$VG_LOG" | tail -1 | sed -E 's/.*Outputs in: //')"
if [[ -z "$OUT_DIR" || ! -d "$OUT_DIR" ]]; then
  echo "Could not determine pipeline output directory from log." >&2
  exit 1
fi
echo "Pipeline output: $OUT_DIR"

if [[ "${SKIP_CONTACT_GRASPNET:-0}" == "1" ]]; then
  echo "SKIP_CONTACT_GRASPNET=1, skipping Contact-GraspNet."
  exit 0
fi

shopt -s nullglob
NPZ_FILES=("$OUT_DIR"/contact_graspnet_input_*.npz)
shopt -u nullglob
if [[ ${#NPZ_FILES[@]} -eq 0 ]]; then
  echo "No contact_graspnet_input_*.npz under $OUT_DIR." >&2
  echo "Re-run with --export-contact-graspnet-input (and --enable-sam2)." >&2
  exit 1
fi

if [[ ! -f "$CGN_DIR/inference.py" ]]; then
  echo "Contact-GraspNet inference.py not found: $CGN_DIR/inference.py" >&2
  echo "Set CONTACT_GRASPNET_DIR to your contact_graspnet_pytorch/contact_graspnet_pytorch path." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -z "$CONDA_BASE" || ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  echo "conda not found; activate $CGN_ENV yourself or set CONTACT_GRASPNET_ENV." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CGN_ENV"

# Step 1 may leave vla-grasp-server .venv on PATH; conda activate alone does not
# always override `python`. Always use the env interpreter explicitly.
CGN_PYTHON="${CONTACT_GRASPNET_PYTHON:-$CONDA_PREFIX/bin/python}"
if [[ ! -x "$CGN_PYTHON" ]]; then
  echo "Contact-GraspNet python not found: $CGN_PYTHON" >&2
  exit 1
fi
export PYTHONPATH="${CGN_REPO}${PYTHONPATH:+:$PYTHONPATH}"

echo "==> [2/2] Contact-GraspNet (${#NPZ_FILES[@]} input(s), env=$CGN_ENV, python=$CGN_PYTHON)"
cd "$CGN_DIR"
for npz in "${NPZ_FILES[@]}"; do
  echo "    inference: $npz"
  "$CGN_PYTHON" inference.py --np_path "$npz" --no_vis
done

if [[ "${RUN_GRASP_REPORT:-0}" == "1" ]]; then
  echo "==> [3/3] HTML grasp report(s)"
  cd "$REPO_ROOT"
  for npz in "${NPZ_FILES[@]}"; do
    stem="${npz%.npz}"
    pred="$OUT_DIR/predictions_${stem##*/}.npz"
    if [[ ! -f "$pred" ]]; then
      echo "    skip report (missing $pred)"
      continue
    fi
    "$VG_PYTHON" scripts/generate_grasp_report.py \
      --run-dir "$OUT_DIR" \
      --predictions-npz "$pred"
  done
fi

echo "Done. Outputs: $OUT_DIR"
