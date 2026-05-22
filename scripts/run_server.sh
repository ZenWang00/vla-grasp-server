#!/usr/bin/env bash
# Launch the VLA grasp HTTP service (FastAPI + persistent Contact-GraspNet worker).
#
# Environment overrides:
#   GRASP_SERVER_HOST          default: 0.0.0.0
#   GRASP_SERVER_PORT          default: 8765
#   GRASP_SERVER_OUTPUT_BASE   default: <repo>/output_vg
#   GRASP_SERVER_PROVIDER      default: gemini
#   GRASP_SERVER_MODEL         default: gemini-robotics-er-1.6-preview
#   CONTACT_GRASPNET_REPO      default: ~/contact_graspnet_pytorch
#   CONTACT_GRASPNET_ENV       default: contact_graspnet
#   CONTACT_GRASPNET_CKPT      default: $CONTACT_GRASPNET_REPO/contact_graspnet_pytorch/checkpoints/contact_graspnet
#   CONTACT_GRASPNET_PYTHON    optional explicit python path (skips `conda run`)
#   GEMINI_API_KEY / GOOGLE_API_KEY   required for the VLM step
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VG_PYTHON="${VG_PYTHON:-$REPO_ROOT/.venv/bin/python}"

if [[ ! -x "$VG_PYTHON" ]]; then
  echo "VG_PYTHON not found: $VG_PYTHON (create .venv or set VG_PYTHON)" >&2
  exit 1
fi

HOST="${GRASP_SERVER_HOST:-0.0.0.0}"
PORT="${GRASP_SERVER_PORT:-8765}"

cd "$REPO_ROOT"
exec "$VG_PYTHON" -m uvicorn grasp_server.app:app \
  --host "$HOST" \
  --port "$PORT" \
  "$@"
