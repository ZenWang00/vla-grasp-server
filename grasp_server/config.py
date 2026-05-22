"""Centralized environment configuration for the grasp server."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _env_path(name: str, default: Path | str) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else Path(default).expanduser().resolve()


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw if raw else default


def _env_matrix4x4(name: str) -> np.ndarray | None:
    """Parse a 4×4 homogeneous transform from a JSON env var, or return None if unset."""
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON: {exc}") from exc
    mat = np.asarray(payload, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"{name} must be a 4×4 matrix, got shape {mat.shape}")
    if not np.all(np.isfinite(mat)):
        raise ValueError(f"{name} contains non-finite values")
    return mat


@dataclass(frozen=True)
class ServerConfig:
    output_base: Path
    host: str
    port: int
    provider: str
    model: str

    cgn_repo: Path
    cgn_env: str
    cgn_ckpt: Path
    cgn_conda_python: Path | None

    # Static hand-eye calibration: transforms a point from camera frame to gripper frame.
    # Set via GRASP_T_GRIPPER_CAMERA env var as a JSON 4×4 matrix (row-major).
    # Example: '[[1,0,0,0.05],[0,1,0,0.02],[0,0,1,-0.03],[0,0,0,1]]'
    T_gripper_camera: np.ndarray | None

    @classmethod
    def from_env(cls) -> "ServerConfig":
        repo_root = Path(__file__).resolve().parent.parent
        cgn_repo = _env_path("CONTACT_GRASPNET_REPO", Path.home() / "contact_graspnet_pytorch")
        cgn_ckpt_default = cgn_repo / "contact_graspnet_pytorch" / "checkpoints" / "contact_graspnet"
        cgn_ckpt = _env_path("CONTACT_GRASPNET_CKPT", cgn_ckpt_default)

        conda_python_raw = os.environ.get("CONTACT_GRASPNET_PYTHON")
        cgn_conda_python = (
            Path(conda_python_raw).expanduser().resolve()
            if conda_python_raw
            else None
        )

        return cls(
            output_base=_env_path("GRASP_SERVER_OUTPUT_BASE", repo_root / "output_vg"),
            host=_env_str("GRASP_SERVER_HOST", "0.0.0.0"),
            port=int(_env_str("GRASP_SERVER_PORT", "8765")),
            provider=_env_str("GRASP_SERVER_PROVIDER", "gemini"),
            model=_env_str("GRASP_SERVER_MODEL", "gemini-robotics-er-1.6-preview"),
            cgn_repo=cgn_repo,
            cgn_env=_env_str("CONTACT_GRASPNET_ENV", "contact_graspnet"),
            cgn_ckpt=cgn_ckpt,
            cgn_conda_python=cgn_conda_python,
            T_gripper_camera=_env_matrix4x4("GRASP_T_GRIPPER_CAMERA"),
        )
