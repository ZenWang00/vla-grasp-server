"""Centralized environment configuration for the grasp server."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_path(name: str, default: Path | str) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else Path(default).expanduser().resolve()


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw if raw else default


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
        )
