"""Async client for the persistent Contact-GraspNet worker subprocess.

Spawns ``grasp_server/cgn_worker.py`` inside the ``contact_graspnet`` conda env,
waits for its ``{"event":"ready"}`` line, and exposes a single-flight
``predict(input_npz, output_npz)`` coroutine guarded by an ``asyncio.Lock`` so
only one inference runs at a time (GPU contention with VLM / SAM2).
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
from pathlib import Path
from typing import Any

from .config import ServerConfig


WORKER_SCRIPT = Path(__file__).resolve().parent / "cgn_worker.py"


class WorkerError(RuntimeError):
    """Raised when the worker reports an error or dies unexpectedly."""


class CgnWorker:
    def __init__(self, config: ServerConfig) -> None:
        self._config = config
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._stderr_task: asyncio.Task[None] | None = None
        self._ready = False
        self._stderr_buffer: list[str] = []
        self._stderr_max_lines = 200

    @property
    def ready(self) -> bool:
        return self._ready and self._proc is not None and self._proc.returncode is None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def recent_stderr(self) -> list[str]:
        return list(self._stderr_buffer[-50:])

    def _build_command(self) -> list[str]:
        cfg = self._config
        ckpt = str(cfg.cgn_ckpt)
        worker_args = [str(WORKER_SCRIPT), "--ckpt-dir", ckpt]
        if cfg.cgn_conda_python is not None:
            return [str(cfg.cgn_conda_python), "-u", *worker_args]

        conda_exe = shutil.which("conda")
        if conda_exe is None:
            raise FileNotFoundError(
                "conda not found on PATH. Set CONTACT_GRASPNET_PYTHON to the python "
                "interpreter inside the contact_graspnet env, or install conda."
            )
        return [
            conda_exe,
            "run",
            "--no-capture-output",
            "-n",
            cfg.cgn_env,
            "python",
            "-u",
            *worker_args,
        ]

    def _build_env(self) -> dict[str, str]:
        cfg = self._config
        cgn_pkg_root = cfg.cgn_repo
        cgn_inner = cgn_pkg_root / "contact_graspnet_pytorch"
        env = os.environ.copy()
        extra_paths = [str(cgn_pkg_root), str(cgn_inner)]
        existing = env.get("PYTHONPATH")
        if existing:
            env["PYTHONPATH"] = os.pathsep.join([*extra_paths, existing])
        else:
            env["PYTHONPATH"] = os.pathsep.join(extra_paths)
        return env

    async def start(self, *, ready_timeout: float = 180.0) -> None:
        if self._proc is not None and self._proc.returncode is None:
            return
        cfg = self._config
        if not cfg.cgn_ckpt.is_dir():
            raise FileNotFoundError(f"Contact-GraspNet checkpoint dir not found: {cfg.cgn_ckpt}")
        if not WORKER_SCRIPT.is_file():
            raise FileNotFoundError(f"Worker script missing: {WORKER_SCRIPT}")

        cwd = cfg.cgn_repo / "contact_graspnet_pytorch"
        if not cwd.is_dir():
            cwd = cfg.cgn_repo
        command = self._build_command()
        env = self._build_env()

        self._proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._ready = False
        self._stderr_buffer.clear()
        self._stderr_task = asyncio.create_task(self._drain_stderr(), name="cgn-worker-stderr")
        try:
            await asyncio.wait_for(self._await_ready(), timeout=ready_timeout)
        except asyncio.TimeoutError:
            await self.stop()
            raise WorkerError(
                f"CGN worker did not signal ready within {ready_timeout:.0f}s. "
                f"Recent stderr: {self.recent_stderr()}"
            )

    async def _drain_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        try:
            while True:
                line = await self._proc.stderr.readline()
                if not line:
                    return
                text = line.decode("utf-8", errors="replace").rstrip()
                self._stderr_buffer.append(text)
                if len(self._stderr_buffer) > self._stderr_max_lines:
                    del self._stderr_buffer[: -self._stderr_max_lines]
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            self._stderr_buffer.append(f"[client] stderr drain error: {exc}")

    async def _await_ready(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                raise WorkerError(
                    f"CGN worker exited before signaling ready. Recent stderr: {self.recent_stderr()}"
                )
            try:
                payload = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                # Older versions of conda/run sometimes interleave non-JSON
                # banner lines; ignore those until the worker emits valid JSON.
                continue
            event = payload.get("event")
            if event == "ready":
                self._ready = True
                return
            if event == "fatal":
                raise WorkerError(f"CGN worker fatal during startup: {payload.get('error')}")

    async def stop(self, *, timeout: float = 10.0) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.returncode is None:
            try:
                if proc.stdin is not None and not proc.stdin.is_closing():
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.send_signal(signal.SIGTERM)
                await asyncio.wait_for(proc.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except (asyncio.CancelledError, Exception):
                pass
        self._stderr_task = None
        self._proc = None
        self._ready = False

    async def predict(self, input_npz: Path, output_npz: Path) -> dict[str, Any]:
        """Send one request to the worker, return its response dict.

        Holds an asyncio lock so only one inference runs at a time even if the
        FastAPI app receives concurrent requests.
        """
        async with self._lock:
            if not self.ready:
                await self.start()
            assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None

            request = json.dumps({
                "input_npz": str(input_npz.resolve()),
                "output_npz": str(output_npz.resolve()),
            }) + "\n"
            try:
                self._proc.stdin.write(request.encode("utf-8"))
                await self._proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                await self.stop()
                raise WorkerError(f"CGN worker stdin closed: {exc}") from exc

            response: dict[str, Any] | None = None
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    await self.stop()
                    raise WorkerError(
                        "CGN worker exited unexpectedly while waiting for a response. "
                        f"Recent stderr: {self.recent_stderr()}"
                    )
                stripped = line.strip()
                if not stripped or not stripped.startswith(b"{"):
                    continue
                try:
                    response = json.loads(stripped.decode("utf-8"))
                    break
                except json.JSONDecodeError:
                    continue
            assert response is not None

            if response.get("status") != "ok":
                raise WorkerError(
                    f"CGN worker error: {response.get('error', 'unknown')}"
                )
            return response
