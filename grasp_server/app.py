"""FastAPI app: RGBD -> VLM + SAM2 + Contact-GraspNet -> top-K 6-DoF grasps.

Endpoints:

- ``POST /grasp`` — multipart form (``rgb``, ``depth``, ``K``, ``task_spec``,
  optional ``top_k``, ``num_candidates``, ``provider``, ``model``) returns a
  JSON list of grasps in the camera frame.
- ``GET /health`` — worker liveness, configuration summary.

Concurrency: a single ``asyncio.Lock`` serializes the full request handler so
the VLM, SAM2 and CGN stages never compete for the GPU simultaneously.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from vg_pipeline import new_run_id

from .cgn_client import CgnWorker, WorkerError
from .config import ServerConfig
from .grasp_selection import predicted_npz_paths, select_top_k_grasps
from .pipeline_runner import run_vg_pipeline
from .request_handling import materialize_request


logger = logging.getLogger("grasp_server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = ServerConfig.from_env()
    worker = CgnWorker(config)
    app.state.config = config
    app.state.worker = worker
    app.state.request_lock = asyncio.Lock()
    app.state.last_request_ms = None
    app.state.last_run_dir = None
    logger.info(
        "starting Contact-GraspNet worker (env=%s, ckpt=%s)", config.cgn_env, config.cgn_ckpt
    )
    try:
        await worker.start()
    except Exception:
        logger.exception("CGN worker failed to start; the /grasp endpoint will return 503")
    try:
        yield
    finally:
        logger.info("shutting down CGN worker")
        await worker.stop()


app = FastAPI(
    title="VLA Grasp Server",
    version="0.1.0",
    description=(
        "Wraps the visual-grounding + SAM2 + Contact-GraspNet pipeline behind a "
        "single POST /grasp endpoint."
    ),
    lifespan=lifespan,
)


def _get_state(app: FastAPI):
    return app.state.config, app.state.worker, app.state.request_lock


@app.get("/health")
async def health() -> dict[str, Any]:
    config: ServerConfig = app.state.config
    worker: CgnWorker = app.state.worker
    return {
        "status": "ok" if worker.ready else "degraded",
        "worker_ready": worker.ready,
        "worker_pid": worker.pid,
        "last_request_ms": app.state.last_request_ms,
        "last_run_dir": str(app.state.last_run_dir) if app.state.last_run_dir else None,
        "config": {
            "provider": config.provider,
            "model": config.model,
            "output_base": str(config.output_base),
            "cgn_env": config.cgn_env,
            "cgn_ckpt": str(config.cgn_ckpt),
        },
        "recent_worker_stderr": worker.recent_stderr(),
    }


@app.post("/grasp")
async def grasp(
    rgb: UploadFile = File(..., description="RGB image (PNG/JPG)."),
    depth: UploadFile = File(..., description="Depth as .npy float32 HxW in meters."),
    K: str = Form(..., description='3x3 intrinsics, JSON e.g. "[[fx,0,cx],[0,fy,cy],[0,0,1]]".'),
    task_spec: str = Form(..., description='Natural-language task, e.g. "Target: the cup".'),
    frame_id: str = Form(
        ...,
        description=(
            "TF frame the returned camera-frame poses are expressed in. "
            "Typically copied from the source Image message header (e.g. "
            "'camera_color_optical_frame' for realsense2_camera)."
        ),
    ),
    top_k: int = Form(1, description="How many top-scoring grasps to return."),
    num_candidates: int = Form(
        1,
        description=(
            "How many grasp_region_box proposals the VLM emits before CGN scoring."
        ),
    ),
    provider: str | None = Form(None, description="Optional VLM provider override."),
    model: str | None = Form(None, description="Optional VLM model override."),
    T_base_camera: str | None = Form(
        None,
        description=(
            "Optional 4x4 camera-to-base transform at image capture time, JSON encoded "
            "(queried from the ROS2 TF tree at the moment of image capture). "
            "When provided, each grasp also includes a 'base_frame' field with the pose "
            "expressed in the robot base frame."
        ),
    ),
) -> JSONResponse:
    config, worker, request_lock = _get_state(app)
    if not worker.ready:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "contact_graspnet worker not ready",
                "recent_worker_stderr": worker.recent_stderr(),
            },
        )

    rgb_bytes = await rgb.read()
    depth_bytes = await depth.read()

    run_id = new_run_id()
    capture_dir = (config.output_base / f"api_{run_id}" / "capture").resolve()

    try:
        parsed = materialize_request(
            rgb_bytes=rgb_bytes,
            rgb_filename=rgb.filename,
            depth_bytes=depth_bytes,
            K_json=K,
            task_spec=task_spec,
            top_k=top_k,
            num_candidates=num_candidates,
            provider=provider or config.provider,
            model=model or config.model,
            frame_id=frame_id,
            capture_dir=capture_dir,
            T_base_camera_json=T_base_camera,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async with request_lock:
        t_start = time.perf_counter()
        try:
            pipeline_result = await asyncio.to_thread(
                run_vg_pipeline,
                capture_dir=parsed.capture_dir,
                task_spec=parsed.task_spec,
                out_base=config.output_base,
                run_id=run_id,
                provider=parsed.provider,
                model=parsed.model,
                num_candidates=parsed.num_candidates,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=f"pipeline file error: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"pipeline rejected the inputs: {exc}",
            ) from exc
        except Exception as exc:
            logger.exception("vg_pipeline failed")
            raise HTTPException(status_code=500, detail=f"vg_pipeline failed: {exc}") from exc

        try:
            for input_npz in pipeline_result.exported_npzs:
                output_npz = input_npz.parent / f"predictions_{input_npz.stem}.npz"
                await worker.predict(input_npz, output_npz)
        except WorkerError as exc:
            logger.exception("CGN worker error")
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "contact_graspnet worker error",
                    "message": str(exc),
                    "recent_worker_stderr": worker.recent_stderr(),
                },
            ) from exc

        prediction_npzs = predicted_npz_paths(pipeline_result.exported_npzs)
        try:
            grasps_json = await asyncio.to_thread(
                select_top_k_grasps,
                prediction_npzs,
                parsed.top_k,
                T_base_camera=parsed.T_base_camera,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"failed to load Contact-GraspNet predictions: {exc}",
            ) from exc

        elapsed_ms = int((time.perf_counter() - t_start) * 1000)

    app.state.last_request_ms = elapsed_ms
    app.state.last_run_dir = pipeline_result.run_dir

    return JSONResponse({
        "run_id": run_id,
        "run_dir": str(pipeline_result.run_dir),
        "frame_id": parsed.frame_id,
        "elapsed_ms": elapsed_ms,
        "rgb_shape": list(parsed.rgb_shape),
        "depth_shape": list(parsed.depth_shape),
        "num_candidates": parsed.num_candidates,
        "top_k": parsed.top_k,
        "grasps": grasps_json,
    })


def _ensure_output_base_exists() -> None:
    cfg = ServerConfig.from_env()
    cfg.output_base.mkdir(parents=True, exist_ok=True)


_ensure_output_base_exists()
