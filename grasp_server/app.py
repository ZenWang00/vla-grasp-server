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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

import numpy as np
from PIL import Image

from vg_pipeline import new_run_id
from vg_pipeline.align import parse_align_result, parse_align_results_multi
from vg_pipeline.io import load_observation_npy, resolve_capture_dir
from vg_pipeline.prompting import build_align_prompt, build_align_prompt_multi
from vg_pipeline.providers import run_vg_inference

from .align_grasp import build_align_grasp
from .cgn_client import CgnWorker, WorkerError
from .config import ServerConfig
from .grasp_filter import filter_grasps
from .grasp_scorer import rank_grasps
from .grasp_selection import predicted_npz_paths, select_top_k_grasps
from .grasp_viz import save_grasp_viz, save_grasp_viz_3d, save_grasp_viz_best
from .pipeline_runner import run_vg_pipeline
from .request_handling import materialize_capture, materialize_request
from .ui import UI_HTML


@dataclass
class PendingCapture:
    capture_dir: Path
    frame_id: str
    uploaded_at: float


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
    app.state.pending_capture = None  # PendingCapture | None
    app.state.upload_lock = asyncio.Lock()
    app.state.last_grasp_result = None  # last /run_grasp payload, for /trigger_publish
    app.state.last_grasp_viz_path = None       # Path to the latest grasp_viz.jpg
    app.state.last_grasp_viz_3d_path = None    # Path to the latest grasp_viz_3d.html
    app.state.last_grasp_viz_best_path = None  # Path to the latest grasp_viz_best.jpg (post-IK selection)
    app.state.pending_publish = None    # set by /trigger_publish or /select_and_execute, cleared by /poll_publish
    app.state.publish_lock = asyncio.Lock()
    app.state.ik_results = {}           # trace_id -> {status: "pending"|"complete", grasps, run_id, run_dir, …}
    app.state.ik_results_lock = asyncio.Lock()
    app.state.latest_frame_bytes = None  # latest JPEG pushed by the ROS2 node
    app.state.frame_lock = asyncio.Lock()
    app.state.capture_requested = False  # set by /request_capture, cleared by /poll_capture_request
    app.state.capture_lock = asyncio.Lock()
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


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(UI_HTML)


@app.post("/upload_capture")
async def upload_capture(
    rgb: UploadFile = File(..., description="RGB image (PNG/JPG)."),
    depth: UploadFile = File(..., description="Depth as .npy float32 HxW in meters."),
    K: str = Form(..., description='3x3 intrinsics JSON, e.g. "[[fx,0,cx],[0,fy,cy],[0,0,1]]".'),
    frame_id: str = Form(
        "camera_color_optical_frame",
        description="TF frame of the camera.",
    ),
) -> JSONResponse:
    config: ServerConfig = app.state.config
    rgb_bytes = await rgb.read()
    depth_bytes = await depth.read()

    capture_dir = (config.output_base / "pending_capture").resolve()
    try:
        result = materialize_capture(
            rgb_bytes=rgb_bytes,
            rgb_filename=rgb.filename,
            depth_bytes=depth_bytes,
            K_json=K,
            frame_id=frame_id,
            capture_dir=capture_dir,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async with app.state.upload_lock:
        app.state.pending_capture = PendingCapture(
            capture_dir=result.capture_dir,
            frame_id=result.frame_id,
            uploaded_at=time.time(),
        )

    logger.info("upload_capture: saved %s frame_id=%s", result.rgb_shape, result.frame_id)
    return JSONResponse({
        "status": "ok",
        "shape": list(result.rgb_shape),
        "frame_id": result.frame_id,
        "uploaded_at": app.state.pending_capture.uploaded_at,
    })


@app.get("/latest_image")
async def latest_image() -> Response:
    pending: PendingCapture | None = app.state.pending_capture
    if pending is None:
        raise HTTPException(status_code=404, detail="no capture uploaded yet")
    preview_path = pending.capture_dir / "color_preview.jpg"
    if not preview_path.is_file():
        raise HTTPException(status_code=404, detail="preview file missing")
    return Response(content=preview_path.read_bytes(), media_type="image/jpeg")


@app.get("/grasp_viz_image")
async def grasp_viz_image() -> Response:
    viz_path: Path | None = app.state.last_grasp_viz_path
    if viz_path is None or not viz_path.is_file():
        raise HTTPException(status_code=404, detail="no grasp visualization available yet")
    return Response(content=viz_path.read_bytes(), media_type="image/jpeg")


@app.get("/grasp_viz_best_image")
async def grasp_viz_best_image() -> Response:
    """Gold-overlay image of the selected best grasp; generated by /select_and_execute."""
    viz_path: Path | None = app.state.last_grasp_viz_best_path
    if viz_path is None or not viz_path.is_file():
        raise HTTPException(status_code=404, detail="no best-grasp visualization available yet")
    return Response(content=viz_path.read_bytes(), media_type="image/jpeg")


@app.get("/grasp_viz_3d", response_class=HTMLResponse)
async def grasp_viz_3d() -> HTMLResponse:
    viz_path: Path | None = app.state.last_grasp_viz_3d_path
    if viz_path is None or not viz_path.is_file():
        raise HTTPException(status_code=404, detail="no 3-D grasp visualization available yet")
    return HTMLResponse(viz_path.read_text(encoding="utf-8"))


@app.post("/run_grasp")
async def run_grasp(
    task_spec: str = Form(..., description='Natural-language task, e.g. "grasp the cup".'),
    num_candidates: int = Form(1, description="VLM grasp_region_box proposals."),
    top_k: int = Form(1, description="Top-K grasps to return."),
    provider: str | None = Form(None, description="Optional VLM provider override."),
    model: str | None = Form(None, description="Optional VLM model override."),
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

    pending: PendingCapture | None = app.state.pending_capture
    if pending is None:
        raise HTTPException(
            status_code=400,
            detail="no capture uploaded yet — press the right button on the 3D mouse first",
        )

    task_spec = task_spec.strip()
    if not task_spec:
        raise HTTPException(status_code=400, detail="task_spec must be a non-empty string")
    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be >= 1")
    if num_candidates < 1:
        raise HTTPException(status_code=400, detail="num_candidates must be >= 1")

    run_id = new_run_id()
    effective_provider = provider or config.provider
    effective_model = model or config.model

    async with request_lock:
        t_start = time.perf_counter()
        try:
            pipeline_result = await asyncio.to_thread(
                run_vg_pipeline,
                capture_dir=pending.capture_dir,
                task_spec=task_spec,
                out_base=config.output_base,
                run_id=run_id,
                provider=effective_provider,
                model=effective_model,
                num_candidates=num_candidates,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=f"pipeline file error: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"pipeline rejected the inputs: {exc}") from exc
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
                top_k,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"failed to load Contact-GraspNet predictions: {exc}",
            ) from exc

        elapsed_ms = int((time.perf_counter() - t_start) * 1000)

    app.state.last_request_ms = elapsed_ms
    app.state.last_run_dir = pipeline_result.run_dir

    viz_path = save_grasp_viz(pending.capture_dir, pipeline_result.run_dir, grasps_json)
    if viz_path:
        app.state.last_grasp_viz_path = viz_path

    viz_3d_path = save_grasp_viz_3d(pending.capture_dir, pipeline_result.run_dir, grasps_json)
    if viz_3d_path:
        app.state.last_grasp_viz_3d_path = viz_3d_path

    result_payload = {
        "run_id": run_id,
        "run_dir": str(pipeline_result.run_dir),
        "frame_id": pending.frame_id,
        "elapsed_ms": elapsed_ms,
        "num_candidates": num_candidates,
        "top_k": top_k,
        "grasps": grasps_json,
        "grasp_viz": str(viz_path) if viz_path else None,
    }
    app.state.last_grasp_result = result_payload
    return JSONResponse(result_payload)


def _run_align_pipeline(
    *,
    capture_dir: Path,
    task_spec: str,
    provider: str,
    model: str,
    num_candidates: int = 1,
) -> list[dict[str, Any]]:
    """VLM align-point inference -> depth back-projection -> list of 6-DoF grasp dicts.

    Runs in a worker thread (blocking HTTP + numpy). No SAM2 / Contact-GraspNet.
    When num_candidates == 1, uses the single-point prompt for backward compatibility.
    """
    npy_path, scene_path, _ = resolve_capture_dir(capture_dir)
    obs = load_observation_npy(npy_path)
    depth = np.asarray(obs["depth"])
    K = np.asarray(obs["K"])
    rgb = np.asarray(Image.open(scene_path).convert("RGB"), dtype=np.uint8)
    h, w = int(depth.shape[0]), int(depth.shape[1])

    if num_candidates == 1:
        prompt = build_align_prompt(task_spec, w, h)
        raw_model_text = run_vg_inference(
            provider=provider,
            images=[Image.fromarray(rgb)],
            task_spec=task_spec,
            model_path=model,
            openai_image_mime_types=["image/png"],
            gemini_image_mime_types=["image/png"],
            prompt=prompt,
        )
        parsed_list = [parse_align_result(raw_model_text, canvas_h=h, canvas_w=w, rgb_h=h)]
    else:
        prompt = build_align_prompt_multi(task_spec, w, h, num_candidates)
        raw_model_text = run_vg_inference(
            provider=provider,
            images=[Image.fromarray(rgb)],
            task_spec=task_spec,
            model_path=model,
            openai_image_mime_types=["image/png"],
            gemini_image_mime_types=["image/png"],
            prompt=prompt,
        )
        parsed_list = parse_align_results_multi(raw_model_text, canvas_h=h, canvas_w=w, rgb_h=h)

    grasps: list[dict[str, Any]] = []
    for parsed in parsed_list:
        grasps.extend(build_align_grasp(
            depth,
            K,
            point_yx=parsed.point_yx,
            angle_deg=parsed.angle_deg,
            width_m=parsed.width_m,
        ))
    return grasps


@app.post("/run_align")
async def run_align(
    task_spec: str = Form(..., description='Natural-language target, e.g. "the rail".'),
    num_candidates: int = Form(1, description="Number of alignment candidates to request from the VLM."),
    provider: str | None = Form(None, description="Optional VLM provider override."),
    model: str | None = Form(None, description="Optional VLM model override."),
) -> JSONResponse:
    """Lightweight flow: VLA returns a 2D alignment point + gripper angle; the server
    recovers the 3D camera-frame 6-DoF pose from the captured depth map. Same response
    schema as ``/run_grasp`` so the downstream ROS2 client is unaffected. Does not use
    the Contact-GraspNet worker."""
    config, _, request_lock = _get_state(app)

    pending: PendingCapture | None = app.state.pending_capture
    if pending is None:
        raise HTTPException(
            status_code=400,
            detail="no capture uploaded yet — press the right button on the 3D mouse first",
        )

    task_spec = task_spec.strip()
    if not task_spec:
        raise HTTPException(status_code=400, detail="task_spec must be a non-empty string")
    if num_candidates < 1:
        raise HTTPException(status_code=400, detail="num_candidates must be >= 1")

    run_id = new_run_id()
    effective_provider = provider or config.provider
    effective_model = model or config.model

    async with request_lock:
        t_start = time.perf_counter()
        try:
            grasps_json = await asyncio.to_thread(
                _run_align_pipeline,
                capture_dir=pending.capture_dir,
                task_spec=task_spec,
                provider=effective_provider,
                model=effective_model,
                num_candidates=num_candidates,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=f"capture file error: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"align flow rejected the inputs: {exc}") from exc
        except Exception as exc:
            logger.exception("align flow failed")
            raise HTTPException(status_code=500, detail=f"align flow failed: {exc}") from exc

        elapsed_ms = int((time.perf_counter() - t_start) * 1000)

    app.state.last_request_ms = elapsed_ms

    viz_path = save_grasp_viz(pending.capture_dir, pending.capture_dir, grasps_json)
    if viz_path:
        app.state.last_grasp_viz_path = viz_path

    result_payload = {
        "run_id": run_id,
        "run_dir": str(pending.capture_dir),
        "frame_id": pending.frame_id,
        "elapsed_ms": elapsed_ms,
        "num_candidates": num_candidates,
        "top_k": len(grasps_json),
        "grasps": grasps_json,
        "grasp_viz": str(viz_path) if viz_path else None,
    }
    app.state.last_grasp_result = result_payload
    return JSONResponse(result_payload)


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
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"failed to load Contact-GraspNet predictions: {exc}",
            ) from exc

        elapsed_ms = int((time.perf_counter() - t_start) * 1000)

    app.state.last_request_ms = elapsed_ms
    app.state.last_run_dir = pipeline_result.run_dir

    viz_path = save_grasp_viz(parsed.capture_dir, pipeline_result.run_dir, grasps_json)
    save_grasp_viz_3d(parsed.capture_dir, pipeline_result.run_dir, grasps_json)

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
        "grasp_viz": str(viz_path) if viz_path else None,
    })


@app.post("/request_capture")
async def request_capture() -> JSONResponse:
    """Web UI calls this to ask the ROS2 node to take a snapshot."""
    async with app.state.capture_lock:
        app.state.capture_requested = True
    return JSONResponse({"status": "ok"})


@app.get("/poll_capture_request")
async def poll_capture_request() -> JSONResponse:
    """ROS2 node polls this; returns requested=true once then clears the flag."""
    async with app.state.capture_lock:
        requested = app.state.capture_requested
        if requested:
            app.state.capture_requested = False
    return JSONResponse({"requested": requested})


@app.post("/push_frame")
async def push_frame(frame: UploadFile = File(...)) -> JSONResponse:
    """ROS2 node pushes a JPEG frame here; served back by /latest_frame for the live UI stream."""
    data = await frame.read()
    async with app.state.frame_lock:
        app.state.latest_frame_bytes = data
    return JSONResponse({"status": "ok"})


@app.get("/latest_frame")
async def latest_frame() -> Response:
    """Return the most recent JPEG frame pushed by the ROS2 node."""
    async with app.state.frame_lock:
        data = app.state.latest_frame_bytes
    if data is None:
        raise HTTPException(status_code=404, detail="no frame received yet")
    return Response(content=data, media_type="image/jpeg")


@app.get("/capture_status")
async def capture_status() -> JSONResponse:
    """Return the timestamp of the last uploaded capture so the UI can detect new snapshots."""
    pending: PendingCapture | None = app.state.pending_capture
    return JSONResponse({"uploaded_at": pending.uploaded_at if pending else None})


@app.post("/trigger_publish")
async def trigger_publish() -> JSONResponse:
    """Mark the last /run_grasp result as pending for the ROS2 node to pick up (direct, no IK)."""
    if app.state.last_grasp_result is None:
        raise HTTPException(status_code=404, detail="no grasp result available — run a grasp first")
    result = app.state.last_grasp_result
    async with app.state.publish_lock:
        app.state.pending_publish = {
            **result,
            "mode": "execute",
            "trace_id": result.get("trace_id", result["run_id"]),
        }
    return JSONResponse({"status": "ok", "run_id": result["run_id"]})


@app.post("/trigger_ik_check")
async def trigger_ik_check() -> JSONResponse:
    """Pre-filter grasps server-side, then queue for IK checking by the ROS2 client."""
    if app.state.last_grasp_result is None:
        raise HTTPException(status_code=404, detail="no grasp result available — run a grasp first")
    pending: PendingCapture | None = app.state.pending_capture
    if pending is None:
        raise HTTPException(status_code=400, detail="no capture available — upload before IK check")

    npy_path, _, _ = resolve_capture_dir(pending.capture_dir)
    obs = load_observation_npy(npy_path)
    depth = np.asarray(obs["depth"])
    K = np.asarray(obs["K"])

    raw_grasps: list[dict] = list(app.state.last_grasp_result.get("grasps", []))

    try:
        filter_report = await asyncio.to_thread(filter_grasps, raw_grasps, depth, K)
    except NotImplementedError:
        filter_report = None  # stubs not yet implemented — pass through all grasps

    if filter_report is not None and filter_report.num_passed == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"all {filter_report.total} grasps rejected by pre-IK filter "
                f"(width={len(filter_report.rejected_width)}, "
                f"clearance={len(filter_report.rejected_clearance)}, "
                f"collision={len(filter_report.rejected_collision)})"
            ),
        )

    filtered_grasps = filter_report.passed if filter_report is not None else raw_grasps
    trace_id = new_run_id()

    # Store metadata + filtered grasps in ik_results with status="pending".
    # capture_dir is snapshotted here so select_and_execute can generate the
    # best-grasp viz independently of the live pending_capture state.
    # submit_ik_result will update grasps and flip status to "complete".
    entry = {
        **app.state.last_grasp_result,
        "grasps": filtered_grasps,
        "trace_id": trace_id,
        "status": "pending",
        "capture_dir": str(pending.capture_dir),
    }
    async with app.state.ik_results_lock:
        app.state.ik_results[trace_id] = entry
    async with app.state.publish_lock:
        app.state.pending_publish = {**entry, "mode": "ik_check"}

    return JSONResponse({
        "status": "ok",
        "run_id": entry["run_id"],
        "trace_id": trace_id,
        "num_before_filter": len(raw_grasps),
        "num_after_filter": len(filtered_grasps),
    })


@app.post("/submit_ik_result")
async def submit_ik_result(payload: dict = Body(...)) -> JSONResponse:
    """ROS2 returns IK-passing grasps. Updates ik_results to 'complete'; does NOT execute."""
    trace_id = payload.get("trace_id", "")
    run_id = payload.get("run_id", "")
    grasps = payload.get("grasps", [])

    async with app.state.ik_results_lock:
        entry = app.state.ik_results.get(trace_id)
        if entry is None or entry.get("status") != "pending":
            raise HTTPException(
                status_code=409,
                detail=f"no pending IK check for trace_id={trace_id!r}",
            )
        app.state.ik_results[trace_id] = {
            **entry,
            "run_id": run_id,
            "grasps": grasps,
            "status": "complete",
        }

    return JSONResponse({"status": "ok", "trace_id": trace_id, "num_passing": len(grasps)})


@app.get("/ik_result_status")
async def ik_result_status(trace_id: str = Query(...)) -> JSONResponse:
    """Web UI polls this to know when the ROS2 client has submitted IK results.

    Response:
      count=None  → IK not yet back
      count=N>0   → N grasps passed IK; proceed to select_and_execute
      count=0     → all grasps failed IK; UI should prompt re-capture
    """
    async with app.state.ik_results_lock:
        entry = app.state.ik_results.get(trace_id)
    complete = entry is not None and entry.get("status") == "complete"
    count = len(entry["grasps"]) if complete else None
    return JSONResponse({"trace_id": trace_id, "ready": complete, "count": count})


@app.post("/select_and_execute")
async def select_and_execute(payload: dict = Body(...)) -> JSONResponse:
    """Score IK-passing grasps, select the single best, queue for execution."""
    trace_id = payload.get("trace_id", "")
    async with app.state.ik_results_lock:
        entry = app.state.ik_results.pop(trace_id, None)
    if entry is None or entry.get("status") != "complete":
        raise HTTPException(status_code=404, detail=f"no completed IK result for trace_id={trace_id!r}")

    ik_grasps: list[dict] = entry.get("grasps", [])
    if not ik_grasps:
        raise HTTPException(
            status_code=422,
            detail="IK rejected all candidates — re-capture from a different angle",
        )

    report = rank_grasps(ik_grasps)
    best_grasp = report.best_grasp

    scored_grasps = [
        {**ik_grasps[i], "composite_score": report.scores[i]}
        for i in range(len(ik_grasps))
    ]

    async with app.state.publish_lock:
        app.state.pending_publish = {**entry, "grasps": [best_grasp], "mode": "execute"}

    # Generate gold-overlay best-grasp image using the snapshotted capture_dir.
    capture_dir_str = entry.get("capture_dir")
    if capture_dir_str:
        run_dir = Path(entry.get("run_dir", capture_dir_str))
        best_viz_path = await asyncio.to_thread(
            save_grasp_viz_best,
            Path(capture_dir_str),
            run_dir,
            best_grasp,
            report.scores[report.best_index],
        )
        if best_viz_path:
            app.state.last_grasp_viz_best_path = best_viz_path

    return JSONResponse({
        "status": "ok",
        "trace_id": trace_id,
        "num_ik_passing": len(ik_grasps),
        "num_selected": 1,
        "scored_grasps": scored_grasps,
        "ranked_indices": report.ranked_indices,
        "best_index": report.best_index,
    })


@app.get("/poll_publish")
async def poll_publish() -> JSONResponse:
    """ROS2 node polls this; returns the pending result once then clears it."""
    async with app.state.publish_lock:
        result = app.state.pending_publish
        if result is not None:
            app.state.pending_publish = None
            return JSONResponse(result)
    raise HTTPException(status_code=404, detail="no pending publish")


def _ensure_output_base_exists() -> None:
    cfg = ServerConfig.from_env()
    cfg.output_base.mkdir(parents=True, exist_ok=True)


_ensure_output_base_exists()
