"""
api/routes/video.py
Video upload and background analysis endpoint for Suraksha AI.
"""

import os
import uuid
import threading
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from api.schemas import VideoUploadResponse, MessageResponse
from utils.video_utils import validate_video_file
from utils.logger import get_api_logger

logger = get_api_logger()

router = APIRouter(prefix="/api/video", tags=["Video"])

UPLOAD_DIR      = "outputs/uploads"
MAX_SIZE_BYTES  = 500 * 1024 * 1024   # 500 MB

# Track active background analysis jobs {task_id: status}
_analysis_jobs: dict = {}


def get_pipeline():
    from api.main_api import app
    return app.state.pipeline


def get_config():
    from api.main_api import app
    return app.state.config


# ── POST /api/video/upload ────────────────────────────────────────────────────

@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file to analyse (.mp4, .avi, .mov)"),
    camera_id: str = "CAM-01",
):
    """
    Upload a video file and start background harassment detection.

    Returns a task_id that can be polled for status.
    """
    # ── Validate extension ────────────────────────────────────
    ALLOWED = {"mp4", "avi", "mov", "mkv"}
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '.{ext}'. Allowed: {ALLOWED}",
        )

    # ── Read & size-check ─────────────────────────────────────
    content = await file.read()
    if len(content) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content)//1024//1024} MB). Max: 500 MB.",
        )

    # ── Save to disk ──────────────────────────────────────────
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    task_id   = uuid.uuid4().hex[:10].upper()
    safe_name = f"{task_id}_{file.filename}"
    save_path = os.path.join(UPLOAD_DIR, safe_name)

    with open(save_path, "wb") as f:
        f.write(content)

    # ── Validate with OpenCV ──────────────────────────────────
    if not validate_video_file(save_path):
        os.remove(save_path)
        raise HTTPException(status_code=422, detail="Uploaded file could not be opened as a video.")

    # ── Schedule background analysis ──────────────────────────
    _analysis_jobs[task_id] = {"status": "queued", "source": save_path}
    background_tasks.add_task(_run_analysis, task_id, save_path, camera_id)

    logger.info(f"Video uploaded: {save_path} | task_id={task_id}")
    return VideoUploadResponse(
        message="Video uploaded successfully. Analysis started in background.",
        filename=file.filename or safe_name,
        saved_path=save_path,
        task_id=task_id,
    )


# ── GET /api/video/status/{task_id} ──────────────────────────────────────────

@router.get("/status/{task_id}", response_model=MessageResponse)
async def get_task_status(task_id: str):
    """Poll the status of a background video analysis task."""
    job = _analysis_jobs.get(task_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found.")
    return MessageResponse(message=job["status"], detail=job)


# ── Background Worker ─────────────────────────────────────────────────────────

def _run_analysis(task_id: str, video_path: str, camera_id: str) -> None:
    """Run full inference pipeline on uploaded video (background thread)."""
    try:
        _analysis_jobs[task_id]["status"] = "running"
        pipeline = get_pipeline()
        pipeline.run(source=video_path, camera_id=camera_id, display=False)
        _analysis_jobs[task_id]["status"] = "completed"
        logger.info(f"Background analysis complete | task_id={task_id}")
    except Exception as e:
        _analysis_jobs[task_id]["status"] = f"error: {e}"
        logger.error(f"Background analysis failed | task_id={task_id} | error={e}")
