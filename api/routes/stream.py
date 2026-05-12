"""
api/routes/stream.py
Live stream control endpoints (webcam / RTSP start-stop) for Suraksha AI.
"""

import threading
from fastapi import APIRouter, HTTPException

from api.schemas import (
    StartWebcamRequest, StartRTSPRequest,
    DetectionStatus, MessageResponse,
)
from utils.logger import get_api_logger

logger = get_api_logger()
router = APIRouter(prefix="/api/stream", tags=["Stream"])

# ── Shared pipeline state (set by main_api on startup) ───────────────────────
_pipeline_thread: threading.Thread = None
_stream_state = {
    "running":       False,
    "mode":          "idle",
    "source":        None,
    "camera_id":     None,
    "threat_status": "Safe",
    "threat_score":  0.0,
    "active_tracks": 0,
    "fps":           0.0,
}


def get_pipeline():
    from api.main_api import app
    return app.state.pipeline


def get_db():
    from api.main_api import app
    return app.state.db


# ── POST /api/stream/start/webcam ─────────────────────────────────────────────

@router.post("/start/webcam", response_model=MessageResponse)
async def start_webcam(body: StartWebcamRequest):
    """Start live detection from a webcam device."""
    global _pipeline_thread

    if _stream_state["running"]:
        raise HTTPException(
            status_code=409,
            detail="A stream is already running. Stop it first via /api/stream/stop.",
        )

    _stream_state.update({
        "running":   True,
        "mode":      "webcam",
        "source":    str(body.webcam_index),
        "camera_id": body.camera_id,
    })

    def _run():
        try:
            get_pipeline().run(
                source=body.webcam_index,
                camera_id=body.camera_id,
                display=False,
            )
        except Exception as e:
            logger.error(f"Webcam stream error: {e}")
        finally:
            _stream_state["running"] = False
            _stream_state["mode"]    = "idle"

    _pipeline_thread = threading.Thread(target=_run, daemon=True, name="webcam-stream")
    _pipeline_thread.start()

    logger.info(f"Webcam stream started | index={body.webcam_index} | id={body.camera_id}")
    return MessageResponse(message=f"Webcam {body.webcam_index} detection started.", detail=body.camera_id)


# ── POST /api/stream/start/rtsp ───────────────────────────────────────────────

@router.post("/start/rtsp", response_model=MessageResponse)
async def start_rtsp(body: StartRTSPRequest):
    """Start live detection from an RTSP camera stream."""
    global _pipeline_thread

    if _stream_state["running"]:
        raise HTTPException(status_code=409, detail="A stream is already running.")

    if not body.rtsp_url.startswith("rtsp://"):
        raise HTTPException(status_code=422, detail="rtsp_url must start with 'rtsp://'")

    _stream_state.update({
        "running":   True,
        "mode":      "rtsp",
        "source":    body.rtsp_url,
        "camera_id": body.camera_id,
    })

    def _run():
        try:
            get_pipeline().run(
                source=body.rtsp_url,
                camera_id=body.camera_id,
                display=False,
            )
        except Exception as e:
            logger.error(f"RTSP stream error: {e}")
        finally:
            _stream_state["running"] = False
            _stream_state["mode"]    = "idle"

    _pipeline_thread = threading.Thread(target=_run, daemon=True, name="rtsp-stream")
    _pipeline_thread.start()

    logger.info(f"RTSP stream started | url={body.rtsp_url} | id={body.camera_id}")
    return MessageResponse(message="RTSP stream detection started.", detail=body.rtsp_url)


# ── POST /api/stream/stop ─────────────────────────────────────────────────────

@router.post("/stop", response_model=MessageResponse)
async def stop_stream():
    """Stop the currently running stream or video analysis."""
    if not _stream_state["running"]:
        raise HTTPException(status_code=409, detail="No stream is currently running.")

    get_pipeline().stop()
    _stream_state["running"] = False
    _stream_state["mode"]    = "idle"

    logger.info("Stream stopped via API.")
    return MessageResponse(message="Detection stream stopped successfully.")


# ── GET /api/stream/status ────────────────────────────────────────────────────

@router.get("/status", response_model=DetectionStatus)
async def get_status():
    """Return the real-time detection status and metrics."""
    db = get_db()
    total = db.count_incidents() if db else 0

    return DetectionStatus(
        running=        _stream_state["running"],
        mode=           _stream_state["mode"],
        source=         _stream_state.get("source"),
        camera_id=      _stream_state.get("camera_id"),
        threat_status=  _stream_state.get("threat_status", "Safe"),
        threat_score=   _stream_state.get("threat_score",  0.0),
        active_tracks=  _stream_state.get("active_tracks", 0),
        fps=            _stream_state.get("fps",            0.0),
        total_incidents=total,
    )
