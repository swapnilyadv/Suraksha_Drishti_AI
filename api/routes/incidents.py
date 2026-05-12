"""
api/routes/incidents.py
Incident management endpoints for Suraksha AI.
"""

import os
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from api.schemas import (
    IncidentResponse, IncidentListResponse,
    IncidentStatusUpdate, MessageResponse, ExportRequest
)
from utils.logger import get_api_logger
from utils.export import export_incidents_csv, export_incidents_json, export_summary

logger = get_api_logger()

router = APIRouter(prefix="/api/incidents", tags=["Incidents"])


def get_db():
    """Dependency: get DBManager from app state."""
    from api.main_api import app
    return app.state.db


# ── GET /api/incidents ────────────────────────────────────────────────────────

@router.get("", response_model=IncidentListResponse)
async def list_incidents(
    threat_level: Optional[str] = Query(None, description="'High' | 'Medium' | 'Low'"),
    status:       Optional[str] = Query(None, description="'New' | 'Reviewed' | 'Dismissed'"),
    camera_id:    Optional[str] = Query(None),
    limit:        int           = Query(50,  ge=1, le=500),
    offset:       int           = Query(0,   ge=0),
):
    """Return paginated list of incidents with optional filters."""
    db = get_db()
    incidents = db.get_incidents(
        threat_level=threat_level,
        status=status,
        camera_id=camera_id,
        limit=limit,
        offset=offset,
    )
    total = db.count_incidents(threat_level=threat_level)
    logger.info(f"GET /incidents | returned={len(incidents)} | total={total}")
    return IncidentListResponse(total=total, incidents=incidents)


# ── GET /api/incidents/{incident_id} ─────────────────────────────────────────

@router.get("/{incident_id}", response_model=IncidentResponse)
async def get_incident(incident_id: str):
    """Fetch a single incident by its string ID."""
    db  = get_db()
    inc = db.get_incident_by_id(incident_id)
    if not inc:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id!r} not found.")
    return inc


# ── PATCH /api/incidents/{incident_id}/status ─────────────────────────────────

@router.patch("/{incident_id}/status", response_model=MessageResponse)
async def update_status(incident_id: str, body: IncidentStatusUpdate):
    """Update the review status and/or notes for an incident."""
    valid_statuses = {"New", "Reviewed", "Dismissed"}
    if body.status not in valid_statuses:
        raise HTTPException(status_code=422, detail=f"status must be one of {valid_statuses}")

    db = get_db()
    ok = db.update_status(incident_id, body.status, body.notes or "")
    if not ok:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id!r} not found.")

    return MessageResponse(message=f"Status updated to '{body.status}'.")


# ── GET /api/incidents/{incident_id}/clip ─────────────────────────────────────

@router.get("/{incident_id}/clip")
async def download_clip(incident_id: str):
    """Download the saved MP4 clip for an incident."""
    db  = get_db()
    inc = db.get_incident_by_id(incident_id)
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found.")

    clip_path = inc.get("clip_path")
    if not clip_path or not os.path.exists(clip_path):
        raise HTTPException(status_code=404, detail="Clip file not found on disk.")

    return FileResponse(clip_path, media_type="video/mp4", filename=f"{incident_id}.mp4")


# ── GET /api/incidents/{incident_id}/screenshot ───────────────────────────────

@router.get("/{incident_id}/screenshot")
async def download_screenshot(incident_id: str):
    """Download the saved screenshot JPEG for an incident."""
    db  = get_db()
    inc = db.get_incident_by_id(incident_id)
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found.")

    ss_path = inc.get("screenshot_path")
    if not ss_path or not os.path.exists(ss_path):
        raise HTTPException(status_code=404, detail="Screenshot not found on disk.")

    return FileResponse(ss_path, media_type="image/jpeg", filename=f"{incident_id}.jpg")


# ── POST /api/incidents/export ────────────────────────────────────────────────

@router.post("/export/report")
async def export_report(body: ExportRequest):
    """Export all incidents as CSV or JSON and return the file."""
    db        = get_db()
    incidents = db.get_all_as_dicts()

    if not incidents:
        raise HTTPException(status_code=404, detail="No incidents to export.")

    export_dir = "outputs/incidents"
    os.makedirs(export_dir, exist_ok=True)

    if body.format == "csv":
        path = os.path.join(export_dir, "incidents_export.csv")
        export_incidents_csv(incidents, path)
        return FileResponse(path, media_type="text/csv", filename="incidents_export.csv")

    elif body.format == "json":
        path = os.path.join(export_dir, "incidents_export.json")
        export_incidents_json(incidents, path)
        return FileResponse(path, media_type="application/json", filename="incidents_export.json")

    else:
        raise HTTPException(status_code=422, detail="format must be 'csv' or 'json'")
