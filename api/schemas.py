"""
api/schemas.py
Pydantic request/response schemas for Suraksha AI FastAPI backend.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# ── Incident Schemas ──────────────────────────────────────────────────────────

class IncidentResponse(BaseModel):
    id:               int
    incident_id:      str
    timestamp:        Optional[str]
    threat_level:     str
    threat_score:     float
    person_ids:       List[int]
    camera_id:        Optional[str]
    screenshot_path:  Optional[str]
    clip_path:        Optional[str]
    status:           str
    notes:            Optional[str]


class IncidentListResponse(BaseModel):
    total:     int
    incidents: List[IncidentResponse]


class IncidentStatusUpdate(BaseModel):
    status: str = Field(..., description="'New' | 'Reviewed' | 'Dismissed'")
    notes:  Optional[str] = Field(None, description="Operator notes")


# ── Detection Status Schemas ──────────────────────────────────────────────────

class DetectionStatus(BaseModel):
    running:         bool
    mode:            str               # 'idle' | 'webcam' | 'rtsp' | 'video'
    source:          Optional[str]
    camera_id:       Optional[str]
    threat_status:   str               # 'Safe' | 'Suspicious' | 'Harassment Detected'
    threat_score:    float
    active_tracks:   int
    fps:             float
    total_incidents: int


# ── Stream Control Schemas ────────────────────────────────────────────────────

class StartWebcamRequest(BaseModel):
    webcam_index: int  = Field(0,        description="Camera device index")
    camera_id:    str  = Field("CAM-01", description="Camera label for reports")


class StartRTSPRequest(BaseModel):
    rtsp_url:  str = Field(..., description="RTSP stream URL")
    camera_id: str = Field("CAM-01")


# ── Upload Response ───────────────────────────────────────────────────────────

class VideoUploadResponse(BaseModel):
    message:    str
    filename:   str
    saved_path: str
    task_id:    str


# ── Export Schemas ────────────────────────────────────────────────────────────

class ExportRequest(BaseModel):
    format:       str            = Field("csv", description="'csv' | 'json'")
    threat_level: Optional[str] = None
    from_date:    Optional[str] = None
    to_date:      Optional[str] = None


# ── Generic Response ──────────────────────────────────────────────────────────

class MessageResponse(BaseModel):
    message: str
    detail:  Optional[Any] = None
