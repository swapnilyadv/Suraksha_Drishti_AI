"""
database/models.py
SQLAlchemy ORM models for Suraksha AI incident database.
"""

import json
from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text, create_engine
)
from sqlalchemy.orm import declarative_base, Session

Base = declarative_base()


class Incident(Base):
    """
    Stores every detected harassment incident.

    Columns:
        id            — Auto-increment primary key.
        incident_id   — Human-readable unique ID (e.g. INC-20260512-001-ABCDEF).
        timestamp     — UTC datetime of detection.
        threat_level  — 'High' | 'Medium' | 'Low'.
        threat_score  — 0.0–1.0 float.
        person_ids    — JSON-encoded list of involved track IDs.
        camera_id     — Camera label (e.g. 'CAM-01').
        screenshot_path — Path to saved screenshot JPEG.
        clip_path       — Path to saved MP4 clip.
        status          — 'New' | 'Reviewed' | 'Dismissed'.
        notes           — Optional operator notes.
    """

    __tablename__ = "incidents"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    incident_id     = Column(String(64), unique=True, nullable=False, index=True)
    timestamp       = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    threat_level    = Column(String(16), nullable=False, default="Low")
    threat_score    = Column(Float,      nullable=False, default=0.0)
    person_ids      = Column(Text,       nullable=True)   # JSON list
    camera_id       = Column(String(32), nullable=True,  default="CAM-01")
    screenshot_path = Column(String(512), nullable=True)
    clip_path       = Column(String(512), nullable=True)
    status          = Column(String(16),  nullable=False, default="New")
    notes           = Column(Text,        nullable=True)

    def set_person_ids(self, ids: list) -> None:
        self.person_ids = json.dumps(ids)

    def get_person_ids(self) -> list:
        if not self.person_ids:
            return []
        return json.loads(self.person_ids)

    def to_dict(self) -> dict:
        return {
            "id":              self.id,
            "incident_id":     self.incident_id,
            "timestamp":       self.timestamp.isoformat() if self.timestamp else None,
            "threat_level":    self.threat_level,
            "threat_score":    self.threat_score,
            "person_ids":      self.get_person_ids(),
            "camera_id":       self.camera_id,
            "screenshot_path": self.screenshot_path,
            "clip_path":       self.clip_path,
            "status":          self.status,
            "notes":           self.notes,
        }

    def __repr__(self) -> str:
        return (
            f"<Incident id={self.id} incident_id={self.incident_id!r} "
            f"level={self.threat_level} score={self.threat_score:.2f}>"
        )
