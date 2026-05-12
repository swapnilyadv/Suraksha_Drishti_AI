"""
database/db_manager.py
SQLite database manager for Suraksha AI.
Provides CRUD operations for incident storage and retrieval.
"""

import os
from typing import List, Optional, Dict, Any
from datetime import datetime

from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker, Session

from database.models import Base, Incident
from utils.logger import get_logger

logger = get_logger(__name__)


class DBManager:
    """
    Manages SQLite database operations for incident persistence.

    Usage:
        db = DBManager(db_path="database/incidents.db")
        db.save_incident(incident_dict)
        incidents = db.get_incidents(threat_level="High", limit=50)
    """

    def __init__(self, db_path: str = "database/incidents.db"):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        engine_url = f"sqlite:///{db_path}"
        self._engine = create_engine(engine_url, connect_args={"check_same_thread": False})
        self._Session = sessionmaker(bind=self._engine)

        # Auto-create tables if they don't exist
        Base.metadata.create_all(self._engine)
        logger.info(f"DBManager ready | {engine_url}")

    # ── Write ─────────────────────────────────────────────────────────────────

    def save_incident(self, incident_dict: Dict[str, Any]) -> Incident:
        """
        Persist a new incident to the database.

        Args:
            incident_dict: Dict as produced by build_incident_json().

        Returns:
            The saved Incident ORM object.
        """
        with self._Session() as session:
            inc = Incident(
                incident_id     = incident_dict["incident_id"],
                timestamp       = self._parse_ts(incident_dict.get("timestamp")),
                threat_level    = incident_dict.get("threat_level", "Low"),
                threat_score    = incident_dict.get("threat_score", 0.0),
                camera_id       = incident_dict.get("camera_id", "CAM-01"),
                screenshot_path = incident_dict.get("screenshot_path"),
                clip_path       = incident_dict.get("clip_path"),
                status          = "New",
            )
            inc.set_person_ids(incident_dict.get("person_ids", []))
            session.add(inc)
            session.commit()
            session.refresh(inc)
            logger.info(f"Incident saved to DB: {inc.incident_id}")
            return inc

    def update_status(self, incident_id: str, status: str, notes: str = "") -> bool:
        """Update the review status of an existing incident."""
        with self._Session() as session:
            inc = session.query(Incident).filter_by(incident_id=incident_id).first()
            if not inc:
                logger.warning(f"Incident not found for status update: {incident_id}")
                return False
            inc.status = status
            inc.notes  = notes
            session.commit()
            return True

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_incidents(
        self,
        threat_level: Optional[str] = None,
        status: Optional[str] = None,
        camera_id: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Query incidents with optional filters.

        Args:
            threat_level: 'High' | 'Medium' | 'Low' | None (all)
            status:       'New' | 'Reviewed' | 'Dismissed' | None (all)
            camera_id:    Camera label filter.
            from_date:    Start of date range filter.
            to_date:      End of date range filter.
            limit:        Max results to return.
            offset:       Pagination offset.

        Returns:
            List of incident dicts.
        """
        with self._Session() as session:
            q = session.query(Incident)

            if threat_level:
                q = q.filter(Incident.threat_level == threat_level)
            if status:
                q = q.filter(Incident.status == status)
            if camera_id:
                q = q.filter(Incident.camera_id == camera_id)
            if from_date:
                q = q.filter(Incident.timestamp >= from_date)
            if to_date:
                q = q.filter(Incident.timestamp <= to_date)

            incidents = (
                q.order_by(desc(Incident.timestamp))
                 .limit(limit)
                 .offset(offset)
                 .all()
            )
            return [i.to_dict() for i in incidents]

    def get_incident_by_id(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single incident by its string ID."""
        with self._Session() as session:
            inc = session.query(Incident).filter_by(incident_id=incident_id).first()
            return inc.to_dict() if inc else None

    def count_incidents(self, threat_level: Optional[str] = None) -> int:
        """Return total incident count, optionally filtered by threat level."""
        with self._Session() as session:
            q = session.query(Incident)
            if threat_level:
                q = q.filter(Incident.threat_level == threat_level)
            return q.count()

    def get_all_as_dicts(self) -> List[Dict[str, Any]]:
        """Return all incidents for export."""
        with self._Session() as session:
            return [i.to_dict() for i in session.query(Incident).all()]

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_ts(ts_str) -> Optional[datetime]:
        """Parse ISO timestamp string to datetime object."""
        if ts_str is None:
            return None
        if isinstance(ts_str, datetime):
            return ts_str
        try:
            return datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
        except Exception:
            return None
