"""
alerts/alert_manager.py
Alert generation and incident persistence for Suraksha AI.
Handles sound alarms, screenshots, video clip saving, JSON reports, and DB storage.
"""

import os
import json
import uuid
import threading
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import numpy as np

from utils.logger import get_alert_logger
from utils.video_utils import save_clip, save_screenshot
from utils.export import build_incident_json

logger = get_alert_logger()

# Minimum seconds between consecutive alerts to avoid spam
ALERT_COOLDOWN_SEC = 10.0


class AlertManager:
    """
    Manages all alert side-effects when harassment is detected.

    Responsibilities:
    - Play alarm sound (non-blocking thread)
    - Save incident screenshot
    - Save pre-incident video clip
    - Write JSON incident report
    - Store incident in database (via DBManager)
    - Enforce cooldown to prevent alert flooding

    Usage:
        am = AlertManager(config, db_manager)
        am.trigger(frame, frame_buffer, threat_result, tracks, camera_id)
    """

    def __init__(self, config: dict, db_manager=None):
        """
        Args:
            config:     Loaded config dict (from config.yaml paths section).
            db_manager: Optional DBManager instance for DB persistence.
        """
        self.cfg       = config
        self.db        = db_manager
        self._last_alert_time: float = 0.0
        self._sound_path = config.get("paths", {}).get("alarm_sound", "alerts/sounds/alarm.wav")

        # Ensure output directories exist
        for key in ["screenshots_dir", "clips_dir", "incidents_dir"]:
            path = config.get("paths", {}).get(key, f"outputs/{key.replace('_dir','')}")
            os.makedirs(path, exist_ok=True)

        logger.info("AlertManager ready.")

    # ── Public Interface ──────────────────────────────────────────────────────

    def trigger(
        self,
        frame: np.ndarray,
        frame_buffer: list,
        threat_result: Dict[str, Any],
        tracks: List[Dict[str, Any]],
        camera_id: str = "CAM-01",
        fps: float = 25.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Fire all alert actions when a harassment event is detected.

        Args:
            frame:        Current annotated BGR frame.
            frame_buffer: List of recent raw frames for clip saving.
            threat_result: Output from HarassmentEngine.evaluate().
            tracks:       Current tracked persons.
            camera_id:    Camera identifier string.
            fps:          Video FPS for clip writer.

        Returns:
            Incident dict if alert was fired, None if still in cooldown.
        """
        import time
        now = time.time()
        if now - self._last_alert_time < ALERT_COOLDOWN_SEC:
            return None   # Cooldown active — skip
        self._last_alert_time = now

        # ── Build incident metadata ───────────────────────────
        incident_id    = f"INC-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        timestamp      = datetime.now(timezone.utc).isoformat()
        threat_level   = self._map_threat_level(threat_result["threat_score"])
        threat_score   = threat_result["threat_score"]
        involved_ids   = threat_result.get("involved_ids", [])

        paths = self.cfg.get("paths", {})
        screenshot_path = os.path.join(paths.get("screenshots_dir", "outputs/screenshots"), f"{incident_id}.jpg")
        clip_path       = os.path.join(paths.get("clips_dir",       "outputs/clips"),       f"{incident_id}.mp4")
        report_path     = os.path.join(paths.get("incidents_dir",   "outputs/incidents"),   f"{incident_id}.json")

        # ── 1. Play alarm sound (non-blocking) ────────────────
        self._play_alarm_async()

        # ── 2. Save screenshot ────────────────────────────────
        save_screenshot(frame, screenshot_path)

        # ── 3. Save video clip ────────────────────────────────
        if frame_buffer:
            h, w = frame_buffer[0].shape[:2]
            save_clip(frame_buffer, clip_path, fps=fps, size=(w, h))

        # ── 4. Build & write JSON report ──────────────────────
        incident = build_incident_json(
            incident_id=incident_id,
            timestamp=timestamp,
            threat_level=threat_level,
            threat_score=threat_score,
            person_ids=involved_ids,
            screenshot_path=screenshot_path,
            clip_path=clip_path,
            camera_id=camera_id,
        )
        self._write_json_report(incident, report_path)

        # ── 5. Persist to database ────────────────────────────
        if self.db:
            try:
                self.db.save_incident(incident)
            except Exception as e:
                logger.error(f"Failed to save incident to DB: {e}")

        logger.warning(
            f"ALERT FIRED | {incident_id} | level={threat_level} | "
            f"score={threat_score:.2f} | IDs={involved_ids}"
        )
        return incident

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _map_threat_level(self, score: float) -> str:
        """Map 0–1 threat score to High/Medium/Low label."""
        if score >= 0.70:
            return "High"
        elif score >= 0.40:
            return "Medium"
        return "Low"

    def _play_alarm_async(self) -> None:
        """Play alarm sound in a daemon thread so it doesn't block inference."""
        def _play():
            try:
                if not os.path.exists(self._sound_path):
                    logger.warning(f"Alarm sound not found: {self._sound_path}")
                    return
                # Try pygame first, fall back to playsound
                try:
                    import pygame
                    pygame.mixer.init()
                    pygame.mixer.music.load(self._sound_path)
                    pygame.mixer.music.play()
                except ImportError:
                    from playsound import playsound
                    playsound(self._sound_path, block=False)
            except Exception as e:
                logger.error(f"Alarm sound error: {e}")

        t = threading.Thread(target=_play, daemon=True)
        t.start()

    def _write_json_report(self, incident: dict, path: str) -> None:
        """Write incident dict to a JSON file."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(incident, f, indent=2, default=str)
            logger.info(f"Incident report saved: {path}")
        except Exception as e:
            logger.error(f"Failed to write incident JSON: {e}")
