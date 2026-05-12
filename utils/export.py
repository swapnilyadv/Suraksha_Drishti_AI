"""
utils/export.py
CSV / JSON export utilities for incident reports and summaries.
"""

import csv
import json
import os
from datetime import datetime
from typing import List, Dict, Any

from utils.logger import get_logger

logger = get_logger(__name__)


def export_incidents_csv(incidents: List[Dict[str, Any]], output_path: str) -> bool:
    """
    Export a list of incident dicts to a CSV file.

    Args:
        incidents:   List of incident dictionaries.
        output_path: Destination .csv file path.

    Returns:
        True on success.
    """
    if not incidents:
        logger.warning("No incidents to export.")
        return False

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fieldnames = [
        "incident_id", "timestamp", "threat_level",
        "threat_score", "person_ids", "camera_id",
        "screenshot_path", "clip_path",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for inc in incidents:
            row = dict(inc)
            # Flatten person_ids list to string
            if isinstance(row.get("person_ids"), list):
                row["person_ids"] = ",".join(str(i) for i in row["person_ids"])
            writer.writerow(row)

    logger.info(f"Exported {len(incidents)} incidents → {output_path}")
    return True


def export_incidents_json(incidents: List[Dict[str, Any]], output_path: str) -> bool:
    """
    Export a list of incident dicts to a JSON file.

    Args:
        incidents:   List of incident dictionaries.
        output_path: Destination .json file path.

    Returns:
        True on success.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(incidents, f, indent=2, default=str)

    logger.info(f"Exported {len(incidents)} incidents → {output_path}")
    return True


def export_summary(incidents: List[Dict[str, Any]], output_path: str) -> bool:
    """
    Generate an incident summary report.

    Summary includes:
    - Total incidents
    - Breakdown by threat level
    - Date range

    Args:
        incidents:   List of incident dictionaries.
        output_path: Destination .json file path.
    """
    if not incidents:
        logger.warning("No incidents to summarise.")
        return False

    counts = {"High": 0, "Medium": 0, "Low": 0, "Unknown": 0}
    for inc in incidents:
        level = inc.get("threat_level", "Unknown")
        counts[level] = counts.get(level, 0) + 1

    timestamps = [inc.get("timestamp") for inc in incidents if inc.get("timestamp")]
    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_incidents": len(incidents),
        "by_threat_level": counts,
        "date_range": {
            "from": str(min(timestamps)) if timestamps else None,
            "to":   str(max(timestamps)) if timestamps else None,
        },
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Summary exported → {output_path}")
    return True


def build_incident_json(
    incident_id: str,
    timestamp: str,
    threat_level: str,
    threat_score: float,
    person_ids: List[int],
    screenshot_path: str,
    clip_path: str,
    camera_id: str = "CAM-01",
) -> Dict[str, Any]:
    """
    Build a structured incident dictionary ready for JSON serialisation.
    """
    return {
        "incident_id":     incident_id,
        "timestamp":       timestamp,
        "threat_level":    threat_level,
        "threat_score":    round(threat_score, 4),
        "person_ids":      person_ids,
        "camera_id":       camera_id,
        "screenshot_path": screenshot_path,
        "clip_path":       clip_path,
    }
