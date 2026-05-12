"""
utils/drawing.py
All OpenCV overlay drawing utilities for Suraksha AI.
Handles bounding boxes, labels, HUD, skeleton, and threat overlays.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List

POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
POSE_LANDMARKS = mp.solutions.pose.PoseLandmark


# ── Color palette (BGR) ──────────────────────────────────────────────────────
COLORS = {
    "safe":       (0,   200,  0),    # Green
    "suspicious": (0,   200, 255),   # Yellow-Orange
    "harassment": (0,    30, 220),   # Red
    "weapon":     (0,     0, 255),   # Pure Red
    "male":       (255, 120,  50),   # Blue-ish
    "female":     (180,  60, 255),   # Purple
    "white":      (255, 255, 255),
    "black":      (0,     0,   0),
    "dark_gray":  (30,   30,  30),
}

THREAT_COLOR_MAP = {
    "Safe":                COLORS["safe"],
    "Suspicious":          COLORS["suspicious"],
    "Harassment Detected": COLORS["harassment"],
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def get_threat_color(status: str) -> Tuple[int, int, int]:
    """Return BGR color corresponding to threat status."""
    return THREAT_COLOR_MAP.get(status, COLORS["white"])


def draw_person_box(
    frame: np.ndarray,
    bbox: List[int],
    track_id: int,
    gender: str = "",
    gender_conf: float = 0.0,
    threat_status: str = "Safe",
    threat_score: float = 0.0,
) -> np.ndarray:
    """
    Draw a colored bounding box with track ID, gender, and threat info.

    Args:
        frame:        BGR frame to draw on.
        bbox:         [x1, y1, x2, y2] bounding box.
        track_id:     Unique tracking ID.
        gender:       'Male' or 'Female' (empty if unknown).
        gender_conf:  Gender confidence (0–1).
        threat_status: 'Safe' | 'Suspicious' | 'Harassment Detected'
        threat_score:  0.0–1.0

    Returns:
        Annotated frame.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = get_threat_color(threat_status)
    thickness = 3 if threat_status == "Harassment Detected" else 2

    # Main bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # ── Top label background ──────────────────────────────────
    gender_icon = "♀" if gender == "Female" else ("♂" if gender == "Male" else "?")
    label = f"ID:{track_id} {gender_icon}"
    if gender_conf > 0:
        label += f" {gender_conf*100:.0f}%"

    (lw, lh), _ = cv2.getTextSize(label, FONT, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4), FONT, 0.5, COLORS["white"], 1, cv2.LINE_AA)

    # ── Bottom threat score ───────────────────────────────────
    if threat_status != "Safe":
        score_label = f"{threat_status} {threat_score*100:.0f}%"
        cv2.putText(frame, score_label, (x1, y2 + 18), FONT, 0.5, color, 2, cv2.LINE_AA)

    return frame


def draw_weapon_box(
    frame: np.ndarray,
    bbox: List[int],
    weapon_type: str,
    confidence: float,
    suspect_id: Optional[int] = None,
) -> np.ndarray:
    """
    Draw a thick red bounding box for weapon detections.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = COLORS["weapon"]

    # Thick red box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # Corner accent marks
    mark_len = 15
    for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(frame, (px, py), (px + dx * mark_len, py), color, 3)
        cv2.line(frame, (px, py), (px, py + dy * mark_len), color, 3)

    # Label
    label = f"⚠ {weapon_type} {confidence*100:.0f}%"
    if suspect_id is not None:
        label += f" [ID:{suspect_id}]"
    (lw, lh), _ = cv2.getTextSize(label, FONT, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 4), FONT, 0.55, COLORS["white"], 2, cv2.LINE_AA)

    return frame


def draw_harassment_alert(frame: np.ndarray, message: str = "⚠ HARASSMENT DETECTED") -> np.ndarray:
    """
    Draw a full-frame red warning overlay when harassment is detected.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Red border flashing effect
    cv2.rectangle(overlay, (0, 0), (w, h), COLORS["harassment"], 20)

    # Alert banner at top
    banner_h = 60
    cv2.rectangle(overlay, (0, 0), (w, banner_h), COLORS["harassment"], -1)

    # Message
    (tw, _), _ = cv2.getTextSize(message, FONT, 1.2, 3)
    tx = (w - tw) // 2
    cv2.putText(overlay, message, (tx, 42), FONT, 1.2, COLORS["white"], 3, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    return frame


def draw_hud(
    frame: np.ndarray,
    fps: float,
    track_count: int,
    threat_status: str,
    mode: str = "LIVE",
) -> np.ndarray:
    """
    Draw a semi-transparent HUD panel in the top-left corner.

    Args:
        frame:         BGR frame.
        fps:           Current frames per second.
        track_count:   Number of active tracks.
        threat_status: Overall scene threat level.
        mode:          'LIVE' | 'VIDEO' | 'RTSP'
    """
    h, w = frame.shape[:2]
    panel_w, panel_h = 250, 110
    overlay = frame.copy()

    # Dark background panel
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), COLORS["dark_gray"], -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    color = get_threat_color(threat_status)
    lines = [
        f"MODE:   {mode}",
        f"FPS:    {fps:.1f}",
        f"TRACKS: {track_count}",
        f"STATUS: {threat_status}",
    ]
    for i, line in enumerate(lines):
        y = 30 + i * 22
        text_color = color if "STATUS" in line else COLORS["white"]
        cv2.putText(frame, line, (14, y), FONT, 0.52, text_color, 1, cv2.LINE_AA)

    return frame


def draw_skeleton(frame: np.ndarray, landmarks, connections, color=(0, 255, 120)) -> np.ndarray:
    """
    Draw MediaPipe pose skeleton on frame.

    Args:
        frame:       BGR frame.
        landmarks:   MediaPipe NormalizedLandmarkList.
        connections: MediaPipe POSE_CONNECTIONS.
        color:       BGR color for skeleton lines.
    """
    h, w = frame.shape[:2]
    if landmarks is None:
        return frame

    points = {}
    for idx, lm in enumerate(landmarks.landmark):
        cx, cy = int(lm.x * w), int(lm.y * h)
        points[idx] = (cx, cy)
        cv2.circle(frame, (cx, cy), 4, color, -1)

    for conn in connections:
        if conn[0] in points and conn[1] in points:
            cv2.line(frame, points[conn[0]], points[conn[1]], color, 2, cv2.LINE_AA)

    return frame
