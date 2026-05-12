"""
models/pose_estimator.py
MediaPipe Pose estimation module for Suraksha AI.
Detects 33 body landmarks per person and extracts movement flags.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from utils.logger import get_logger

logger = get_logger(__name__)

# MediaPipe landmark indices (subset used for analysis)
LM = mp.solutions.pose.PoseLandmark
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

KEY_POINTS = {
    "nose":           LM.NOSE,
    "left_shoulder":  LM.LEFT_SHOULDER,
    "right_shoulder": LM.RIGHT_SHOULDER,
    "left_elbow":     LM.LEFT_ELBOW,
    "right_elbow":    LM.RIGHT_ELBOW,
    "left_wrist":     LM.LEFT_WRIST,
    "right_wrist":    LM.RIGHT_WRIST,
    "left_hip":       LM.LEFT_HIP,
    "right_hip":      LM.RIGHT_HIP,
    "left_knee":      LM.LEFT_KNEE,
    "right_knee":     LM.RIGHT_KNEE,
}

POSE_CACHE_INTERVAL = 3   # Re-run per track every N frames


class PoseEstimator:
    """
    Per-person MediaPipe Pose estimator.

    Processes each person's ROI independently and returns:
    - 33 landmark positions (normalized + pixel)
    - Movement flags (hand raised, body bending, sudden motion)
    - Movement vectors for action analysis
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        try:
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                smooth_landmarks=True,
            )
            logger.info("PoseEstimator ready (MediaPipe).")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Pose: {e}")
            logger.warning("PoseEstimator running in STUB mode (skeleton features disabled).")
            self._pose = None
        # Stores previous landmark positions per track for motion vectors
        self._prev_landmarks: Dict[int, np.ndarray] = {}
        self._cache: Dict[int, Tuple[Any, int]] = {}   # track_id → (result, frame_idx)

        logger.info(f"PoseEstimator ready | complexity={model_complexity}")

    def _crop_roi(
        self, frame: np.ndarray, bbox: List[int]
    ) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        h, w = frame.shape[:2]
        x2, y2 = min(x2, w), min(y2, h)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or min(roi.shape[:2]) < 30:
            return None
        return cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    def _landmarks_to_array(self, landmarks, h: int, w: int) -> np.ndarray:
        """Convert MediaPipe landmarks to (33, 2) pixel array."""
        pts = np.zeros((33, 2), dtype=np.float32)
        for i, lm in enumerate(landmarks.landmark):
            pts[i] = [lm.x * w, lm.y * h]
        return pts

    def _compute_angle(
        self, a: np.ndarray, b: np.ndarray, c: np.ndarray
    ) -> float:
        """Compute angle (degrees) at joint b given points a-b-c."""
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

    def _extract_flags(
        self,
        landmarks,
        pts: np.ndarray,
        prev_pts: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Derive pose-based flags from landmark positions.

        Returns:
            Dict with boolean flags and numeric values.
        """
        h, w = pts.max(axis=0)   # Approximate scale

        # ── Hand raised ──────────────────────────────────────
        lw = pts[LM.LEFT_WRIST]
        rw = pts[LM.RIGHT_WRIST]
        ls = pts[LM.LEFT_SHOULDER]
        rs = pts[LM.RIGHT_SHOULDER]
        hand_raised = bool(lw[1] < ls[1] or rw[1] < rs[1])

        # ── Body bending (hip-shoulder angle) ────────────────
        lh = pts[LM.LEFT_HIP]
        rh = pts[LM.RIGHT_HIP]
        mid_shoulder = (ls + rs) / 2
        mid_hip = (lh + rh) / 2
        torso_vec = mid_shoulder - mid_hip
        torso_angle = float(np.degrees(np.arctan2(abs(torso_vec[0]), abs(torso_vec[1]) + 1e-6)))
        body_bending = torso_angle > 30.0

        # ── Sudden movement (optical-flow proxy via landmark delta) ──
        sudden_movement = False
        total_motion = 0.0
        if prev_pts is not None:
            delta = np.linalg.norm(pts - prev_pts, axis=1)
            total_motion = float(delta.mean())
            sudden_movement = total_motion > 20.0    # Threshold: ~20px mean shift

        # ── Elbow angles ─────────────────────────────────────
        left_elbow_angle  = self._compute_angle(pts[LM.LEFT_SHOULDER],  pts[LM.LEFT_ELBOW],  pts[LM.LEFT_WRIST])
        right_elbow_angle = self._compute_angle(pts[LM.RIGHT_SHOULDER], pts[LM.RIGHT_ELBOW], pts[LM.RIGHT_WRIST])

        return {
            "hand_raised":      hand_raised,
            "body_bending":     body_bending,
            "sudden_movement":  sudden_movement,
            "total_motion":     round(total_motion, 2),
            "torso_angle":      round(torso_angle, 2),
            "left_elbow_angle": round(left_elbow_angle, 2),
            "right_elbow_angle":round(right_elbow_angle, 2),
        }

    def estimate(
        self,
        frame: np.ndarray,
        bbox: List[int],
        track_id: int,
        frame_idx: int,
    ) -> Dict[str, Any]:
        """
        Run pose estimation on a single person's ROI.

        Args:
            frame:     Full BGR frame.
            bbox:      [x1, y1, x2, y2] bounding box.
            track_id:  Tracker ID for caching + motion tracking.
            frame_idx: Current frame index.

        Returns:
            Dict with keys: 'landmarks', 'pts', 'flags', 'track_id'
            Returns empty flags dict if pose not detected.
        """
        # ── Cache check ──────────────────────────────────────
        if self._pose is None:
            return {"landmarks": None, "flags": {}, "track_id": track_id}

        if track_id in self._cache:
            cached_result, cached_frame = self._cache[track_id]
            if frame_idx - cached_frame < POSE_CACHE_INTERVAL:
                return cached_result

        roi = self._crop_roi(frame, bbox)
        if roi is None:
            return {"track_id": track_id, "landmarks": None, "pts": None, "flags": {}}

        roi_h, roi_w = roi.shape[:2]

        try:
            results = self._pose.process(roi)
        except Exception as e:
            logger.error(f"Pose estimation error (track {track_id}): {e}")
            return {"track_id": track_id, "landmarks": None, "pts": None, "flags": {}}

        if not results.pose_landmarks:
            return {"track_id": track_id, "landmarks": None, "pts": None, "flags": {}}

        pts = self._landmarks_to_array(results.pose_landmarks, roi_h, roi_w)
        prev_pts = self._prev_landmarks.get(track_id)
        flags = self._extract_flags(results.pose_landmarks, pts, prev_pts)

        # Update previous landmarks
        self._prev_landmarks[track_id] = pts.copy()

        output = {
            "track_id":  track_id,
            "landmarks": results.pose_landmarks,
            "pts":       pts,
            "flags":     flags,
        }
        self._cache[track_id] = (output, frame_idx)
        return output

    def pose_score(self, flags: Dict[str, Any]) -> float:
        """
        Compute a 0–1 pose threat score from extracted flags.

        Higher score = more suspicious body language.
        """
        if not flags:
            return 0.0
        score = 0.0
        if flags.get("sudden_movement"):  score += 0.40
        if flags.get("hand_raised"):      score += 0.30
        if flags.get("body_bending"):     score += 0.15
        motion = flags.get("total_motion", 0.0)
        score += min(motion / 100.0, 0.15)   # Motion intensity bonus
        return round(min(score, 1.0), 4)

    def invalidate_track(self, track_id: int) -> None:
        """Clean up state for a lost track."""
        self._prev_landmarks.pop(track_id, None)
        self._cache.pop(track_id, None)
