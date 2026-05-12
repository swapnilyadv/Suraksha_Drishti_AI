"""
tracking/tracker.py
DeepSORT-based multi-person tracker for Suraksha AI.
Wraps deep_sort_realtime to assign stable track IDs across frames.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from deep_sort_realtime.deepsort_tracker import DeepSort

from utils.logger import get_logger

logger = get_logger(__name__)


class PersonTracker:
    """
    Wraps DeepSORT to maintain persistent track IDs for detected persons.

    Usage:
        tracker = PersonTracker()
        tracks = tracker.update(detections, frame)
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_distance: float = 0.4,
        nn_budget: int = 100,
    ):
        """
        Args:
            max_age:              Frames before a lost track is deleted.
            n_init:               Frames needed to confirm a new track.
            max_cosine_distance:  Re-ID cosine distance threshold.
            nn_budget:            Max descriptors per class in gallery.
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            override_track_class=None,
            embedder="mobilenet",          # Built-in re-ID embedder
            half=True,
            bgr=True,
            embedder_gpu=True,
        )
        logger.info(
            f"PersonTracker ready | max_age={max_age} | n_init={n_init}"
        )

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections and return confirmed tracks.

        Args:
            detections: List of dicts with keys 'bbox' and 'confidence'.
                        bbox format: [x1, y1, x2, y2]
            frame:      Current BGR frame (used for appearance embedding).

        Returns:
            List of track dicts:
            [
                {
                    "track_id": int,
                    "bbox":     [x1, y1, x2, y2],
                    "confidence": float,
                }
            ]
        """
        if frame is None or frame.size == 0:
            return []

        # Convert detections to DeepSORT format:
        # ([left, top, w, h], confidence, class_label)
        ds_inputs = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1
            conf = det.get("confidence", 1.0)
            ds_inputs.append(([x1, y1, w, h], conf, "person"))

        try:
            raw_tracks = self.tracker.update_tracks(ds_inputs, frame=frame)
        except Exception as e:
            logger.error(f"Tracker update failed: {e}")
            return []

        tracks = []
        for track in raw_tracks:
            if not track.is_confirmed():
                continue

            ltrb = track.to_ltrb()  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = [int(v) for v in ltrb]

            tracks.append({
                "track_id":   int(track.track_id),
                "bbox":       [x1, y1, x2, y2],
                "confidence": round(track.det_conf or 1.0, 4),
            })

        return tracks

    def reset(self) -> None:
        """Reset tracker state (call between video clips)."""
        self.tracker = DeepSort(
            max_age=self.tracker.max_age,
            n_init=self.tracker.n_init,
        )
        logger.info("Tracker state reset.")
