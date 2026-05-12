"""
models/weapon_detector.py
YOLOv8-based weapon detection module for Suraksha AI.
Detects guns, knives, bats, and other weapons in real-time.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional
from ultralytics import YOLO

from utils.logger import get_logger

logger = get_logger(__name__)

WEAPON_LABELS = {
    0: "Gun",
    1: "Knife",
    2: "Pistol",
    3: "Rifle",
    4: "Bat",
    5: "Sharp Object",
    6: "Metallic Object",
}


class WeaponDetector:
    """
    YOLOv8 weapon detector with suspect association.

    If a custom weapon model is unavailable, the class runs in
    'stub mode' (returns empty detections) — the rest of the
    pipeline still works for harassment-only detection.

    Usage:
        wd = WeaponDetector(weights="saved_models/weapon_detector.pt")
        weapons = wd.detect(frame, tracks)
    """

    def __init__(
        self,
        weights: Optional[str] = None,
        confidence: float = 0.5,
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        self.confidence = confidence
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_fp16 = use_fp16 and self.device == "cuda"
        self._stub = False

        if not weights:
            logger.warning(
                "WeaponDetector: no weights provided — running in stub mode. "
                "Train and supply saved_models/weapon_detector.pt to enable weapon detection."
            )
            self._stub = True
            return

        try:
            self.model = YOLO(weights)
            self.model.to(self.device)
            logger.info(f"WeaponDetector ready | weights={weights} | device={self.device}")
        except Exception as e:
            logger.error(f"WeaponDetector failed to load ({e}). Switching to stub mode.")
            self._stub = True

    def detect(
        self,
        frame: np.ndarray,
        tracks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Detect weapons in frame and associate with nearest tracked person.

        Args:
            frame:  BGR numpy frame.
            tracks: Current list of tracked persons (with 'bbox', 'track_id').

        Returns:
            List of weapon dicts:
            [
                {
                    "weapon_type":  str,
                    "confidence":   float,
                    "bbox":         [x1, y1, x2, y2],
                    "suspect_id":   int | None,
                }
            ]
        """
        if self._stub or frame is None or frame.size == 0:
            return []

        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                half=self.use_fp16,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"WeaponDetector inference error: {e}")
            return []

        weapons = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf  = float(box.conf[0])
                cls   = int(box.cls[0])
                label = WEAPON_LABELS.get(cls, "Unknown")

                # ── Associate with nearest person ─────────────
                suspect_id = self._find_nearest_person(
                    weapon_center=((x1 + x2) / 2, (y1 + y2) / 2),
                    tracks=tracks,
                )

                weapons.append({
                    "weapon_type": label,
                    "confidence":  round(conf, 4),
                    "bbox":        [x1, y1, x2, y2],
                    "suspect_id":  suspect_id,
                })

                logger.warning(
                    f"WEAPON DETECTED: {label} ({conf:.2f}) "
                    f"near suspect ID={suspect_id}"
                )

        return weapons

    def _find_nearest_person(
        self,
        weapon_center: tuple,
        tracks: List[Dict[str, Any]],
        max_distance: float = 200.0,
    ) -> Optional[int]:
        """
        Find the track ID of the person closest to the weapon centre.

        Args:
            weapon_center: (cx, cy) of weapon bounding box.
            tracks:        Current person tracks.
            max_distance:  Maximum pixel distance to associate.

        Returns:
            track_id of nearest person, or None if no one is close enough.
        """
        wx, wy = weapon_center
        best_id, best_dist = None, float("inf")

        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            pcx = (x1 + x2) / 2
            pcy = (y1 + y2) / 2
            dist = ((wx - pcx) ** 2 + (wy - pcy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_id = t["track_id"]

        return best_id if best_dist <= max_distance else None
