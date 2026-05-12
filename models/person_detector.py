"""
models/person_detector.py
YOLOv8-based person detection module for Suraksha AI.
Detects all humans in a frame and returns bounding boxes with confidence scores.
"""

import numpy as np
import torch
from typing import List, Dict, Any
from ultralytics import YOLO

from utils.logger import get_logger

logger = get_logger(__name__)

# COCO class index for 'person'
PERSON_CLASS_ID = 0


class PersonDetector:
    """
    Wraps YOLOv8 for real-time person detection.

    Usage:
        detector = PersonDetector(weights="yolov8n.pt", device="cuda")
        detections = detector.detect(frame)
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        confidence: float = 0.5,
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        """
        Args:
            weights:    Path to YOLOv8 .pt weights file.
            confidence: Minimum detection confidence threshold.
            device:     'cuda' or 'cpu'.
            use_fp16:   Use half-precision on CUDA for faster inference.
        """
        self.confidence = confidence
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_fp16 = use_fp16 and self.device == "cuda"

        logger.info(f"Loading PersonDetector | weights={weights} | device={self.device}")
        self.model = YOLO(weights)
        self.model.to(self.device)
        logger.info("PersonDetector ready.")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on a single BGR frame and return person detections.

        Args:
            frame: BGR numpy array (H, W, 3).

        Returns:
            List of detection dicts:
            [
                {
                    "bbox":       [x1, y1, x2, y2],  # ints
                    "confidence": float,
                    "class_id":   0,
                }
            ]
        """
        if frame is None or frame.size == 0:
            return []

        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                classes=[PERSON_CLASS_ID],
                half=self.use_fp16,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"PersonDetector inference error: {e}")
            return []

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                if cls == PERSON_CLASS_ID and conf >= self.confidence:
                    detections.append({
                        "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": round(conf, 4),
                        "class_id":   cls,
                    })

        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        Run batched inference on multiple frames.

        Args:
            frames: List of BGR numpy arrays.

        Returns:
            List of detection lists (one per frame).
        """
        if not frames:
            return []

        try:
            results = self.model.predict(
                source=frames,
                conf=self.confidence,
                classes=[PERSON_CLASS_ID],
                half=self.use_fp16,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"PersonDetector batch inference error: {e}")
            return [[] for _ in frames]

        all_detections = []
        for result in results:
            frame_dets = []
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls  = int(box.cls[0])
                    if cls == PERSON_CLASS_ID:
                        frame_dets.append({
                            "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": round(conf, 4),
                            "class_id":   cls,
                        })
            all_detections.append(frame_dets)

        return all_detections
