"""
models/gender_classifier.py
Lightweight MobileNetV2-based gender classification for Suraksha AI.
Classifies each tracked person's ROI as Male or Female.
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from typing import Dict, Tuple, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

GENDER_LABELS = ["Male", "Female"]
GENDER_CACHE_INTERVAL = 5   # Re-run inference every N frames per track


class GenderClassifier:
    """
    MobileNetV2-based gender classifier.

    Usage:
        clf = GenderClassifier(weights_path="saved_models/gender_classifier.pt")
        gender, conf = clf.classify(frame, bbox)
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.6,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = confidence_threshold
        self._cache: Dict[int, Tuple[str, float, int]] = {}  # track_id → (gender, conf, last_frame)

        # ── Build model ──────────────────────────────────────
        self.model = self._build_model()
        self.model.to(self.device)

        # ── Load weights ─────────────────────────────────────
        if weights_path:
            try:
                state = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state)
                logger.info(f"GenderClassifier weights loaded: {weights_path}")
            except Exception as e:
                logger.warning(f"Could not load gender weights ({e}). Running with random init.")
        else:
            logger.warning("No gender weights provided. Predictions will be random until trained.")

        self.model.eval()

        # ── Preprocessing ────────────────────────────────────
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        logger.info(f"GenderClassifier ready | device={self.device}")

    def _build_model(self) -> nn.Module:
        """Build MobileNetV2 with a custom 2-class head."""
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 2),
        )
        return backbone

    def _crop_roi(self, frame: np.ndarray, bbox: list) -> Optional[np.ndarray]:
        """Crop and validate person ROI from frame."""
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
            return None
        return cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    @torch.no_grad()
    def classify(
        self, frame: np.ndarray, bbox: list
    ) -> Tuple[str, float]:
        """
        Classify gender from a bounding box ROI.

        Args:
            frame: Full BGR frame.
            bbox:  [x1, y1, x2, y2] person bounding box.

        Returns:
            (gender_label, confidence)  e.g. ("Female", 0.91)
        """
        roi = self._crop_roi(frame, bbox)
        if roi is None:
            return "Unknown", 0.0

        try:
            tensor = self.transform(roi).unsqueeze(0).to(self.device)
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            idx = int(probs.argmax())
            conf = float(probs[idx])
            label = GENDER_LABELS[idx] if conf >= self.confidence_threshold else "Unknown"
            return label, round(conf, 4)
        except Exception as e:
            logger.error(f"Gender inference error: {e}")
            return "Unknown", 0.0

    def classify_with_cache(
        self,
        frame: np.ndarray,
        bbox: list,
        track_id: int,
        frame_idx: int,
    ) -> Tuple[str, float]:
        """
        Classify gender with per-track caching to reduce compute cost.

        Returns cached result if within GENDER_CACHE_INTERVAL frames.
        """
        if track_id in self._cache:
            cached_gender, cached_conf, last_frame = self._cache[track_id]
            if frame_idx - last_frame < GENDER_CACHE_INTERVAL:
                return cached_gender, cached_conf

        gender, conf = self.classify(frame, bbox)
        self._cache[track_id] = (gender, conf, frame_idx)
        return gender, conf

    def invalidate_track(self, track_id: int) -> None:
        """Remove a track from the gender cache when it disappears."""
        self._cache.pop(track_id, None)
