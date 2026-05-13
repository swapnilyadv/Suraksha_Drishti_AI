"""
models/action_recognizer.py
CNN + LSTM action recognition model for Suraksha AI.
Classifies sequences of person ROI frames into harassment-related action categories.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from collections import defaultdict, deque
from torchvision import models, transforms
from typing import Dict, List, Tuple, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

ACTION_LABELS = {
    0: "Normal",
    1: "Harassment",
    2: "Weapon",
    3: "Male Faces",
    4: "Female Faces",
}

SEQUENCE_LENGTH = 16    # Number of frames per action window
INPUT_SIZE      = 112   # Spatial resolution fed to CNN


# ── CNN Feature Extractor ─────────────────────────────────────────────────────
class CNNEncoder(nn.Module):
    """ResNet18 backbone truncated at the avgpool layer, outputs 512-d feature."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        # Remove the final FC layer — keep up to avgpool
        self.features = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, 512)
        """
        out = self.features(x)
        return out.flatten(1)


# ── CNN + LSTM Classifier ─────────────────────────────────────────────────────
class ActionRecognizerModel(nn.Module):
    """
    Combines CNNEncoder + LSTM for video sequence classification.

    Input:  (B, T, C, H, W)  — batch of frame sequences
    Output: (B, num_classes)  — class logits
    """

    def __init__(
        self,
        num_classes: int = 5,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.5,
        pretrained: bool = True,
    ):
        super().__init__()
        self.cnn    = CNNEncoder(pretrained=pretrained)
        self.lstm   = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        # Encode each frame
        x = x.view(B * T, C, H, W)
        features = self.cnn(x)               # (B*T, 512)
        features = features.view(B, T, -1)   # (B, T, 512)
        # Sequence modelling
        out, _ = self.lstm(features)         # (B, T, hidden)
        out = self.head(out[:, -1, :])       # Use last timestep
        return out


# ── Runtime Inference Wrapper ─────────────────────────────────────────────────
class ActionRecognizer:
    """
    Manages per-track frame sliding windows and runs action recognition.

    Usage:
        ar = ActionRecognizer(weights_path="saved_models/action_recognizer.pt")
        action, conf = ar.classify(frame, bbox, track_id)
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.55,
        sequence_length: int = SEQUENCE_LENGTH,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = confidence_threshold
        self.sequence_length = sequence_length

        # Per-track sliding window of preprocessed tensors
        self._windows: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=sequence_length)
        )

        # ── Build + load model ───────────────────────────────
        self.model = ActionRecognizerModel(num_classes=len(ACTION_LABELS))
        self.model.to(self.device)

        if weights_path:
            try:
                state = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state)
                logger.info(f"ActionRecognizer weights loaded: {weights_path}")
            except Exception as e:
                logger.warning(f"Could not load action weights ({e}). Using random init.")
        else:
            logger.warning("No action weights provided. Train first via training/train.py.")

        self.model.eval()

        # ── Preprocessing ────────────────────────────────────
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        logger.info(f"ActionRecognizer ready | device={self.device} | seq_len={sequence_length}")

    def _preprocess_roi(self, frame: np.ndarray, bbox: List[int]) -> Optional[torch.Tensor]:
        """Crop ROI and convert to normalised tensor."""
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or min(roi.shape[:2]) < 20:
            return None
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        return self.transform(roi_rgb)

    @torch.no_grad()
    def classify(
        self,
        frame: np.ndarray,
        bbox: List[int],
        track_id: int,
        frame_idx: int = 0,
    ) -> Tuple[str, float]:
        """
        Append current frame to track's sliding window and classify if full.

        Args:
            frame:    Full BGR frame.
            bbox:     [x1, y1, x2, y2] person bbox.
            track_id: Tracker ID.

        Returns:
            (action_label, confidence) — "Normal", 0.0 if window not full yet.
        """
        tensor = self._preprocess_roi(frame, bbox)
        if tensor is not None:
            self._windows[track_id].append(tensor)

        window = self._windows[track_id]
        if len(window) < self.sequence_length:
            return "Normal", 0.0   # Not enough frames yet

        # Stack sequence: (1, T, C, H, W)
        seq = torch.stack(list(window)).unsqueeze(0).to(self.device)

        try:
            logits = self.model(seq)
            probs  = torch.softmax(logits, dim=1)[0]
            idx    = int(probs.argmax())
            conf   = float(probs[idx])
            label  = ACTION_LABELS.get(idx, "Unknown")
            
            # DIAGNOSTIC LOG: Show raw probability of Harassment (Class 1)
            h_prob = float(probs[1]) if len(probs) > 1 else 0.0
            if frame_idx % 10 == 0:
                logger.info(f"Track {track_id} | Harassment Prob: {h_prob:.4f} | Prediction: {label}")

            if conf < self.confidence_threshold:
                label = "Normal"
            return label, round(conf, 4)
        except Exception as e:
            logger.error(f"Action inference error (track {track_id}): {e}")
            return "Normal", 0.0

    def action_score(self, action_label: str, confidence: float) -> float:
        """
        Map action label + confidence to a 0–1 threat contribution score.
        """
        # Threat weights per action class
        THREAT_WEIGHTS = {
            "Normal":           0.0,
            "Harassment":       1.0,
            "Weapon":           1.0,
            "Male Faces":       0.0,
            "Female Faces":     0.0,
        }
        w = THREAT_WEIGHTS.get(action_label, 0.0)
        return round(w * confidence, 4)

    def invalidate_track(self, track_id: int) -> None:
        """Clear sliding window for a lost track."""
        self._windows.pop(track_id, None)
