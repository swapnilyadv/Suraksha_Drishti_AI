"""
training/augmentation.py
Data augmentation pipeline for Suraksha AI action recognition training.
"""

import cv2
import numpy as np
import random
from typing import List


def augment_frame(frame: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Apply a random combination of augmentations to a single BGR frame.

    Augmentations:
        - Horizontal flip
        - Brightness / contrast jitter
        - Gaussian blur
        - Rotation (±15°)
        - Gaussian noise

    Args:
        frame: BGR numpy array (H, W, 3).
        seed:  Optional random seed for reproducibility.

    Returns:
        Augmented BGR frame.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # ── Horizontal flip ───────────────────────────────────────
    if random.random() < 0.5:
        frame = cv2.flip(frame, 1)

    # ── Brightness / contrast jitter ─────────────────────────
    if random.random() < 0.5:
        alpha = random.uniform(0.7, 1.3)   # Contrast
        beta  = random.uniform(-30, 30)    # Brightness
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # ── Gaussian blur ─────────────────────────────────────────
    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        frame = cv2.GaussianBlur(frame, (ksize, ksize), 0)

    # ── Rotation ──────────────────────────────────────────────
    if random.random() < 0.4:
        h, w = frame.shape[:2]
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # ── Gaussian noise ────────────────────────────────────────
    if random.random() < 0.3:
        noise = np.random.normal(0, 10, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return frame


def augment_sequence(frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    Apply consistent augmentations to a sequence of frames.
    Spatial augmentations (flip, rotation) use the same parameters for all frames.
    Temporal augmentations (brightness, noise) vary per frame.

    Args:
        frames: List of BGR frames (one video clip).

    Returns:
        Augmented list of frames.
    """
    if not frames:
        return frames

    # Decide spatial transforms once for the whole sequence
    do_flip      = random.random() < 0.5
    do_rotate    = random.random() < 0.4
    rotate_angle = random.uniform(-15, 15) if do_rotate else 0.0
    h, w = frames[0].shape[:2]
    rot_M = cv2.getRotationMatrix2D((w / 2, h / 2), rotate_angle, 1.0) if do_rotate else None

    augmented = []
    for frame in frames:
        # ── Consistent spatial transforms ─────────────────────
        if do_flip:
            frame = cv2.flip(frame, 1)
        if do_rotate and rot_M is not None:
            frame = cv2.warpAffine(frame, rot_M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # ── Per-frame temporal transforms ──────────────────────
        if random.random() < 0.4:
            alpha = random.uniform(0.8, 1.2)
            beta  = random.uniform(-20, 20)
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        if random.random() < 0.2:
            noise = np.random.normal(0, 8, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        augmented.append(frame)

    return augmented
