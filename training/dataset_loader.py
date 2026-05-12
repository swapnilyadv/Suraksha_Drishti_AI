"""
training/dataset_loader.py
PyTorch Dataset for loading harassment / normal video clips for training.
Supports frame-sequence extraction from video files and raw frame directories.
"""

import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List, Tuple, Optional

from training.augmentation import augment_sequence
from utils.logger import get_logger

logger = get_logger(__name__)

# Default class mapping (matches action_recognizer.py for binary training)
BINARY_LABELS = {"normal": 0, "harassment": 1}

INPUT_SIZE      = 112
SEQUENCE_LENGTH = 16


class HarassmentDataset(Dataset):
    """
    Video-sequence dataset for binary (normal / harassment) classification.

    Expected directory structure:
        root/
        ├── train/
        │   ├── harassment/   ← .mp4 / .avi / .mov clips OR frame folders
        │   └── normal/
        ├── val/
        └── test/

    Each clip is sampled as a fixed-length sequence of frames.

    Args:
        root:            Path to split root (e.g. 'datasets/train').
        sequence_length: Number of frames per sample.
        input_size:      Spatial resolution (square crop).
        augment:         Whether to apply data augmentation.
    """

    def __init__(
        self,
        root: str,
        sequence_length: int = SEQUENCE_LENGTH,
        input_size: int = INPUT_SIZE,
        augment: bool = True,
    ):
        self.root            = root
        self.sequence_length = sequence_length
        self.input_size      = input_size
        self.augment         = augment

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.samples: List[Tuple[str, int]] = []  # (clip_path, label)
        self._build_sample_list()
        logger.info(
            f"Dataset loaded | root={root} | samples={len(self.samples)} | augment={augment}"
        )

    def _build_sample_list(self) -> None:
        """Walk the root directory and collect all video file paths with labels."""
        VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

        for class_name, label in BINARY_LABELS.items():
            class_dir = os.path.join(self.root, class_name)
            if not os.path.isdir(class_dir):
                logger.warning(f"Missing class directory: {class_dir}")
                continue

            for fname in os.listdir(class_dir):
                ext = os.path.splitext(fname)[1].lower()
                if ext in VIDEO_EXTS:
                    self.samples.append((os.path.join(class_dir, fname), label))

        if not self.samples:
            logger.error(
                f"No video samples found in {self.root}. "
                "Ensure clips are placed in harassment/ and normal/ subdirectories."
            )

    def _load_frames(self, video_path: str) -> Optional[List[np.ndarray]]:
        """
        Sample `sequence_length` evenly-spaced frames from a video file.

        Returns None if the video cannot be read.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Cannot open video: {video_path}")
            return None

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1:
            cap.release()
            return None

        # Evenly space frame indices across the clip
        if total <= self.sequence_length:
            indices = list(range(total)) + [total - 1] * (self.sequence_length - total)
        else:
            step = total / self.sequence_length
            indices = [int(i * step) for i in range(self.sequence_length)]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                # Duplicate last valid frame on read failure
                if frames:
                    frames.append(frames[-1].copy())

        cap.release()
        return frames if len(frames) == self.sequence_length else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            (sequence_tensor, label)
            sequence_tensor shape: (T, C, H, W)
        """
        video_path, label = self.samples[idx]
        frames = self._load_frames(video_path)

        # Fallback: return zeros on bad video
        if frames is None:
            dummy = torch.zeros(self.sequence_length, 3, self.input_size, self.input_size)
            return dummy, label

        # Augmentation
        if self.augment:
            frames = augment_sequence(frames)

        # Convert to tensor sequence
        tensors = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(rgb))

        sequence = torch.stack(tensors)   # (T, C, H, W)
        return sequence, label


# ── Factory functions ─────────────────────────────────────────────────────────

def get_dataloaders(
    dataset_root: str = "datasets",
    sequence_length: int = SEQUENCE_LENGTH,
    input_size: int = INPUT_SIZE,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders.

    Args:
        dataset_root:    Root path containing train/, val/, test/ folders.
        sequence_length: Frames per sample.
        input_size:      Square spatial resolution.
        batch_size:      Samples per mini-batch.
        num_workers:     DataLoader worker processes.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = HarassmentDataset(
        root=os.path.join(dataset_root, "train"),
        sequence_length=sequence_length,
        input_size=input_size,
        augment=True,
    )
    val_ds = HarassmentDataset(
        root=os.path.join(dataset_root, "val"),
        sequence_length=sequence_length,
        input_size=input_size,
        augment=False,
    )
    test_ds = HarassmentDataset(
        root=os.path.join(dataset_root, "test"),
        sequence_length=sequence_length,
        input_size=input_size,
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    logger.info(
        f"DataLoaders ready | train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}"
    )
    return train_loader, val_loader, test_loader
