"""
utils/video_utils.py
Video I/O helpers — frame extraction, resizing, clip saving, stream validation.
"""

import cv2
import os
import time
import threading
from collections import deque
from typing import Optional, Generator, Tuple
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


# ── Frame Buffer (circular, thread-safe) ─────────────────────────────────────
class FrameBuffer:
    """
    Thread-safe circular buffer for storing recent frames.
    Used to save pre-incident video clips.
    """

    def __init__(self, maxlen: int = 450):  # ~15s at 30fps
        self._buf: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, frame: np.ndarray) -> None:
        with self._lock:
            self._buf.append(frame.copy())

    def get_all(self) -> list:
        with self._lock:
            return list(self._buf)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


# ── Source Opener ─────────────────────────────────────────────────────────────
def open_video_source(source) -> cv2.VideoCapture:
    """
    Open a webcam index, RTSP URL, or file path as a VideoCapture.

    Args:
        source: int (webcam index), str (file path or RTSP URL)

    Returns:
        Opened cv2.VideoCapture object.

    Raises:
        RuntimeError if the source cannot be opened.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    logger.info(f"Opened video source: {source}")
    return cap


def get_video_properties(cap: cv2.VideoCapture) -> dict:
    """Return FPS, width, height of an opened VideoCapture."""
    return {
        "fps":    cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


# ── Frame Generator ───────────────────────────────────────────────────────────
def frame_generator(
    cap: cv2.VideoCapture,
    frame_skip: int = 1,
    resize: Optional[Tuple[int, int]] = None,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Yield (frame_index, frame) from a VideoCapture with optional skipping and resize.

    Args:
        cap:        Open VideoCapture.
        frame_skip: Yield every N-th frame (1 = no skip).
        resize:     Optional (width, height) to resize each frame.

    Yields:
        (frame_index, frame_bgr)
    """
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("Video stream ended or frame read failed.")
            break

        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        if resize:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_LINEAR)

        yield frame_idx, frame


# ── Clip Saver ────────────────────────────────────────────────────────────────
def save_clip(
    frames: list,
    output_path: str,
    fps: float = 25.0,
    size: Optional[Tuple[int, int]] = None,
) -> bool:
    """
    Save a list of BGR frames as an MP4 video file.

    Args:
        frames:      List of numpy BGR frames.
        output_path: Destination .mp4 path.
        fps:         Frames per second for output video.
        size:        (width, height); inferred from first frame if None.

    Returns:
        True on success, False on failure.
    """
    if not frames:
        logger.warning("save_clip called with empty frame list.")
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if size is None:
        h, w = frames[0].shape[:2]
        size = (w, h)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, size)

    for frame in frames:
        if frame.shape[1] != size[0] or frame.shape[0] != size[1]:
            frame = cv2.resize(frame, size)
        writer.write(frame)

    writer.release()
    logger.info(f"Clip saved: {output_path} ({len(frames)} frames)")
    return True


def save_screenshot(frame: np.ndarray, output_path: str) -> bool:
    """Save a single frame as a JPEG screenshot."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ok = cv2.imwrite(output_path, frame)
    if ok:
        logger.info(f"Screenshot saved: {output_path}")
    else:
        logger.error(f"Failed to save screenshot: {output_path}")
    return ok


# ── Validation ────────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}


def validate_video_file(path: str) -> bool:
    """
    Check file extension and that OpenCV can open it.

    Returns:
        True if valid, False otherwise.
    """
    ext = path.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"Rejected file extension: .{ext}")
        return False

    cap = cv2.VideoCapture(path)
    valid = cap.isOpened()
    cap.release()

    if not valid:
        logger.warning(f"OpenCV could not open file: {path}")
    return valid


# ── FPS Counter ───────────────────────────────────────────────────────────────
class FPSCounter:
    """Sliding-window FPS tracker."""

    def __init__(self, window: int = 30):
        self._times: deque = deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0
