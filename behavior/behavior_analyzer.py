"""
behavior/behavior_analyzer.py
Long-duration behavioral pattern analysis for Suraksha AI.
Detects stalking, unsafe proximity, cornering, crowd formation, and fast approach.
"""

import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


class BehaviorAnalyzer:
    """
    Analyzes multi-frame behavioral patterns between tracked persons.

    Tracks trajectories, proximity durations, and movement intentions
    to flag suspicious long-term behaviors.

    Usage:
        ba = BehaviorAnalyzer(fps=25)
        flags = ba.analyze(tracks, genders, frame_idx)
    """

    def __init__(
        self,
        fps: float = 25.0,
        history_frames: int = 90,
        unsafe_proximity_px: float = 120.0,
        unsafe_proximity_sec: float = 5.0,
        stalking_frames: int = 90,
        crowd_male_count: int = 3,
        fast_approach_px_per_frame: float = 15.0,
    ):
        """
        Args:
            fps:                      Video FPS (used for time calculations).
            history_frames:           How many past frames to store per track.
            unsafe_proximity_px:      Pixel distance below which proximity is flagged.
            unsafe_proximity_sec:     Duration (s) of unsafe proximity to trigger alert.
            stalking_frames:          Lookback window (frames) for stalking detection.
            crowd_male_count:         Min number of males to trigger crowd alert.
            fast_approach_px_per_frame: Speed threshold for fast approach.
        """
        self.fps = fps
        self.history_frames = history_frames
        self.unsafe_proximity_px = unsafe_proximity_px
        self.unsafe_proximity_frames = int(unsafe_proximity_sec * fps)
        self.stalking_frames = stalking_frames
        self.crowd_male_count = crowd_male_count
        self.fast_approach_px = fast_approach_px_per_frame

        # Per-track trajectory: deque of (frame_idx, cx, cy)
        self._trajectories: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=history_frames)
        )

        # Proximity timer: (male_id, female_id) → consecutive close frames
        self._proximity_counters: Dict[Tuple[int, int], int] = defaultdict(int)

        logger.info("BehaviorAnalyzer ready.")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _center(self, bbox: List[int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def _distance(self, c1: Tuple, c2: Tuple) -> float:
        return float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))

    def _update_trajectories(
        self, tracks: List[Dict[str, Any]], frame_idx: int
    ) -> None:
        """Append current centroid to each track's history."""
        for t in tracks:
            cx, cy = self._center(t["bbox"])
            self._trajectories[t["track_id"]].append((frame_idx, cx, cy))

    def _get_females(
        self, tracks: List[Dict[str, Any]], genders: Dict[int, str]
    ) -> List[Dict[str, Any]]:
        return [t for t in tracks if genders.get(t["track_id"]) == "Female"]

    def _get_males(
        self, tracks: List[Dict[str, Any]], genders: Dict[int, str]
    ) -> List[Dict[str, Any]]:
        return [t for t in tracks if genders.get(t["track_id"]) == "Male"]

    # ── Detection Methods ────────────────────────────────────────────────────

    def _check_unsafe_proximity(
        self,
        males: List[Dict],
        females: List[Dict],
    ) -> Tuple[bool, List[Tuple[int, int]]]:
        """
        Detect any male-female pair within unsafe proximity for too long.

        Returns:
            (flag, list of (male_id, female_id) pairs that are unsafe)
        """
        unsafe_pairs = []
        all_pairs = set()

        for m in males:
            for f in females:
                pair = (m["track_id"], f["track_id"])
                all_pairs.add(pair)
                dist = self._distance(
                    self._center(m["bbox"]),
                    self._center(f["bbox"])
                )
                if dist < self.unsafe_proximity_px:
                    self._proximity_counters[pair] += 1
                else:
                    self._proximity_counters[pair] = 0

                if self._proximity_counters[pair] >= self.unsafe_proximity_frames:
                    unsafe_pairs.append(pair)

        # Clean up stale pairs
        stale = [k for k in self._proximity_counters if k not in all_pairs]
        for k in stale:
            del self._proximity_counters[k]

        return bool(unsafe_pairs), unsafe_pairs

    def _check_stalking(
        self,
        males: List[Dict],
        females: List[Dict],
    ) -> bool:
        """
        Detect if a male is consistently moving toward a female over time.

        Logic: compare direction vectors of male trajectory and
               vector from male to female across the last N frames.
        """
        for m in males:
            mid = m["track_id"]
            m_hist = list(self._trajectories.get(mid, []))
            if len(m_hist) < self.stalking_frames // 3:
                continue

            for f in females:
                fid = f["track_id"]
                f_hist = list(self._trajectories.get(fid, []))
                if len(f_hist) < self.stalking_frames // 3:
                    continue

                # Compare direction of male movement to direction toward female
                m_old = np.array(m_hist[0][1:])
                m_new = np.array(m_hist[-1][1:])
                f_pos = np.array(f_hist[-1][1:])

                move_vec   = m_new - m_old
                toward_vec = f_pos - m_old

                move_norm   = np.linalg.norm(move_vec)
                toward_norm = np.linalg.norm(toward_vec)

                if move_norm < 5 or toward_norm < 5:
                    continue   # Too little movement

                cos_sim = np.dot(move_vec, toward_vec) / (move_norm * toward_norm + 1e-6)

                # High cosine similarity = male consistently moving toward female
                if cos_sim > 0.8:
                    return True

        return False

    def _check_fast_approach(
        self,
        males: List[Dict],
        females: List[Dict],
    ) -> bool:
        """Detect rapid movement of a male directly toward a female."""
        for m in males:
            mid = m["track_id"]
            m_hist = list(self._trajectories.get(mid, []))
            if len(m_hist) < 5:
                continue

            # Recent movement speed
            m_old = np.array(m_hist[-5][1:])
            m_new = np.array(m_hist[-1][1:])
            speed = np.linalg.norm(m_new - m_old) / 5.0   # px per frame

            if speed < self.fast_approach_px:
                continue

            # Is he moving toward any female?
            for f in females:
                f_center = np.array(self._center(f["bbox"]))
                toward_vec = f_center - m_old
                move_vec   = m_new - m_old
                norm_t = np.linalg.norm(toward_vec)
                norm_m = np.linalg.norm(move_vec)
                if norm_t < 1 or norm_m < 1:
                    continue
                cos_sim = np.dot(move_vec, toward_vec) / (norm_m * norm_t + 1e-6)
                if cos_sim > 0.7:
                    return True

        return False

    def _check_crowd_surrounding(
        self,
        males: List[Dict],
        females: List[Dict],
    ) -> bool:
        """
        Detect ≥N males within unsafe distance of the same female simultaneously.
        """
        for f in females:
            f_center = self._center(f["bbox"])
            nearby_males = sum(
                1 for m in males
                if self._distance(self._center(m["bbox"]), f_center) < self.unsafe_proximity_px * 2
            )
            if nearby_males >= self.crowd_male_count:
                return True
        return False

    def _check_cornering(
        self,
        males: List[Dict],
        females: List[Dict],
        frame_w: int = 1280,
        frame_h: int = 720,
    ) -> bool:
        """
        Detect if a female's movement is restricted (near wall + males nearby).
        """
        WALL_MARGIN = 80   # pixels from edge = "near wall"
        for f in females:
            x1, y1, x2, y2 = f["bbox"]
            cx, cy = self._center(f["bbox"])
            near_wall = (
                x1 < WALL_MARGIN or x2 > frame_w - WALL_MARGIN or
                y1 < WALL_MARGIN or y2 > frame_h - WALL_MARGIN
            )
            if not near_wall:
                continue
            nearby = sum(
                1 for m in males
                if self._distance(self._center(m["bbox"]), (cx, cy)) < self.unsafe_proximity_px * 1.5
            )
            if nearby >= 1:
                return True
        return False

    # ── Main Analysis Entry Point ─────────────────────────────────────────────

    def analyze(
        self,
        tracks: List[Dict[str, Any]],
        genders: Dict[int, str],
        frame_idx: int,
        frame_shape: Tuple[int, int] = (720, 1280),
    ) -> Dict[str, Any]:
        """
        Run all behavioral checks for the current frame.

        Args:
            tracks:       Current list of tracked persons.
            genders:      Dict mapping track_id → 'Male'|'Female'|'Unknown'.
            frame_idx:    Current frame number.
            frame_shape:  (height, width) of the frame.

        Returns:
            Behavior flags dict:
            {
                "stalking":          bool,
                "unsafe_proximity":  bool,
                "fast_approach":     bool,
                "crowd_surrounding": bool,
                "cornering":         bool,
                "unsafe_pairs":      list of (male_id, female_id),
                "behavior_score":    float (0–1),
            }
        """
        self._update_trajectories(tracks, frame_idx)

        males   = self._get_males(tracks, genders)
        females = self._get_females(tracks, genders)

        # Skip if no female present — no harassment possible
        if not females or not males:
            return self._empty_flags()

        frame_h, frame_w = frame_shape

        prox_flag, unsafe_pairs = self._check_unsafe_proximity(males, females)
        stalking  = self._check_stalking(males, females)
        fast_app  = self._check_fast_approach(males, females)
        crowd     = self._check_crowd_surrounding(males, females)
        cornering = self._check_cornering(males, females, frame_w, frame_h)

        # ── Behavior threat score ────────────────────────────
        score = 0.0
        if prox_flag: score += 0.35
        if stalking:  score += 0.30
        if fast_app:  score += 0.20
        if crowd:     score += 0.25
        if cornering: score += 0.20
        score = round(min(score, 1.0), 4)

        flags = {
            "stalking":          stalking,
            "unsafe_proximity":  prox_flag,
            "fast_approach":     fast_app,
            "crowd_surrounding": crowd,
            "cornering":         cornering,
            "unsafe_pairs":      unsafe_pairs,
            "behavior_score":    score,
        }

        if score > 0.3:
            logger.warning(f"Behavior alert | frame={frame_idx} | score={score} | {flags}")

        return flags

    def _empty_flags(self) -> Dict[str, Any]:
        return {
            "stalking":          False,
            "unsafe_proximity":  False,
            "fast_approach":     False,
            "crowd_surrounding": False,
            "cornering":         False,
            "unsafe_pairs":      [],
            "behavior_score":    0.0,
        }

    def invalidate_track(self, track_id: int) -> None:
        """Clean up state for a lost track."""
        self._trajectories.pop(track_id, None)
        stale_pairs = [k for k in self._proximity_counters if track_id in k]
        for k in stale_pairs:
            del self._proximity_counters[k]
