"""
models/harassment_engine.py
Harassment Classification Engine for Suraksha AI.
Fuses all module outputs into a single weighted threat score and status.
"""

from typing import Dict, List, Any, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Threat status thresholds ──────────────────────────────────────────────────
THRESHOLD_SUSPICIOUS  = 0.40
THRESHOLD_HARASSMENT  = 0.70  # Default (overridden by config)

# ── Default scoring weights (configurable via config.yaml) ────────────────────
DEFAULT_WEIGHTS = {
    "w_action":   0.35,
    "w_behavior": 0.30,
    "w_pose":     0.15,
    "w_gender":   0.10,
    "w_weapon":   0.10,
}


class HarassmentEngine:
    """
    Central threat fusion engine.

    Takes outputs from all AI modules per frame and produces:
    - A final 0–1 threat score
    - A status label: 'Safe' | 'Suspicious' | 'Harassment Detected'
    - A list of involved person track IDs

    Usage:
        engine = HarassmentEngine()
        result = engine.evaluate(action_data, behavior_flags, pose_flags, genders, weapons, tracks)
    """

    def __init__(self, weights: Dict[str, float] = None):
        """
        Args:
            weights: Optional dict overriding default scoring weights.
                     Keys: w_action, w_behavior, w_pose, w_gender, w_weapon
        """
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        logger.info(f"HarassmentEngine ready | weights={self.weights}")

    def evaluate(
        self,
        action_data: Dict[int, Tuple[str, float]],
        behavior_flags: Dict[str, Any],
        pose_data: Dict[int, Dict[str, Any]],
        genders: Dict[int, str],
        weapons: List[Dict[str, Any]],
        tracks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compute threat score and status for the current frame.

        Args:
            action_data:     {track_id: (action_label, confidence)}
            behavior_flags:  Output from BehaviorAnalyzer.analyze()
            pose_data:       {track_id: pose_result_dict}
            genders:         {track_id: 'Male'|'Female'|'Unknown'}
            weapons:         List of weapon detection dicts
            tracks:          Current person tracks

        Returns:
            {
                "status":       str,        # 'Safe' | 'Suspicious' | 'Harassment Detected'
                "threat_score": float,      # 0.0 – 1.0
                "involved_ids": List[int],
                "breakdown":    Dict,       # Per-component scores
            }
        """
        w = self.weights

        # ── 1. Action score (worst-case across all tracks) ────────────────────
        action_score = 0.0
        for track_id, (action_label, conf) in action_data.items():
            from models.action_recognizer import ActionRecognizer
            per_score = self._action_threat_score(action_label, conf)
            action_score = max(action_score, per_score)

        # ── 2. Behavior score ─────────────────────────────────────────────────
        behavior_score = behavior_flags.get("behavior_score", 0.0)

        # ── 3. Pose score (worst-case across all tracks) ──────────────────────
        pose_score = 0.0
        for track_id, pose_result in pose_data.items():
            flags = pose_result.get("flags", {})
            ps = self._pose_threat_score(flags)
            pose_score = max(pose_score, ps)

        # ── 4. Gender factor (raises score if females are present) ────────────
        has_female = any(g == "Female" for g in genders.values())
        has_male   = any(g == "Male"   for g in genders.values())
        gender_factor = 1.0 if (has_female and has_male) else 0.0

        # ── 5. Weapon flag ────────────────────────────────────────────────────
        weapon_flag = 1.0 if weapons else 0.0

        # ── Weighted sum ──────────────────────────────────────────────────────
        threat_score = (
            w["w_action"]   * action_score   +
            w["w_behavior"] * behavior_score +
            w["w_pose"]     * pose_score      +
            w["w_gender"]   * gender_factor   +
            w["w_weapon"]   * weapon_flag
        )
        threat_score = round(min(max(threat_score, 0.0), 1.0), 4)

        # ── Status classification (using weights/config values) ─────────
        threshold = self.weights.get("threat_score_harassment", THRESHOLD_HARASSMENT)
        if threat_score >= threshold:
            status = "Harassment Detected"
        elif threat_score >= THRESHOLD_SUSPICIOUS:
            status = "Suspicious"
        else:
            status = "Safe"

        # ── Identify involved persons ─────────────────────────────────────────
        involved_ids = self._get_involved_ids(
            action_data, behavior_flags, weapons, genders
        )

        breakdown = {
            "action_score":   round(action_score, 4),
            "behavior_score": round(behavior_score, 4),
            "pose_score":     round(pose_score, 4),
            "gender_factor":  gender_factor,
            "weapon_flag":    weapon_flag,
        }

        if status != "Safe":
            logger.warning(
                f"THREAT | status={status} | score={threat_score} | "
                f"involved={involved_ids} | breakdown={breakdown}"
            )

        return {
            "status":       status,
            "threat_score": threat_score,
            "involved_ids": involved_ids,
            "breakdown":    breakdown,
        }

    # ── Scoring Helpers ───────────────────────────────────────────────────────

    def _action_threat_score(self, action_label: str, confidence: float) -> float:
        """Map action label + confidence to a threat score component."""
        ACTION_WEIGHTS = {
            "Normal":            0.0,
            "Harassment":        1.0,
            "Pushing":           0.7,
            "Pulling":           0.7,
            "Chasing":           0.6,
            "Hitting":           0.9,
            "Grabbing":          0.8,
            "Group Surrounding": 0.65,
            "Running Behind":    0.55,
        }
        return round(ACTION_WEIGHTS.get(action_label, 0.0) * confidence, 4)

    def _pose_threat_score(self, flags: Dict[str, Any]) -> float:
        """Derive a pose threat contribution from pose flags."""
        if not flags:
            return 0.0
        score = 0.0
        if flags.get("sudden_movement"): score += 0.40
        if flags.get("hand_raised"):     score += 0.30
        if flags.get("body_bending"):    score += 0.15
        motion = flags.get("total_motion", 0.0)
        score += min(motion / 100.0, 0.15)
        return round(min(score, 1.0), 4)

    def _get_involved_ids(
        self,
        action_data: Dict[int, Tuple[str, float]],
        behavior_flags: Dict[str, Any],
        weapons: List[Dict[str, Any]],
        genders: Dict[int, str],
    ) -> List[int]:
        """Collect IDs of persons contributing to the threat."""
        ids = set()

        # Anyone performing a non-normal action
        for tid, (label, conf) in action_data.items():
            if label != "Normal" and conf > 0.4:
                ids.add(tid)

        # IDs from unsafe behavior pairs
        for pair in behavior_flags.get("unsafe_pairs", []):
            ids.update(pair)

        # Suspects from weapon detections
        for w in weapons:
            if w.get("suspect_id") is not None:
                ids.add(w["suspect_id"])

        return sorted(ids)

    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Dynamically update scoring weights at runtime."""
        self.weights.update(new_weights)
        logger.info(f"HarassmentEngine weights updated: {self.weights}")
