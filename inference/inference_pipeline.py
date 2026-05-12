"""
inference/inference_pipeline.py
Real-time inference pipeline for Suraksha AI.
Orchestrates all AI modules in a multi-threaded loop for maximum throughput.
"""

import cv2
import time
import threading
import queue
import signal
import sys
from typing import Dict, List, Any, Optional

import numpy as np

from models.person_detector   import PersonDetector
from models.gender_classifier import GenderClassifier
from models.pose_estimator    import PoseEstimator
from models.action_recognizer import ActionRecognizer
from models.weapon_detector   import WeaponDetector
from models.harassment_engine import HarassmentEngine
from tracking.tracker         import PersonTracker
from behavior.behavior_analyzer import BehaviorAnalyzer
from alerts.alert_manager     import AlertManager
from utils.video_utils        import open_video_source, FrameBuffer, FPSCounter, get_video_properties
from utils.drawing            import (
    draw_person_box, draw_weapon_box,
    draw_harassment_alert, draw_hud, draw_skeleton,
    POSE_CONNECTIONS,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class InferencePipeline:
    """
    Main real-time inference pipeline.

    Architecture:
        Thread-1 (capture):   Read frames from source → raw_queue
        Thread-2 (inference): Pop frames → run all AI → annotated_queue
        Thread-3 (display):   Pop annotated frames → cv2.imshow

    Usage:
        pipeline = InferencePipeline(config)
        pipeline.run(source=0)           # Webcam
        pipeline.run(source="rtsp://..") # RTSP
        pipeline.run(source="video.mp4") # File
    """

    def __init__(self, config: dict, db_manager=None):
        self.cfg = config
        self._running = False

        # ── Queues ────────────────────────────────────────────
        self._raw_q  = queue.Queue(maxsize=4)
        self._disp_q = queue.Queue(maxsize=4)

        # ── Frame buffer (for clip saving) ────────────────────
        buffer_sec = config.get("video", {}).get("buffer_seconds", 15)
        fps_hint   = config.get("video", {}).get("fps", 25)
        self._frame_buffer = FrameBuffer(maxlen=int(buffer_sec * fps_hint))

        # ── State ─────────────────────────────────────────────
        self._genders: Dict[int, str]   = {}   # track_id → gender
        self._action_data: Dict         = {}   # track_id → (label, conf)
        self._pose_data: Dict           = {}   # track_id → pose result
        self._fps_counter = FPSCounter()

        # ── Load all modules ──────────────────────────────────
        device    = config.get("device", "cpu")
        paths     = config.get("paths", {})
        det_cfg   = config.get("detection", {})

        logger.info("Initialising AI modules ...")

        self.detector  = PersonDetector(
            weights=paths.get("yolo_weights", "yolov8n.pt"),
            confidence=det_cfg.get("person_confidence", 0.5),
            device=device,
            use_fp16=config.get("use_fp16", False),
        )
        self.tracker   = PersonTracker(
            max_age=config.get("tracker", {}).get("max_age", 30),
            n_init=config.get("tracker", {}).get("n_init", 3),
        )
        self.gender_clf = GenderClassifier(
            weights_path=paths.get("gender_weights"),
            device=device,
            confidence_threshold=det_cfg.get("gender_confidence", 0.6),
        )
        self.pose_est  = PoseEstimator(
            model_complexity=config.get("pose_estimator", {}).get("model_complexity", 1),
        )
        self.action_rec = ActionRecognizer(
            weights_path=paths.get("action_weights"),
            device=device,
            confidence_threshold=det_cfg.get("action_confidence", 0.55),
        )
        self.weapon_det = WeaponDetector(
            weights=paths.get("weapon_weights"),
            confidence=det_cfg.get("weapon_confidence", 0.5),
            device=device,
        )
        self.behavior  = BehaviorAnalyzer(
            unsafe_proximity_px=config.get("behavior", {}).get("unsafe_proximity_pixels", 120),
            unsafe_proximity_sec=config.get("behavior", {}).get("unsafe_proximity_duration", 5.0),
            crowd_male_count=config.get("behavior", {}).get("crowd_male_count", 3),
        )
        engine_weights = config.get("scoring", {}).copy()
        engine_weights.update(config.get("detection", {}))
        self.engine    = HarassmentEngine(weights=engine_weights)
        self.alerts    = AlertManager(config, db_manager)

        self._frame_skip = config.get("video", {}).get("frame_skip", 2)
        logger.info("All AI modules ready.")

        # ── Graceful shutdown on Ctrl+C ───────────────────────
        signal.signal(signal.SIGINT, self._signal_handler)

    # ── Thread: Frame Capture ─────────────────────────────────────────────────

    def _capture_thread(self, cap: cv2.VideoCapture) -> None:
        """Read frames from source and push to raw_queue."""
        frame_idx = 0
        while self._running:
            ret, frame = cap.read()
            if not ret:
                logger.info("Source exhausted — stopping capture thread.")
                self._running = False
                break

            frame_idx += 1
            self._frame_buffer.append(frame)

            if frame_idx % self._frame_skip != 0:
                continue   # Skip frame for speed

            try:
                self._raw_q.put((frame_idx, frame.copy()), timeout=1.0)
            except queue.Full:
                pass  # Drop frame if inference is lagging

        cap.release()

    # ── Thread: AI Inference ──────────────────────────────────────────────────

    def _inference_thread(self, fps: float, mode: str) -> None:
        """Pop frames → run full AI pipeline → push annotated frames."""
        while self._running:
            try:
                frame_idx, frame = self._raw_q.get(timeout=1.0)
            except queue.Empty:
                continue

            annotated = self._process_frame(frame, frame_idx, fps, mode)
            self._fps_counter.tick()

            try:
                self._disp_q.put(annotated, timeout=1.0)
            except queue.Full:
                pass

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        mode: str,
    ) -> np.ndarray:
        """Run all AI modules on one frame and return the annotated frame."""
        display = frame.copy()
        h, w = frame.shape[:2]

        # ── 1. Detect persons ─────────────────────────────────
        detections = self.detector.detect(frame)

        # ── 2. Track persons ──────────────────────────────────
        tracks = self.tracker.update(detections, frame)

        # ── 3. Per-track: gender, pose, action ────────────────
        active_ids = set()
        for track in tracks:
            tid  = track["track_id"]
            bbox = track["bbox"]
            active_ids.add(tid)

            # Gender (cached)
            gender, g_conf = self.gender_clf.classify_with_cache(frame, bbox, tid, frame_idx)
            self._genders[tid] = gender

            # Pose estimation
            pose_result = self.pose_est.estimate(frame, bbox, tid, frame_idx)
            self._pose_data[tid] = pose_result

            # Draw skeleton if landmarks found
            if pose_result.get("landmarks"):
                display = draw_skeleton(display, pose_result["landmarks"], POSE_CONNECTIONS)

            # Action recognition
            action_label, action_conf = self.action_rec.classify(frame, bbox, tid)
            self._action_data[tid] = (action_label, action_conf)

        # ── Clean up lost tracks ──────────────────────────────
        lost = set(self._genders.keys()) - active_ids
        for tid in lost:
            self._genders.pop(tid, None)
            self._action_data.pop(tid, None)
            self._pose_data.pop(tid, None)
            self.gender_clf.invalidate_track(tid)
            self.pose_est.invalidate_track(tid)
            self.action_rec.invalidate_track(tid)
            self.behavior.invalidate_track(tid)

        # ── 4. Weapon detection ───────────────────────────────
        weapons = self.weapon_det.detect(frame, tracks)
        for w in weapons:
            display = draw_weapon_box(
                display, w["bbox"], w["weapon_type"], w["confidence"], w.get("suspect_id")
            )

        # ── 5. Behavioral analysis ────────────────────────────
        behavior_flags = self.behavior.analyze(
            tracks, self._genders, frame_idx, frame_shape=(h, w)
        )

        # ── 6. Harassment scoring ─────────────────────────────
        threat_result = self.engine.evaluate(
            action_data=self._action_data,
            behavior_flags=behavior_flags,
            pose_data=self._pose_data,
            genders=self._genders,
            weapons=weapons,
            tracks=tracks,
        )
        
        # ── Frame Diagnostics ─────────────────────────────────
        if frame_idx % 10 == 0:
            p_count = len(tracks)
            score   = threat_result["threat_score"]
            status  = threat_result["status"]
            logger.info(f"Frame {frame_idx} | Persons: {p_count} | Score: {score:.4f} | Status: {status}")
            if p_count == 0 and frame_idx > 30:
                logger.warning("No persons detected. Ensure the video clearly shows people.")

        # ── 7. Draw person boxes ──────────────────────────────
        for track in tracks:
            tid    = track["track_id"]
            gender = self._genders.get(tid, "")
            g_conf = 0.0
            action_label, _ = self._action_data.get(tid, ("Normal", 0.0))
            display = draw_person_box(
                display,
                bbox=track["bbox"],
                track_id=tid,
                gender=gender,
                gender_conf=g_conf,
                threat_status=threat_result["status"],
                threat_score=threat_result["threat_score"],
            )

        # ── 8. Alert overlay + trigger ────────────────────────
        if threat_result["status"] == "Harassment Detected":
            display = draw_harassment_alert(display)
            self.alerts.trigger(
                frame=display,
                frame_buffer=self._frame_buffer.get_all(),
                threat_result=threat_result,
                tracks=tracks,
                fps=fps,
            )

        # ── 9. HUD overlay ────────────────────────────────────
        current_fps = self._fps_counter.tick()
        display = draw_hud(display, current_fps, len(tracks), threat_result["status"], mode)

        return display

    # ── Thread: Display ───────────────────────────────────────────────────────

    def _display_thread(self, window_title: str) -> None:
        """Pop annotated frames and display in OpenCV window."""
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        while self._running:
            try:
                frame = self._disp_q.get(timeout=1.0)
            except queue.Empty:
                continue

            cv2.imshow(window_title, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                logger.info("User pressed Q — stopping.")
                self._running = False
                break

        cv2.destroyAllWindows()

    # ── Public Interface ──────────────────────────────────────────────────────

    def run(
        self,
        source,
        camera_id: str = "CAM-01",
        display: bool = True,
    ) -> None:
        """
        Start the inference pipeline.

        Args:
            source:    int (webcam), str (file path or RTSP URL).
            camera_id: Label for alerts/reports.
            display:   Whether to show the OpenCV window.
        """
        cap  = open_video_source(source)
        props = get_video_properties(cap)
        fps  = props["fps"]

        mode = "LIVE" if isinstance(source, int) or "rtsp" in str(source).lower() else "VIDEO"
        window_title = f"Suraksha AI — {camera_id}"

        self._running = True
        logger.info(f"Pipeline starting | source={source} | mode={mode} | fps={fps:.1f}")

        threads = [
            threading.Thread(target=self._capture_thread,   args=(cap,),              daemon=True, name="capture"),
            threading.Thread(target=self._inference_thread, args=(fps, mode),          daemon=True, name="inference"),
        ]

        for t in threads:
            t.start()

        if display:
            self._display_thread(window_title)
        else:
            try:
                while self._running:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt — shutting down.")
            finally:
                self._running = False

        # ── Cleanup ──────────────────────────────────────────
        self._running = False
        for t in threads:
            if t.is_alive():
                t.join(timeout=1.0)
        
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Pipeline stopped.")

    def stop(self) -> None:
        """Programmatically stop the pipeline (e.g. from API)."""
        self._running = False
        logger.info("Pipeline stop requested.")

    def _signal_handler(self, sig, frame):
        logger.info("Signal received — stopping pipeline.")
        self._running = False
