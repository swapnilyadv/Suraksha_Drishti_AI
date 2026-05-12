"""
main.py
Suraksha AI — Unified CLI Entry Point

Usage:
    # Live webcam detection (with display window)
    python main.py --mode webcam

    # Specific webcam index
    python main.py --mode webcam --source 1

    # RTSP stream
    python main.py --mode rtsp --source rtsp://192.168.1.100:554/stream

    # Uploaded video file
    python main.py --mode video --source path/to/video.mp4

    # Launch FastAPI backend server
    python main.py --mode api

    # No display (headless server mode)
    python main.py --mode webcam --no-display
"""

import argparse
import os
import sys
import yaml

from utils.logger import get_logger

logger = get_logger("suraksha.main")


# ── Config Loader ─────────────────────────────────────────────────────────────
def load_config(config_path: str = "configs/config.yaml") -> dict:
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded from: {config_path}")
    return cfg


# ── Argument Parser ───────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Suraksha AI — Women Harassment Detection System",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["webcam", "rtsp", "video", "api"],
        help=(
            "Operation mode:\n"
            "  webcam  — Live webcam detection\n"
            "  rtsp    — RTSP IP camera stream\n"
            "  video   — Process uploaded video file\n"
            "  api     — Launch FastAPI HTTP server\n"
        ),
    )
    p.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video source: webcam index (int), RTSP URL, or file path.",
    )
    p.add_argument(
        "--camera-id",
        type=str,
        default="CAM-01",
        help="Camera label used in alerts and incident reports (default: CAM-01).",
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file (default: configs/config.yaml).",
    )
    p.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV display window (headless / server mode).",
    )
    p.add_argument(
        "--host",
        type=str,
        default=None,
        help="API server host (overrides config). Used with --mode api.",
    )
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help="API server port (overrides config). Used with --mode api.",
    )

    return p.parse_args()


# ── Mode Handlers ─────────────────────────────────────────────────────────────
def run_detection(config: dict, source, camera_id: str, display: bool) -> None:
    """Initialise and run the full inference pipeline."""
    from database.db_manager import DBManager
    from inference.inference_pipeline import InferencePipeline

    db_path = config.get("paths", {}).get("db_path", "database/incidents.db")
    db      = DBManager(db_path=db_path)

    pipeline = InferencePipeline(config=config, db_manager=db)
    pipeline.run(source=source, camera_id=camera_id, display=display)


def run_api(config: dict, host: str = None, port: int = None) -> None:
    """Launch the FastAPI server with Uvicorn."""
    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn is not installed. Run: pip install uvicorn")
        sys.exit(1)

    api_cfg  = config.get("api", {})
    _host = host or api_cfg.get("host", "0.0.0.0")
    _port = port or api_cfg.get("port", 8000)

    logger.info(f"Starting FastAPI server at http://{_host}:{_port}")
    logger.info(f"API docs available at: http://{_host}:{_port}/docs")

    uvicorn.run(
        "api.main_api:app",
        host=_host,
        port=_port,
        reload=False,
        log_level="info",
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    args   = parse_args()
    config = load_config(args.config)

    # ── Print startup banner ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  🛡️  SURAKSHA AI — Women Harassment Detection System")
    print("=" * 60)
    print(f"  Mode:      {args.mode.upper()}")
    if args.source:
        print(f"  Source:    {args.source}")
    print(f"  Camera ID: {args.camera_id}")
    print(f"  Config:    {args.config}")
    print(f"  Display:   {'OFF (headless)' if args.no_display else 'ON'}")
    print("=" * 60 + "\n")

    # ── Dispatch to mode ──────────────────────────────────────
    if args.mode == "webcam":
        source = int(args.source) if args.source and args.source.isdigit() else (
            config.get("video", {}).get("webcam_index", 0)
        )
        run_detection(config, source, args.camera_id, not args.no_display)

    elif args.mode == "rtsp":
        source = args.source or config.get("video", {}).get("rtsp_url", "")
        if not source:
            logger.error("RTSP mode requires --source <rtsp_url> or rtsp_url in config.yaml")
            sys.exit(1)
        run_detection(config, source, args.camera_id, not args.no_display)

    elif args.mode == "video":
        if not args.source:
            logger.error("Video mode requires --source <path_to_video>")
            sys.exit(1)
        if not os.path.exists(args.source):
            logger.error(f"Video file not found: {args.source}")
            sys.exit(1)
        run_detection(config, args.source, args.camera_id, not args.no_display)

    elif args.mode == "api":
        run_api(config, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
