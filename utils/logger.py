"""
utils/logger.py
Centralised logging setup for Suraksha AI.
Provides both console and rotating file logging.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime


def get_logger(name: str, log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    """
    Create and return a named logger with console + rotating file handlers.

    Args:
        name:    Logger name (usually __name__ of the calling module).
        log_dir: Directory where log files will be saved.
        level:   Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').

    Returns:
        Configured logging.Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # ── Formatter ───────────────────────────────────────────────
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console Handler ─────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(console_handler)

    # ── Rotating File Handler ───────────────────────────────────
    log_file = os.path.join(log_dir, f"suraksha_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
    logger.addHandler(file_handler)

    return logger


def get_api_logger() -> logging.Logger:
    """Logger specifically for FastAPI request/response logging."""
    return get_logger("suraksha.api")


def get_detection_logger() -> logging.Logger:
    """Logger specifically for detection events."""
    return get_logger("suraksha.detection")


def get_alert_logger() -> logging.Logger:
    """Logger specifically for alert events."""
    return get_logger("suraksha.alerts")
