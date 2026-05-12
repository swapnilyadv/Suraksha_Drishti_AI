"""
api/main_api.py
FastAPI application factory for Suraksha AI backend.
"""

import os
import yaml
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from database.db_manager import DBManager
from inference.inference_pipeline import InferencePipeline
from api.routes import incidents, video, stream
from utils.logger import get_api_logger

logger = get_api_logger()

CONFIG_PATH = "configs/config.yaml"


def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup, clean up on shutdown."""
    logger.info("Suraksha AI API starting up ...")

    config = _load_config(CONFIG_PATH)
    app.state.config = config

    db_path = config.get("paths", {}).get("db_path", "database/incidents.db")
    app.state.db = DBManager(db_path=db_path)

    app.state.pipeline = InferencePipeline(config=config, db_manager=app.state.db)

    logger.info("Suraksha AI API ready.")
    yield

    logger.info("Suraksha AI API shutting down ...")
    try:
        app.state.pipeline.stop()
    except Exception:
        pass


# ── App Factory ───────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title="Suraksha AI — Women Harassment Detection API",
        description=(
            "Real-time AI-powered surveillance API for detecting women harassment "
            "incidents from CCTV, webcam, and uploaded videos."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────
    app.include_router(video.router)
    app.include_router(stream.router)
    app.include_router(incidents.router)

    # ── Health check ──────────────────────────────────────────
    @app.get("/health", tags=["Health"])
    async def health():
        return JSONResponse({"status": "ok", "service": "Suraksha AI"})

    # ── Root ──────────────────────────────────────────────────
    @app.get("/", tags=["Root"])
    async def root():
        return {
            "message": "Suraksha AI Harassment Detection System",
            "docs":    "/docs",
            "version": "1.0.0",
        }

    return app


app = create_app()
