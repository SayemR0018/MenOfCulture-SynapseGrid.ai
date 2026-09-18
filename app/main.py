"""FastAPI application entry point.

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging

from fastapi import FastAPI

from app.config import get_settings
from app.routes import router

_settings = get_settings()
logging.basicConfig(
    level=getattr(logging, _settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(
    title="SynapseGrid.ai — Smart Campus Energy Optimization",
    description=(
        "LLM-assisted operator directive interpretation + deterministic "
        "energy optimization for the BUP CSE FEST 2026 preliminary round."
    ),
    version="1.0.0",
)

app.include_router(router)
