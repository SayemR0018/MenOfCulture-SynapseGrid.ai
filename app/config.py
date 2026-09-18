"""Environment-driven configuration. No secrets are ever hard-coded."""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is a light convenience only
    pass


@dataclass(frozen=True)
class Settings:
    model_provider: str
    model_name: str
    api_key: str | None
    api_base_url: str | None
    max_retries: int
    log_level: str
    llm_timeout_seconds: float


def get_settings() -> Settings:
    return Settings(
        model_provider=os.getenv("MODEL_PROVIDER", "mock"),
        model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        api_key=os.getenv("API_KEY"),
        api_base_url=os.getenv("API_BASE_URL") or None,
        max_retries=int(os.getenv("MAX_RETRIES", "1")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "20")),
    )
