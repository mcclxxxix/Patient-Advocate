"""
Patient Advocate — Environment-Based Configuration
====================================================

Uses pydantic-settings to load configuration from environment variables
or .env files. This ensures secrets never appear in source code.

All thresholds have clinically-grounded defaults with citations.

Reference:
  - NCCN Guidelines v2024: ANC thresholds for neutropenia grading.
  - Bower et al. (2014), J Clin Oncol: Fatigue threshold of 4/10 as
    clinically significant cancer-related fatigue.
  - Crawford et al. (2004), NEJM: Nadir window tolerance defaults.

Author: Patient Advocate Platform Team
License: Apache-2.0
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Usage:
        settings = Settings()  # reads from .env automatically
        engine = create_async_engine(settings.database_url)
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- Database ---
    database_url: str = Field(
        default="sqlite+aiosqlite:///./patient_advocate.db",
        description="Async database URL (PostgreSQL recommended for production)"
    )

    # --- Tesseract OCR ---
    tesseract_cmd: Optional[str] = Field(
        default=None,
        description="Path to Tesseract executable (auto-detected if on PATH)"
    )

    # --- LLM Configuration (Open Source First per WHO 2024) ---
    llm_provider: str = Field(
        default="ollama",
        description="LLM provider: 'ollama', 'openai', 'anthropic'"
    )
    llm_model: str = Field(
        default="llama3",
        description="Model name (prefer open-source: llama3, deepseek-coder)"
    )
    llm_base_url: str = Field(
        default="http://localhost:11434",
        description="LLM API base URL"
    )

    # --- Clinical Thresholds ---
    anc_severe_neutropenia: float = Field(
        default=500.0,
        description="ANC threshold for severe neutropenia (cells/uL). "
                    "Source: NCCN Guidelines v2024."
    )
    anc_moderate_neutropenia: float = Field(
        default=1000.0,
        description="ANC threshold for moderate neutropenia (cells/uL). "
                    "Source: NCCN Guidelines v2024."
    )
    anc_normal_minimum: float = Field(
        default=1500.0,
        description="ANC minimum for normal range (cells/uL). "
                    "Source: NCCN Guidelines v2024."
    )
    fatigue_threshold: float = Field(
        default=4.0,
        description="Fatigue score (0-10) above which scheduling caution applies. "
                    "Source: Bower et al. (2014), J Clin Oncol."
    )
    nadir_buffer_days: int = Field(
        default=2,
        description="Extra buffer days around the nadir window for safety margin."
    )
    hitl_confidence_threshold: float = Field(
        default=0.7,
        description="Suggestions with confidence below this require mandatory HITL review."
    )

    # --- HITL ---
    hitl_timeout_seconds: int = Field(
        default=86400,
        description="Seconds to wait for human approval before auto-deferring."
    )

    # --- Firebase Cloud Messaging ---
    fcm_credentials_path: Optional[str] = Field(
        default=None,
        description="Path to Firebase service account JSON for push notifications."
    )

    # --- Logging ---
    log_level: str = Field(default="INFO")
