"""
Patient Advocate — FastAPI Application Factory
===============================================

``create_app`` is the single entry-point for both production (``uvicorn``)
and test-time (``TestClient`` / ``AsyncClient``) instantiation.

Lifespan responsibilities:
  1. Load ``Settings`` (from env / .env file).
  2. Initialise ``SuggestionStore`` and ``RLHFStore`` singletons.
  3. Build ``HITLService`` with the singletons.
  4. Wire the service into FastAPI's dependency-override system so the router's
     ``Depends(_get_hitl_service)`` placeholder is replaced transparently.
  5. Start a background eviction task (every 5 minutes) that removes terminal
     decisions from the in-memory store to prevent unbounded memory growth.
  6. Clean up on shutdown.

Router layout:
  /hitl/*       – HITL decision gate (hitl_router.py)
  /ethics/*     – Ethics dashboard (placeholder; implemented in Phase 2)
  /health       – Liveness probe (no dependencies; always returns 200)

Author: Patient Advocate Platform Team
License: Apache-2.0
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from patient_advocate.api.hitl_router import _get_hitl_service, router as hitl_router
from patient_advocate.api.hitl_service import HITLService
from patient_advocate.api.rlhf_store import RLHFStore
from patient_advocate.api.suggestion_store import SuggestionStore
from patient_advocate.core.config import Settings

logger = logging.getLogger(__name__)

# How often the background eviction task runs (seconds).
_EVICTION_INTERVAL_SECONDS = 300  # 5 minutes


async def _run_eviction(store: SuggestionStore) -> None:
    """Background coroutine that periodically evicts terminal decisions."""
    while True:
        await asyncio.sleep(_EVICTION_INTERVAL_SECONDS)
        try:
            evicted = await store.evict_decided()
            if evicted:
                logger.info("SuggestionStore: evicted %d terminal decision(s)", evicted)
        except Exception:
            logger.exception("Eviction task encountered an unexpected error")


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """
    FastAPI application factory.

    Parameters
    ----------
    settings:
        Optional pre-constructed ``Settings`` object.  If ``None``, settings
        are loaded from environment variables / .env file at call time.
        Passing a custom instance is useful in tests to override timeout
        values, RLHF path, etc. without touching the environment.

    Returns
    -------
    FastAPI
        A fully configured application ready to be served by uvicorn.
    """
    if settings is None:
        settings = Settings()

    # ------------------------------------------------------------------
    # Lifespan: initialise singletons, start background tasks, clean up.
    # ------------------------------------------------------------------
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # --- Startup ---
        logger.info("Patient Advocate API starting up")

        rlhf_path = Path("rlhf_feedback.jsonl")
        suggestion_store = SuggestionStore()
        rlhf_store = RLHFStore(path=rlhf_path)
        hitl_service = HITLService(
            suggestion_store=suggestion_store,
            rlhf_store=rlhf_store,
            hitl_timeout_seconds=settings.hitl_timeout_seconds,
        )

        # Wire the singleton into the dependency override system.
        app.dependency_overrides[_get_hitl_service] = lambda: hitl_service

        # Background eviction task.
        eviction_task = asyncio.create_task(
            _run_eviction(suggestion_store),
            name="suggestion_store_eviction",
        )

        logger.info(
            "HITL service ready (timeout=%ds, rlhf_path=%s)",
            settings.hitl_timeout_seconds,
            rlhf_path,
        )

        yield  # Application is live.

        # --- Shutdown ---
        logger.info("Patient Advocate API shutting down")
        eviction_task.cancel()
        try:
            await eviction_task
        except asyncio.CancelledError:
            pass
        app.dependency_overrides.clear()

    # ------------------------------------------------------------------
    # App construction
    # ------------------------------------------------------------------
    app = FastAPI(
        title="Patient Advocate API",
        description=(
            "Level 2 Multi-Agent Personal Medical AI — "
            "Human-in-the-Loop decision gate and ethics dashboard."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    app.include_router(hitl_router)

    # Placeholder for Phase 2 ethics dashboard — imported lazily to avoid
    # circular imports during unit tests that stub ethics modules.
    try:
        from patient_advocate.ethics.router import router as ethics_router  # type: ignore[import]

        app.include_router(ethics_router, prefix="/ethics", tags=["ethics"])
        logger.info("Ethics dashboard router registered")
    except ImportError:
        logger.warning(
            "Ethics router not found — /ethics endpoints will return 404. "
            "Implement patient_advocate.ethics.router to enable the dashboard."
        )

    # ------------------------------------------------------------------
    # Core endpoints
    # ------------------------------------------------------------------
    @app.get("/health", tags=["ops"], summary="Liveness probe")
    async def health() -> JSONResponse:
        """
        Simple liveness probe.  Returns 200 with ``{"status": "ok"}`` as long
        as the event loop is running.  Does NOT check database connectivity or
        RLHF file writability — use a dedicated readiness probe for that.
        """
        return JSONResponse(content={"status": "ok"})

    return app


# ---------------------------------------------------------------------------
# ASGI entry-point for ``uvicorn patient_advocate.api.app:app``
# ---------------------------------------------------------------------------
app = create_app()
