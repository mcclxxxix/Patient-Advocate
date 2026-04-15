"""
Patient Advocate — HITL Business Logic Service
===============================================

Pure business logic for the Human-in-the-Loop decision pipeline.  This class
has no FastAPI dependency; it can be instantiated and tested in plain pytest
without spinning up an HTTP server.

Pipeline for ``process_decision``:

  1. Lookup suggestion in ``SuggestionStore``.
  2. Verify patient ownership.
     HIPAA note: wrong patient → 404, *not* 403.  Confirming that a suggestion
     *exists* for a different patient leaks PHI; a 404 is the safe response.
  3. Check timeout.  If expired, auto-defer the suggestion and raise
     ``HITLTimeoutError`` after writing the audit record.
  4. Write ``TransparencyRecord`` (Glass Box guarantee — every path writes).
  5. Write ``RLHFFeedbackRecord`` if decision == REJECTED.
  6. Mark suggestion as decided in the store.
  7. Return ``HITLDecisionResponse``.

Author: Patient Advocate Platform Team
License: Apache-2.0
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

from patient_advocate.api.rlhf_store import RLHFStore
from patient_advocate.api.schemas import (
    HITLDecisionRequest,
    HITLDecisionResponse,
    SuggestionPayload,
)
from patient_advocate.api.suggestion_store import SuggestionStore
from patient_advocate.core.exceptions import (
    AuditWriteError,
    HITLRejectionError,
    HITLTimeoutError,
)
from patient_advocate.core.models import (
    HITLDecision,
    RLHFFeedbackRecord,
    TransparencyRecord,
)

logger = logging.getLogger(__name__)


class HITLService:
    """
    Orchestrates the full HITL decision lifecycle.

    All dependencies are injected so that unit tests can substitute fakes
    for ``SuggestionStore``, ``RLHFStore``, and the audit writer.

    Parameters
    ----------
    suggestion_store:
        Registry of pending/decided suggestions.
    rlhf_store:
        Append-only feedback log for rejected suggestions.
    hitl_timeout_seconds:
        Seconds after registration before a suggestion is considered timed-out.
    audit_writer:
        Async callable ``(TransparencyRecord) -> None`` used to persist audit
        records.  Defaults to a structured logging stub so the service can be
        run without a database.
    """

    def __init__(
        self,
        suggestion_store: SuggestionStore,
        rlhf_store: RLHFStore,
        hitl_timeout_seconds: int = 86400,
        audit_writer=None,
    ) -> None:
        self._store = suggestion_store
        self._rlhf = rlhf_store
        self._timeout = hitl_timeout_seconds
        self._audit_writer = audit_writer or _default_audit_writer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_decision(
        self, request: HITLDecisionRequest
    ) -> HITLDecisionResponse:
        """
        Process a patient's HITL decision.

        Every exit path — approved, rejected, deferred, timed-out, not-found —
        writes a ``TransparencyRecord`` to satisfy the Glass Box guarantee.

        Returns
        -------
        HITLDecisionResponse
            The recorded decision with ``acknowledged=True``.

        Raises
        ------
        KeyError
            When the suggestion does not exist (HTTP layer maps this to 404).
        HITLTimeoutError
            When the timeout window has expired (suggestion is auto-deferred).
        HITLRejectionError
            When the decision is REJECTED (informational; the HTTP layer may
            choose to return 200 anyway).
        AuditWriteError
            When the audit write fails (HTTP layer maps to 500).
        """
        suggestion_id = request.suggestion_id
        patient_id = request.patient_id

        # ----------------------------------------------------------------
        # Step 1: Lookup
        # ----------------------------------------------------------------
        payload = await self._store.get(suggestion_id)
        if payload is None:
            # HIPAA: 404 regardless of *why* we can't serve this record.
            await self._write_audit(
                suggestion_id=suggestion_id,
                payload=None,
                decision=None,
                reasoning=f"Suggestion {suggestion_id} not found during HITL lookup",
                agent_name="hitl_service",
            )
            raise KeyError(f"Suggestion {suggestion_id} not found")

        # ----------------------------------------------------------------
        # Step 2: Patient ownership — HIPAA-compliant 404
        # ----------------------------------------------------------------
        if payload.patient_id != patient_id:
            await self._write_audit(
                suggestion_id=suggestion_id,
                payload=payload,
                decision=None,
                reasoning=(
                    f"Patient ownership mismatch: requestor={patient_id}, "
                    f"owner=<redacted> — returning 404 per HIPAA safe-harbour"
                ),
                agent_name="hitl_service",
            )
            # Surface as a KeyError → 404 (not 403, which leaks PHI)
            raise KeyError(f"Suggestion {suggestion_id} not found")

        # ----------------------------------------------------------------
        # Step 3: Timeout check
        # ----------------------------------------------------------------
        timed_out = await self._store.is_timed_out(suggestion_id, self._timeout)
        if timed_out:
            # Auto-defer and surface as timeout error.
            try:
                await self._store.mark_decided(suggestion_id, HITLDecision.DEFERRED)
            except ValueError:
                # Already decided by a concurrent request; that's fine.
                pass

            await self._write_audit(
                suggestion_id=suggestion_id,
                payload=payload,
                decision=HITLDecision.DEFERRED,
                reasoning=(
                    f"HITL timeout after {self._timeout}s — auto-deferred. "
                    f"No patient action received within the review window."
                ),
                agent_name="hitl_service",
            )
            raise HITLTimeoutError(
                f"Suggestion {suggestion_id} exceeded the {self._timeout}s review window "
                "and has been auto-deferred.",
                agent_name="hitl_service",
            )

        # ----------------------------------------------------------------
        # Step 4: Write TransparencyRecord (Glass Box — every path)
        # ----------------------------------------------------------------
        await self._write_audit(
            suggestion_id=suggestion_id,
            payload=payload,
            decision=request.decision,
            reasoning=(
                f"Patient {patient_id} submitted decision '{request.decision.value}'. "
                + (f"Feedback: {request.feedback_text}" if request.feedback_text else "No feedback text.")
            ),
            agent_name="hitl_service",
        )

        # ----------------------------------------------------------------
        # Step 5: RLHF record for rejections
        # ----------------------------------------------------------------
        if request.decision == HITLDecision.REJECTED:
            rlhf_record = RLHFFeedbackRecord(
                feedback_id=uuid4(),
                suggestion_id=suggestion_id,
                patient_id=patient_id,
                decision=HITLDecision.REJECTED,
                suggestion_snapshot=payload.model_dump(mode="json"),
                feedback_text=request.feedback_text,
                timestamp=datetime.now(tz=timezone.utc),
            )
            try:
                await self._rlhf.append(rlhf_record)
            except OSError as exc:
                raise AuditWriteError(
                    f"Failed to write RLHF record for suggestion {suggestion_id}: {exc}",
                    agent_name="hitl_service",
                ) from exc

        # ----------------------------------------------------------------
        # Step 6: Mark decided in store
        # ----------------------------------------------------------------
        await self._store.mark_decided(suggestion_id, request.decision)

        # ----------------------------------------------------------------
        # Step 7: Build and return response
        # ----------------------------------------------------------------
        response = HITLDecisionResponse(
            suggestion_id=suggestion_id,
            decision=request.decision,
            acknowledged=True,
        )

        if request.decision == HITLDecision.REJECTED:
            # Raise *after* returning is not possible — so we log and let
            # the router decide whether to surface this as a non-200 status.
            # The service raises so callers have the option to handle it.
            raise HITLRejectionError(
                f"Suggestion {suggestion_id} was rejected by patient {patient_id}.",
                agent_name="hitl_service",
            )

        return response

    async def register_suggestion(self, payload: SuggestionPayload) -> None:
        """Delegate registration to the suggestion store."""
        await self._store.register(payload)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _write_audit(
        self,
        *,
        suggestion_id: UUID,
        payload: SuggestionPayload | None,
        decision: HITLDecision | None,
        reasoning: str,
        agent_name: str,
    ) -> None:
        """
        Build and persist a ``TransparencyRecord``.

        Errors here are wrapped in ``AuditWriteError`` and re-raised because
        audit integrity is non-negotiable per the Glass Box Imperative.
        """
        record = TransparencyRecord(
            event_id=uuid4(),
            suggestion_id=suggestion_id,
            agent_name=agent_name,
            conflict_type=None,  # set from payload if available
            confidence=payload.confidence if payload else None,
            scholarly_basis=payload.scholarly_basis if payload else None,
            patient_decision=decision,
            reasoning_chain=reasoning,
            timestamp=datetime.now(tz=timezone.utc),
        )
        try:
            await self._audit_writer(record)
        except Exception as exc:
            raise AuditWriteError(
                f"Audit write failed for suggestion {suggestion_id}: {exc}",
                agent_name=agent_name,
            ) from exc


# ---------------------------------------------------------------------------
# Default audit writer: structured logging (no-op for production without DB)
# ---------------------------------------------------------------------------

async def _default_audit_writer(record: TransparencyRecord) -> None:
    """
    Fallback audit writer that emits the record as a structured log line.

    Replace this in ``app.py`` with a database writer once the DB layer is
    wired in.
    """
    logger.info(
        "AUDIT|event_id=%s|suggestion_id=%s|agent=%s|decision=%s|confidence=%s",
        record.event_id,
        record.suggestion_id,
        record.agent_name,
        record.patient_decision.value if record.patient_decision else "none",
        record.confidence,
    )
