"""
Patient Advocate — In-Memory Suggestion Store
=============================================

An async-safe, in-process registry of all scheduling suggestions that are
awaiting (or have received) a HITL decision.

Design decisions:
  - A single ``asyncio.Lock`` guards all mutations, giving us correct
    serialisation under concurrent FastAPI requests without the overhead
    of a full database round-trip for what is typically a small working set.
  - Terminal decisions (APPROVED / REJECTED) are immutable; a DEFERRED
    decision can be overridden.  This mirrors the domain rule that once a
    patient has committed to an action it cannot be silently undone.
  - ``evict_decided`` is safe to call from a background ``asyncio.Task``
    (or APScheduler job) to keep memory bounded.  It removes all terminal
    decisions so only live work items remain.

Author: Patient Advocate Platform Team
License: Apache-2.0
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from patient_advocate.api.schemas import SuggestionPayload
from patient_advocate.core.models import HITLDecision

# Terminal decisions that cannot be overridden once set.
_TERMINAL_DECISIONS: frozenset[HITLDecision] = frozenset(
    {HITLDecision.APPROVED, HITLDecision.REJECTED}
)


class SuggestionStore:
    """
    Async-safe in-memory registry for suggestion lifecycle management.

    Thread-safety guarantee: every public method acquires ``_lock`` before
    reading or writing ``_payloads`` / ``_decisions`` / ``_registered_at``.
    FastAPI's event loop runs single-threaded, so this is sufficient.
    """

    def __init__(self) -> None:
        self._lock: asyncio.Lock = asyncio.Lock()
        # suggestion_id -> payload
        self._payloads: dict[UUID, SuggestionPayload] = {}
        # suggestion_id -> HITLDecision (absent = pending)
        self._decisions: dict[UUID, HITLDecision] = {}
        # suggestion_id -> UTC datetime when registered
        self._registered_at: dict[UUID, datetime] = {}

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def register(self, payload: SuggestionPayload) -> None:
        """
        Register a new suggestion.

        Idempotent: re-registering the same ``suggestion_id`` is a no-op so
        that the Calendar Engine can safely retry without duplicating entries.
        """
        async with self._lock:
            if payload.suggestion_id not in self._payloads:
                self._payloads[payload.suggestion_id] = payload
                self._registered_at[payload.suggestion_id] = datetime.now(tz=timezone.utc)

    async def mark_decided(
        self, suggestion_id: UUID, decision: HITLDecision
    ) -> None:
        """
        Record a HITL decision on a suggestion.

        Rules:
          - APPROVED and REJECTED are *terminal*: once set they cannot be
            changed, even to DEFERRED.
          - DEFERRED can be overridden by any decision, including itself.

        Raises:
            KeyError: if ``suggestion_id`` is not registered.
            ValueError: if an attempt is made to override a terminal decision.
        """
        async with self._lock:
            if suggestion_id not in self._payloads:
                raise KeyError(f"Suggestion {suggestion_id} not registered")
            existing = self._decisions.get(suggestion_id)
            if existing in _TERMINAL_DECISIONS:
                raise ValueError(
                    f"Cannot override terminal decision '{existing.value}' "
                    f"on suggestion {suggestion_id}"
                )
            self._decisions[suggestion_id] = decision

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get(self, suggestion_id: UUID) -> Optional[SuggestionPayload]:
        """Return the payload for a suggestion, or ``None`` if not found."""
        async with self._lock:
            return self._payloads.get(suggestion_id)

    async def get_pending_for_patient(self, patient_id: UUID) -> list[SuggestionPayload]:
        """
        Return all suggestions for ``patient_id`` that have no decision yet
        (i.e., status == pending).
        """
        async with self._lock:
            return [
                payload
                for sid, payload in self._payloads.items()
                if payload.patient_id == patient_id
                and sid not in self._decisions
            ]

    async def get_decision(self, suggestion_id: UUID) -> Optional[HITLDecision]:
        """Return the recorded decision, or ``None`` if still pending."""
        async with self._lock:
            return self._decisions.get(suggestion_id)

    async def is_timed_out(self, suggestion_id: UUID, timeout_seconds: int) -> bool:
        """
        Return ``True`` if:
          - the suggestion is registered, AND
          - it has no decision yet (still pending), AND
          - it was registered more than ``timeout_seconds`` ago.

        Returns ``False`` for unknown suggestion IDs so callers can distinguish
        "not found" from "timed out" at a higher layer.
        """
        async with self._lock:
            registered_at = self._registered_at.get(suggestion_id)
            if registered_at is None:
                return False
            if suggestion_id in self._decisions:
                # Already decided — timeout is irrelevant.
                return False
            elapsed = (datetime.now(tz=timezone.utc) - registered_at).total_seconds()
            return elapsed > timeout_seconds

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def evict_decided(self) -> int:
        """
        Remove all suggestions that have a terminal decision.

        Suggestions with DEFERRED are *not* evicted because they may still
        receive a final decision.  Only APPROVED and REJECTED are evicted.

        Returns the number of entries removed.  Safe to call from a
        background asyncio Task.
        """
        async with self._lock:
            to_evict = [
                sid
                for sid, decision in self._decisions.items()
                if decision in _TERMINAL_DECISIONS
            ]
            for sid in to_evict:
                self._payloads.pop(sid, None)
                self._decisions.pop(sid, None)
                self._registered_at.pop(sid, None)
            return len(to_evict)

    # ------------------------------------------------------------------
    # Introspection (test / debug helpers)
    # ------------------------------------------------------------------

    async def size(self) -> int:
        """Return the total number of registered suggestions (any state)."""
        async with self._lock:
            return len(self._payloads)
