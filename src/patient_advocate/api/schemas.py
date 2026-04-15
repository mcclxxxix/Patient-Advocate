"""
Patient Advocate — HITL API Wire Schemas (Pydantic v2)
=======================================================

These are *transport* types, not domain types.  Domain types live in
``patient_advocate.core.models``.  Keeping them separate means the API
contract can evolve independently from the internal domain model, and
validation errors surface at the HTTP boundary before touching business logic.

Design choices:
  - All UUIDs are ``UUID`` objects (FastAPI serialises them to strings).
  - ``feedback_text`` is capped at 2 000 characters per RLHF data hygiene guidelines.
  - ``status`` on ``SuggestionStatusResponse`` is a plain string because the set
    of values includes runtime-generated states (e.g. ``timed_out``) that do not
    belong in the domain enum.

Author: Patient Advocate Platform Team
License: Apache-2.0
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

from patient_advocate.core.models import HITLDecision


class HITLDecisionRequest(BaseModel):
    """Patient or case-manager submits a decision on a pending suggestion."""

    suggestion_id: UUID = Field(description="UUID of the suggestion being decided")
    patient_id: UUID = Field(description="UUID of the patient who owns the suggestion")
    decision: HITLDecision = Field(description="APPROVED | REJECTED | DEFERRED")
    feedback_text: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Optional plain-text reason (required for REJECTED per RLHF guidelines)",
    )


class SuggestionPayload(BaseModel):
    """
    Inbound registration payload for a new scheduling suggestion.

    Produced by the Calendar Intelligence Engine and pushed to the HITL
    queue so that the patient can review it within the timeout window.
    """

    suggestion_id: UUID = Field(description="Stable UUID set by the producing agent")
    patient_id: UUID = Field(description="UUID of the patient this suggestion belongs to")
    event_id: Optional[str] = Field(
        default=None, description="Calendar event ID this suggestion modifies (if any)"
    )
    original_date: str = Field(description="ISO 8601 date of the current appointment")
    suggested_date: str = Field(description="ISO 8601 date of the proposed reschedule")
    conflict_type: str = Field(description="nadir_window | fatigue_threshold | …")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Model confidence [0, 1]"
    )
    reasoning: str = Field(description="Plain-language explanation of why to reschedule")
    scholarly_basis: str = Field(
        description="Peer-reviewed citation supporting the suggestion"
    )


class HITLDecisionResponse(BaseModel):
    """Response returned to the caller after a decision is recorded."""

    suggestion_id: UUID
    decision: HITLDecision
    acknowledged: bool = Field(
        description="True when the decision has been durably written to all stores"
    )


class PendingListResponse(BaseModel):
    """All suggestions still awaiting a decision for a given patient."""

    patient_id: UUID
    pending: list[SuggestionPayload]


class SuggestionStatusResponse(BaseModel):
    """
    Current lifecycle status of a suggestion.

    Possible values:
      pending       – awaiting human decision
      approved      – patient approved (terminal)
      rejected      – patient rejected (terminal)
      deferred      – deferred for later review (can be overridden)
      timed_out     – no decision within ``hitl_timeout_seconds``
    """

    suggestion_id: UUID
    status: str
