"""
Patient Advocate — LangGraph Shared State Definition
=====================================================

The AdvocateState is the single source of truth passed through every node
in the LangGraph routing graph. It replaces the untyped dict-based state
from the prototype with fully typed Pydantic models.

Design:
  - TypedDict (not a class) because LangGraph requires TypedDict for state.
  - Annotated lists with operator.add enable automatic accumulation of
    messages and audit records across nodes.
  - Optional fields allow progressive enrichment as the graph executes.

Reference:
  - LangGraph StateGraph documentation (langchain-ai/langgraph, 2024):
    "Build resilient language agents as graphs."
  - Bengio et al. (2019): System 2 deliberate reasoning patterns.

Author: Patient Advocate Platform Team
License: Apache-2.0
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import BaseMessage

from patient_advocate.core.models import (
    CalendarEvent,
    CalendarSuggestion,
    EthicsFlag,
    PatientProfile,
    TransparencyRecord,
)


class AdvocateState(TypedDict):
    """
    Shared state object passed through the LangGraph workflow.

    Fields:
        messages: Accumulated conversation history (auto-appended via operator.add).
        patient_profile: The patient's clinical and demographic data.
        audit_log: Immutable transparency records (auto-appended via operator.add).
        pending_suggestions: Calendar suggestions awaiting HITL approval.
        calendar_events: Current calendar events for the patient.
        next_agent: Routing decision from the Master Brain.
        image_path: Optional path to an uploaded image for the Vision Agent.
        ethics_flags: Accumulated ethics concerns (auto-appended).
        hitl_required: Whether the current pipeline step needs human approval.
        calendar_data: Raw calendar data dict (backward compat with prototypes).
        ethics_data: Raw ethics data dict (backward compat with prototypes).
        legal_data: Raw legal data dict (backward compat with prototypes).
        documents_generated: List of generated document paths/identifiers.
    """
    messages: Annotated[list[BaseMessage], operator.add]
    patient_profile: PatientProfile
    audit_log: Annotated[list[TransparencyRecord], operator.add]
    pending_suggestions: list[CalendarSuggestion]
    calendar_events: list[CalendarEvent]
    next_agent: str
    image_path: Optional[str]
    ethics_flags: Annotated[list[EthicsFlag], operator.add]
    hitl_required: bool
    # Backward compatibility with prototype data structures
    calendar_data: Optional[dict]
    ethics_data: Optional[dict]
    legal_data: Optional[dict]
    documents_generated: Optional[list[str]]
