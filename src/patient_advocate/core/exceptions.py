"""
Patient Advocate — Exception Hierarchy
========================================

Structured exception classes for clear error handling and audit logging.
Every exception carries context sufficient for the GlassBoxLogger to
produce a meaningful TransparencyRecord.

Design:
  - All exceptions inherit from PatientAdvocateError for catch-all handling.
  - Clinical exceptions carry the agent_name for audit attribution.
  - HITL exceptions distinguish timeout from rejection for RLHF recording.

Author: Patient Advocate Platform Team
License: Apache-2.0
"""

from __future__ import annotations


class PatientAdvocateError(Exception):
    """Base exception for all Patient Advocate errors."""

    def __init__(self, message: str, agent_name: str = "unknown") -> None:
        self.agent_name = agent_name
        super().__init__(message)


# --- Clinical Errors ---

class ClinicalDataError(PatientAdvocateError):
    """Raised when clinical data is missing, invalid, or out of range."""
    pass


class OCRExtractionError(PatientAdvocateError):
    """Raised when the Vision Agent fails to extract text from an image."""
    pass


class NadirWindowViolation(PatientAdvocateError):
    """
    Raised when an appointment is scheduled during the nadir window.

    This is a warning, not a blocker — the HITL gate allows the patient
    to override if they acknowledge the risk.
    """
    pass


# --- HITL Errors ---

class HITLTimeoutError(PatientAdvocateError):
    """Raised when human approval is not received within the timeout window."""
    pass


class HITLRejectionError(PatientAdvocateError):
    """
    Raised when a patient or case manager explicitly rejects a suggestion.

    The rejection is captured in the RLHF store for future model retraining.
    """
    pass


# --- Routing Errors ---

class RoutingError(PatientAdvocateError):
    """Raised when the Master Brain cannot determine the correct agent route."""
    pass


class AgentNotFoundError(RoutingError):
    """Raised when the requested agent does not exist in the graph."""
    pass


# --- Database Errors ---

class AuditWriteError(PatientAdvocateError):
    """Raised when the GlassBoxLogger fails to write an audit record."""
    pass


# --- Ethics & Legal Errors ---

class EthicsViolationError(PatientAdvocateError):
    """
    Raised when the ethics screener detects a pattern requiring escalation.

    Flagged items populate the legal_action_queue for case manager review.
    """
    pass


class AttorneyReviewRequired(PatientAdvocateError):
    """
    Raised when a legal action requires attorney-in-the-loop review.

    Per claude.md §5: Always require attorney review before filing
    any legal motion, complaint, or appeal.
    """
    pass
