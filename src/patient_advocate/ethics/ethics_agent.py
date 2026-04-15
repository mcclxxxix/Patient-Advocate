"""
Patient Advocate — Ethics Complaint Agent Node
===============================================

Implements the ``ethics_complaint_agent_node`` for the LangGraph routing
graph.  This node replaces the Phase 1 stub and wires the deterministic
EthicsScreener into the shared AdvocateState.

Design Principles
-----------------
1. **Glass Box Imperative** — Every flag produced is recorded in the
   ``audit_log`` with full provenance (agent name, timestamp, citation).
2. **Human-in-the-Loop by Default** — ``hitl_required`` is unconditionally
   set to ``True``; no automated filing occurs without an attorney gate.
3. **Three-Step Chain** — Patient-facing messages always include the
   mandatory three-step next-action chain so that patients understand
   the procedural safeguards before any formal complaint is filed:

       Step 1  → Audit trail locked (Glass Box Imperative)
       Step 2  → Case manager review within 2 business days
       Step 3  → Attorney gate required before any external filing

4. **Append-Only State** — ``ethics_flags`` uses ``operator.add`` in
   AdvocateState; this node returns a list to be appended, never the
   full list, preventing double-counting in multi-node runs.

References
----------
- Hantel et al. (2024), JAMA Network Open — patient autonomy and
  transparency as core requirements for clinical AI deployment.
- Carey (2024), Advances in Consumer Research — audit logs as the
  legal record for AI-assisted decisions.
- claude.md §4.4 — attorney-in-the-loop required before any legal filing.

Author: Patient Advocate Platform Team
License: Apache-2.0
"""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from patient_advocate.core.models import (
    AgentType,
    ComplaintType,
    EthicsFlag,
    TransparencyRecord,
)
from patient_advocate.ethics.ethics_screener import EthicsScreener, ScreenerResult


# ---------------------------------------------------------------------------
# Severity → plain-English label map for patient-facing messages
# ---------------------------------------------------------------------------
_SEVERITY_LABEL: dict[str, str] = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "critical": "CRITICAL",
}

_COMPLAINT_LABEL: dict[ComplaintType, str] = {
    ComplaintType.HIPAA_VIOLATION: "HIPAA Privacy Violation",
    ComplaintType.INFORMED_CONSENT: "Informed Consent Failure",
    ComplaintType.DISCRIMINATION: "Discrimination",
    ComplaintType.DENIAL_OF_CARE: "Denial of Care",
    ComplaintType.BILLING_FRAUD: "Billing Fraud",
    ComplaintType.NEGLIGENCE: "Medical Negligence / Malpractice",
}

# The mandatory three-step chain shown to the patient on every ethics response
_THREE_STEP_CHAIN = """\
MANDATORY 3-STEP NEXT-ACTION CHAIN (no step may be skipped):
  Step 1 — Audit Trail Locked: Your concern has been logged in our
            immutable Glass Box audit trail and cannot be altered.
  Step 2 — Case Manager Review: A licensed case manager will review
            your flag within 2 business days and contact you.
  Step 3 — Attorney Gate: An attorney must review and approve any
            external complaint, regulatory filing, or legal action
            before it is submitted on your behalf."""

_SCREENER = EthicsScreener()


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

def ethics_complaint_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node — ethics_complaint_agent.

    Reads the latest HumanMessage from ``state["messages"]``, runs the
    deterministic EthicsScreener, and produces:

    * ``ethics_flags``  — list of EthicsFlag objects to be appended to state
    * ``legal_data``    — updated dict; action_queue populated for escalations
    * ``audit_log``     — list of TransparencyRecord objects to be appended
    * ``messages``      — list containing one AIMessage for the patient
    * ``hitl_required`` — always True

    Parameters
    ----------
    state:
        The current AdvocateState dict.

    Returns
    -------
    dict
        Partial state update compatible with LangGraph's ``operator.add``
        accumulation for list fields.
    """
    # ------------------------------------------------------------------
    # 1. Extract the latest human message
    # ------------------------------------------------------------------
    messages: list = state.get("messages", [])
    text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    # ------------------------------------------------------------------
    # 2. Resolve patient_id
    # ------------------------------------------------------------------
    patient_profile = state.get("patient_profile")
    if patient_profile is not None and hasattr(patient_profile, "patient_id"):
        patient_id = patient_profile.patient_id
    else:
        patient_id = uuid4()

    # ------------------------------------------------------------------
    # 3. Run the deterministic screener
    # ------------------------------------------------------------------
    screener_results: list[ScreenerResult] = _SCREENER.screen(text, patient_id)

    # ------------------------------------------------------------------
    # 4. Build EthicsFlag objects
    # ------------------------------------------------------------------
    now = datetime.now(timezone.utc)
    new_flags: list[EthicsFlag] = []
    for result in screener_results:
        flag = EthicsFlag(
            flag_id=uuid4(),
            patient_id=patient_id,
            flag_type=result.complaint_type,
            description=(
                f"{_COMPLAINT_LABEL[result.complaint_type]} detected. "
                f'Matched text: "{result.matched_text}". '
                f"Basis: {result.scholarly_basis}"
            ),
            severity=result.severity,
            source_agent=AgentType.ETHICS_COMPLAINT_AGENT.value,
            timestamp=now,
        )
        new_flags.append(flag)

    # ------------------------------------------------------------------
    # 5. Populate legal_data action_queue for escalation-required flags
    # ------------------------------------------------------------------
    existing_legal_data: dict = state.get("legal_data") or {}
    legal_data = dict(existing_legal_data)  # shallow copy — preserve existing keys
    action_queue: list = list(legal_data.get("action_queue") or [])

    for flag, result in zip(new_flags, screener_results):
        if result.requires_escalation:
            action_queue.append(
                {
                    "flag_id": str(flag.flag_id),
                    "patient_id": str(flag.patient_id),
                    "flag_type": flag.flag_type.value,
                    "severity": flag.severity,
                    "description": flag.description,
                    "scholarly_basis": result.scholarly_basis,
                    "requires_attorney_review": True,
                    "queued_at": now.isoformat(),
                    "status": "pending_case_manager_review",
                }
            )

    legal_data["action_queue"] = action_queue

    # ------------------------------------------------------------------
    # 6. Build audit records
    # ------------------------------------------------------------------
    new_audit_records: list[TransparencyRecord] = []
    for flag, result in zip(new_flags, screener_results):
        record = TransparencyRecord(
            event_id=uuid4(),
            agent_name=AgentType.ETHICS_COMPLAINT_AGENT.value,
            scholarly_basis=result.scholarly_basis,
            reasoning_chain=(
                f"EthicsScreener detected '{result.complaint_type.value}' "
                f"(severity={result.severity}) via pattern match on text: "
                f'"{result.matched_text}". '
                f"requires_escalation={result.requires_escalation}. "
                f"flag_id={flag.flag_id}."
            ),
            timestamp=now,
        )
        new_audit_records.append(record)

    # ------------------------------------------------------------------
    # 7. Compose patient-facing AIMessage
    # ------------------------------------------------------------------
    if not screener_results:
        patient_message_text = (
            "Thank you for sharing your concern. Our ethics screener did not "
            "detect a specific category of ethics violation in your message, "
            "but your concern has been logged for case manager review.\n\n"
            + _THREE_STEP_CHAIN
        )
    else:
        flag_lines: list[str] = []
        for i, (flag, result) in enumerate(zip(new_flags, screener_results), start=1):
            escalation_note = (
                " [ESCALATION REQUIRED — attorney review mandatory]"
                if result.requires_escalation
                else ""
            )
            flag_lines.append(
                f"  Flag {i}: {_COMPLAINT_LABEL[result.complaint_type]}\n"
                f"           Severity: {_SEVERITY_LABEL[result.severity]}"
                f"{escalation_note}\n"
                f"           Basis: {result.scholarly_basis}"
            )

        flags_block = "\n".join(flag_lines)
        patient_message_text = (
            f"We have identified {len(new_flags)} ethics concern(s) in your "
            f"message and flagged them for immediate review:\n\n"
            f"{flags_block}\n\n"
            f"Your patient rights are being protected. "
            f"Here is exactly what happens next:\n\n"
            + _THREE_STEP_CHAIN
            + "\n\nAll records are append-only and cannot be altered or deleted."
        )

    ai_message = AIMessage(
        content=patient_message_text,
        name=AgentType.ETHICS_COMPLAINT_AGENT.value,
    )

    # ------------------------------------------------------------------
    # 8. Return partial state update
    # ------------------------------------------------------------------
    return {
        "audit_log": new_audit_records,          # operator.add accumulates
        "messages": [ai_message],                # operator.add accumulates
        "ethics_flags": new_flags,               # operator.add accumulates
        "legal_data": legal_data,
        "hitl_required": True,
    }
