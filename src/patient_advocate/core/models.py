"""
Patient Advocate — Core Data Models (Pydantic v2, Frozen/Immutable)
===================================================================

All domain objects are defined here as frozen Pydantic models to enforce:
  1. Type safety at runtime (no silent dict-key typos)
  2. Immutability for audit integrity (Glass Box Imperative)
  3. Validation with clinical bounds (e.g., ANC >= 0)
  4. Serialization for database persistence and API transport

Design Principles:
  - Every model that participates in the audit trail is frozen (immutable).
  - Scholarly citations are embedded in docstrings, not comments.
  - Field constraints reflect clinical thresholds from peer-reviewed literature.

References:
  - NCCN Guidelines for Myelosuppressive Therapy, v2024:
    ANC < 500 = severe neutropenia; 500-1000 = moderate neutropenia.
  - Bower et al. (2014), J Clin Oncol: Cancer-related fatigue thresholds.
  - Crawford et al. (2004), NEJM: G-CSF timing relative to nadir windows.
  - NCI-CTCAE v5.0: Common Terminology Criteria for Adverse Events.
  - Carey (2024), Advances in Consumer Research: Model facts labels and audit logs.
  - Hantel et al. (2024), JAMA Network Open: Transparency and accountability in clinical AI.

Author: Patient Advocate Platform Team
License: Apache-2.0
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# ENUMERATIONS
# =============================================================================

class AgentType(str, Enum):
    """Identifiers for each specialized agent in the LangGraph routing graph."""
    MASTER_BRAIN = "master_brain"
    VISION_AGENT = "vision_agent"
    CALENDAR_AGENT = "calendar_agent"
    CLINICAL_TRIALS_AGENT = "clinical_trials_agent"
    ETHICS_COMPLAINT_AGENT = "ethics_complaint_agent"
    LEGAL_MOTIONS_AGENT = "legal_motions_agent"
    END = "end"


class ConflictType(str, Enum):
    """
    Types of scheduling conflicts detected by the Calendar Intelligence Engine.

    Each type maps to a specific clinical heuristic:
      - NADIR_WINDOW: Appointment falls within the expected chemotherapy nadir
        period when ANC is at its lowest (Crawford et al., 2004, NEJM).
      - FATIGUE_THRESHOLD: Patient-reported or model-predicted fatigue exceeds
        the safe threshold for the appointment type (Bower et al., 2014).
      - APPOINTMENT_OVERLAP: Two events occupy the same time slot.
      - RECOVERY_PERIOD: Insufficient recovery time between high-intensity events.
    """
    NADIR_WINDOW = "nadir_window"
    FATIGUE_THRESHOLD = "fatigue_threshold"
    APPOINTMENT_OVERLAP = "appointment_overlap"
    RECOVERY_PERIOD = "recovery_period"


class HITLDecision(str, Enum):
    """
    Human-in-the-Loop decision on an AI suggestion.

    Per the Glass Box Imperative, every high-stakes suggestion requires explicit
    patient or case-manager approval before execution.

    Reference: Hantel et al. (2024), JAMA Network Open — patient autonomy as
    a core ethical requirement for clinical AI deployment.
    """
    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class ComplaintType(str, Enum):
    """
    Categories of ethics complaints that trigger the legal_action_queue.

    Aligned with HIPAA, AMA ethics codes, and state licensing board standards.
    Attorney-in-the-loop required before any filing (claude.md §5).
    """
    HIPAA_VIOLATION = "hipaa_violation"
    INFORMED_CONSENT = "informed_consent_failure"
    DISCRIMINATION = "discrimination"
    DENIAL_OF_CARE = "denial_of_care"
    BILLING_FRAUD = "billing_fraud"
    NEGLIGENCE = "negligence"


class LegalMotionType(str, Enum):
    """Legal motion categories for the Phase 2 legal action module."""
    MOTION_TO_COMPEL = "motion_to_compel"
    TEMPORARY_RESTRAINING_ORDER = "temporary_restraining_order"
    CLASS_ACTION_COMPLAINT = "class_action_complaint"
    INSURANCE_APPEAL = "insurance_appeal"
    MEDICAL_MALPRACTICE = "medical_malpractice"
    ADA_VIOLATION = "ada_violation"


class SymptomSeverity(int, Enum):
    """
    NCI-CTCAE v5.0 aligned grading (simplified for patient self-report).

    Reference: National Cancer Institute Common Terminology Criteria for
    Adverse Events, Version 5.0 (2017).
    """
    NONE = 0
    MILD = 1               # Awareness of symptom but easily tolerated
    MODERATE = 2            # Discomfort, interferes with some daily activities
    SEVERE = 3              # Significant interference, may need medical intervention
    LIFE_THREATENING = 4    # Urgent medical intervention required
    DEATH_RELATED = 5       # Death related to adverse event


class MoodScale(int, Enum):
    """Patient emotional state — 5-point Likert scale for ePRO diary."""
    VERY_LOW = 1
    LOW = 2
    NEUTRAL = 3
    GOOD = 4
    VERY_GOOD = 5


class ActivityLevel(str, Enum):
    """
    ECOG Performance Status (simplified for patient diary).

    Reference: Oken et al. (1982), Am J Clin Oncol — Eastern Cooperative
    Oncology Group performance status scale.
    """
    FULLY_ACTIVE = "ecog_0"
    RESTRICTED_STRENUOUS = "ecog_1"
    AMBULATORY_SELF_CARE = "ecog_2"
    LIMITED_SELF_CARE = "ecog_3"
    COMPLETELY_DISABLED = "ecog_4"


# =============================================================================
# CLINICAL DATA MODELS (Frozen / Immutable)
# =============================================================================

class LabValues(BaseModel):
    """
    Immutable lab value record extracted from OCR or manual entry.

    Clinical Thresholds (NCCN Guidelines, v2024):
      - ANC < 500 cells/uL  -> Severe neutropenia (defer non-urgent appointments)
      - ANC 500-1000         -> Moderate neutropenia (schedule with caution)
      - ANC > 1500           -> Normal range
      - WBC < 3.0 x10^3/uL  -> Leukopenia
      - Hemoglobin < 8.0 g/dL -> Severe anemia (consider transfusion)
      - Platelets < 50 x10^3/uL -> Thrombocytopenia risk

    The `source` field tracks provenance for audit:
      - 'ocr': Extracted via PyTesseract (confidence should be verified by HITL)
      - 'manual': Patient or clinician entered
      - 'ehr_api': Direct EHR integration (highest confidence)
    """
    model_config = ConfigDict(frozen=True)

    anc: Optional[float] = Field(
        None, ge=0, description="Absolute Neutrophil Count (cells/uL)"
    )
    wbc: Optional[float] = Field(
        None, ge=0, description="White Blood Cell count (10^3/uL)"
    )
    hemoglobin: Optional[float] = Field(
        None, ge=0, description="Hemoglobin (g/dL)"
    )
    platelets: Optional[int] = Field(
        None, ge=0, description="Platelet count (10^3/uL)"
    )
    fatigue_score: Optional[int] = Field(
        None, ge=0, le=10, description="Patient-reported fatigue on 0-10 scale"
    )
    collection_date: date = Field(description="Date labs were collected")
    source: str = Field(
        description="Provenance: 'ocr', 'manual', or 'ehr_api'"
    )


class ChemoRegimen(BaseModel):
    """
    Immutable chemotherapy regimen descriptor.

    The nadir window (expected_nadir_day_start to expected_nadir_day_end) is
    the period post-infusion when blood counts reach their lowest point.
    Scheduling non-urgent appointments during this window is contraindicated.

    Reference: Crawford et al. (2004), NEJM — G-CSF administration timing
    relative to the chemotherapy nadir window.
    """
    model_config = ConfigDict(frozen=True)

    regimen_name: str = Field(description="e.g., 'AC-T (Adriamycin/Cyclophosphamide -> Taxol)'")
    cycle_number: int = Field(ge=1, description="Current cycle number")
    total_cycles: int = Field(ge=1, description="Total planned cycles")
    cycle_start_date: date = Field(description="Start date of current cycle")
    cycle_length_days: int = Field(ge=1, description="Days per cycle")
    expected_nadir_day_start: int = Field(
        ge=1, description="Day post-infusion when nadir begins (typically day 7-10)"
    )
    expected_nadir_day_end: int = Field(
        ge=1, description="Day post-infusion when nadir ends (typically day 12-14)"
    )


class PatientProfile(BaseModel):
    """
    Mutable patient profile aggregating clinical and demographic data.

    Not frozen because profiles are updated as new labs arrive, preferences
    change, and treatment progresses. However, all mutations are logged via
    the GlassBoxLogger audit trail.
    """
    patient_id: UUID = Field(default_factory=uuid4)
    name: str = Field(default="", description="Patient display name")
    diagnosis: str = Field(description="Primary diagnosis")
    diagnosis_date: Optional[date] = None
    oncologist: Optional[str] = None
    hospital: Optional[str] = None
    regimen: Optional[ChemoRegimen] = None
    latest_labs: Optional[LabValues] = None
    fatigue_baseline: float = Field(
        default=3.0, ge=0.0, le=10.0,
        description="Personalized fatigue baseline (adjusted by RLHF)"
    )
    preferences: dict = Field(
        default_factory=dict,
        description="Per-patient scheduling preferences learned via RLHF"
    )
    medications: list[dict] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)


# =============================================================================
# CALENDAR & SCHEDULING MODELS (Frozen / Immutable)
# =============================================================================

class CalendarEvent(BaseModel):
    """A single calendar event with clinical priority."""
    model_config = ConfigDict(frozen=True)

    event_id: str = Field(description="Unique event identifier")
    title: str
    event_date: date
    time: str = Field(description="HH:MM format")
    duration_hours: float = Field(ge=0)
    location: str = ""
    provider: str = ""
    prep_notes: str = ""
    status: str = Field(default="confirmed", description="confirmed/tentative/action_required")
    priority: str = Field(default="medium", description="critical/high/medium/low")


class CalendarSuggestion(BaseModel):
    """
    Immutable AI-generated scheduling suggestion with full provenance.

    The Glass Box Imperative requires that every suggestion includes:
      - confidence: Numeric confidence bound (anti-automation bias)
      - reasoning: Natural-language explanation of the AI's logic
      - scholarly_basis: At least one peer-reviewed citation supporting the suggestion

    Reference: Carey (2024), Advances in Consumer Research — 'model facts labels'
    with uncertainty bounds and failure-mode documentation.
    """
    model_config = ConfigDict(frozen=True)

    suggestion_id: UUID = Field(default_factory=uuid4)
    event_id: Optional[str] = Field(None, description="Calendar event this modifies")
    original_date: date
    suggested_date: date
    conflict_type: ConflictType
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model confidence [0,1]. Values < 0.7 trigger mandatory HITL review."
    )
    reasoning: str = Field(
        description="Plain-language explanation of why this change is suggested"
    )
    scholarly_basis: str = Field(
        description="Peer-reviewed citation supporting this suggestion"
    )
    requires_hitl: bool = Field(
        default=True,
        description="Whether this suggestion requires human approval before execution"
    )


# =============================================================================
# AUDIT & TRANSPARENCY MODELS (Frozen / Immutable)
# =============================================================================

class TransparencyRecord(BaseModel):
    """
    Immutable audit record per the Glass Box Imperative.

    Every agent action — routing decisions, OCR extractions, scheduling
    suggestions, ethics flags — produces a TransparencyRecord that is
    INSERT-only in the database. These records form the legal evidence
    trail supporting future complaints or litigation.

    Required columns per claude.md §4.2:
      event_id, suggestion_id, conflict_type, confidence,
      scholarly_basis, patient_decision, timestamp.

    Reference: Carey (2024) — audit logs support ex post explanations
    in complaint or litigation; the Glass Box design IS the legal record.
    """
    model_config = ConfigDict(frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    suggestion_id: Optional[UUID] = None
    agent_name: str = Field(description="Which agent produced this record")
    conflict_type: Optional[ConflictType] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    scholarly_basis: Optional[str] = None
    patient_decision: Optional[HITLDecision] = None
    reasoning_chain: str = Field(
        description="Full reasoning chain that led to this action"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EthicsFlag(BaseModel):
    """
    An ethics concern flagged by the ethics screener for case manager review.

    Flagged items populate the legal_action_queue per claude.md §4.4.
    """
    model_config = ConfigDict(frozen=True)

    flag_id: UUID = Field(default_factory=uuid4)
    patient_id: UUID
    flag_type: ComplaintType
    description: str
    severity: str = Field(default="medium", description="low/medium/high/critical")
    source_agent: str = Field(description="Agent that raised this flag")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RLHFFeedbackRecord(BaseModel):
    """
    Captures patient approval/rejection for RLHF retraining.

    Rejected suggestions are stored with the full suggestion snapshot
    to retrain per-patient scheduling heuristics (claude.md §4.3).
    """
    model_config = ConfigDict(frozen=True)

    feedback_id: UUID = Field(default_factory=uuid4)
    suggestion_id: UUID
    patient_id: UUID
    decision: HITLDecision
    suggestion_snapshot: dict = Field(
        description="Frozen CalendarSuggestion as dict for persistence"
    )
    feedback_text: Optional[str] = Field(
        None, description="Optional patient comment on why they rejected"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
