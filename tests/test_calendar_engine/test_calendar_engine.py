"""
Tests for Patient Advocate — Calendar Intelligence Engine
=========================================================

All external packages (pydantic, pydantic_settings, langchain_core, langgraph)
are stubbed at module level so this test suite runs without any third-party
dependencies installed.

Inline dataclass-based replicas of the core models match the field signatures
and frozen semantics used by the production Pydantic models.

Coverage targets:
  - compute_nadir_window: correct date arithmetic with and without buffer
  - assess_nadir: in-window, days_until_clear, ANC threshold (independent)
  - assess_fatigue: above/below threshold, patient baseline respected
  - detect_overlaps: exact overlap, boundary touch not counted, multi-event
  - detect_recovery_violation: <24h gap, ≥24h safe, non-high-intensity ignored
  - analyse_event: priority ordering, all four conflict types, empty result
  - suggest_safe_date: clears nadir, clears fatigue, 30-day fallback
  - calendar_agent_node: no-profile path, no-events path, conflicts path

Author: Patient Advocate Platform Team
License: Apache-2.0
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

# =============================================================================
# STUB ALL EXTERNAL PACKAGES
# =============================================================================

# ---- pydantic ---------------------------------------------------------------
pydantic_mod = types.ModuleType("pydantic")


class _BaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            object.__setattr__(self, k, v)


class _ConfigDict(dict):
    pass


def _Field(default=None, **kwargs):
    return default


pydantic_mod.BaseModel = _BaseModel
pydantic_mod.ConfigDict = _ConfigDict
pydantic_mod.Field = _Field
sys.modules["pydantic"] = pydantic_mod

# ---- pydantic_settings ------------------------------------------------------
ps_mod = types.ModuleType("pydantic_settings")


class _BaseSettings(_BaseModel):
    pass


ps_mod.BaseSettings = _BaseSettings
ps_mod.SettingsConfigDict = _ConfigDict
sys.modules["pydantic_settings"] = ps_mod

# ---- langchain_core ---------------------------------------------------------
lc_mod = types.ModuleType("langchain_core")
lc_messages_mod = types.ModuleType("langchain_core.messages")


class _BaseMessage:
    def __init__(self, content: str = "", **kwargs):
        self.content = content


class _AIMessage(_BaseMessage):
    pass


lc_messages_mod.BaseMessage = _BaseMessage
lc_messages_mod.AIMessage = _AIMessage
lc_mod.messages = lc_messages_mod
sys.modules["langchain_core"] = lc_mod
sys.modules["langchain_core.messages"] = lc_messages_mod

# ---- langgraph --------------------------------------------------------------
lg_mod = types.ModuleType("langgraph")
sys.modules["langgraph"] = lg_mod

# ---- patient_advocate.core (stub — we define minimal models below) ----------
pa_mod = types.ModuleType("patient_advocate")
pa_core_mod = types.ModuleType("patient_advocate.core")
pa_models_mod = types.ModuleType("patient_advocate.core.models")
pa_config_mod = types.ModuleType("patient_advocate.core.config")

# Enums needed by models module
class ConflictType(str, Enum):
    NADIR_WINDOW = "nadir_window"
    FATIGUE_THRESHOLD = "fatigue_threshold"
    APPOINTMENT_OVERLAP = "appointment_overlap"
    RECOVERY_PERIOD = "recovery_period"


class HITLDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class AgentType(str, Enum):
    CALENDAR_AGENT = "calendar_agent"


# ---------------------------------------------------------------------------
# Minimal inline model replicas (dataclasses matching production field names)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LabValues:
    collection_date: date
    source: str
    anc: Optional[float] = None
    wbc: Optional[float] = None
    hemoglobin: Optional[float] = None
    platelets: Optional[int] = None
    fatigue_score: Optional[int] = None


@dataclass(frozen=True)
class ChemoRegimen:
    regimen_name: str
    cycle_number: int
    total_cycles: int
    cycle_start_date: date
    cycle_length_days: int
    expected_nadir_day_start: int
    expected_nadir_day_end: int


@dataclass(frozen=True)
class CalendarEvent:
    event_id: str
    title: str
    event_date: date
    time: str
    duration_hours: float
    location: str = ""
    provider: str = ""
    prep_notes: str = ""
    status: str = "confirmed"
    priority: str = "medium"


@dataclass(frozen=True)
class CalendarSuggestion:
    original_date: date
    suggested_date: date
    conflict_type: ConflictType
    confidence: float
    reasoning: str
    scholarly_basis: str
    suggestion_id: UUID = field(default_factory=uuid4)
    event_id: Optional[str] = None
    requires_hitl: bool = True


@dataclass(frozen=True)
class TransparencyRecord:
    agent_name: str
    reasoning_chain: str
    event_id: UUID = field(default_factory=uuid4)
    suggestion_id: Optional[UUID] = None
    conflict_type: Optional[ConflictType] = None
    confidence: Optional[float] = None
    scholarly_basis: Optional[str] = None
    patient_decision: Optional[HITLDecision] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


# Inject into stub modules so calendar_engine.py imports resolve
for _name, _obj in [
    ("AgentType", AgentType),
    ("CalendarEvent", CalendarEvent),
    ("CalendarSuggestion", CalendarSuggestion),
    ("ChemoRegimen", ChemoRegimen),
    ("ConflictType", ConflictType),
    ("HITLDecision", HITLDecision),
    ("LabValues", LabValues),
    ("TransparencyRecord", TransparencyRecord),
]:
    setattr(pa_models_mod, _name, _obj)

sys.modules["patient_advocate"] = pa_mod
sys.modules["patient_advocate.core"] = pa_core_mod
sys.modules["patient_advocate.core.models"] = pa_models_mod
sys.modules["patient_advocate.core.config"] = pa_config_mod

# =============================================================================
# NOW import the module under test
# =============================================================================
from patient_advocate.calendar_engine.calendar_engine import (  # noqa: E402
    CalendarIntelligenceEngine,
    NadirAssessment,
    NadirWindow,
    calendar_agent_node,
)

# =============================================================================
# HELPERS
# =============================================================================

BASE_DATE = date(2024, 3, 1)


def make_regimen(
    *,
    cycle_start: date = BASE_DATE,
    nadir_start: int = 7,
    nadir_end: int = 14,
    cycle_length: int = 21,
) -> ChemoRegimen:
    return ChemoRegimen(
        regimen_name="AC-T",
        cycle_number=1,
        total_cycles=4,
        cycle_start_date=cycle_start,
        cycle_length_days=cycle_length,
        expected_nadir_day_start=nadir_start,
        expected_nadir_day_end=nadir_end,
    )


def make_labs(
    *,
    anc: Optional[float] = 1200.0,
    fatigue_score: Optional[int] = 2,
    collection_date: date = BASE_DATE,
) -> LabValues:
    return LabValues(
        anc=anc,
        fatigue_score=fatigue_score,
        collection_date=collection_date,
        source="manual",
    )


def make_event(
    *,
    event_id: str = "evt-001",
    title: str = "Oncology Follow-Up",
    event_date: date = BASE_DATE,
    time: str = "10:00",
    duration_hours: float = 1.0,
    priority: str = "medium",
) -> CalendarEvent:
    return CalendarEvent(
        event_id=event_id,
        title=title,
        event_date=event_date,
        time=time,
        duration_hours=duration_hours,
        priority=priority,
    )


engine = CalendarIntelligenceEngine()


# =============================================================================
# TESTS: compute_nadir_window
# =============================================================================

class TestComputeNadirWindow:
    """compute_nadir_window: date arithmetic with various buffers."""

    def test_default_buffer_start(self):
        regimen = make_regimen(cycle_start=date(2024, 1, 1), nadir_start=7, nadir_end=14)
        window = engine.compute_nadir_window(regimen)
        # day 7 (index 6) minus buffer 2 = day 5 → Jan 5
        assert window.start == date(2024, 1, 5)

    def test_default_buffer_end(self):
        regimen = make_regimen(cycle_start=date(2024, 1, 1), nadir_start=7, nadir_end=14)
        window = engine.compute_nadir_window(regimen)
        # day 14 (index 13) plus buffer 2 = day 16 → Jan 16
        assert window.end == date(2024, 1, 16)

    def test_zero_buffer_start(self):
        regimen = make_regimen(cycle_start=date(2024, 1, 1), nadir_start=7, nadir_end=14)
        window = engine.compute_nadir_window(regimen, buffer_days=0)
        assert window.start == date(2024, 1, 7)

    def test_zero_buffer_end(self):
        regimen = make_regimen(cycle_start=date(2024, 1, 1), nadir_start=7, nadir_end=14)
        window = engine.compute_nadir_window(regimen, buffer_days=0)
        assert window.end == date(2024, 1, 14)

    def test_custom_buffer_days(self):
        regimen = make_regimen(cycle_start=date(2024, 1, 1), nadir_start=10, nadir_end=14)
        window = engine.compute_nadir_window(regimen, buffer_days=3)
        # start: day 10 (idx 9) - 3 = day 7 → Jan 7
        # end:   day 14 (idx 13) + 3 = day 17 → Jan 17
        assert window.start == date(2024, 1, 7)
        assert window.end == date(2024, 1, 17)

    def test_returns_nadir_window_type(self):
        regimen = make_regimen()
        window = engine.compute_nadir_window(regimen)
        assert isinstance(window, NadirWindow)

    def test_start_before_end(self):
        regimen = make_regimen(nadir_start=7, nadir_end=14)
        window = engine.compute_nadir_window(regimen)
        assert window.start < window.end

    def test_large_buffer_extends_window(self):
        regimen = make_regimen(cycle_start=date(2024, 1, 10), nadir_start=7, nadir_end=10)
        window_small = engine.compute_nadir_window(regimen, buffer_days=1)
        window_large = engine.compute_nadir_window(regimen, buffer_days=5)
        assert window_large.start < window_small.start
        assert window_large.end > window_small.end

    def test_single_day_nadir_with_buffer(self):
        regimen = make_regimen(cycle_start=date(2024, 2, 1), nadir_start=10, nadir_end=10)
        window = engine.compute_nadir_window(regimen, buffer_days=2)
        # raw start=end=Feb 10 (day 10 = idx 9 → Feb 10)
        assert window.start == date(2024, 2, 8)
        assert window.end == date(2024, 2, 12)


# =============================================================================
# TESTS: assess_nadir
# =============================================================================

class TestAssessNadir:
    """assess_nadir: in-window detection, days_until_clear, ANC threshold."""

    def _regimen(self):
        return make_regimen(cycle_start=date(2024, 1, 1), nadir_start=7, nadir_end=14)

    def test_date_inside_window_is_in_nadir(self):
        regimen = self._regimen()
        # Window: Jan 5 – Jan 16 (buffer=2)
        result = engine.assess_nadir(date(2024, 1, 10), regimen)
        assert result.is_in_nadir is True

    def test_date_before_window_not_in_nadir(self):
        regimen = self._regimen()
        result = engine.assess_nadir(date(2024, 1, 1), regimen)
        assert result.is_in_nadir is False

    def test_date_after_window_not_in_nadir(self):
        regimen = self._regimen()
        result = engine.assess_nadir(date(2024, 1, 20), regimen)
        assert result.is_in_nadir is False

    def test_window_start_inclusive(self):
        regimen = self._regimen()
        window = engine.compute_nadir_window(regimen)
        result = engine.assess_nadir(window.start, regimen)
        assert result.is_in_nadir is True

    def test_window_end_inclusive(self):
        regimen = self._regimen()
        window = engine.compute_nadir_window(regimen)
        result = engine.assess_nadir(window.end, regimen)
        assert result.is_in_nadir is True

    def test_day_before_window_not_in_nadir(self):
        regimen = self._regimen()
        window = engine.compute_nadir_window(regimen)
        result = engine.assess_nadir(window.start - timedelta(days=1), regimen)
        assert result.is_in_nadir is False

    def test_day_after_window_not_in_nadir(self):
        regimen = self._regimen()
        window = engine.compute_nadir_window(regimen)
        result = engine.assess_nadir(window.end + timedelta(days=1), regimen)
        assert result.is_in_nadir is False

    def test_days_until_clear_on_first_day(self):
        regimen = self._regimen()
        window = engine.compute_nadir_window(regimen)
        result = engine.assess_nadir(window.start, regimen)
        expected = (window.end - window.start).days + 1
        assert result.days_until_clear == expected

    def test_days_until_clear_on_last_day(self):
        regimen = self._regimen()
        window = engine.compute_nadir_window(regimen)
        result = engine.assess_nadir(window.end, regimen)
        assert result.days_until_clear == 1

    def test_days_until_clear_zero_outside_window(self):
        regimen = self._regimen()
        result = engine.assess_nadir(date(2024, 1, 25), regimen)
        assert result.days_until_clear == 0

    def test_anc_below_threshold_flagged(self):
        regimen = self._regimen()
        labs = make_labs(anc=300.0)
        result = engine.assess_nadir(date(2024, 1, 25), regimen, labs)
        assert result.anc_below_threshold is True

    def test_anc_above_threshold_not_flagged(self):
        regimen = self._regimen()
        labs = make_labs(anc=800.0)
        result = engine.assess_nadir(date(2024, 1, 25), regimen, labs)
        assert result.anc_below_threshold is False

    def test_anc_exactly_500_not_flagged(self):
        regimen = self._regimen()
        labs = make_labs(anc=500.0)
        result = engine.assess_nadir(date(2024, 1, 25), regimen, labs)
        assert result.anc_below_threshold is False

    def test_anc_499_flagged(self):
        regimen = self._regimen()
        labs = make_labs(anc=499.0)
        result = engine.assess_nadir(date(2024, 1, 25), regimen, labs)
        assert result.anc_below_threshold is True

    def test_no_labs_anc_not_flagged(self):
        regimen = self._regimen()
        result = engine.assess_nadir(date(2024, 1, 10), regimen, labs=None)
        assert result.anc_below_threshold is False

    def test_labs_none_anc_field_not_flagged(self):
        regimen = self._regimen()
        labs = make_labs(anc=None)
        result = engine.assess_nadir(date(2024, 1, 10), regimen, labs)
        assert result.anc_below_threshold is False

    def test_returns_nadir_assessment_type(self):
        regimen = self._regimen()
        result = engine.assess_nadir(date(2024, 1, 10), regimen)
        assert isinstance(result, NadirAssessment)

    def test_anc_low_outside_date_window_still_flagged(self):
        """ANC check is independent of the date-based nadir window."""
        regimen = self._regimen()
        labs = make_labs(anc=100.0)
        result = engine.assess_nadir(date(2024, 1, 25), regimen, labs)
        assert result.is_in_nadir is False
        assert result.anc_below_threshold is True


# =============================================================================
# TESTS: assess_fatigue
# =============================================================================

class TestAssessFatigue:
    """assess_fatigue: threshold checks, patient baseline respected."""

    def test_above_default_threshold_returns_true(self):
        labs = make_labs(fatigue_score=5)
        assert engine.assess_fatigue(labs) is True

    def test_below_default_threshold_returns_false(self):
        labs = make_labs(fatigue_score=3)
        assert engine.assess_fatigue(labs) is False

    def test_at_default_threshold_returns_true(self):
        labs = make_labs(fatigue_score=4)
        assert engine.assess_fatigue(labs) is True

    def test_none_fatigue_score_returns_false(self):
        labs = make_labs(fatigue_score=None)
        assert engine.assess_fatigue(labs) is False

    def test_lower_patient_baseline_is_used(self):
        """If patient baseline < 4.0, threshold should be patient baseline."""
        labs = make_labs(fatigue_score=3)
        # With baseline 3.0, score 3 >= 3.0 → True
        assert engine.assess_fatigue(labs, fatigue_baseline=3.0) is True

    def test_system_threshold_used_when_lower(self):
        """If system threshold (4.0) < patient baseline, system threshold wins."""
        labs = make_labs(fatigue_score=4)
        # Baseline 6.0, system 4.0 → min = 4.0 → score 4 >= 4.0 → True
        assert engine.assess_fatigue(labs, fatigue_baseline=6.0) is True

    def test_patient_baseline_6_score_5_no_concern(self):
        """Score 5 < system threshold 4.0 — wait, 5 > 4.0 so still flagged."""
        labs = make_labs(fatigue_score=5)
        # system threshold 4.0 < baseline 6.0 → effective threshold = 4.0
        # score 5 >= 4.0 → True
        assert engine.assess_fatigue(labs, fatigue_baseline=6.0) is True

    def test_fatigue_score_zero_returns_false(self):
        labs = make_labs(fatigue_score=0)
        assert engine.assess_fatigue(labs) is False

    def test_fatigue_score_ten_returns_true(self):
        labs = make_labs(fatigue_score=10)
        assert engine.assess_fatigue(labs) is True

    def test_baseline_2_score_2_triggers(self):
        labs = make_labs(fatigue_score=2)
        assert engine.assess_fatigue(labs, fatigue_baseline=2.0) is True

    def test_baseline_2_score_1_no_trigger(self):
        labs = make_labs(fatigue_score=1)
        assert engine.assess_fatigue(labs, fatigue_baseline=2.0) is False


# =============================================================================
# TESTS: detect_overlaps
# =============================================================================

class TestDetectOverlaps:
    """detect_overlaps: strict overlap, boundary touch, multi-event."""

    def test_same_time_same_day_overlaps(self):
        ev = make_event(event_id="a", event_date=date(2024, 3, 1), time="10:00", duration_hours=1.0)
        other = make_event(event_id="b", event_date=date(2024, 3, 1), time="10:00", duration_hours=1.0)
        result = engine.detect_overlaps(ev, [ev, other])
        assert other in result

    def test_partial_overlap_detected(self):
        ev = make_event(event_id="a", event_date=date(2024, 3, 1), time="10:00", duration_hours=2.0)
        other = make_event(event_id="b", event_date=date(2024, 3, 1), time="11:00", duration_hours=2.0)
        result = engine.detect_overlaps(ev, [ev, other])
        assert other in result

    def test_contained_event_overlaps(self):
        """Shorter event fully inside longer event."""
        ev = make_event(event_id="a", event_date=date(2024, 3, 1), time="09:00", duration_hours=4.0)
        other = make_event(event_id="b", event_date=date(2024, 3, 1), time="10:00", duration_hours=1.0)
        result = engine.detect_overlaps(ev, [ev, other])
        assert other in result

    def test_boundary_touching_not_overlap(self):
        """Event A ends at 11:00, Event B starts at 11:00 — NOT an overlap."""
        ev = make_event(event_id="a", event_date=date(2024, 3, 1), time="10:00", duration_hours=1.0)
        other = make_event(event_id="b", event_date=date(2024, 3, 1), time="11:00", duration_hours=1.0)
        result = engine.detect_overlaps(ev, [ev, other])
        assert other not in result

    def test_completely_before_not_overlap(self):
        ev = make_event(event_id="a", event_date=date(2024, 3, 1), time="14:00", duration_hours=1.0)
        other = make_event(event_id="b", event_date=date(2024, 3, 1), time="10:00", duration_hours=2.0)
        result = engine.detect_overlaps(ev, [ev, other])
        assert other not in result

    def test_completely_after_not_overlap(self):
        ev = make_event(event_id="a", event_date=date(2024, 3, 1), time="10:00", duration_hours=1.0)
        other = make_event(event_id="b", event_date=date(2024, 3, 1), time="14:00", duration_hours=1.0)
        result = engine.detect_overlaps(ev, [ev, other])
        assert other not in result

    def test_different_dates_not_overlap(self):
        ev = make_event(event_id="a", event_date=date(2024, 3, 1), time="10:00", duration_hours=2.0)
        other = make_event(event_id="b", event_date=date(2024, 3, 2), time="10:00", duration_hours=2.0)
        result = engine.detect_overlaps(ev, [ev, other])
        assert other not in result

    def test_event_does_not_conflict_with_itself(self):
        ev = make_event(event_id="a", event_date=date(2024, 3, 1), time="10:00", duration_hours=1.0)
        result = engine.detect_overlaps(ev, [ev])
        assert ev not in result

    def test_multiple_overlapping_events_all_returned(self):
        ev = make_event(event_id="a", event_date=date(2024, 3, 1), time="10:00", duration_hours=3.0)
        b = make_event(event_id="b", event_date=date(2024, 3, 1), time="10:30", duration_hours=1.0)
        c = make_event(event_id="c", event_date=date(2024, 3, 1), time="11:00", duration_hours=1.0)
        result = engine.detect_overlaps(ev, [ev, b, c])
        assert b in result
        assert c in result

    def test_empty_events_list_returns_empty(self):
        ev = make_event()
        result = engine.detect_overlaps(ev, [])
        assert result == []

    def test_minutes_precision_overlap(self):
        """Overlap decided by minutes, not just hours."""
        ev = make_event(event_id="a", event_date=date(2024, 3, 1), time="10:00", duration_hours=1.5)
        # Ends at 11:30; other starts at 11:00 — overlap of 30 min
        other = make_event(event_id="b", event_date=date(2024, 3, 1), time="11:00", duration_hours=1.0)
        result = engine.detect_overlaps(ev, [ev, other])
        assert other in result

    def test_minutes_precision_boundary_no_overlap(self):
        """ev ends at 10:30, other starts at 10:30 — boundary touch."""
        ev = make_event(event_id="a", event_date=date(2024, 3, 1), time="10:00", duration_hours=0.5)
        other = make_event(event_id="b", event_date=date(2024, 3, 1), time="10:30", duration_hours=1.0)
        result = engine.detect_overlaps(ev, [ev, other])
        assert other not in result


# =============================================================================
# TESTS: detect_recovery_violation
# =============================================================================

class TestDetectRecoveryViolation:
    """detect_recovery_violation: <24h gap, ≥24h safe, non-high-intensity."""

    def test_chemo_less_than_24h_before_flagged(self):
        prior = make_event(
            event_id="prior", title="Chemotherapy Session",
            event_date=date(2024, 3, 1), time="09:00", duration_hours=4.0,
        )
        # Prior ends 13:00; next starts 14:00 (gap = 1h)
        next_ev = make_event(
            event_id="next", title="Oncology Follow-Up",
            event_date=date(2024, 3, 1), time="14:00", duration_hours=1.0,
        )
        result = engine.detect_recovery_violation(next_ev, [prior, next_ev])
        assert result is not None
        assert result.event_id == "prior"

    def test_exactly_24h_gap_not_flagged(self):
        prior = make_event(
            event_id="prior", title="Infusion Therapy",
            event_date=date(2024, 3, 1), time="09:00", duration_hours=3.0,
        )
        # Prior ends Mar 1 12:00; next starts Mar 2 12:00 (exactly 24h gap)
        next_ev = make_event(
            event_id="next", title="Follow-Up",
            event_date=date(2024, 3, 2), time="12:00", duration_hours=1.0,
        )
        result = engine.detect_recovery_violation(next_ev, [prior, next_ev])
        assert result is None

    def test_more_than_24h_gap_not_flagged(self):
        prior = make_event(
            event_id="prior", title="Radiation Therapy",
            event_date=date(2024, 3, 1), time="09:00", duration_hours=1.0,
        )
        # Prior ends 10:00 Mar 1; next starts 14:00 Mar 2 (gap = 28h)
        next_ev = make_event(
            event_id="next", title="Consultation",
            event_date=date(2024, 3, 2), time="14:00", duration_hours=1.0,
        )
        result = engine.detect_recovery_violation(next_ev, [prior, next_ev])
        assert result is None

    def test_non_high_intensity_not_flagged(self):
        prior = make_event(
            event_id="prior", title="Dietary Consultation",
            event_date=date(2024, 3, 1), time="09:00", duration_hours=1.0,
        )
        next_ev = make_event(
            event_id="next", title="Follow-Up",
            event_date=date(2024, 3, 1), time="11:00", duration_hours=1.0,
        )
        result = engine.detect_recovery_violation(next_ev, [prior, next_ev])
        assert result is None

    def test_surgery_keyword_flagged(self):
        prior = make_event(
            event_id="prior", title="Port Surgery",
            event_date=date(2024, 3, 1), time="08:00", duration_hours=2.0,
        )
        next_ev = make_event(
            event_id="next", title="Lab Draw",
            event_date=date(2024, 3, 1), time="12:00", duration_hours=0.5,
        )
        result = engine.detect_recovery_violation(next_ev, [prior, next_ev])
        assert result is not None

    def test_infusion_keyword_case_insensitive(self):
        prior = make_event(
            event_id="prior", title="IV INFUSION",
            event_date=date(2024, 3, 1), time="08:00", duration_hours=6.0,
        )
        next_ev = make_event(
            event_id="next", title="Nurse Visit",
            event_date=date(2024, 3, 1), time="15:00", duration_hours=0.5,
        )
        result = engine.detect_recovery_violation(next_ev, [prior, next_ev])
        assert result is not None

    def test_biopsy_keyword_flagged(self):
        prior = make_event(
            event_id="prior", title="Bone Marrow Biopsy",
            event_date=date(2024, 3, 1), time="10:00", duration_hours=1.0,
        )
        next_ev = make_event(
            event_id="next", title="Follow-Up",
            event_date=date(2024, 3, 1), time="12:00", duration_hours=1.0,
        )
        result = engine.detect_recovery_violation(next_ev, [prior, next_ev])
        assert result is not None

    def test_event_not_flagged_against_itself(self):
        ev = make_event(event_id="a", title="Chemotherapy", event_date=date(2024, 3, 1),
                        time="10:00", duration_hours=4.0)
        result = engine.detect_recovery_violation(ev, [ev])
        assert result is None

    def test_no_prior_events_returns_none(self):
        ev = make_event(event_id="a", title="Follow-Up", event_date=date(2024, 3, 1),
                        time="10:00", duration_hours=1.0)
        result = engine.detect_recovery_violation(ev, [ev])
        assert result is None

    def test_future_high_intensity_not_flagged(self):
        """Recovery violation only flags events that are PRIOR (gap > 0)."""
        next_high = make_event(
            event_id="future", title="Chemotherapy",
            event_date=date(2024, 3, 2), time="10:00", duration_hours=4.0,
        )
        ev = make_event(
            event_id="now", title="Follow-Up",
            event_date=date(2024, 3, 1), time="10:00", duration_hours=1.0,
        )
        result = engine.detect_recovery_violation(ev, [ev, next_high])
        assert result is None


# =============================================================================
# TESTS: analyse_event
# =============================================================================

class TestAnalyseEvent:
    """analyse_event: priority ordering, all four conflict types, empty result."""

    def _regimen(self) -> ChemoRegimen:
        return make_regimen(cycle_start=date(2024, 1, 1), nadir_start=7, nadir_end=14)

    def test_no_conflicts_returns_empty_list(self):
        event = make_event(event_date=date(2024, 1, 25), time="10:00")
        labs = make_labs(anc=1500.0, fatigue_score=2)
        result = engine.analyse_event(event, [event], regimen=self._regimen(), labs=labs)
        assert result == []

    def test_nadir_window_conflict_detected(self):
        # Window: Jan 5 – Jan 16 with buffer=2
        event = make_event(event_date=date(2024, 1, 10), time="10:00")
        result = engine.analyse_event(event, [event], regimen=self._regimen())
        assert len(result) == 1
        assert result[0].conflict_type == ConflictType.NADIR_WINDOW

    def test_nadir_confidence_095_with_anc(self):
        event = make_event(event_date=date(2024, 1, 10), time="10:00")
        labs = make_labs(anc=200.0)
        result = engine.analyse_event(event, [event], regimen=self._regimen(), labs=labs)
        assert result[0].confidence == 0.95

    def test_nadir_confidence_085_without_anc(self):
        event = make_event(event_date=date(2024, 1, 10), time="10:00")
        result = engine.analyse_event(event, [event], regimen=self._regimen(), labs=None)
        assert result[0].confidence == 0.85

    def test_nadir_priority_over_fatigue(self):
        """NADIR_WINDOW should be returned even when fatigue is also concerning."""
        event = make_event(event_date=date(2024, 1, 10), time="10:00")
        labs = make_labs(anc=200.0, fatigue_score=8)
        result = engine.analyse_event(event, [event], regimen=self._regimen(), labs=labs)
        assert result[0].conflict_type == ConflictType.NADIR_WINDOW

    def test_fatigue_conflict_detected(self):
        event = make_event(event_date=date(2024, 1, 25), time="10:00")
        labs = make_labs(anc=1500.0, fatigue_score=7)
        result = engine.analyse_event(event, [event], regimen=self._regimen(), labs=labs)
        assert len(result) == 1
        assert result[0].conflict_type == ConflictType.FATIGUE_THRESHOLD

    def test_fatigue_confidence_080(self):
        event = make_event(event_date=date(2024, 1, 25), time="10:00")
        labs = make_labs(anc=1500.0, fatigue_score=7)
        result = engine.analyse_event(event, [event], regimen=self._regimen(), labs=labs)
        assert result[0].confidence == 0.80

    def test_fatigue_priority_over_overlap(self):
        """FATIGUE_THRESHOLD should be returned over APPOINTMENT_OVERLAP."""
        ev = make_event(event_id="a", event_date=date(2024, 1, 25), time="10:00", duration_hours=2.0)
        other = make_event(event_id="b", event_date=date(2024, 1, 25), time="11:00", duration_hours=1.0)
        labs = make_labs(anc=1500.0, fatigue_score=7)
        result = engine.analyse_event(ev, [ev, other], regimen=self._regimen(), labs=labs)
        assert result[0].conflict_type == ConflictType.FATIGUE_THRESHOLD

    def test_overlap_conflict_detected(self):
        ev = make_event(event_id="a", event_date=date(2024, 1, 25), time="10:00", duration_hours=2.0)
        other = make_event(event_id="b", event_date=date(2024, 1, 25), time="11:00", duration_hours=1.0)
        labs = make_labs(anc=1500.0, fatigue_score=2)
        result = engine.analyse_event(ev, [ev, other], regimen=self._regimen(), labs=labs)
        assert len(result) == 1
        assert result[0].conflict_type == ConflictType.APPOINTMENT_OVERLAP

    def test_overlap_confidence_099(self):
        ev = make_event(event_id="a", event_date=date(2024, 1, 25), time="10:00", duration_hours=2.0)
        other = make_event(event_id="b", event_date=date(2024, 1, 25), time="11:00", duration_hours=1.0)
        labs = make_labs(anc=1500.0, fatigue_score=2)
        result = engine.analyse_event(ev, [ev, other], regimen=self._regimen(), labs=labs)
        assert result[0].confidence == 0.99

    def test_overlap_priority_over_recovery(self):
        """APPOINTMENT_OVERLAP should be returned over RECOVERY_PERIOD."""
        prior = make_event(
            event_id="prior", title="Chemotherapy",
            event_date=date(2024, 1, 25), time="08:00", duration_hours=2.0,
        )
        ev = make_event(
            event_id="a", event_date=date(2024, 1, 25), time="10:00", duration_hours=2.0,
        )
        overlap = make_event(
            event_id="b", event_date=date(2024, 1, 25), time="11:00", duration_hours=1.0,
        )
        labs = make_labs(anc=1500.0, fatigue_score=2)
        result = engine.analyse_event(ev, [prior, ev, overlap], regimen=self._regimen(), labs=labs)
        assert result[0].conflict_type == ConflictType.APPOINTMENT_OVERLAP

    def test_recovery_conflict_detected(self):
        prior = make_event(
            event_id="prior", title="Chemotherapy",
            event_date=date(2024, 1, 25), time="09:00", duration_hours=4.0,
        )
        ev = make_event(
            event_id="a", event_date=date(2024, 1, 25), time="15:00", duration_hours=1.0,
        )
        labs = make_labs(anc=1500.0, fatigue_score=2)
        result = engine.analyse_event(ev, [prior, ev], regimen=self._regimen(), labs=labs)
        assert len(result) == 1
        assert result[0].conflict_type == ConflictType.RECOVERY_PERIOD

    def test_recovery_confidence_075(self):
        prior = make_event(
            event_id="prior", title="Infusion",
            event_date=date(2024, 1, 25), time="09:00", duration_hours=4.0,
        )
        ev = make_event(
            event_id="a", event_date=date(2024, 1, 25), time="15:00", duration_hours=1.0,
        )
        labs = make_labs(anc=1500.0, fatigue_score=2)
        result = engine.analyse_event(ev, [prior, ev], regimen=self._regimen(), labs=labs)
        assert result[0].confidence == 0.75

    def test_suggestion_requires_hitl(self):
        event = make_event(event_date=date(2024, 1, 10), time="10:00")
        result = engine.analyse_event(event, [event], regimen=self._regimen())
        assert result[0].requires_hitl is True

    def test_suggestion_has_event_id(self):
        event = make_event(event_id="evt-xyz", event_date=date(2024, 1, 10), time="10:00")
        result = engine.analyse_event(event, [event], regimen=self._regimen())
        assert result[0].event_id == "evt-xyz"

    def test_suggested_date_after_nadir_clears(self):
        event = make_event(event_date=date(2024, 1, 10), time="10:00")
        window = engine.compute_nadir_window(self._regimen())
        result = engine.analyse_event(event, [event], regimen=self._regimen())
        assert result[0].suggested_date > window.end

    def test_no_regimen_no_nadir_conflict(self):
        """Without a regimen, no nadir conflict can be raised."""
        event = make_event(event_date=date(2024, 1, 10), time="10:00")
        labs = make_labs(anc=1500.0, fatigue_score=2)
        result = engine.analyse_event(event, [event], regimen=None, labs=labs)
        assert all(s.conflict_type != ConflictType.NADIR_WINDOW for s in result)


# =============================================================================
# TESTS: suggest_safe_date
# =============================================================================

class TestSuggestSafeDate:
    """suggest_safe_date: clears nadir, clears fatigue, 30-day fallback."""

    def _regimen(self) -> ChemoRegimen:
        return make_regimen(cycle_start=date(2024, 1, 1), nadir_start=7, nadir_end=14)

    def test_returns_date_after_nadir_window(self):
        window = engine.compute_nadir_window(self._regimen())
        # Start inside window
        safe = engine.suggest_safe_date(window.start, regimen=self._regimen())
        assert safe > window.end

    def test_safe_date_outside_window(self):
        regimen = self._regimen()
        window = engine.compute_nadir_window(regimen)
        safe = engine.suggest_safe_date(window.start, regimen=regimen)
        assert not (window.start <= safe <= window.end)

    def test_already_safe_date_returned_immediately(self):
        safe_start = date(2024, 1, 25)  # well outside nadir window
        result = engine.suggest_safe_date(safe_start, regimen=self._regimen())
        assert result == safe_start

    def test_fatigue_does_not_block_forever(self):
        """With fatigue concerning and no regimen, safe_date skips fatigue concern
        but since fatigue is static (not date-dependent), engine returns day 0."""
        # When labs show high fatigue and no regimen, the engine should still
        # find the first date that clears all checks — but fatigue is static
        # in this engine (same score every day), so it will exhaust the window.
        labs = make_labs(anc=1500.0, fatigue_score=8)
        # With no regimen, fatigue_concerning=True always → should fall back
        result = engine.suggest_safe_date(date(2024, 3, 1), labs=labs, fatigue_baseline=4.0)
        # Expect fallback: original + 30 days
        assert result == date(2024, 3, 1) + timedelta(days=30)

    def test_no_regimen_no_labs_returns_original(self):
        """No constraints → first date is immediately safe."""
        start = date(2024, 3, 15)
        result = engine.suggest_safe_date(start)
        assert result == start

    def test_anc_low_skips_all_30_days_fallback(self):
        """Low ANC is static — engine should exhaust window and fall back."""
        labs = make_labs(anc=100.0, fatigue_score=0)
        result = engine.suggest_safe_date(
            date(2024, 3, 1), regimen=None, labs=labs, fatigue_baseline=4.0
        )
        assert result == date(2024, 3, 1) + timedelta(days=30)

    def test_nadir_fallback_uses_window_end_plus_1(self):
        """When nadir covers entire search window, fallback = nadir_end + 1."""
        # Create a very long nadir to exceed 30 days
        regimen = make_regimen(
            cycle_start=date(2024, 1, 1),
            nadir_start=1,   # Day 1 - buffer=2 → can't go negative, starts Jan 1
            nadir_end=40,    # Window end: Jan 40 + 2 buffer = well past 30-day search
        )
        window = engine.compute_nadir_window(regimen)
        result = engine.suggest_safe_date(date(2024, 1, 1), regimen=regimen)
        assert result == window.end + timedelta(days=1)


# =============================================================================
# TESTS: calendar_agent_node
# =============================================================================

class TestCalendarAgentNode:
    """calendar_agent_node: no-profile, no-events, conflicts found paths."""

    def _base_state(self, *, profile=None, events=None) -> dict:
        return {
            "patient_profile": profile,
            "calendar_events": events or [],
            "audit_log": [],
            "messages": [],
            "pending_suggestions": [],
            "hitl_required": False,
        }

    def _make_profile(self, *, regimen=None, labs=None, fatigue_baseline=4.0):
        class FakeProfile:
            pass
        p = FakeProfile()
        p.regimen = regimen
        p.latest_labs = labs
        p.fatigue_baseline = fatigue_baseline
        return p

    # --- No profile path ---

    def test_no_profile_returns_dict(self):
        state = self._base_state()
        result = calendar_agent_node(state)
        assert isinstance(result, dict)

    def test_no_profile_hitl_false(self):
        result = calendar_agent_node(self._base_state())
        assert result["hitl_required"] is False

    def test_no_profile_empty_suggestions(self):
        result = calendar_agent_node(self._base_state())
        assert result["pending_suggestions"] == []

    def test_no_profile_has_audit_entry(self):
        result = calendar_agent_node(self._base_state())
        assert len(result["audit_log"]) >= 1

    def test_no_profile_has_message(self):
        result = calendar_agent_node(self._base_state())
        assert len(result["messages"]) >= 1

    def test_no_profile_message_mentions_profile(self):
        result = calendar_agent_node(self._base_state())
        msg_content = result["messages"][0].content.lower()
        assert "profile" in msg_content

    # --- No events path ---

    def test_no_events_returns_dict(self):
        profile = self._make_profile()
        result = calendar_agent_node(self._base_state(profile=profile))
        assert isinstance(result, dict)

    def test_no_events_hitl_false(self):
        profile = self._make_profile()
        result = calendar_agent_node(self._base_state(profile=profile))
        assert result["hitl_required"] is False

    def test_no_events_empty_suggestions(self):
        profile = self._make_profile()
        result = calendar_agent_node(self._base_state(profile=profile))
        assert result["pending_suggestions"] == []

    def test_no_events_has_audit_entry(self):
        profile = self._make_profile()
        result = calendar_agent_node(self._base_state(profile=profile))
        assert len(result["audit_log"]) >= 1

    def test_no_events_has_message(self):
        profile = self._make_profile()
        result = calendar_agent_node(self._base_state(profile=profile))
        assert len(result["messages"]) >= 1

    # --- Conflicts found path ---

    def test_conflicts_sets_hitl_true(self):
        regimen = make_regimen(cycle_start=date(2024, 1, 1), nadir_start=7, nadir_end=14)
        profile = self._make_profile(regimen=regimen)
        # Event inside nadir window (Jan 5–16 with buffer=2)
        ev = make_event(event_id="evt-1", event_date=date(2024, 1, 10), time="10:00")
        result = calendar_agent_node(self._base_state(profile=profile, events=[ev]))
        assert result["hitl_required"] is True

    def test_conflicts_returns_suggestions(self):
        regimen = make_regimen(cycle_start=date(2024, 1, 1), nadir_start=7, nadir_end=14)
        profile = self._make_profile(regimen=regimen)
        ev = make_event(event_id="evt-1", event_date=date(2024, 1, 10), time="10:00")
        result = calendar_agent_node(self._base_state(profile=profile, events=[ev]))
        assert len(result["pending_suggestions"]) >= 1

    def test_conflicts_audit_log_populated(self):
        regimen = make_regimen(cycle_start=date(2024, 1, 1), nadir_start=7, nadir_end=14)
        profile = self._make_profile(regimen=regimen)
        ev = make_event(event_id="evt-1", event_date=date(2024, 1, 10), time="10:00")
        result = calendar_agent_node(self._base_state(profile=profile, events=[ev]))
        assert len(result["audit_log"]) >= 1

    def test_conflicts_message_mentions_conflict(self):
        regimen = make_regimen(cycle_start=date(2024, 1, 1), nadir_start=7, nadir_end=14)
        profile = self._make_profile(regimen=regimen)
        ev = make_event(event_id="evt-1", event_date=date(2024, 1, 10), time="10:00")
        result = calendar_agent_node(self._base_state(profile=profile, events=[ev]))
        content = result["messages"][0].content.lower()
        assert "conflict" in content

    def test_clean_calendar_hitl_false(self):
        """All events outside nadir with good labs → no conflicts → hitl_required False."""
        regimen = make_regimen(cycle_start=date(2024, 1, 1), nadir_start=7, nadir_end=14)
        labs = make_labs(anc=1500.0, fatigue_score=2)
        profile = self._make_profile(regimen=regimen, labs=labs)
        ev = make_event(event_id="evt-1", event_date=date(2024, 1, 25), time="10:00")
        result = calendar_agent_node(self._base_state(profile=profile, events=[ev]))
        assert result["hitl_required"] is False

    def test_clean_calendar_empty_suggestions(self):
        regimen = make_regimen(cycle_start=date(2024, 1, 1), nadir_start=7, nadir_end=14)
        labs = make_labs(anc=1500.0, fatigue_score=2)
        profile = self._make_profile(regimen=regimen, labs=labs)
        ev = make_event(event_id="evt-1", event_date=date(2024, 1, 25), time="10:00")
        result = calendar_agent_node(self._base_state(profile=profile, events=[ev]))
        assert result["pending_suggestions"] == []

    def test_result_keys_present(self):
        """Verify all required output keys are always returned."""
        result = calendar_agent_node(self._base_state())
        required = {"audit_log", "messages", "pending_suggestions", "hitl_required"}
        assert required.issubset(result.keys())

    def test_multiple_events_analysed(self):
        """Multiple events, some conflicting, some not."""
        regimen = make_regimen(cycle_start=date(2024, 1, 1), nadir_start=7, nadir_end=14)
        labs = make_labs(anc=1500.0, fatigue_score=2)
        profile = self._make_profile(regimen=regimen, labs=labs)
        ev_conflict = make_event(event_id="e1", event_date=date(2024, 1, 10), time="10:00")
        ev_safe = make_event(event_id="e2", event_date=date(2024, 1, 25), time="10:00")
        result = calendar_agent_node(
            self._base_state(profile=profile, events=[ev_conflict, ev_safe])
        )
        assert result["hitl_required"] is True
        # One suggestion for the conflicting event
        assert len(result["pending_suggestions"]) >= 1
