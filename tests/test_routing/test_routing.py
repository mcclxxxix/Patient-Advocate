"""
Tests for patient_advocate.routing.patient_advocate_routing_system
==================================================================

All external packages are stubbed BEFORE the module under test is imported,
so the test suite runs without langchain, langgraph, or pydantic-settings
installed in the test environment.

Coverage targets (40+ tests):
  - Signal classification for each of the 5 agent types
  - Plural-form matching (trials, studies, eligibilities …)
  - Confidence bounds enforcement ([0.55, 1.0])
  - HITL gate fires when confidence < 0.7
  - master_brain_node state mutations:
      audit_log appended, messages appended, next_agent set, hitl_required set
  - master_brain_node does NOT mutate calendar_events / pending_suggestions
  - route_to_agent returns correct node names
  - HITL override in route_to_agent
  - Stub agent nodes return TransparencyRecord with correct agent_name
  - reasoning_chain includes required scholarly citations

Author: Patient Advocate Platform Team
License: Apache-2.0
"""

from __future__ import annotations

# =============================================================================
# STUB EXTERNAL PACKAGES — must happen before any package import
# =============================================================================

import sys
import types
from datetime import datetime
from uuid import UUID, uuid4

# ---------------------------------------------------------------------------
# 1. Stub langchain_core.messages
# ---------------------------------------------------------------------------
_langchain_core = types.ModuleType("langchain_core")
_langchain_core_messages = types.ModuleType("langchain_core.messages")


class _BaseMessage:
    def __init__(self, content: str = "", **kwargs):
        self.content = content
    def __repr__(self):
        return f"{self.__class__.__name__}(content={self.content!r})"


class _HumanMessage(_BaseMessage):
    pass


class _AIMessage(_BaseMessage):
    pass


_langchain_core_messages.BaseMessage = _BaseMessage
_langchain_core_messages.HumanMessage = _HumanMessage
_langchain_core_messages.AIMessage = _AIMessage
_langchain_core.messages = _langchain_core_messages
sys.modules["langchain_core"] = _langchain_core
sys.modules["langchain_core.messages"] = _langchain_core_messages

# ---------------------------------------------------------------------------
# 2. Stub langgraph.graph
# ---------------------------------------------------------------------------
_langgraph = types.ModuleType("langgraph")
_langgraph_graph = types.ModuleType("langgraph.graph")
_GRAPH_END_SENTINEL = "end"


class _StateGraph:
    def __init__(self, state_schema=None):
        self._nodes: dict = {}
        self._edges: list = []
        self._conditional_edges: list = []
        self._entry_point = None

    def add_node(self, name, fn):
        self._nodes[name] = fn

    def add_edge(self, src, dst):
        self._edges.append((src, dst))

    def add_conditional_edges(self, src, fn, mapping):
        self._conditional_edges.append((src, fn, mapping))

    def set_entry_point(self, name):
        self._entry_point = name

    def compile(self):
        return self


_langgraph_graph.StateGraph = _StateGraph
_langgraph_graph.END = _GRAPH_END_SENTINEL
_langgraph.graph = _langgraph_graph
sys.modules["langgraph"] = _langgraph
sys.modules["langgraph.graph"] = _langgraph_graph

# ---------------------------------------------------------------------------
# 3. Stub pydantic_settings
# ---------------------------------------------------------------------------
_pydantic_settings = types.ModuleType("pydantic_settings")


class _BaseSettings:
    """Minimal pydantic-settings stub."""
    model_config = {}

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _SettingsConfigDict(dict):
    pass


_pydantic_settings.BaseSettings = _BaseSettings
_pydantic_settings.SettingsConfigDict = _SettingsConfigDict
sys.modules["pydantic_settings"] = _pydantic_settings

# ---------------------------------------------------------------------------
# 4. Stub patient_advocate.core.config with Settings.hitl_confidence_threshold
# ---------------------------------------------------------------------------
_pa_core_config = types.ModuleType("patient_advocate.core.config")


class _Settings:
    hitl_confidence_threshold: float = 0.7

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


_pa_core_config.Settings = _Settings
sys.modules["patient_advocate.core.config"] = _pa_core_config

# ---------------------------------------------------------------------------
# 5. Stub aiosqlite / sqlalchemy / anything else that might be pulled in
#    transitively by patient_advocate.core.* imports
# ---------------------------------------------------------------------------
for _stub_mod in (
    "aiosqlite",
    "sqlalchemy",
    "sqlalchemy.ext",
    "sqlalchemy.ext.asyncio",
    "pydantic",
    "pydantic.fields",
):
    if _stub_mod not in sys.modules:
        sys.modules[_stub_mod] = types.ModuleType(_stub_mod)

# Ensure pydantic.BaseModel / ConfigDict stubs exist (models.py uses them)
import pydantic as _pydantic_stub  # noqa: E402 — this is our stub

if not hasattr(_pydantic_stub, "BaseModel"):
    class _PydanticBaseModel:
        model_config = {}
        def __init_subclass__(cls, **kwargs): ...
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                object.__setattr__(self, k, v)

    class _Field:
        def __init__(self, default=None, **kwargs):
            self.default = default

    class _ConfigDict(dict):
        pass

    _pydantic_stub.BaseModel = _PydanticBaseModel
    _pydantic_stub.Field = _Field
    _pydantic_stub.ConfigDict = _ConfigDict

# ---------------------------------------------------------------------------
# 6. Patch patient_advocate.core.models to use our stub BaseMessage
# ---------------------------------------------------------------------------
# The real models.py imports from pydantic — re-use stub above.
# We need TransparencyRecord to actually work as a real object in tests,
# so we define a lightweight version that the module can import.
# ---------------------------------------------------------------------------

_pa_core_models = types.ModuleType("patient_advocate.core.models")


class AgentType(str):
    """Minimal AgentType enum-like class."""
    MASTER_BRAIN: "AgentType"
    VISION_AGENT: "AgentType"
    CALENDAR_AGENT: "AgentType"
    CLINICAL_TRIALS_AGENT: "AgentType"
    ETHICS_COMPLAINT_AGENT: "AgentType"
    LEGAL_MOTIONS_AGENT: "AgentType"
    END: "AgentType"

    def __new__(cls, value):
        obj = str.__new__(cls, value)
        obj._value_ = value
        return obj

    @property
    def value(self):
        return self._value_


# Instantiate constants
AgentType.MASTER_BRAIN = AgentType("master_brain")
AgentType.VISION_AGENT = AgentType("vision_agent")
AgentType.CALENDAR_AGENT = AgentType("calendar_agent")
AgentType.CLINICAL_TRIALS_AGENT = AgentType("clinical_trials_agent")
AgentType.ETHICS_COMPLAINT_AGENT = AgentType("ethics_complaint_agent")
AgentType.LEGAL_MOTIONS_AGENT = AgentType("legal_motions_agent")
AgentType.END = AgentType("end")


class HITLDecision(str):
    APPROVED: "HITLDecision"
    REJECTED: "HITLDecision"
    DEFERRED: "HITLDecision"

    def __new__(cls, value):
        obj = str.__new__(cls, value)
        obj._value_ = value
        return obj

    @property
    def value(self):
        return self._value_


HITLDecision.APPROVED = HITLDecision("approved")
HITLDecision.REJECTED = HITLDecision("rejected")
HITLDecision.DEFERRED = HITLDecision("deferred")


class ConflictType(str):
    NADIR_WINDOW: "ConflictType"
    FATIGUE_THRESHOLD: "ConflictType"
    APPOINTMENT_OVERLAP: "ConflictType"
    RECOVERY_PERIOD: "ConflictType"

    def __new__(cls, value):
        obj = str.__new__(cls, value)
        obj._value_ = value
        return obj

    @property
    def value(self):
        return self._value_


ConflictType.NADIR_WINDOW = ConflictType("nadir_window")
ConflictType.FATIGUE_THRESHOLD = ConflictType("fatigue_threshold")
ConflictType.APPOINTMENT_OVERLAP = ConflictType("appointment_overlap")
ConflictType.RECOVERY_PERIOD = ConflictType("recovery_period")


class TransparencyRecord:
    """Lightweight TransparencyRecord matching the frozen Pydantic model interface."""
    def __init__(
        self,
        *,
        event_id=None,
        suggestion_id=None,
        agent_name: str = "",
        conflict_type=None,
        confidence=None,
        scholarly_basis=None,
        patient_decision=None,
        reasoning_chain: str = "",
        timestamp=None,
    ):
        self.event_id = event_id or uuid4()
        self.suggestion_id = suggestion_id
        self.agent_name = agent_name
        self.conflict_type = conflict_type
        self.confidence = confidence
        self.scholarly_basis = scholarly_basis
        self.patient_decision = patient_decision
        self.reasoning_chain = reasoning_chain
        self.timestamp = timestamp or datetime.utcnow()

    def __repr__(self):
        return f"TransparencyRecord(agent_name={self.agent_name!r}, confidence={self.confidence})"


class EthicsFlag:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class PatientProfile:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class CalendarEvent:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class CalendarSuggestion:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


_pa_core_models.AgentType = AgentType
_pa_core_models.HITLDecision = HITLDecision
_pa_core_models.ConflictType = ConflictType
_pa_core_models.TransparencyRecord = TransparencyRecord
_pa_core_models.EthicsFlag = EthicsFlag
_pa_core_models.PatientProfile = PatientProfile
_pa_core_models.CalendarEvent = CalendarEvent
_pa_core_models.CalendarSuggestion = CalendarSuggestion
sys.modules["patient_advocate.core.models"] = _pa_core_models

# ---------------------------------------------------------------------------
# 7. Stub patient_advocate.core.state
# ---------------------------------------------------------------------------
_pa_core_state = types.ModuleType("patient_advocate.core.state")


class AdvocateState(dict):
    """TypedDict-like stub for AdvocateState."""


_pa_core_state.AdvocateState = AdvocateState
sys.modules["patient_advocate.core.state"] = _pa_core_state

# Ensure patient_advocate package path exists in sys.modules
for _mod_name in (
    "patient_advocate",
    "patient_advocate.core",
    "patient_advocate.routing",
):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = types.ModuleType(_mod_name)

# =============================================================================
# NOW import the module under test
# =============================================================================
import importlib  # noqa: E402
import unittest  # noqa: E402

# Force fresh import in case the module was partially loaded
if "patient_advocate.routing.patient_advocate_routing_system" in sys.modules:
    del sys.modules["patient_advocate.routing.patient_advocate_routing_system"]

from patient_advocate.routing.patient_advocate_routing_system import (  # noqa: E402
    IntentClassifier,
    build_advocate_graph,
    calendar_agent_node,
    clinical_trials_agent_node,
    ethics_complaint_agent_node,
    legal_motions_agent_node,
    master_brain_node,
    route_to_agent,
    vision_agent_node,
    _ROUTING_CITATIONS,
    _classifier,
)


# =============================================================================
# HELPERS
# =============================================================================

def _make_state(
    human_text: str = "",
    next_agent: str = "",
    hitl_required: bool = False,
    calendar_events: list | None = None,
    pending_suggestions: list | None = None,
    image_path: str | None = None,
) -> dict:
    """Build a minimal AdvocateState dict for testing."""
    msgs = []
    if human_text:
        msgs.append(_HumanMessage(content=human_text))
    return {
        "messages": msgs,
        "audit_log": [],
        "next_agent": next_agent,
        "hitl_required": hitl_required,
        "calendar_events": calendar_events if calendar_events is not None else [],
        "pending_suggestions": pending_suggestions if pending_suggestions is not None else [],
        "patient_profile": None,
        "image_path": image_path,
        "ethics_flags": [],
        "calendar_data": None,
        "ethics_data": None,
        "legal_data": None,
        "documents_generated": None,
    }


# =============================================================================
# TEST CASES
# =============================================================================

class TestIntentClassifierCalendar(unittest.TestCase):
    """Tests that calendar-related phrases route to CALENDAR_AGENT."""

    def setUp(self):
        self.clf = IntentClassifier()

    def test_schedule_appointment_routes_to_calendar(self):
        agent, conf = self.clf.classify("I need to schedule an appointment next week")
        self.assertEqual(agent, AgentType.CALENDAR_AGENT)

    def test_reschedule_routes_to_calendar(self):
        agent, conf = self.clf.classify("Can you help me reschedule my chemo session?")
        self.assertEqual(agent, AgentType.CALENDAR_AGENT)

    def test_nadir_routes_to_calendar(self):
        agent, conf = self.clf.classify("My nadir window starts on day 8 of the cycle")
        self.assertEqual(agent, AgentType.CALENDAR_AGENT)

    def test_fatigue_routes_to_calendar(self):
        agent, conf = self.clf.classify("My fatigue is too high to attend this appointment")
        self.assertEqual(agent, AgentType.CALENDAR_AGENT)

    def test_treatment_date_routes_to_calendar(self):
        agent, conf = self.clf.classify("What is my treatment date for next cycle?")
        self.assertEqual(agent, AgentType.CALENDAR_AGENT)

    def test_follow_up_routes_to_calendar(self):
        agent, conf = self.clf.classify("I need to book a follow-up appointment with my oncologist")
        self.assertEqual(agent, AgentType.CALENDAR_AGENT)

    def test_calendar_confidence_in_bounds(self):
        _, conf = self.clf.classify("Please schedule my infusion date on the calendar")
        self.assertGreaterEqual(conf, 0.55)
        self.assertLessEqual(conf, 1.0)


class TestIntentClassifierTrials(unittest.TestCase):
    """Tests that clinical-trial phrases route to CLINICAL_TRIALS_AGENT."""

    def setUp(self):
        self.clf = IntentClassifier()

    def test_clinical_trials_plural_routes_correctly(self):
        agent, conf = self.clf.classify("Are there any clinical trials I can join?")
        self.assertEqual(agent, AgentType.CLINICAL_TRIALS_AGENT)

    def test_clinical_trial_singular_routes_correctly(self):
        agent, conf = self.clf.classify("I heard about a clinical trial for my diagnosis")
        self.assertEqual(agent, AgentType.CLINICAL_TRIALS_AGENT)

    def test_eligibility_routes_to_trials(self):
        agent, conf = self.clf.classify("What is my eligibility for the new study?")
        self.assertEqual(agent, AgentType.CLINICAL_TRIALS_AGENT)

    def test_eligibilities_plural_routes_to_trials(self):
        agent, conf = self.clf.classify("Check my eligibilities for open trials")
        self.assertEqual(agent, AgentType.CLINICAL_TRIALS_AGENT)

    def test_enroll_routes_to_trials(self):
        agent, conf = self.clf.classify("How do I enroll in a phase 2 trial?")
        self.assertEqual(agent, AgentType.CLINICAL_TRIALS_AGENT)

    def test_clinical_studies_plural_routes_to_trials(self):
        agent, conf = self.clf.classify("Are there clinical studies for my cancer type?")
        self.assertEqual(agent, AgentType.CLINICAL_TRIALS_AGENT)

    def test_nct_identifier_routes_to_trials(self):
        agent, conf = self.clf.classify("Tell me about NCT12345678")
        self.assertEqual(agent, AgentType.CLINICAL_TRIALS_AGENT)

    def test_trials_confidence_in_bounds(self):
        _, conf = self.clf.classify("I am eligible for multiple clinical trials")
        self.assertGreaterEqual(conf, 0.55)
        self.assertLessEqual(conf, 1.0)


class TestIntentClassifierEthics(unittest.TestCase):
    """Tests that ethics-related phrases route to ETHICS_COMPLAINT_AGENT."""

    def setUp(self):
        self.clf = IntentClassifier()

    def test_hipaa_routes_to_ethics(self):
        agent, conf = self.clf.classify("I think there was a HIPAA violation at my clinic")
        self.assertEqual(agent, AgentType.ETHICS_COMPLAINT_AGENT)

    def test_privacy_violation_routes_to_ethics(self):
        agent, conf = self.clf.classify("My privacy was violated when they shared my records")
        self.assertEqual(agent, AgentType.ETHICS_COMPLAINT_AGENT)

    def test_billing_fraud_routes_to_ethics(self):
        agent, conf = self.clf.classify("I believe there is billing fraud on my account")
        self.assertEqual(agent, AgentType.ETHICS_COMPLAINT_AGENT)

    def test_negligence_routes_to_ethics(self):
        agent, conf = self.clf.classify("This is medical negligence — they ignored my symptoms")
        self.assertEqual(agent, AgentType.ETHICS_COMPLAINT_AGENT)

    def test_informed_consent_routes_to_ethics(self):
        agent, conf = self.clf.classify("I never gave informed consent for that procedure")
        self.assertEqual(agent, AgentType.ETHICS_COMPLAINT_AGENT)

    def test_discrimination_routes_to_ethics(self):
        agent, conf = self.clf.classify("I experienced discrimination when seeking care")
        self.assertEqual(agent, AgentType.ETHICS_COMPLAINT_AGENT)

    def test_ethics_complaint_routes_to_ethics(self):
        agent, conf = self.clf.classify("I want to file an ethics complaint against the hospital")
        self.assertEqual(agent, AgentType.ETHICS_COMPLAINT_AGENT)

    def test_ethics_confidence_in_bounds(self):
        _, conf = self.clf.classify("HIPAA violation and billing fraud both occurred")
        self.assertGreaterEqual(conf, 0.55)
        self.assertLessEqual(conf, 1.0)


class TestIntentClassifierLegal(unittest.TestCase):
    """Tests that legal-related phrases route to LEGAL_MOTIONS_AGENT."""

    def setUp(self):
        self.clf = IntentClassifier()

    def test_lawsuit_routes_to_legal(self):
        agent, conf = self.clf.classify("I want to file a lawsuit against the hospital")
        self.assertEqual(agent, AgentType.LEGAL_MOTIONS_AGENT)

    def test_attorney_routes_to_legal(self):
        agent, conf = self.clf.classify("I need to speak with an attorney about my case")
        self.assertEqual(agent, AgentType.LEGAL_MOTIONS_AGENT)

    def test_malpractice_routes_to_legal(self):
        agent, conf = self.clf.classify("Is this considered medical malpractice?")
        self.assertEqual(agent, AgentType.LEGAL_MOTIONS_AGENT)

    def test_motion_routes_to_legal(self):
        agent, conf = self.clf.classify("Can you draft a motion to compel release of my records?")
        self.assertEqual(agent, AgentType.LEGAL_MOTIONS_AGENT)

    def test_appeal_routes_to_legal(self):
        agent, conf = self.clf.classify("I need to file an insurance appeal")
        self.assertEqual(agent, AgentType.LEGAL_MOTIONS_AGENT)

    def test_class_action_routes_to_legal(self):
        agent, conf = self.clf.classify("Is there a class action lawsuit I can join?")
        self.assertEqual(agent, AgentType.LEGAL_MOTIONS_AGENT)

    def test_legal_confidence_in_bounds(self):
        _, conf = self.clf.classify("I need legal advice — malpractice attorney")
        self.assertGreaterEqual(conf, 0.55)
        self.assertLessEqual(conf, 1.0)


class TestIntentClassifierVision(unittest.TestCase):
    """Tests that image / file references route to VISION_AGENT at confidence 1.0."""

    def setUp(self):
        self.clf = IntentClassifier()

    def test_png_path_routes_to_vision(self):
        agent, conf = self.clf.classify("Here is my lab result: /home/user/labs.png")
        self.assertEqual(agent, AgentType.VISION_AGENT)
        self.assertEqual(conf, 1.0)

    def test_jpg_path_routes_to_vision(self):
        agent, conf = self.clf.classify("See attached file labs.jpg")
        self.assertEqual(agent, AgentType.VISION_AGENT)
        self.assertEqual(conf, 1.0)

    def test_jpeg_path_routes_to_vision(self):
        agent, conf = self.clf.classify("Uploading results.jpeg now")
        self.assertEqual(agent, AgentType.VISION_AGENT)
        self.assertEqual(conf, 1.0)

    def test_pdf_path_routes_to_vision(self):
        agent, conf = self.clf.classify("I have a report.pdf to share")
        self.assertEqual(agent, AgentType.VISION_AGENT)
        self.assertEqual(conf, 1.0)

    def test_image_keyword_routes_to_vision(self):
        agent, conf = self.clf.classify("Can you extract my labs from this image?")
        self.assertEqual(agent, AgentType.VISION_AGENT)
        self.assertEqual(conf, 1.0)

    def test_vision_confidence_always_1(self):
        for text in ("scan.tif", "photo.bmp", "upload image", "ocr"):
            _, conf = self.clf.classify(text)
            self.assertEqual(conf, 1.0, msg=f"Expected 1.0 for '{text}'")


class TestConfidenceBounds(unittest.TestCase):
    """Tests that normalised confidence always stays within [0.55, 1.0]."""

    def setUp(self):
        self.clf = IntentClassifier()

    def test_confidence_never_below_055_when_routing(self):
        """When a route IS chosen, confidence must be >= 0.55."""
        texts = [
            "I need to schedule an appointment",
            "clinical trial eligibility",
            "HIPAA violation complaint",
            "malpractice lawsuit attorney",
        ]
        for text in texts:
            agent, conf = self.clf.classify(text)
            if agent != AgentType.END:
                self.assertGreaterEqual(conf, 0.55, msg=f"Failed for '{text}'")

    def test_confidence_never_above_1(self):
        """Confidence must never exceed 1.0."""
        texts = [
            "schedule appointment calendar nadir fatigue chemo treatment date reschedule",
            "clinical trials eligibility enroll study phase 2 trial NCT12345678",
        ]
        for text in texts:
            _, conf = self.clf.classify(text)
            self.assertLessEqual(conf, 1.0, msg=f"Exceeded 1.0 for '{text}'")

    def test_empty_text_returns_end_with_zero_confidence(self):
        agent, conf = self.clf.classify("")
        self.assertEqual(agent, AgentType.END)
        self.assertEqual(conf, 0.0)

    def test_whitespace_only_returns_end(self):
        agent, conf = self.clf.classify("   ")
        self.assertEqual(agent, AgentType.END)
        self.assertEqual(conf, 0.0)

    def test_gibberish_returns_end_with_zero_confidence(self):
        agent, conf = self.clf.classify("xkcd qwerty zxcvbnm asdfghjkl")
        self.assertEqual(agent, AgentType.END)
        self.assertEqual(conf, 0.0)


class TestHITLGate(unittest.TestCase):
    """Tests that the HITL gate fires when confidence < 0.7."""

    def setUp(self):
        self.clf = IntentClassifier()

    def test_low_confidence_routes_to_end(self):
        """
        A message with exactly one weak signal should produce confidence <= 0.7
        and therefore route to END if normalised confidence < threshold.
        We can verify by checking the raw hit count normalises below threshold.
        """
        # One signal hit out of ~20 patterns → normalised ≈ 0.55 + 1/20 * 0.45 ≈ 0.57
        agent, conf = self.clf.classify("schedule")
        # With only one hit the classifier MAY still route to calendar, but
        # the normalised confidence = 0.55 + (1 / N) * 0.45.
        # N for calendar is ~21 patterns, so conf ≈ 0.571 < 0.7 → END
        if conf < 0.7:
            self.assertEqual(agent, AgentType.END)

    def test_above_threshold_does_not_route_to_end(self):
        """Multiple strong signals should produce confidence >= 0.7."""
        text = (
            "I need to schedule an appointment, reschedule my chemo session, "
            "and check my nadir window on the calendar"
        )
        agent, conf = self.clf.classify(text)
        self.assertGreaterEqual(conf, 0.7)
        self.assertNotEqual(agent, AgentType.END)

    def test_single_weak_signal_confidence_below_threshold(self):
        """Verify raw confidence computation for a single-signal text."""
        hits = self.clf.get_signal_hits("schedule", AgentType.CALENDAR_AGENT)
        n = len(self.clf._compiled[AgentType.CALENDAR_AGENT])
        normalised = 0.55 + (hits / n) * 0.45
        self.assertLess(normalised, 0.7)


class TestMasterBrainNode(unittest.TestCase):
    """Tests for master_brain_node state mutations and immutability contract."""

    def test_audit_log_is_appended(self):
        state = _make_state("I need to schedule an appointment with my oncologist next Tuesday")
        result = master_brain_node(state)
        self.assertIn("audit_log", result)
        self.assertEqual(len(result["audit_log"]), 1)

    def test_audit_log_entry_is_transparency_record(self):
        state = _make_state("Can you reschedule my chemo on the calendar?")
        result = master_brain_node(state)
        record = result["audit_log"][0]
        self.assertIsInstance(record, TransparencyRecord)

    def test_audit_log_agent_name_is_master_brain(self):
        state = _make_state("reschedule appointment calendar nadir")
        result = master_brain_node(state)
        record = result["audit_log"][0]
        self.assertEqual(record.agent_name, AgentType.MASTER_BRAIN.value)
        self.assertEqual(record.agent_name, "master_brain")

    def test_messages_list_has_ai_message(self):
        state = _make_state("I need to reschedule my appointment on the calendar")
        result = master_brain_node(state)
        self.assertIn("messages", result)
        self.assertEqual(len(result["messages"]), 1)
        msg = result["messages"][0]
        self.assertIsInstance(msg, _AIMessage)

    def test_next_agent_is_set(self):
        state = _make_state("Schedule my infusion date on the calendar for next week")
        result = master_brain_node(state)
        self.assertIn("next_agent", result)
        self.assertIsInstance(result["next_agent"], str)
        self.assertTrue(len(result["next_agent"]) > 0)

    def test_hitl_required_is_set(self):
        state = _make_state("I need to schedule my chemo appointment please")
        result = master_brain_node(state)
        self.assertIn("hitl_required", result)
        self.assertIsInstance(result["hitl_required"], bool)

    def test_calendar_events_not_mutated(self):
        original_events = [object(), object()]
        state = _make_state(
            "schedule appointment",
            calendar_events=list(original_events),
        )
        master_brain_node(state)
        # calendar_events must remain untouched in the state dict
        self.assertEqual(state["calendar_events"], original_events)

    def test_pending_suggestions_not_mutated(self):
        original_suggestions = [object()]
        state = _make_state(
            "reschedule chemo",
            pending_suggestions=list(original_suggestions),
        )
        master_brain_node(state)
        self.assertEqual(state["pending_suggestions"], original_suggestions)

    def test_patient_profile_not_in_result(self):
        state = _make_state("schedule appointment calendar nadir window fatigue chemo")
        result = master_brain_node(state)
        # patient_profile should NOT appear in the returned patch dict
        self.assertNotIn("patient_profile", result)

    def test_calendar_routing_sets_correct_next_agent(self):
        text = "schedule appointment calendar nadir fatigue chemo treatment reschedule"
        state = _make_state(text)
        result = master_brain_node(state)
        self.assertEqual(result["next_agent"], AgentType.CALENDAR_AGENT.value)

    def test_trials_routing_sets_correct_next_agent(self):
        text = "clinical trials eligibility enroll phase 2 cancer study NCT12345678"
        state = _make_state(text)
        result = master_brain_node(state)
        self.assertEqual(result["next_agent"], AgentType.CLINICAL_TRIALS_AGENT.value)

    def test_vision_routing_sets_correct_next_agent(self):
        state = _make_state("Please analyse this file: labs.pdf")
        result = master_brain_node(state)
        self.assertEqual(result["next_agent"], AgentType.VISION_AGENT.value)

    def test_hitl_required_true_when_low_confidence(self):
        state = _make_state("xkcd zxcvbnm qwerty")  # gibberish → no hits → END
        result = master_brain_node(state)
        self.assertTrue(result["hitl_required"])

    def test_hitl_required_false_for_high_confidence_calendar(self):
        text = (
            "schedule appointment calendar nadir fatigue chemo treatment date "
            "reschedule follow-up book infusion"
        )
        state = _make_state(text)
        result = master_brain_node(state)
        # With many hits confidence should be >= 0.7
        if result["next_agent"] == AgentType.CALENDAR_AGENT.value:
            self.assertFalse(result["hitl_required"])

    def test_reasoning_chain_cites_bengio(self):
        state = _make_state("schedule appointment calendar follow-up nadir chemo")
        result = master_brain_node(state)
        record = result["audit_log"][0]
        self.assertIn("Bengio", record.reasoning_chain)

    def test_reasoning_chain_cites_hantel(self):
        state = _make_state("schedule appointment calendar follow-up nadir chemo")
        result = master_brain_node(state)
        record = result["audit_log"][0]
        self.assertIn("Hantel", record.reasoning_chain)

    def test_empty_message_list_does_not_raise(self):
        state = _make_state("")
        state["messages"] = []
        result = master_brain_node(state)
        self.assertIn("audit_log", result)

    def test_result_keys_only_expected(self):
        state = _make_state("schedule appointment calendar nadir chemo treatment")
        result = master_brain_node(state)
        expected_keys = {"audit_log", "messages", "next_agent", "hitl_required"}
        self.assertEqual(set(result.keys()), expected_keys)


class TestRouteToAgent(unittest.TestCase):
    """Tests for the route_to_agent conditional edge function."""

    def test_calendar_agent_routed_correctly(self):
        state = _make_state(next_agent=AgentType.CALENDAR_AGENT.value, hitl_required=False)
        self.assertEqual(route_to_agent(state), AgentType.CALENDAR_AGENT.value)

    def test_clinical_trials_agent_routed_correctly(self):
        state = _make_state(next_agent=AgentType.CLINICAL_TRIALS_AGENT.value, hitl_required=False)
        self.assertEqual(route_to_agent(state), AgentType.CLINICAL_TRIALS_AGENT.value)

    def test_vision_agent_routed_correctly(self):
        state = _make_state(next_agent=AgentType.VISION_AGENT.value, hitl_required=False)
        self.assertEqual(route_to_agent(state), AgentType.VISION_AGENT.value)

    def test_ethics_agent_routed_correctly(self):
        state = _make_state(next_agent=AgentType.ETHICS_COMPLAINT_AGENT.value, hitl_required=False)
        self.assertEqual(route_to_agent(state), AgentType.ETHICS_COMPLAINT_AGENT.value)

    def test_legal_agent_routed_correctly(self):
        state = _make_state(next_agent=AgentType.LEGAL_MOTIONS_AGENT.value, hitl_required=False)
        self.assertEqual(route_to_agent(state), AgentType.LEGAL_MOTIONS_AGENT.value)

    def test_hitl_overrides_to_end_for_calendar(self):
        state = _make_state(next_agent=AgentType.CALENDAR_AGENT.value, hitl_required=True)
        self.assertEqual(route_to_agent(state), "end")

    def test_hitl_overrides_to_end_for_trials(self):
        state = _make_state(next_agent=AgentType.CLINICAL_TRIALS_AGENT.value, hitl_required=True)
        self.assertEqual(route_to_agent(state), "end")

    def test_hitl_overrides_to_end_for_ethics(self):
        state = _make_state(next_agent=AgentType.ETHICS_COMPLAINT_AGENT.value, hitl_required=True)
        self.assertEqual(route_to_agent(state), "end")

    def test_hitl_overrides_to_end_for_legal(self):
        state = _make_state(next_agent=AgentType.LEGAL_MOTIONS_AGENT.value, hitl_required=True)
        self.assertEqual(route_to_agent(state), "end")

    def test_unknown_agent_falls_back_to_end(self):
        state = _make_state(next_agent="completely_unknown_agent", hitl_required=False)
        self.assertEqual(route_to_agent(state), "end")

    def test_end_agent_falls_back_to_end(self):
        state = _make_state(next_agent=AgentType.END.value, hitl_required=False)
        self.assertEqual(route_to_agent(state), "end")

    def test_empty_next_agent_falls_back_to_end(self):
        state = _make_state(next_agent="", hitl_required=False)
        self.assertEqual(route_to_agent(state), "end")


class TestStubAgentNodes(unittest.TestCase):
    """Tests that each stub agent node emits the correct TransparencyRecord."""

    def _assert_node_output(self, fn, expected_agent_name: str, state=None):
        if state is None:
            state = _make_state()
        result = fn(state)
        self.assertIn("audit_log", result)
        self.assertIn("messages", result)
        self.assertEqual(len(result["audit_log"]), 1)
        self.assertEqual(len(result["messages"]), 1)
        record = result["audit_log"][0]
        self.assertIsInstance(record, TransparencyRecord)
        self.assertEqual(record.agent_name, expected_agent_name)
        return record

    def test_calendar_node_agent_name(self):
        self._assert_node_output(calendar_agent_node, AgentType.CALENDAR_AGENT.value)

    def test_calendar_node_citations_in_reasoning(self):
        record = self._assert_node_output(calendar_agent_node, AgentType.CALENDAR_AGENT.value)
        self.assertIn("Bengio", record.reasoning_chain)
        self.assertIn("Hantel", record.reasoning_chain)

    def test_trials_node_agent_name(self):
        self._assert_node_output(
            clinical_trials_agent_node, AgentType.CLINICAL_TRIALS_AGENT.value
        )

    def test_trials_node_citations_in_reasoning(self):
        record = self._assert_node_output(
            clinical_trials_agent_node, AgentType.CLINICAL_TRIALS_AGENT.value
        )
        self.assertIn("Bengio", record.reasoning_chain)

    def test_vision_node_agent_name(self):
        state = _make_state(image_path="/tmp/labs.png")
        self._assert_node_output(vision_agent_node, AgentType.VISION_AGENT.value, state=state)

    def test_vision_node_includes_image_path_in_reasoning(self):
        state = _make_state(image_path="/tmp/labs.png")
        result = vision_agent_node(state)
        record = result["audit_log"][0]
        self.assertIn("/tmp/labs.png", record.reasoning_chain)

    def test_ethics_node_agent_name(self):
        self._assert_node_output(
            ethics_complaint_agent_node, AgentType.ETHICS_COMPLAINT_AGENT.value
        )

    def test_ethics_node_citations_in_reasoning(self):
        record = self._assert_node_output(
            ethics_complaint_agent_node, AgentType.ETHICS_COMPLAINT_AGENT.value
        )
        self.assertIn("Hantel", record.reasoning_chain)

    def test_legal_node_agent_name(self):
        self._assert_node_output(legal_motions_agent_node, AgentType.LEGAL_MOTIONS_AGENT.value)

    def test_legal_node_attorney_required_in_reasoning(self):
        result = legal_motions_agent_node(_make_state())
        record = result["audit_log"][0]
        self.assertIn("attorney", record.reasoning_chain.lower())

    def test_all_nodes_return_ai_message(self):
        nodes = [
            calendar_agent_node,
            clinical_trials_agent_node,
            ethics_complaint_agent_node,
            legal_motions_agent_node,
        ]
        for fn in nodes:
            result = fn(_make_state())
            msg = result["messages"][0]
            self.assertIsInstance(msg, _AIMessage, msg=f"Failed for {fn.__name__}")

    def test_stub_nodes_do_not_mutate_calendar_events(self):
        original = [object()]
        nodes = [
            calendar_agent_node,
            clinical_trials_agent_node,
            vision_agent_node,
            ethics_complaint_agent_node,
            legal_motions_agent_node,
        ]
        for fn in nodes:
            state = _make_state(calendar_events=list(original))
            fn(state)
            self.assertEqual(state["calendar_events"], original, msg=f"Mutated by {fn.__name__}")


class TestCitationsConstant(unittest.TestCase):
    """Tests that the module-level citation constant contains required references."""

    def test_routing_citations_contains_bengio(self):
        self.assertIn("Bengio", _ROUTING_CITATIONS)

    def test_routing_citations_contains_hantel(self):
        self.assertIn("Hantel", _ROUTING_CITATIONS)

    def test_routing_citations_contains_year_2024(self):
        self.assertIn("2024", _ROUTING_CITATIONS)

    def test_routing_citations_contains_year_2019(self):
        self.assertIn("2019", _ROUTING_CITATIONS)


class TestBuildAdvocateGraph(unittest.TestCase):
    """Tests that build_advocate_graph compiles without error."""

    def test_build_returns_compiled_graph(self):
        graph = build_advocate_graph()
        self.assertIsNotNone(graph)

    def test_compiled_graph_has_master_brain_node(self):
        graph = build_advocate_graph()
        # With our stub StateGraph the compiled object IS the graph
        self.assertIn("master_brain", graph._nodes)

    def test_compiled_graph_has_all_agent_nodes(self):
        graph = build_advocate_graph()
        expected = {
            "master_brain",
            "calendar_agent",
            "clinical_trials_agent",
            "vision_agent",
            "ethics_complaint_agent",
            "legal_motions_agent",
        }
        self.assertEqual(set(graph._nodes.keys()), expected)

    def test_compiled_graph_entry_point_is_master_brain(self):
        graph = build_advocate_graph()
        self.assertEqual(graph._entry_point, "master_brain")

    def test_compiled_graph_has_conditional_edges(self):
        graph = build_advocate_graph()
        self.assertTrue(len(graph._conditional_edges) > 0)

    def test_compiled_graph_conditional_edge_from_master_brain(self):
        graph = build_advocate_graph()
        src, fn, mapping = graph._conditional_edges[0]
        self.assertEqual(src, "master_brain")
        self.assertIs(fn, route_to_agent)

    def test_compiled_graph_edge_mapping_covers_all_agents(self):
        graph = build_advocate_graph()
        _, _, mapping = graph._conditional_edges[0]
        expected_keys = {
            "calendar_agent",
            "clinical_trials_agent",
            "vision_agent",
            "ethics_complaint_agent",
            "legal_motions_agent",
            "end",
        }
        self.assertEqual(set(mapping.keys()), expected_keys)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
