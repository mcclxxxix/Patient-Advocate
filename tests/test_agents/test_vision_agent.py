"""
Tests for the Vision/OCR Agent.

All external packages are stubbed so no real OCR or ML libraries are required.
The minimal LabValues dataclass is defined inline to match the production
Pydantic model's interface without depending on Pydantic at test time.

Coverage:
  - LabValueParser.parse: clean and noisy text for every lab value field
  - LabValueParser.parse: PLT Count (space-separated label)
  - LabValueParser.parse: partial extraction and empty/garbage input
  - LabValueParser.to_lab_values: source='ocr', default date is today
  - VisionAgent.extract_from_text: return types, confidence, agent_name
  - vision_agent_node: no image_path, with image_path, hitl_required=True
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest


# =============================================================================
# STUB EXTERNAL PACKAGES
# =============================================================================

def _stub_module(name: str, **attrs) -> types.ModuleType:
    """Register a stub module in sys.modules and return it."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# --- pytesseract ---
pytesseract_stub = _stub_module("pytesseract")
pytesseract_stub.image_to_string = MagicMock(return_value="")

# --- PIL ---
pil_stub = _stub_module("PIL")
pil_image_stub = _stub_module("PIL.Image")
pil_filter_stub = _stub_module("PIL.ImageFilter")

_MockImage = MagicMock()
_MockImage.open = MagicMock(return_value=MagicMock())
pil_image_stub.Image = _MockImage
pil_image_stub.open = _MockImage.open

_MockFilter = MagicMock()
pil_filter_stub.SHARPEN = _MockFilter
pil_filter_stub.ImageFilter = _MockFilter

pil_stub.Image = pil_image_stub
pil_stub.ImageFilter = pil_filter_stub

# --- pydantic ---
pydantic_stub = _stub_module("pydantic")

class _BaseModel:  # minimal pydantic.BaseModel shim
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    def model_copy(self, update=None):
        import copy
        obj = copy.copy(self)
        if update:
            for k, v in update.items():
                setattr(obj, k, v)
        return obj

class _ConfigDict(dict):
    pass

pydantic_stub.BaseModel = _BaseModel
pydantic_stub.ConfigDict = _ConfigDict

def _Field(default=None, **kwargs):
    return default

pydantic_stub.Field = _Field

# --- pydantic_settings ---
_stub_module("pydantic_settings")

# --- langchain_core ---
lc_core = _stub_module("langchain_core")
lc_messages = _stub_module("langchain_core.messages")

class _BaseMessage:
    def __init__(self, content: str = "") -> None:
        self.content = content

class _AIMessage(_BaseMessage):
    pass

class _HumanMessage(_BaseMessage):
    pass

lc_messages.BaseMessage = _BaseMessage
lc_messages.AIMessage = _AIMessage
lc_messages.HumanMessage = _HumanMessage
lc_core.messages = lc_messages

# --- langgraph ---
_stub_module("langgraph")
_stub_module("langgraph.graph")

# --- patient_advocate.core.models (inline dataclass stubs) ---

from enum import Enum

class AgentType(str, Enum):
    VISION_AGENT = "vision_agent"

class ConflictType(str, Enum):
    NADIR_WINDOW = "nadir_window"
    FATIGUE_THRESHOLD = "fatigue_threshold"
    APPOINTMENT_OVERLAP = "appointment_overlap"
    RECOVERY_PERIOD = "recovery_period"

class HITLDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    DEFERRED = "deferred"


@dataclass(frozen=True)
class LabValues:
    """Minimal frozen dataclass mirroring the production Pydantic model."""
    anc: Optional[float]
    wbc: Optional[float]
    hemoglobin: Optional[float]
    platelets: Optional[int]
    fatigue_score: Optional[int]
    collection_date: date
    source: str


@dataclass(frozen=True)
class TransparencyRecord:
    """Minimal frozen dataclass mirroring the production Pydantic model."""
    event_id: UUID
    suggestion_id: Optional[UUID]
    agent_name: str
    conflict_type: Optional[ConflictType]
    confidence: Optional[float]
    scholarly_basis: Optional[str]
    patient_decision: Optional[HITLDecision]
    reasoning_chain: str
    timestamp: datetime


# Register stubs in sys.modules before importing the module under test
_models_stub = _stub_module("patient_advocate.core.models")
_models_stub.AgentType = AgentType
_models_stub.ConflictType = ConflictType
_models_stub.HITLDecision = HITLDecision
_models_stub.LabValues = LabValues
_models_stub.TransparencyRecord = TransparencyRecord

_core_stub = _stub_module("patient_advocate.core")
_core_stub.models = _models_stub

class _OCRExtractionError(Exception):
    def __init__(self, message: str = "", agent_name: str = "unknown") -> None:
        self.agent_name = agent_name
        super().__init__(message)

class _PatientAdvocateError(Exception):
    def __init__(self, message: str = "", agent_name: str = "unknown") -> None:
        self.agent_name = agent_name
        super().__init__(message)

_exceptions_stub = _stub_module("patient_advocate.core.exceptions")
_exceptions_stub.OCRExtractionError = _OCRExtractionError
_exceptions_stub.PatientAdvocateError = _PatientAdvocateError

_pa_stub = _stub_module("patient_advocate")
_pa_stub.core = _core_stub

# --- langchain_core.messages shim must be in sys.modules before import ---
sys.modules["langchain_core.messages"] = lc_messages

# =============================================================================
# NOW import the module under test
# =============================================================================

from patient_advocate.agents.vision_agent import (  # noqa: E402
    LabValueParser,
    VisionAgent,
    vision_agent_node,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture()
def parser() -> LabValueParser:
    return LabValueParser()


@pytest.fixture()
def agent() -> VisionAgent:
    return VisionAgent()


# =============================================================================
# LabValueParser.parse — ANC
# =============================================================================

class TestParseANC:
    def test_anc_clean(self, parser):
        result = parser.parse("ANC: 1500")
        assert result["anc"] == 1500.0

    def test_anc_decimal(self, parser):
        result = parser.parse("ANC: 750.5")
        assert result["anc"] == 750.5

    def test_anc_dotted_abbreviation(self, parser):
        """OCR noise: A.N.C."""
        result = parser.parse("A.N.C: 320")
        assert result["anc"] == 320.0

    def test_anc_absolute_neutrophil(self, parser):
        """Full label form."""
        result = parser.parse("Absolute Neutrophil Count: 2100")
        assert result["anc"] == 2100.0

    def test_anc_absolute_neutrophil_no_count(self, parser):
        result = parser.parse("Absolute Neutrophil 980")
        assert result["anc"] == 980.0

    def test_anc_equals_separator(self, parser):
        result = parser.parse("ANC = 450")
        assert result["anc"] == 450.0

    def test_anc_case_insensitive(self, parser):
        result = parser.parse("anc: 1200")
        assert result["anc"] == 1200.0

    def test_anc_out_of_range_clipped_to_none(self, parser):
        """Values > 50 000 are implausible and should be discarded."""
        result = parser.parse("ANC: 999999")
        assert result["anc"] is None

    def test_anc_missing_returns_none(self, parser):
        result = parser.parse("WBC: 5.0 Hemoglobin: 12.0")
        assert result["anc"] is None

    def test_anc_spaced_abbreviation(self, parser):
        """OCR noise: A N C with spaces."""
        result = parser.parse("A N C 600")
        assert result["anc"] == 600.0


# =============================================================================
# LabValueParser.parse — WBC
# =============================================================================

class TestParseWBC:
    def test_wbc_clean(self, parser):
        result = parser.parse("WBC: 5.2")
        assert result["wbc"] == 5.2

    def test_wbc_integer(self, parser):
        result = parser.parse("WBC: 4")
        assert result["wbc"] == 4.0

    def test_wbc_dotted(self, parser):
        """OCR noise: W.B.C"""
        result = parser.parse("W.B.C: 3.8")
        assert result["wbc"] == 3.8

    def test_wbc_white_blood_cell_count(self, parser):
        result = parser.parse("White Blood Cell Count: 7.1")
        assert result["wbc"] == 7.1

    def test_wbc_white_blood_only(self, parser):
        result = parser.parse("White Blood: 6.0")
        assert result["wbc"] == 6.0

    def test_wbc_out_of_range(self, parser):
        result = parser.parse("WBC: 250")
        assert result["wbc"] is None

    def test_wbc_missing_returns_none(self, parser):
        result = parser.parse("ANC: 1500")
        assert result["wbc"] is None


# =============================================================================
# LabValueParser.parse — Hemoglobin
# =============================================================================

class TestParseHemoglobin:
    def test_hgb_label(self, parser):
        result = parser.parse("HGB: 11.5")
        assert result["hemoglobin"] == 11.5

    def test_hgb_mixed_case(self, parser):
        result = parser.parse("Hgb: 13.2")
        assert result["hemoglobin"] == 13.2

    def test_hb_label(self, parser):
        result = parser.parse("HB: 9.8")
        assert result["hemoglobin"] == 9.8

    def test_hemoglobin_full_label(self, parser):
        result = parser.parse("Hemoglobin: 10.4")
        assert result["hemoglobin"] == 10.4

    def test_haemoglobin_british_spelling(self, parser):
        result = parser.parse("Haemoglobin: 12.1")
        assert result["hemoglobin"] == 12.1

    def test_hemoglobin_out_of_range(self, parser):
        result = parser.parse("Hemoglobin: 99.0")
        assert result["hemoglobin"] is None

    def test_hemoglobin_missing_returns_none(self, parser):
        result = parser.parse("ANC: 1500 WBC: 5.0")
        assert result["hemoglobin"] is None


# =============================================================================
# LabValueParser.parse — Platelets
# =============================================================================

class TestParsePlatelets:
    def test_plt_label(self, parser):
        result = parser.parse("PLT: 180")
        assert result["platelets"] == 180

    def test_plt_count_with_space(self, parser):
        """Critical: space between PLT and Count must be handled."""
        result = parser.parse("PLT Count: 225")
        assert result["platelets"] == 225

    def test_plt_count_uppercase(self, parser):
        result = parser.parse("PLT COUNT: 310")
        assert result["platelets"] == 310

    def test_plts_plural(self, parser):
        result = parser.parse("PLTS: 95")
        assert result["platelets"] == 95

    def test_platelet_singular(self, parser):
        result = parser.parse("Platelet: 140")
        assert result["platelets"] == 140

    def test_platelets_plural(self, parser):
        result = parser.parse("Platelets: 275")
        assert result["platelets"] == 275

    def test_platelets_stored_as_int(self, parser):
        """Platelets must be stored as int, not float."""
        result = parser.parse("PLT: 180.0")
        assert result["platelets"] == 180
        assert isinstance(result["platelets"], int)

    def test_platelets_out_of_range(self, parser):
        result = parser.parse("PLT: 9999")
        assert result["platelets"] is None

    def test_platelets_missing_returns_none(self, parser):
        result = parser.parse("ANC: 1500")
        assert result["platelets"] is None


# =============================================================================
# LabValueParser.parse — Fatigue Score
# =============================================================================

class TestParseFatigue:
    def test_fatigue_score_label(self, parser):
        result = parser.parse("Fatigue Score: 6")
        assert result["fatigue_score"] == 6

    def test_fatigue_label_only(self, parser):
        result = parser.parse("Fatigue: 4")
        assert result["fatigue_score"] == 4

    def test_pain_score_label(self, parser):
        result = parser.parse("Pain Score: 8")
        assert result["fatigue_score"] == 8

    def test_pain_label_only(self, parser):
        result = parser.parse("Pain: 3")
        assert result["fatigue_score"] == 3

    def test_fatigue_boundary_zero(self, parser):
        result = parser.parse("Fatigue: 0")
        assert result["fatigue_score"] == 0

    def test_fatigue_boundary_ten(self, parser):
        result = parser.parse("Fatigue: 10")
        assert result["fatigue_score"] == 10

    def test_fatigue_out_of_range_above_ten(self, parser):
        result = parser.parse("Fatigue: 11")
        assert result["fatigue_score"] is None

    def test_fatigue_missing_returns_none(self, parser):
        result = parser.parse("ANC: 1500 WBC: 5.0")
        assert result["fatigue_score"] is None


# =============================================================================
# LabValueParser.parse — Partial extraction and edge cases
# =============================================================================

class TestParsePartialAndEdgeCases:
    def test_empty_string_all_none(self, parser):
        result = parser.parse("")
        assert result == {
            "anc": None,
            "wbc": None,
            "hemoglobin": None,
            "platelets": None,
            "fatigue_score": None,
        }

    def test_whitespace_only_all_none(self, parser):
        result = parser.parse("   \n\t  ")
        assert result == {
            "anc": None,
            "wbc": None,
            "hemoglobin": None,
            "platelets": None,
            "fatigue_score": None,
        }

    def test_garbage_text_all_none(self, parser):
        result = parser.parse("@#$%^& foo bar baz qux 9999")
        assert result == {
            "anc": None,
            "wbc": None,
            "hemoglobin": None,
            "platelets": None,
            "fatigue_score": None,
        }

    def test_only_anc_present(self, parser):
        result = parser.parse("ANC: 1200")
        assert result["anc"] == 1200.0
        assert result["wbc"] is None
        assert result["hemoglobin"] is None
        assert result["platelets"] is None
        assert result["fatigue_score"] is None

    def test_anc_and_wbc_only(self, parser):
        result = parser.parse("ANC: 800 WBC: 3.5")
        assert result["anc"] == 800.0
        assert result["wbc"] == 3.5
        assert result["hemoglobin"] is None
        assert result["platelets"] is None

    def test_multiline_lab_report(self, parser):
        text = (
            "Patient Lab Results\n"
            "Date: 2024-03-15\n"
            "ANC: 1450\n"
            "WBC: 4.8\n"
            "Hemoglobin: 11.2\n"
            "PLT Count: 195\n"
            "Fatigue Score: 5\n"
        )
        result = parser.parse(text)
        assert result["anc"] == 1450.0
        assert result["wbc"] == 4.8
        assert result["hemoglobin"] == 11.2
        assert result["platelets"] == 195
        assert result["fatigue_score"] == 5

    def test_all_fields_present(self, parser):
        text = "ANC 1800 WBC 6.2 Hemoglobin 12.5 PLT 220 Fatigue 3"
        result = parser.parse(text)
        assert result["anc"] == 1800.0
        assert result["wbc"] == 6.2
        assert result["hemoglobin"] == 12.5
        assert result["platelets"] == 220
        assert result["fatigue_score"] == 3

    def test_returns_five_keys_always(self, parser):
        """parse() must always return all five keys regardless of content."""
        result = parser.parse("nothing here")
        assert set(result.keys()) == {"anc", "wbc", "hemoglobin", "platelets", "fatigue_score"}


# =============================================================================
# LabValueParser.to_lab_values
# =============================================================================

class TestToLabValues:
    def test_source_is_ocr(self, parser):
        parsed = {"anc": 1500.0, "wbc": 5.0, "hemoglobin": 12.0, "platelets": 200, "fatigue_score": 4}
        lv = parser.to_lab_values(parsed)
        assert lv.source == "ocr"

    def test_collection_date_defaults_to_today(self, parser):
        parsed = {"anc": None, "wbc": None, "hemoglobin": None, "platelets": None, "fatigue_score": None}
        lv = parser.to_lab_values(parsed)
        assert lv.collection_date == date.today()

    def test_explicit_collection_date(self, parser):
        parsed = {"anc": None, "wbc": None, "hemoglobin": None, "platelets": None, "fatigue_score": None}
        explicit = date(2024, 3, 15)
        lv = parser.to_lab_values(parsed, collection_date=explicit)
        assert lv.collection_date == explicit

    def test_values_propagated_correctly(self, parser):
        parsed = {"anc": 950.0, "wbc": 3.2, "hemoglobin": 10.1, "platelets": 88, "fatigue_score": 7}
        lv = parser.to_lab_values(parsed)
        assert lv.anc == 950.0
        assert lv.wbc == 3.2
        assert lv.hemoglobin == 10.1
        assert lv.platelets == 88
        assert lv.fatigue_score == 7

    def test_none_values_propagated(self, parser):
        parsed = {"anc": 1200.0, "wbc": None, "hemoglobin": None, "platelets": None, "fatigue_score": None}
        lv = parser.to_lab_values(parsed)
        assert lv.anc == 1200.0
        assert lv.wbc is None


# =============================================================================
# VisionAgent.extract_from_text
# =============================================================================

class TestExtractFromText:
    def test_returns_tuple_of_lab_values_and_transparency_record(self, agent):
        lv, rec = agent.extract_from_text("ANC: 1500 WBC: 5.0")
        assert isinstance(lv, LabValues)
        assert isinstance(rec, TransparencyRecord)

    def test_lab_values_source_is_ocr(self, agent):
        lv, _ = agent.extract_from_text("ANC: 1200")
        assert lv.source == "ocr"

    def test_transparency_record_agent_name_is_vision_agent(self, agent):
        _, rec = agent.extract_from_text("ANC: 1500")
        assert rec.agent_name == "vision_agent"

    def test_confidence_all_five_fields(self, agent):
        text = "ANC 1500 WBC 5.0 Hemoglobin 12.0 PLT 200 Fatigue 4"
        _, rec = agent.extract_from_text(text)
        assert rec.confidence == pytest.approx(1.0, abs=1e-4)

    def test_confidence_zero_fields(self, agent):
        _, rec = agent.extract_from_text("nothing here")
        assert rec.confidence == pytest.approx(0.0, abs=1e-4)

    def test_confidence_one_field(self, agent):
        _, rec = agent.extract_from_text("ANC: 1500")
        assert rec.confidence == pytest.approx(0.2, abs=1e-4)

    def test_confidence_three_of_five(self, agent):
        _, rec = agent.extract_from_text("ANC: 800 WBC: 3.5 Hemoglobin: 9.0")
        assert rec.confidence == pytest.approx(0.6, abs=1e-4)

    def test_confidence_two_of_five(self, agent):
        _, rec = agent.extract_from_text("PLT Count: 150 Fatigue: 6")
        assert rec.confidence == pytest.approx(0.4, abs=1e-4)

    def test_transparency_record_has_event_id(self, agent):
        _, rec = agent.extract_from_text("ANC: 1500")
        assert isinstance(rec.event_id, UUID)

    def test_transparency_record_has_scholarly_basis(self, agent):
        _, rec = agent.extract_from_text("ANC: 1500")
        assert rec.scholarly_basis is not None
        assert len(rec.scholarly_basis) > 0

    def test_transparency_record_has_timestamp(self, agent):
        _, rec = agent.extract_from_text("ANC: 1500")
        assert isinstance(rec.timestamp, datetime)

    def test_transparency_record_no_conflict_type(self, agent):
        _, rec = agent.extract_from_text("ANC: 1500")
        assert rec.conflict_type is None

    def test_transparency_record_no_patient_decision(self, agent):
        _, rec = agent.extract_from_text("ANC: 1500")
        assert rec.patient_decision is None

    def test_explicit_collection_date_propagated(self, agent):
        explicit = date(2024, 6, 1)
        lv, _ = agent.extract_from_text("ANC: 1500", collection_date=explicit)
        assert lv.collection_date == explicit

    def test_collection_date_defaults_to_today_when_none(self, agent):
        lv, _ = agent.extract_from_text("ANC: 1500")
        assert lv.collection_date == date.today()

    def test_reasoning_chain_is_non_empty_string(self, agent):
        _, rec = agent.extract_from_text("ANC: 1200 WBC: 4.0")
        assert isinstance(rec.reasoning_chain, str)
        assert len(rec.reasoning_chain) > 10

    def test_reasoning_chain_mentions_confidence(self, agent):
        _, rec = agent.extract_from_text("ANC: 1200")
        assert "confidence" in rec.reasoning_chain.lower()

    def test_empty_text_all_lab_values_none(self, agent):
        lv, rec = agent.extract_from_text("")
        assert lv.anc is None
        assert lv.wbc is None
        assert lv.hemoglobin is None
        assert lv.platelets is None
        assert lv.fatigue_score is None
        assert rec.confidence == pytest.approx(0.0, abs=1e-4)


# =============================================================================
# vision_agent_node
# =============================================================================

class TestVisionAgentNode:
    def test_no_image_path_returns_message(self):
        state = {"image_path": None, "patient_profile": None}
        result = vision_agent_node(state)
        assert "messages" in result
        assert len(result["messages"]) == 1

    def test_no_image_path_message_explains_what_is_needed(self):
        state = {"image_path": None, "patient_profile": None}
        result = vision_agent_node(state)
        content = result["messages"][0].content.lower()
        # Message should tell patient what to provide
        assert "image" in content or "upload" in content or "lab" in content

    def test_no_image_path_hitl_required_true(self):
        state = {"image_path": None, "patient_profile": None}
        result = vision_agent_node(state)
        assert result["hitl_required"] is True

    def test_no_image_path_audit_log_empty(self):
        state = {"image_path": None, "patient_profile": None}
        result = vision_agent_node(state)
        assert result["audit_log"] == []

    def test_image_path_calls_extract_from_image(self):
        """When image_path is set, extract_from_image should be called."""
        fake_lv = LabValues(
            anc=1500.0, wbc=5.0, hemoglobin=12.0, platelets=200,
            fatigue_score=4, collection_date=date.today(), source="ocr"
        )
        fake_rec = TransparencyRecord(
            event_id=uuid4(), suggestion_id=None,
            agent_name="vision_agent", conflict_type=None,
            confidence=0.8, scholarly_basis="test", patient_decision=None,
            reasoning_chain="test chain", timestamp=datetime.utcnow()
        )

        with patch(
            "patient_advocate.agents.vision_agent.VisionAgent.extract_from_image",
            return_value=(fake_lv, fake_rec),
        ):
            state = {"image_path": "/tmp/labs.png", "patient_profile": None}
            result = vision_agent_node(state)

        assert len(result["audit_log"]) == 1
        assert result["audit_log"][0] is fake_rec

    def test_image_path_hitl_required_always_true(self):
        """HITL must always be True regardless of confidence."""
        fake_lv = LabValues(
            anc=1500.0, wbc=5.0, hemoglobin=12.0, platelets=200,
            fatigue_score=4, collection_date=date.today(), source="ocr"
        )
        fake_rec = TransparencyRecord(
            event_id=uuid4(), suggestion_id=None,
            agent_name="vision_agent", conflict_type=None,
            confidence=1.0,  # maximum confidence still requires HITL
            scholarly_basis="test", patient_decision=None,
            reasoning_chain="test chain", timestamp=datetime.utcnow()
        )

        with patch(
            "patient_advocate.agents.vision_agent.VisionAgent.extract_from_image",
            return_value=(fake_lv, fake_rec),
        ):
            state = {"image_path": "/tmp/labs.png", "patient_profile": None}
            result = vision_agent_node(state)

        assert result["hitl_required"] is True

    def test_image_path_returns_message(self):
        """A patient-facing message is always returned."""
        fake_lv = LabValues(
            anc=1500.0, wbc=None, hemoglobin=None, platelets=None,
            fatigue_score=None, collection_date=date.today(), source="ocr"
        )
        fake_rec = TransparencyRecord(
            event_id=uuid4(), suggestion_id=None,
            agent_name="vision_agent", conflict_type=None,
            confidence=0.2, scholarly_basis="test", patient_decision=None,
            reasoning_chain="chain", timestamp=datetime.utcnow()
        )

        with patch(
            "patient_advocate.agents.vision_agent.VisionAgent.extract_from_image",
            return_value=(fake_lv, fake_rec),
        ):
            state = {"image_path": "/tmp/labs.png", "patient_profile": None}
            result = vision_agent_node(state)

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0].content, str)

    def test_image_path_missing_file_returns_error_message(self):
        """OCRExtractionError → error message returned, not a raised exception."""
        # Use the same OCRExtractionError class that vision_agent imported.
        from patient_advocate.agents import vision_agent as _va_mod

        with patch(
            "patient_advocate.agents.vision_agent.VisionAgent.extract_from_image",
            side_effect=_va_mod.OCRExtractionError("File not found: /tmp/bad.png"),
        ):
            state = {"image_path": "/tmp/bad.png", "patient_profile": None}
            result = vision_agent_node(state)

        assert result["hitl_required"] is True
        assert result["audit_log"] == []
        assert len(result["messages"]) == 1

    def test_missing_image_path_key_treated_as_no_image(self):
        """State dict without 'image_path' key should behave like None."""
        state = {"patient_profile": None}
        result = vision_agent_node(state)
        assert result["hitl_required"] is True
        assert result["audit_log"] == []
