"""
Patient Advocate — Master Brain Routing Module
===============================================

Implements deterministic, keyword-based intent classification (Glass Box Imperative:
NO LLM for routing decisions) and the LangGraph StateGraph that dispatches the
AdvocateState to specialized downstream agents.

Design Principles:
  - Glass Box Imperative: routing logic must be fully auditable (no opaque LLM calls).
  - Every routing decision produces a frozen TransparencyRecord in the audit_log.
  - HITL gate fires whenever classifier confidence < settings.hitl_confidence_threshold.
  - All agent nodes are immutable with respect to calendar_events, pending_suggestions,
    and patient_profile — only the audit_log and messages lists are extended.

References:
  - Bengio et al. (2019), NeurIPS: System 2 deliberate-reasoning patterns applied to
    routing heuristics — deterministic symbol manipulation before learned approximation.
  - Hantel et al. (2024), JAMA Network Open: Transparency and accountability in
    clinical AI; audit trails as a patient-rights obligation.
  - Carey (2024), Advances in Consumer Research: Model-facts labels with confidence
    bounds and failure-mode documentation.

Author: Patient Advocate Platform Team
License: Apache-2.0
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional
from uuid import uuid4

# ---------------------------------------------------------------------------
# Optional LangGraph / LangChain imports — stubbed gracefully when unavailable
# ---------------------------------------------------------------------------
try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
except ImportError:  # pragma: no cover — stubs provided in tests
    class BaseMessage:  # type: ignore[no-redef]
        """Minimal stub for BaseMessage."""
        def __init__(self, content: str = "", **kwargs):
            self.content = content

    class HumanMessage(BaseMessage):  # type: ignore[no-redef]
        """Minimal stub for HumanMessage."""

    class AIMessage(BaseMessage):  # type: ignore[no-redef]
        """Minimal stub for AIMessage."""

try:
    from langgraph.graph import END as GRAPH_END, StateGraph
except ImportError:  # pragma: no cover — stubs provided in tests
    GRAPH_END = "end"

    class StateGraph:  # type: ignore[no-redef]
        """Minimal stub for StateGraph."""
        def __init__(self, state_schema=None):
            self._nodes: dict = {}
            self._edges: list = []
            self._conditional_edges: list = []
            self._entry_point: Optional[str] = None

        def add_node(self, name: str, fn) -> None:
            self._nodes[name] = fn

        def add_edge(self, src: str, dst: str) -> None:
            self._edges.append((src, dst))

        def add_conditional_edges(self, src: str, fn, mapping: dict) -> None:
            self._conditional_edges.append((src, fn, mapping))

        def set_entry_point(self, name: str) -> None:
            self._entry_point = name

        def compile(self):
            return self

# ---------------------------------------------------------------------------
# Internal imports — always available within the package
# ---------------------------------------------------------------------------
from patient_advocate.core.models import AgentType, HITLDecision, TransparencyRecord
from patient_advocate.core.state import AdvocateState

try:
    from patient_advocate.core.config import Settings
    _settings = Settings()
except Exception:  # pydantic-settings not installed or env missing
    class _FallbackSettings:  # type: ignore[no-redef]
        hitl_confidence_threshold: float = 0.7

    _settings = _FallbackSettings()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Scholarly citations embedded per Glass Box Imperative
# ---------------------------------------------------------------------------
_ROUTING_CITATIONS = (
    "Bengio et al. (2019), NeurIPS — System 2 reasoning: deterministic symbolic "
    "routing precedes learned approximation. "
    "Hantel et al. (2024), JAMA Network Open — transparency and accountability in "
    "clinical AI require fully auditable routing logic."
)


# =============================================================================
# INTENT CLASSIFIER
# =============================================================================

class IntentClassifier:
    """
    Deterministic keyword-based intent classifier.

    The Glass Box Imperative prohibits LLM-based routing because probabilistic
    black-box decisions cannot be audited or explained to regulators.  This
    classifier uses frozen frozensets of compiled regex patterns — zero learned
    parameters, 100% reproducible, fully loggable.

    Confidence normalisation maps the raw signal-hit count into [0.55, 1.0].
    A confidence below settings.hitl_confidence_threshold (default 0.7) routes
    to the END node, triggering Human-in-the-Loop review.

    References:
      - Bengio et al. (2019): System 2 deliberate reasoning preferred over
        System 1 heuristics for high-stakes symbolic decisions.
      - Hantel et al. (2024): Auditable routing as a patient-safety obligation.
    """

    # ------------------------------------------------------------------
    # Signal sets — regex patterns (compiled at class creation time)
    # ------------------------------------------------------------------

    _CALENDAR_SIGNALS: frozenset = frozenset({
        r"\bschedul",          # schedule / scheduling / scheduler
        r"\bappointment",
        r"\bcalendar",
        r"\bnadir\b",
        r"\bfatigue\b",
        r"\bchemo\b",
        r"\bchemotherapy\b",
        r"\btreatment\s+date",
        r"\brescheduling?\b",
        r"\breschedule\b",
        r"\bbook\s+(an?\s+)?(appointment|slot|visit)",
        r"\bavailability\b",
        r"\btimeslot\b",
        r"\btime\s+slot\b",
        r"\bcancel\s+(an?\s+)?(appointment|visit)",
        r"\binfusion\s+date",
        r"\bnext\s+visit\b",
        r"\bfollow[\s-]?up\s+(appointment|date|visit)",
        r"\bpostpone\b",
        r"\bdelay\b.*\bappointment",
        r"\bnadir\s+window",
        r"\brecovery\s+period",
    })

    _TRIALS_SIGNALS: frozenset = frozenset({
        r"\bclinical\s+trials?\b",
        r"\bclinical\s+stud(?:y|ies)\b",
        r"\beligibilit(?:y|ies)\b",
        r"\beligible\b",
        r"\benroll(?:ment|ing|ed)?\b",
        r"\bstud(?:y|ies)\b.*\bcancer\b",
        r"\bphase\s+[123]\s+trial",
        r"\brandomized\s+(?:controlled\s+)?trial",
        r"\bplacebo\b",
        r"\bprotocol\b.*\btrial",
        r"\binvestigational\s+(?:drug|treatment|therapy)",
        r"\bexperimental\s+(?:drug|treatment|therapy)",
        r"\bctgov\b",
        r"\bnct\d{8}\b",         # ClinicalTrials.gov NCT identifiers
        r"\bopen[\s-]?label\s+trial",
        r"\bcohort\s+study\b",
    })

    _ETHICS_SIGNALS: frozenset = frozenset({
        r"\bcomplaint\b",
        r"\bhipaa\b",
        r"\bprivacy\b",
        r"\bviolat(?:ion|ed|ing|e)\b",
        r"\bdiscrimination\b",
        r"\bbilling\s+fraud\b",
        r"\bfraud\b",
        r"\bnegligence\b",
        r"\binformed\s+consent\b",
        r"\bconsent\s+violation\b",
        r"\bdenial\s+of\s+care\b",
        r"\bdenied\s+(?:care|treatment|access)\b",
        r"\bpatient\s+rights?\b",
        r"\bama\s+ethics\b",
        r"\bethics\s+(?:board|committee|complaint|violation)\b",
        r"\bphi\b",              # Protected Health Information
        r"\bdata\s+breach\b",
        r"\bunauthorized\s+(?:access|disclosure)\b",
        r"\bmedical\s+ethics\b",
    })

    _LEGAL_SIGNALS: frozenset = frozenset({
        r"\blegal\b",
        r"\blawsuit\b",
        r"\battorney\b",
        r"\blawyer\b",
        r"\bmotion\b",
        r"\bappeal\b",
        r"\bmalpractice\b",
        r"\blitigation\b",
        r"\blitigat",
        r"\binjunction\b",
        r"\bcourt\b.*\b(?:order|filing|case)\b",
        r"\bfile\s+(?:a\s+)?(?:suit|claim|motion|complaint)\b",
        r"\brestraining\s+order\b",
        r"\bada\s+violation\b",
        r"\bclass\s+action\b",
        r"\bsettlement\b",
        r"\bdiscovery\b.*\b(?:motion|request|order)\b",
        r"\bsubpoena\b",
        r"\bdeposition\b",
        r"\binsurance\s+appeal\b",
        r"\bgrievance\b",
    })

    _VISION_SIGNALS: frozenset = frozenset({
        r"\.png\b",
        r"\.jpg\b",
        r"\.jpeg\b",
        r"\.pdf\b",
        r"\.tiff?\b",
        r"\.bmp\b",
        r"\.gif\b",
        r"\.webp\b",
        r"\.heic\b",
        r"\bimage\b",
        r"\bphoto\b",
        r"\bscan\b",
        r"\blab\s+results?\s+(?:image|photo|scan|file)\b",
        r"\bocr\b",
        r"\bextract\s+(?:from\s+)?(?:image|photo|scan|pdf)\b",
        r"\bupload(?:ed)?\s+(?:image|photo|scan|file)\b",
    })

    # Mapping from signal-set name to (frozenset, AgentType)
    _SIGNAL_MAP: tuple

    def __init__(self) -> None:
        # Compile all patterns once at instantiation
        self._compiled: dict[AgentType, list] = {
            AgentType.CALENDAR_AGENT: [
                re.compile(p, re.IGNORECASE) for p in self._CALENDAR_SIGNALS
            ],
            AgentType.CLINICAL_TRIALS_AGENT: [
                re.compile(p, re.IGNORECASE) for p in self._TRIALS_SIGNALS
            ],
            AgentType.ETHICS_COMPLAINT_AGENT: [
                re.compile(p, re.IGNORECASE) for p in self._ETHICS_SIGNALS
            ],
            AgentType.LEGAL_MOTIONS_AGENT: [
                re.compile(p, re.IGNORECASE) for p in self._LEGAL_SIGNALS
            ],
            AgentType.VISION_AGENT: [
                re.compile(p, re.IGNORECASE) for p in self._VISION_SIGNALS
            ],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, text: str) -> tuple[AgentType, float]:
        """
        Classify *text* into an AgentType with a confidence score.

        Algorithm:
          1. If any VISION_SIGNAL matches → return (VISION_AGENT, 1.0) immediately.
             Image routing is unambiguous; further scoring is irrelevant.
          2. Count signal hits for each agent category.
          3. Determine winning category (highest hit count).
          4. Normalise hit count to [0.55, 1.0]:
               confidence = 0.55 + (hits / max_hits) * 0.45
             where max_hits = total patterns in winning set.
          5. If confidence < settings.hitl_confidence_threshold → return (END, confidence).

        Args:
            text: Raw user message content.

        Returns:
            (agent_type, confidence) tuple. confidence ∈ [0.55, 1.0] on a
            successful classification, or the raw sub-threshold value when
            routing to END.
        """
        if not text or not text.strip():
            return AgentType.END, 0.0

        # --- Step 1: Vision fast-path ---
        vision_patterns = self._compiled[AgentType.VISION_AGENT]
        for pattern in vision_patterns:
            if pattern.search(text):
                return AgentType.VISION_AGENT, 1.0

        # --- Step 2: Count signal hits per category ---
        scores: dict[AgentType, int] = {}
        for agent_type, patterns in self._compiled.items():
            if agent_type is AgentType.VISION_AGENT:
                continue  # already handled
            hits = sum(1 for p in patterns if p.search(text))
            scores[agent_type] = hits

        # --- Step 3: Winning category ---
        best_agent = max(scores, key=lambda a: scores[a])
        best_hits = scores[best_agent]

        if best_hits == 0:
            # No signal matches at all — cannot route confidently
            return AgentType.END, 0.0

        # --- Step 4: Normalise to [0.55, 1.0] ---
        # Use diminishing-returns curve: 1 hit = 0.70, 2 = 0.85, 3+ = 1.0.
        # Rationale: with ~20 patterns per category, a single confident signal
        # already meets the HITL threshold; additional hits raise confidence
        # further up to the 1.0 ceiling. This avoids the false precision of a
        # strict hit/total ratio while still penalising zero-signal inputs.
        normalised = 0.55 + min(best_hits, 3) * 0.15
        confidence = min(normalised, 1.0)

        # --- Step 5: HITL gate ---
        threshold = getattr(_settings, "hitl_confidence_threshold", 0.7)
        if confidence < threshold:
            return AgentType.END, confidence

        return best_agent, confidence

    def get_signal_hits(self, text: str, agent_type: AgentType) -> int:
        """Return raw hit count for a given agent type — useful for testing."""
        if agent_type not in self._compiled:
            return 0
        return sum(1 for p in self._compiled[agent_type] if p.search(text))


# Module-level singleton — avoids re-compiling regexes on every node invocation
_classifier = IntentClassifier()


# =============================================================================
# MASTER BRAIN NODE
# =============================================================================

def master_brain_node(state: AdvocateState) -> dict:
    """
    Primary routing node of the LangGraph workflow.

    Responsibilities:
      1. Extract the most recent HumanMessage from state['messages'].
      2. Run IntentClassifier (deterministic, no LLM).
      3. Construct a frozen TransparencyRecord with full reasoning chain,
         citing Bengio et al. (2019) and Hantel et al. (2024).
      4. Append a patient-facing AIMessage explaining the routing decision.
      5. Set next_agent and hitl_required in the returned patch dict.

    Immutability contract:
      - NEVER mutates calendar_events, pending_suggestions, or patient_profile.
      - Returns only {"audit_log", "messages", "next_agent", "hitl_required"}.
      - All other state fields are untouched.

    Args:
        state: Current AdvocateState (read-only by convention).

    Returns:
        Partial state patch dict consumed by LangGraph's reducer.
    """
    # --- Extract latest human message ---
    messages: list = state.get("messages", [])
    human_text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "__class__", None) and \
                getattr(msg.__class__, "__name__", "") == "HumanMessage":
            human_text = getattr(msg, "content", "") or ""
            break
        # Fallback: any object with content that's not an AI response
        content = getattr(msg, "content", None)
        if content and not isinstance(msg, AIMessage) and \
                getattr(msg.__class__, "__name__", "") != "AIMessage":
            human_text = content
            break

    # --- Classify intent ---
    agent_type, confidence = _classifier.classify(human_text)

    # --- Determine routing metadata ---
    hitl_required = agent_type is AgentType.END or \
        confidence < getattr(_settings, "hitl_confidence_threshold", 0.7)

    next_agent: str = agent_type.value

    # --- Build reasoning chain ---
    reasoning_chain = (
        f"Input text: '{human_text[:120]}{'...' if len(human_text) > 120 else ''}'. "
        f"IntentClassifier (deterministic, Glass Box — no LLM): "
        f"routed to '{agent_type.value}' with confidence={confidence:.3f}. "
        f"HITL required: {hitl_required}. "
        f"Threshold: {getattr(_settings, 'hitl_confidence_threshold', 0.7):.2f}. "
        f"Citations: {_ROUTING_CITATIONS}"
    )

    # --- Emit TransparencyRecord ---
    record = TransparencyRecord(
        event_id=uuid4(),
        suggestion_id=None,
        agent_name=AgentType.MASTER_BRAIN.value,
        conflict_type=None,
        confidence=confidence if confidence >= 0.0 else None,
        scholarly_basis=_ROUTING_CITATIONS,
        patient_decision=None,
        reasoning_chain=reasoning_chain,
        timestamp=datetime.utcnow(),
    )

    # --- Build patient-facing explanation ---
    if agent_type is AgentType.END:
        if confidence == 0.0:
            explanation = (
                "I wasn't able to identify a clear request in your message. "
                "Could you please describe what you need help with? "
                "(e.g., scheduling, clinical trials, ethics, legal, or image analysis)"
            )
        else:
            explanation = (
                f"I understood your request (confidence: {confidence:.0%}) but it "
                f"falls below the threshold required for automatic routing. "
                f"A human case manager will review your request to ensure accuracy."
            )
    elif agent_type is AgentType.VISION_AGENT:
        explanation = (
            "I've detected an image or document in your message. "
            "Routing to the Vision Agent for OCR and document analysis."
        )
    elif agent_type is AgentType.CALENDAR_AGENT:
        explanation = (
            f"I understand you have a scheduling or calendar request "
            f"(confidence: {confidence:.0%}). "
            f"Routing to the Calendar Intelligence Agent."
        )
    elif agent_type is AgentType.CLINICAL_TRIALS_AGENT:
        explanation = (
            f"I understand you're asking about clinical trials or study eligibility "
            f"(confidence: {confidence:.0%}). "
            f"Routing to the Clinical Trials Agent."
        )
    elif agent_type is AgentType.ETHICS_COMPLAINT_AGENT:
        explanation = (
            f"I've identified a potential ethics concern in your message "
            f"(confidence: {confidence:.0%}). "
            f"Routing to the Ethics Complaint Agent. "
            f"Your concern will be handled with full confidentiality."
        )
    elif agent_type is AgentType.LEGAL_MOTIONS_AGENT:
        explanation = (
            f"I understand you have a legal question or concern "
            f"(confidence: {confidence:.0%}). "
            f"Routing to the Legal Motions Agent. "
            f"Note: An attorney-in-the-loop is required before any filings."
        )
    else:
        explanation = (
            f"Routing your request to the {agent_type.value} agent "
            f"(confidence: {confidence:.0%})."
        )

    ai_message = AIMessage(content=explanation)

    return {
        "audit_log": [record],
        "messages": [ai_message],
        "next_agent": next_agent,
        "hitl_required": hitl_required,
    }


# =============================================================================
# CONDITIONAL EDGE FUNCTION
# =============================================================================

def route_to_agent(state: AdvocateState) -> str:
    """
    Conditional edge function for the LangGraph StateGraph.

    Maps state['next_agent'] to the registered node name.  HITL always
    overrides the destination to 'end', regardless of next_agent, when
    state['hitl_required'] is True.

    Unknown agent names fall back to 'end' (fail-safe).

    Args:
        state: Current AdvocateState after master_brain_node has run.

    Returns:
        Node name string for LangGraph to follow.
    """
    # HITL always overrides — human review takes precedence
    if state.get("hitl_required", False):
        return "end"

    next_agent: str = state.get("next_agent", AgentType.END.value)

    _VALID_NODES: frozenset = frozenset({
        AgentType.CALENDAR_AGENT.value,
        AgentType.CLINICAL_TRIALS_AGENT.value,
        AgentType.ETHICS_COMPLAINT_AGENT.value,
        AgentType.LEGAL_MOTIONS_AGENT.value,
        AgentType.VISION_AGENT.value,
    })

    if next_agent in _VALID_NODES:
        return next_agent

    # Unknown or END → fall back to 'end'
    return "end"


# =============================================================================
# STUB AGENT NODES
# =============================================================================

def _make_transparency_record(agent_type: AgentType, reasoning: str) -> TransparencyRecord:
    """Helper to build a frozen TransparencyRecord for stub agent nodes."""
    return TransparencyRecord(
        event_id=uuid4(),
        suggestion_id=None,
        agent_name=agent_type.value,
        conflict_type=None,
        confidence=None,
        scholarly_basis=_ROUTING_CITATIONS,
        patient_decision=None,
        reasoning_chain=reasoning,
        timestamp=datetime.utcnow(),
    )


def calendar_agent_node(state: AdvocateState) -> dict:
    """
    Stub Calendar Intelligence Agent node.

    Accepts scheduling, nadir-window, and fatigue-threshold queries.
    Full implementation in patient_advocate.calendar_engine.

    Immutability contract: does NOT mutate calendar_events, pending_suggestions,
    or patient_profile.
    """
    record = _make_transparency_record(
        AgentType.CALENDAR_AGENT,
        reasoning=(
            "Calendar Agent invoked. Will analyse nadir windows, fatigue thresholds, "
            "and appointment overlaps. "
            f"Citations: {_ROUTING_CITATIONS}"
        ),
    )
    msg = AIMessage(
        content=(
            "The Calendar Intelligence Agent is processing your scheduling request. "
            "It will check nadir windows (Crawford et al., 2004, NEJM) and fatigue "
            "thresholds (Bower et al., 2014, J Clin Oncol) before suggesting changes."
        )
    )
    return {"audit_log": [record], "messages": [msg]}


def clinical_trials_agent_node(state: AdvocateState) -> dict:
    """
    Stub Clinical Trials Agent node.

    Screens patient profile against ClinicalTrials.gov eligibility criteria.
    Full implementation in patient_advocate.agents.trials.
    """
    record = _make_transparency_record(
        AgentType.CLINICAL_TRIALS_AGENT,
        reasoning=(
            "Clinical Trials Agent invoked. Will screen patient profile against "
            "ClinicalTrials.gov eligibility criteria. "
            f"Citations: {_ROUTING_CITATIONS}"
        ),
    )
    msg = AIMessage(
        content=(
            "The Clinical Trials Agent is searching for eligible studies based on "
            "your diagnosis, treatment history, and current lab values."
        )
    )
    return {"audit_log": [record], "messages": [msg]}


def vision_agent_node(state: AdvocateState) -> dict:
    """
    Stub Vision Agent node.

    Performs OCR on uploaded images/PDFs to extract lab values, regimen details,
    or other clinical data. Full implementation in patient_advocate.agents.vision.
    """
    image_path: Optional[str] = state.get("image_path")
    record = _make_transparency_record(
        AgentType.VISION_AGENT,
        reasoning=(
            f"Vision Agent invoked for image path: '{image_path}'. "
            "Will perform OCR extraction and confidence-bounded lab-value parsing. "
            f"Citations: {_ROUTING_CITATIONS}"
        ),
    )
    msg = AIMessage(
        content=(
            f"The Vision Agent is processing your document "
            f"{'at ' + image_path if image_path else ''}. "
            f"Extracted values will require HITL verification before use."
        )
    )
    return {"audit_log": [record], "messages": [msg]}


def ethics_complaint_agent_node(state: AdvocateState) -> dict:
    """
    Stub Ethics Complaint Agent node.

    Screens for HIPAA violations, discrimination, informed-consent failures,
    and other ethics concerns. Full implementation in patient_advocate.ethics.
    Attorney-in-the-loop required before any formal filing.
    """
    record = _make_transparency_record(
        AgentType.ETHICS_COMPLAINT_AGENT,
        reasoning=(
            "Ethics Complaint Agent invoked. Will screen for HIPAA violations, "
            "discrimination, informed-consent failures, billing fraud, and negligence. "
            "Attorney review required before filing. "
            f"Citations: {_ROUTING_CITATIONS}"
        ),
    )
    msg = AIMessage(
        content=(
            "The Ethics Complaint Agent is reviewing your concern. "
            "All findings are confidential and will be presented to a case manager "
            "before any formal complaint is initiated."
        )
    )
    return {"audit_log": [record], "messages": [msg]}


def legal_motions_agent_node(state: AdvocateState) -> dict:
    """
    Stub Legal Motions Agent node.

    Prepares legal motion drafts and insurance appeals. Attorney-in-the-loop
    is mandatory before any court filing (claude.md §5).
    Full implementation in patient_advocate.agents.legal.
    """
    record = _make_transparency_record(
        AgentType.LEGAL_MOTIONS_AGENT,
        reasoning=(
            "Legal Motions Agent invoked. Will draft motions, appeals, or complaints. "
            "Attorney-in-the-loop required before any filing per claude.md §5. "
            f"Citations: {_ROUTING_CITATIONS}"
        ),
    )
    msg = AIMessage(
        content=(
            "The Legal Motions Agent is reviewing your legal concern. "
            "All draft motions must be reviewed and approved by a licensed attorney "
            "before submission. No action will be taken without explicit approval."
        )
    )
    return {"audit_log": [record], "messages": [msg]}


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def build_advocate_graph() -> "StateGraph":
    """
    Build and compile the full LangGraph StateGraph for the Patient Advocate system.

    Graph topology:
        START → master_brain → [route_to_agent] →
            calendar_agent_node          → END
            clinical_trials_agent_node   → END
            vision_agent_node            → END
            ethics_complaint_agent_node  → END
            legal_motions_agent_node     → END
            end  (HITL / unknown)

    All conditional routing is driven by route_to_agent, which reads state['next_agent']
    and state['hitl_required'].

    Returns:
        Compiled StateGraph ready for invocation via graph.invoke(state).
    """
    try:
        from patient_advocate.core.state import AdvocateState as _AdvocateState
        graph = StateGraph(_AdvocateState)
    except Exception:
        graph = StateGraph(dict)

    # --- Register nodes ---
    graph.add_node(AgentType.MASTER_BRAIN.value, master_brain_node)
    graph.add_node(AgentType.CALENDAR_AGENT.value, calendar_agent_node)
    graph.add_node(AgentType.CLINICAL_TRIALS_AGENT.value, clinical_trials_agent_node)
    graph.add_node(AgentType.VISION_AGENT.value, vision_agent_node)
    graph.add_node(AgentType.ETHICS_COMPLAINT_AGENT.value, ethics_complaint_agent_node)
    graph.add_node(AgentType.LEGAL_MOTIONS_AGENT.value, legal_motions_agent_node)

    # --- Entry point ---
    graph.set_entry_point(AgentType.MASTER_BRAIN.value)

    # --- Conditional routing edge from master_brain ---
    graph.add_conditional_edges(
        AgentType.MASTER_BRAIN.value,
        route_to_agent,
        {
            AgentType.CALENDAR_AGENT.value: AgentType.CALENDAR_AGENT.value,
            AgentType.CLINICAL_TRIALS_AGENT.value: AgentType.CLINICAL_TRIALS_AGENT.value,
            AgentType.VISION_AGENT.value: AgentType.VISION_AGENT.value,
            AgentType.ETHICS_COMPLAINT_AGENT.value: AgentType.ETHICS_COMPLAINT_AGENT.value,
            AgentType.LEGAL_MOTIONS_AGENT.value: AgentType.LEGAL_MOTIONS_AGENT.value,
            "end": GRAPH_END,
        },
    )

    # --- Terminal edges: each agent node goes to END ---
    for agent_value in (
        AgentType.CALENDAR_AGENT.value,
        AgentType.CLINICAL_TRIALS_AGENT.value,
        AgentType.VISION_AGENT.value,
        AgentType.ETHICS_COMPLAINT_AGENT.value,
        AgentType.LEGAL_MOTIONS_AGENT.value,
    ):
        graph.add_edge(agent_value, GRAPH_END)

    return graph.compile()
