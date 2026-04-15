"""
patient_advocate.routing
========================

Master Brain routing module — deterministic intent classification and
LangGraph StateGraph construction for the Patient Advocate system.

Public API:
    IntentClassifier      — Glass Box keyword-based router (no LLM).
    master_brain_node     — Primary LangGraph node; classifies and dispatches.
    route_to_agent        — Conditional edge function with HITL override.
    build_advocate_graph  — Builds and compiles the full StateGraph.

Stub agent nodes (full implementations in their respective sub-packages):
    calendar_agent_node
    clinical_trials_agent_node
    vision_agent_node
    ethics_complaint_agent_node
    legal_motions_agent_node
"""

from patient_advocate.routing.patient_advocate_routing_system import (  # noqa: F401
    IntentClassifier,
    build_advocate_graph,
    calendar_agent_node,
    clinical_trials_agent_node,
    ethics_complaint_agent_node,
    legal_motions_agent_node,
    master_brain_node,
    route_to_agent,
    vision_agent_node,
)

__all__ = [
    "IntentClassifier",
    "build_advocate_graph",
    "calendar_agent_node",
    "clinical_trials_agent_node",
    "ethics_complaint_agent_node",
    "legal_motions_agent_node",
    "master_brain_node",
    "route_to_agent",
    "vision_agent_node",
]
