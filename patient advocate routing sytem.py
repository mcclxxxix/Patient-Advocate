import operator
from typing import Annotated, List, TypedDict, Union
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# --- 1. Define the Shared State ---
# This acts as the "Memory" passing through the diagram
class AdvocateState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    patient_profile: dict         # The "Profile d/b" from your sketch
    audit_log: List[str]          # Transparency Log (Required by sketch)
    next_agent: str               # The decision made by the Master Model

# --- 2. Define the Agents (Nodes) ---

def master_brain_node(state: AdvocateState):
    """
    The 'System 2' Reasoner (Yoshua Bengio inspired).
    It analyzes the user input + patient profile to decide the next step.
    """
    last_message = state["messages"][-1].content
    diagnosis = state["patient_profile"].get("diagnosis", "Unknown")

    print(f"--- Master Brain Thinking (Context: {diagnosis}) ---")

    # SIMULATED LLM LOGIC (Replace with actual ChatModel call)
    # If the user mentions a new serious symptom, check trials.
    # If they mention logistics/time, check calendar.
    if "fatigue" in last_message.lower() or "trial" in last_message.lower():
        decision = "clinical_trials_agent"
        reasoning = "Symptom 'fatigue' flagged for potential clinical trial matching."
    elif "schedule" in last_message.lower() or "appointment" in last_message.lower():
        decision = "calendar_agent"
        reasoning = "User requested schedule modification."
    else:
        decision = "general_chat"
        reasoning = "General inquiry."

    # Update the state with the decision and log the reasoning for transparency
    return {
        "next_agent": decision,
        "audit_log": [f"Master Model routed to {decision}: {reasoning}"]
    }

def clinical_trials_node(state: AdvocateState):
    """
    The RAG Agent (Fareed Khan inspired).
    Retrieves relevant trials based on symptoms.
    """
    print("--- Clinical Trials Agent Activated ---")
    
    # Logic: Search vector DB for matching trials
    found_trials = "Found 2 matching trials for 'Fatigue in Oncology'."
    
    return {
        "messages": [SystemMessage(content=found_trials)],
        "audit_log": ["Query executed against ClinicalTrials.gov database"]
    }

def calendar_node(state: AdvocateState):
    """
    The Optimization Agent.
    Adjusts the regimen schedule.
    """
    print("--- Calendar Agent Activated ---")
    return {
        "messages": [SystemMessage(content="Schedule updated: Rest period added.")],
        "audit_log": ["Calendar repopulated based on fatigue marker"]
    }

# --- 3. Define the Routing Logic (Edges) ---

def router(state: AdvocateState):
    """The conditional edge that directs traffic."""
    return state["next_agent"]

# --- 4. Build the Graph ---

workflow = StateGraph(AdvocateState)

# Add Nodes
workflow.add_node("master_brain", master_brain_node)
workflow.add_node("clinical_trials_agent", clinical_trials_node)
workflow.add_node("calendar_agent", calendar_node)

# Set Entry Point
workflow.set_entry_point("master_brain")

# Add Conditional Edges
workflow.add_conditional_edges(
    "master_brain",
    router,
    {
        "clinical_trials_agent": "clinical_trials_agent",
        "calendar_agent": "calendar_agent",
        "general_chat": END
    }
)

# Add edges back to END (or loop back to Master in a continuous chat)
workflow.add_edge("clinical_trials_agent", END)
workflow.add_edge("calendar_agent", END)

# Compile
app = workflow.compile()

# --- 5. Simulation ---

# Mock Patient Data
patient_data = {
    "diagnosis": "Stage 2 Lymphoma",
    "regimen": "Chemotherapy Cycle A"
}

# User Input: Triggering the "Symptom Loop"
inputs = {
    "messages": [HumanMessage(content="I am feeling extreme fatigue today, is there anything new for this?")],
    "patient_profile": patient_data,
    "audit_log": []
}

# Run the Graph
result = app.invoke(inputs)

print("\n--- Final Output ---")
print(f"Bot Reply: {result['messages'][-1].content}")
print(f"Transparency Log: {result['audit_log']}")