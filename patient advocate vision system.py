import operator
from typing import Annotated, List, TypedDict, Optional
from PIL import Image
import pytesseract

# LangGraph & LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# --- 1. Define the State (Added 'image_path') ---
class AdvocateState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    patient_profile: dict
    audit_log: List[str]
    next_agent: str
    image_path: Optional[str]  # New field to hold the file path of scans/docs

# --- 2. Define the Vision Agent (The "Interpreter") ---

def vision_agent_node(state: AdvocateState):
    """
    The 'Perception' Node (Amit Sethi inspired).
    Uses PyTesseract to extract text from medical screenshots or scanned docs.
    """
    print("--- Vision Agent Activated (OCR) ---")
    image_path = state.get("image_path")
    
    extracted_text = ""
    audit_entry = ""

    if image_path:
        try:
            # 1. Load the image
            img = Image.open(image_path)
            
            # 2. Extract text using PyTesseract
            # Note: Ensure Tesseract engine is installed on your server
            extracted_text = pytesseract.image_to_string(img)
            
            # 3. Create transparency log
            audit_entry = f"Image processed by PyTesseract v5.0. Extracted {len(extracted_text)} characters."
            
        except Exception as e:
            extracted_text = f"Error processing image: {str(e)}"
            audit_entry = "Vision Agent Failed: Image load error."
    else:
        extracted_text = "No image path provided."
        audit_entry = "Vision Agent Skipped: No image found."

    # Return the extracted text as a System Message so the Master Brain can read it later
    return {
        "messages": [SystemMessage(content=f"OCR RESULT: {extracted_text}")],
        "audit_log": [audit_entry]
    }

# --- 3. Define the Master Brain (The Router) ---

def master_brain_node(state: AdvocateState):
    """
    The 'System 2' Reasoner.
    Decides if we need to look at an image or just chat.
    """
    print("--- Master Brain Thinking ---")
    
    # Check if there is an image in the current state
    if state.get("image_path"):
        decision = "vision_agent"
        reasoning = "Detected image attachment. Routing to Vision Agent for OCR."
    
    # Fallback to text analysis (Simulated)
    elif "fatigue" in state["messages"][-1].content.lower():
        decision = "clinical_trials_agent"
        reasoning = "Symptom detected. Routing to Trials Agent."
    else:
        decision = "end"
        reasoning = "No specific action required."

    return {
        "next_agent": decision,
        "audit_log": [f"Master routed to {decision}: {reasoning}"]
    }

# --- 4. Build the Graph ---

workflow = StateGraph(AdvocateState)

workflow.add_node("master_brain", master_brain_node)
workflow.add_node("vision_agent", vision_agent_node)
# (Add other nodes like clinical_trials_agent here...)

workflow.set_entry_point("master_brain")

def router(state: AdvocateState):
    return state["next_agent"]

workflow.add_conditional_edges(
    "master_brain",
    router,
    {
        "vision_agent": "vision_agent",
        "clinical_trials_agent": END, # Placeholder for other agents
        "end": END
    }
)

# Crucially, loop the Vision Agent BACK to the Master Brain
# This allows the Master Brain to "read" the text the Vision Agent just extracted
workflow.add_edge("vision_agent", "master_brain")

app = workflow.compile()

# --- 5. Simulation ---

# Mock Inputs: User uploads a screenshot of a lab report
inputs = {
    "messages": [HumanMessage(content="Here is a screenshot of my blood test results.")],
    "patient_profile": {"name": "John Doe"},
    "audit_log": [],
    "image_path": "lab_report_screenshot.png" # Assume this file exists
}

# In a real run, this would crash if the file doesn't exist, 
# so we mock the function for demonstration if needed.
# result = app.invoke(inputs)