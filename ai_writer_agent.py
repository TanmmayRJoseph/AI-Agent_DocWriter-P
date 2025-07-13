# Import necessary libraries
import os
import time
import datetime
from dotenv import load_dotenv
from typing import Optional, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print("✅ API key loaded!" if GOOGLE_API_KEY else "❌ API key not found!")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.5,
    google_api_key=GOOGLE_API_KEY
)
print("✅ LLM initialized successfully!")

# Define agent state
class AgentState(TypedDict):
    topic: str
    draft: Optional[str]
    feedback: Optional[str]
    is_satisfied: Optional[bool]

# Node 1: ai_draft
def ai_node(s: AgentState) -> AgentState:
    if s['draft'] is None:
        prompt = f"Write a document about {s['topic']}."
    else:
        prompt = f"Refine the following document: {s['draft']}"

    print("🧠 Generating response", end="", flush=True)
    for _ in range(3):
        time.sleep(0.5)
        print(".", end="", flush=True)
    print("\n")

    response = llm.invoke(prompt)
    s['draft'] = response.content
    return s

# Node 2: show_to_human
def show_to_human(s: AgentState) -> AgentState:
    print(f"\n📝 Draft:\n{s['draft']}\n")
    print("Processing complete. Preparing to ask for your feedback...")
    time.sleep(2.5)

    feedback = input("Are you satisfied with the draft? (yes/no): ").strip().lower()
    s['is_satisfied'] = True if feedback == "yes" else False
    return s

# Node 3: take_feedback
def take_feedback(s: AgentState) -> AgentState:
    if not s['is_satisfied']:
        feedback = input("Please provide your feedback on the draft: ")
        s['feedback'] = feedback
    else:
        s['feedback'] = None
    print(f"Feedback recorded: {s['feedback']}")
    return s

# Node 4: refine_draft
def refine_node(s: AgentState) -> AgentState:
    if s['feedback'] is not None:
        prompt = (
            f"Please revise the following draft to better address the feedback provided. "
            f"Be clear, concise, and accurate.\n\n"
            f"Feedback: {s['feedback']}\n\n"
            f"Original Draft:\n{s['draft']}"
        )

        print("🧠 Refining draft", end="", flush=True)
        for _ in range(3):
            time.sleep(0.5)
            print(".", end="", flush=True)
        print("\n")

        response = llm.invoke(prompt)
        s['draft'] = f"Refined Draft:\n{response.content}"
    else:
        print("No feedback provided, skipping refinement.")
    return s

# Node 5: save_draft
def save_draft(s: AgentState):
    if s.get('is_satisfied') == True:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"final_draft_{timestamp}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Topic: {s.get('topic', 'Unknown')}\n\n")
            f.write(f"Draft:\n{s.get('draft', 'No draft')}\n\n")
            f.write(f"Feedback: {s.get('feedback', 'No feedback provided')}\n")
        print(f"✅ Draft saved as {filename} in {os.getcwd()}")
    else:
        print("❌ Draft not saved as it was not approved by the user.")

# LangGraph setup
workflow = StateGraph(AgentState)
workflow.add_node("ai_draft", ai_node)
workflow.add_node("show_to_human", show_to_human)
workflow.add_node("take_feedback", take_feedback)
workflow.add_node("refine_draft", refine_node)
workflow.add_node("save_draft", save_draft)

workflow.set_entry_point("ai_draft")
workflow.add_edge("ai_draft", "show_to_human")

def condition(state: AgentState) -> str:
    return "save_draft" if state["is_satisfied"] else "take_feedback"

workflow.add_conditional_edges(
    "show_to_human",
    condition,
    {
        "save_draft": "save_draft",
        "take_feedback": "take_feedback"
    }
)
workflow.add_edge("take_feedback", "refine_draft")
workflow.add_edge("refine_draft", "ai_draft")

# Compile and run
app = workflow.compile()

if __name__ == "__main__":
    initial_state = {
        "topic": "The impact of AI in Business",
        "draft": None,
        "feedback": None,
        "is_satisfied": None
    }

app.invoke(initial_state)
