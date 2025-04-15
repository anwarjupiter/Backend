from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal, Optional
import random

# --- State definition ---
class AgentState(TypedDict):
    question: str
    file: Optional[str]
    route: Optional[Literal["pdf", "random"]]
    result: Optional[str]

# --- Router Node ---
def router_node(state: AgentState) -> AgentState:
    question = state["question"].lower()
    file = state.get("file")

    if file and file.endswith(".pdf"):
        route = "pdf"
    elif "random" in question:
        route = "random"
    else:
        route = "random"  # default fallback

    return {**state, "route": route}

# --- PDF Tool Node ---
def pdf_tool_node(state: AgentState) -> AgentState:
    question = state["question"]
    file = state["file"]

    # Dummy answer — plug in your RAG logic here
    result = f"📄 Answered from PDF '{file}' for question: '{question}'"
    return {**state, "result": result}

# --- Random Tool Node ---
def random_tool_node(state: AgentState) -> AgentState:
    number = random.randint(1, 100)
    result = f"🎲 Random number: {number}"
    return {**state, "result": result}

# --- Graph construction ---
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("router", router_node)
workflow.add_node("pdf_tool", pdf_tool_node)
workflow.add_node("random_tool", random_tool_node)
workflow.add_node("end", lambda x: x)

# Edges
workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        "pdf": "pdf_tool",
        "random": "random_tool",
    }
)
workflow.add_edge("pdf_tool", "end")
workflow.add_edge("random_tool", "end")

# --- Compile graph ---
graph = workflow.compile()

# --- Example runs ---
print("\n🔍 PDF question:")
result_pdf = graph.invoke({"question": "What is the refund policy?", "file": "contract.pdf"})
print(result_pdf["result"])

print("\n🎲 Random number:")
result_rand = graph.invoke({"question": "Give me a random number"})
print(result_rand["result"])
