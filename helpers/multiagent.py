import random
from typing import TypedDict, Literal, Optional
from pydantic import BaseModel
from constants import *
from langchain_ibm.chat_models import ChatWatsonx
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langgraph.graph import StateGraph, END

# --- STATE MODEL ---
class AgentState(TypedDict):
    question: str
    file: Optional[str]
    route: Optional[str]
    result: Optional[str]

# --- ROUTER OUTPUT SCHEMA ---
AllowedRoutes = Literal["pdf", "csv", "random"]

class RouteOutput(BaseModel):
    route: AllowedRoutes

# --- LLM Router Setup ---
llm = ChatWatsonx(
    model_id=MODEL_GRANITE_13B_CHAT_V2,
    project_id=WATSONX_PROJECT_ID,
    apikey=WATSONX_API_KEY,
    url=SERVER_URL,
    params=WASTSONX_PARAMS
)

parser = PydanticOutputParser(pydantic_object=RouteOutput)

router_prompt = PromptTemplate(
    input_variables=["question", "file"],
    template="""
You are a tool router. Given a user question and an optional file name, select the correct tool.

Tools:
- pdf: Use this if the question is about the content of a PDF document.
- csv: Use this if the question is about tabular data or statistics and a CSV file is provided.
- random: Use this if the user is asking for a random number or no relevant file is present.

Respond with JSON only:
{format_instructions}

Question: {question}
File: {file}
""".strip(),
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

router_chain = LLMChain(llm=llm, prompt=router_prompt)

# --- ROUTER NODE ---
def router_node(state: AgentState) -> AgentState:
    question = state["question"]
    file = state.get("file", "None")
    output = router_chain.run(question=question, file=file)

    try:
        parsed = parser.parse(output)
        route = parsed.route
    except Exception:
        print("⚠️ Routing fallback to 'random'")
        route = "random"

    return {**state, "route": route}

# --- TOOL NODES ---

def pdf_tool(state: AgentState) -> AgentState:
    result = f"📄 PDF Tool: Answered question '{state['question']}' using file '{state['file']}'"
    return {**state, "result": result}

def csv_tool(state: AgentState) -> AgentState:
    result = f"📊 CSV Tool: Analyzed '{state['file']}' for question '{state['question']}'"
    return {**state, "result": result}

def random_tool(state: AgentState) -> AgentState:
    number = random.randint(1, 100)
    result = f"🎲 Random Number Tool: Your number is {number}"
    return {**state, "result": result}

# --- BUILD LANGGRAPH ---
workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("router", router_node)
workflow.add_node("pdf_tool", pdf_tool)
workflow.add_node("csv_tool", csv_tool)
workflow.add_node("random_tool", random_tool)
workflow.add_node("end", lambda x: x)

# Entry & Routing
workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        "pdf": "pdf_tool",
        "csv": "csv_tool",
        "random": "random_tool"
    }
)
workflow.add_edge("pdf_tool", "end")
workflow.add_edge("csv_tool", "end")
workflow.add_edge("random_tool", "end")

graph = workflow.compile()

# 📄 PDF Case
result = graph.invoke({"question": "What is the refund policy?", "file": "contract.pdf"})
print(result["result"])

# 📊 CSV Case
result = graph.invoke({"question": "What is the average sales per month?", "file": "sales.csv"})
print(result["result"])

# 🎲 Random Case
result = graph.invoke({"question": "Give me a random number"})
print(result["result"])
