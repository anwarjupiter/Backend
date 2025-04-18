from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated
from constants import *
from pydantic import BaseModel
import random

# --- Define Tool Functions ---
@tool
def get_joke(category: str = "any", count: int = 1) -> str:
    """Tells one or more jokes from a specific category. 
    Args:
        category: Type of joke like 'tech', 'dad', 'animal', or 'any'.
        count: Number of jokes to return (default 1).
    """
    jokes = {
        "tech": [
            "Why do programmers prefer dark mode? Because light attracts bugs.",
            "There are only 10 kinds of people in this world: those who understand binary and those who don’t."
        ],
        "dad": [
            "I'm reading a book on anti-gravity. It's impossible to put down!",
            "I would avoid the sushi if I was you. It’s a little fishy."
        ],
        "any": [
            "Why did the chicken cross the road? To get to the other side!",
            "Parallel lines have so much in common. It’s a shame they’ll never meet."
        ]
    }
    chosen = jokes.get(category.lower(), jokes["any"])
    return "\n".join(random.choices(chosen, k=min(count, len(chosen))))

@tool
def get_weather(location: str, unit: str = "Celsius") -> str:
    """Provides weather information for a given location.
    Args:
        location: Name of the city or region.
        unit: Unit of temperature. Either 'Celsius' or 'Fahrenheit'.
    """
    temp = 25 if unit == "Celsius" else 77
    return f"The weather in {location} is sunny and {temp}°{unit[0]}."

@tool
def get_quote(topic: str = "life", author: str = None) -> str:
    """Generates an inspirational quote based on topic or author.
    Args:
        topic: Theme like 'life', 'success', 'love'.
        author: Optional author name for a quote.
    """
    quotes = {
        "life": [
            "Life is what happens when you're busy making other plans.",
            "The purpose of our lives is to be happy."
        ],
        "success": [
            "Success usually comes to those who are too busy to be looking for it.",
            "Don't be afraid to give up the good to go for the great."
        ]
    }
    quote = random.choice(quotes.get(topic.lower(), quotes["life"]))
    if author:
        return f"{quote} - {author}"
    return quote

tools = [get_joke, get_weather, get_quote]

# --- LLM (Google Gemini) ---

llm = ChatGoogleGenerativeAI(model=MODEL_FLASH_2_0,api_key=GOOGLE_GEMINI_KEY,temperature=0)

llm_with_tools = llm.bind_tools(tools)

# --- Router Node ---

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    final_output: str

def route_tool(state: GraphState):
    msg = state["messages"][-1]
    response = llm_with_tools.invoke([msg])
    return {"messages": [response]}

# --- Tool Call Node ---

def call_tools(state: GraphState):
    msg = state["messages"][-1]
    tool_calls = msg.tool_calls
    outputs = llm_with_tools.invoke(tool_calls)
    return {"messages": outputs}

# --- LangGraph Assembly ---

builder = StateGraph(GraphState)

builder.add_node("router", route_tool)
builder.add_node("tool_call", ToolNode(tools))  # Directly use the tools list

builder.set_entry_point("router")
builder.add_edge("router", "tool_call")
builder.add_edge("tool_call", END)

graph = builder.compile()

inputs = {"messages": [HumanMessage(content="Tell me a joke about programming ")]}
result = graph.invoke(inputs)
answer = result["messages"][-1].content

print(answer)