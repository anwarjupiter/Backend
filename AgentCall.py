from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
from constants import *
from CustomTools import *

# --- Router Node ---
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    final_output: str

class AgentRouter:
    
    def __init__(self,tools,llm=ChatGoogleGenerativeAI(model=MODEL_FLASH_2_0,api_key=GOOGLE_GEMINI_KEY,temperature=0,convert_system_message_to_human=True)):
        self.llm = llm
        self.tools = tools
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.builder = StateGraph(GraphState)
        self.build()

    def build(self):
        self.builder.add_node("router", self.route_tool)
        self.builder.add_node("tool_call", ToolNode(self.tools))

        self.builder.set_entry_point("router")
        self.builder.add_edge("router", "tool_call")
        self.builder.add_edge("tool_call", END)

        self.graph = self.builder.compile()
        return self.graph

    def route_tool(self,state: GraphState):
        msg = state["messages"][-1]
        response = self.llm_with_tools.invoke([msg])
        return {"messages": [response]}

    def call_tools(self,state: GraphState):
        msg = state["messages"][-1]
        tool_calls = msg.tool_calls
        outputs = self.llm_with_tools.invoke(tool_calls)
        return {"messages": outputs}
    
    def run(self, question: str, **kwargs):
        """
        Accepts question + arbitrary keyword arguments for tool schemas.
        Example: question='What is X?', file='input.pdf', vectorDB='store'
        """
#         user_prompt = f"""
# ou are a tool router. Given a user question, an optional file name, and context, select the correct tool.
# Available Tools:
# {self.tools}
# Question : 
# {question}
# Tool arguments: 
# {kwargs}
# """

        user_prompt = f"""
You are a helpful and knowledgeable assistant. Your task is to understand the user's request and provide a comprehensive and accurate response. You have access to a set of specialized tools that can be used to enhance your answer when appropriate.

Available Tools:
{self.tools}

User Input:
{question}

Tool Arguments (if applicable):
{kwargs}

Based on the user's request and any provided context, determine the most effective way to respond. This might involve:

1.  **Answering directly** using your internal knowledge.
2.  **Selecting and using one or more of the available tools** to gather additional information or perform specific actions.
3.  **A combination of direct answering and tool usage.**

If a tool is selected, ensure the provided arguments are relevant and necessary for the tool to function correctly. If the user's request is a general question that doesn't require a specific tool, provide the best possible answer based on your knowledge.

Your goal is to be informative, helpful, and to fulfill the user's request in the most efficient and effective manner."""
        # print(user_prompt)
        inputs = {"messages": [{"role": "user", "content": user_prompt}]}
        result = self.graph.invoke(inputs)
        return result["messages"][-1].content

if __name__ == "__main__":
    agent = AgentRouter(tools=[mongo_tool,pdf_tool,csv_tool])
    response = agent.run(question="Lookup the data and give statistics",file="input/civil.csv")
    print(response.content)