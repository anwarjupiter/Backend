import requests
from langchain_core.tools import StructuredTool
from pydantic import BaseModel,create_model
from typing import Dict, Any
from constants import *
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI

def create_tool_from_api(api_def: Dict[str, Any]) -> StructuredTool:
    name = api_def["name"].replace(" ", "_").lower()
    description = api_def["description"]
    method = api_def.get("method", "GET").upper()
    url = api_def["url"]
    
    body_schema = api_def.get("body", {})
    default_values = api_def.get("default", {})
    
    # Create a dynamic Pydantic model for tool input
    InputModel = create_model(
        f"{name}_input",
        **{k: (int if v == "number" else str, default_values.get(k, ...)) for k, v in body_schema.items()}
    )

    def tool_func(**kwargs):
        if method == "POST":
            response = requests.post(url, json=kwargs)
        else:
            response = requests.get(url, params=kwargs)
        return response.json()

    return StructuredTool.from_function(
        name=name,
        description=description,
        func=tool_func,
        args_schema=InputModel,
    )

# Your API definitions from Node.js
api_list = [
    {
        "name": "Generate Random Number",
        "description": "Generates a random number between a start and end value.",
        "body":{
            "start":"number",
            "end":"number"
        },
        "default":{
            "start":0,
            "end":5
        },
        "url": "http://192.168.10.124:3500/edchatbot/generaterandom",
        "method": "POST"
    },
    {
        "name": "Get Panel Properties",
        "description": "Return all panel properties",
        "url": "http://192.168.10.124:3500/edchatbot/getpanelproperties",
        "method": "GET"
    }
]

tools = [create_tool_from_api(api) for api in api_list]



llm = ChatGoogleGenerativeAI(model=MODEL_FLASH_2_0,api_key=GOOGLE_GEMINI_KEY,temperature=0)

agent = initialize_agent(
    tools=tools,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True,
)

response = agent.run("Give me a random number")

print(response)