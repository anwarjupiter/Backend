import requests
from langchain_core.tools import StructuredTool
from pydantic import BaseModel,create_model
from typing import Dict, Any
from constants import *
from langchain.agents import initialize_agent, AgentType
from typing import Any, Dict, List, Union
from pydantic import BaseModel, create_model
from langchain_google_genai import ChatGoogleGenerativeAI

class APIAgent:
    def __init__(self,routes,llm):
        self.routes = routes
        self.llm = llm
        self.tools = []
        self.build_tools()

    def create_tool_from_api(self,api_def: Dict[str, Any]) -> StructuredTool:
        name = api_def["name"].replace(" ", "_").lower()
        description = api_def["description"]
        method = api_def.get("method", "GET").upper()
        url = api_def["url"]
        
        body_schema = api_def.get("body", {})
        default_values = api_def.get("default", {})
  
        InputModel = create_model(
            f"{name}_input",
            **{
                key: (
                    self.map_json_type(value),
                    default_values.get(key, ...)
                )
                for key, value in body_schema.items()
            }
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

    def map_json_type(self,json_type: str):
        """
        Maps JSON string type to Python type.
        Extend as needed.
        """
        type_map = {
            "number": int,
            "string": str,
            "boolean": bool,
            "Array Of Objects": List[Dict[str, Any]],
            "array": list,
            "object": dict,
        }
        return type_map.get(json_type, Any)

    def build_tools(self):
        self.tools = [self.create_tool_from_api(api) for api in self.routes]
    
    def invoke(self,question:str):
        agent = initialize_agent(
            tools=self.tools,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            llm=self.llm,
            verbose=False,
        )
        return agent.invoke(input=question)['output']
    

if __name__ == "__main__":
    routes = [
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
    gemini = ChatGoogleGenerativeAI(model=MODEL_FLASH_2_0,api_key=GOOGLE_GEMINI_KEY,temperature=0)

    apiAgent = APIAgent(routes=routes,llm=gemini)

    while True:
        try:
            user = input("You: ")
            if user in ('exit','quit'):
                exit()
            bot = apiAgent.invoke(question=user)
            print("Bot :",bot)
        except Exception as e:
            print(e)