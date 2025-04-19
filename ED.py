from pydantic import BaseModel
from typing import Dict, Optional
import httpx
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from constants import *
from pydantic import BaseModel
from typing import Dict, Optional, Literal
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
import json
from langchain.tools import Tool


class ValidatedAPICall(BaseModel):
    url: str
    method: Literal["GET", "POST"]
    body: Optional[Dict] = {}

class EDAgent:
    def __init__(self,routes:list,llm=ChatGoogleGenerativeAI(
        model=MODEL_FLASH_2_0,
        api_key=GOOGLE_GEMINI_KEY,
        temperature=0,
        convert_system_message_to_human=True
    )):
        logging.info("Routes recieved !")
        self.routes = routes
        self.llm = llm
        self.valid_methods = {"GET", "POST"}
        self.response_parser = PydanticOutputParser(pydantic_object=ValidatedAPICall)
        self.tool = Tool(
            name="smart_api_router",
            func=self.smart_router_tool,
            description="Smart API router that selects and calls an API from the known list. Uses validation."
        )

    def dynamic_api_call(self,url: str, method: str = "POST", body: Dict = {}, headers: Dict = None) -> str:
        try:
            with httpx.Client(timeout=300) as client:
                if method == "GET" and body:
                    response = client.request(method=method, url=url, params=body, headers=headers)
                else:
                    response = client.request(method=method, url=url, json=body, headers=headers)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return f"❌ Error calling API: {e}"
        
    def smart_router_tool(self,query: str) -> str:
        prompt = ChatPromptTemplate.from_template("""
You are an API router assistant.

Here are the available APIs:
{api_list}

Given the user question:
"{query}"

Pick the best API and fill this format:
{valid_request}

Only include keys required by the API. If no input needed, leave body null.
""")

        prompt_filled = prompt.format_messages(
            api_list=json.dumps(self.routes, indent=2),
            query=query,
            valid_request=self.response_parser.get_format_instructions()
        )

        response = self.llm.invoke(prompt_filled)
        
        try:
            parsed = self.response_parser.parse(response.content)
        except Exception as e:
            return f"❌ Invalid API format or response: {e}"

        # If validation passed, call API
        return self.dynamic_api_call(
            url=parsed.url,
            method=parsed.method,
            body=parsed.body
        )

    def build(self):
        return initialize_agent(
            tools=[self.tool],
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False
        )

# if __name__ == "__main__":
#     api_metadata = [
#         {
#             "name": "Generate Random Number",
#             "description": "Generates a random number between a start and end value.",
#             "body":{
#                 "start":"number",
#                 "end":"number"
#             },
#             "default":{
#                 "start":0,
#                 "end":5
#             },
#             "url": "http://192.168.10.124:3500/edchatbot/generaterandom",
#             "method": "POST"
#         },
#         {
#             "name": "Get Panel Properties",
#             "description": "Return all panel properties",
#             "url": "http://192.168.10.124:3500/edchatbot/getpanelproperties",
#             "method": "GET"
#         }
#     ]

#     llm = ChatGoogleGenerativeAI(
#         model=MODEL_FLASH_2_0,
#         api_key=GOOGLE_GEMINI_KEY,
#         temperature=0,
#         convert_system_message_to_human=True
#     )

#     ed_agent = EDAgent(routes=api_metadata,llm=llm)
#     agent = ed_agent.build()
#     response = agent.invoke(input="What are the properties of Panel ?")
#     print(response['output'])
    