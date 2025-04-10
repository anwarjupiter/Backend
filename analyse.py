from langchain_ibm.llms import WatsonxLLM
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.prompts import PromptTemplate
import pandas as pd
from constants import *

df = pd.read_csv("input/civil.csv")

prompt = PromptTemplate(
    input_variables=["input"],
    template="""
You are a smart data analyst working with a DataFrame called `df`.

You must use Python (Pandas) to answer questions accurately and concisely. 
Avoid assumptions. Show your reasoning as comments.

Question: {input}
"""
)

llm = WatsonxLLM(
    model_id=MODEL_GRANITE_8B,
    project_id=WATSONX_PROJECT_ID,
    apikey=WATSONX_API_KEY,
    url=SERVER_URL,
    params=WASTSONX_PARAMS
)

# Create tools (Python REPL is required for pandas agents)
tools = [PythonREPLTool()]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"prefix": prompt.template,'handle_parsing_errors':True},
)

response = agent.run("How many rows are there in the data?")
print(response)
