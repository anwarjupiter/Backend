from langchain_ibm.llms import WatsonxLLM
from langchain_experimental.tools import PythonAstREPLTool
import pandas as pd
from constants import *
from langchain_experimental.agents import create_pandas_dataframe_agent

df = pd.read_csv("input/civil.csv")

custom_prompt = """You are a helpful AI assistant that can answer questions based on the civil.csv file provided as a pandas dataframe.
You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
{agent_scratchpad}"""

llm = WatsonxLLM(
    model_id=MODEL_GRANITE_13B,
    project_id=WATSONX_PROJECT_ID,
    apikey=WATSONX_API_KEY,
    url=SERVER_URL,
    params=WASTSONX_PARAMS
)

repl_tool = PythonAstREPLTool(locals={"df": df})

tools = [repl_tool]

pandas_agent = create_pandas_dataframe_agent(
    llm,
    df=df,
    allow_dangerous_code=True,
    verbose=True,
    prefix=custom_prompt,
    agent_executor_kwargs={'handle_parsing_errors':True},
)

response = pandas_agent.run("how many rows are totally you have ?")

print(response)