from pandasai import SmartDataframe
from langchain_ibm import ChatWatsonx
from constants import *
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent

def run1(question):
    df = pd.read_csv("input/civil.csv")
    model = ChatWatsonx(
        model_id=MODEL_GRANITE_8B,
        project_id=WATSONX_PROJECT_ID,
        apikey=WATSONX_API_KEY,
        url=SERVER_URL,
        params=WASTSONX_PARAMS
    )
    pandas_agent = SmartDataframe(df,config={"llm":model})
    response = pandas_agent.chat(query=question)
    return response

def run(question):
    df = pd.read_csv("input/civil.csv")
    llm = ChatWatsonx(
        model_id=MODEL_GRANITE_8B,
        project_id=WATSONX_PROJECT_ID,
        apikey=WATSONX_API_KEY,
        url=SERVER_URL,
        params=WASTSONX_PARAMS
    )
    agent = create_pandas_dataframe_agent(llm, df, verbose=True,allow_dangerous_code=True)
    response = agent.invoke(question)
    return response

if __name__ == "__main__":
    while True:
        try:
            user = input("You :")
            if user in ('exit','quit'):
                break
            if user == "":
                raise Exception("Input cannot be empty")
            bot = run1(question=user)
            print("Bot :\n",bot)
        except Exception as e:
            print(e)
