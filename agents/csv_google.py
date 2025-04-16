import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import *
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import GoogleGenerativeAI
import pandas as pd

def run(query,file_path="input/HVAC.csv"):

    df = pd.read_csv(file_path,encoding="ISO-8859-1")

    with open("input/prompt.txt", "r") as file:
        custom_prompt = file.read()

    llm = GoogleGenerativeAI(
        model="gemini-2.0-flash",
        api_key=GOOGLE_GEMINI_KEY,
        temperature=0,
        max_tokens=500,
        top_k=10
    )

    # Create a custom agent
    pandas_agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        prefix=custom_prompt,
        allow_dangerous_code=True,
        agent_executor_kwargs={'handle_parsing_errors':True},
        max_iterations=10,
    )
    # print(pandas_agent)

    response = pandas_agent.invoke(query)
    return response['output']

if __name__ == "__main__":
    while True:
        try:
            user = input("You :")
            if user in ('exit','quit'):
                break
            bot = run(query=user)
            print("Bot :\n",bot)
        except Exception as e:
            print(e)