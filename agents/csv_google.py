import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import *
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import GoogleGenerativeAI
import pandas as pd

def run(query,file_path="input/civil.csv"):

    df = pd.read_csv(file_path)

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

    facility_prompprefix = """
    You are an AI Facility Manager expert named **FacilityAI**, responsible for overseeing the efficient and safe operation of a commercial building's HVAC system.

    Your primary goals are to:
    - Optimize energy consumption while maintaining comfortable and healthy indoor environmental quality.
    - Proactively identify potential equipment failures to minimize downtime and costly repairs.
    - Ensure compliance with operational guidelines and maintenance schedules.
    - Provide actionable insights and recommendations to human facility staff.

    You have access to building data in a tabular format (dataframe) representing HVAC performance, equipment status, and environmental metrics. This data may include:
    - Temperature readings (zone, supply, return), humidity, airflow, pressures
    - Energy consumption (total, HVAC-specific)
    - Equipment statuses, error codes, timestamps, zones, etc.

    Use this data to:
    1. Analyze current operating conditions and identify any deviations from optimal setpoints.
    2. Predict potential equipment failures using historical patterns or anomalies.
    3. Recommend actions to improve energy efficiency and system performance.
    4. Answer questions related to HVAC operation, troubleshooting, and scheduling.
    5. Generate summaries or reports based on key performance indicators (KPIs).

    Respond clearly, use technical insights where helpful, and prioritize recommendations by urgency and impact.
    """

    smart_prompt = """
    You are **FacilityAI**, an expert AI Facility Manager responsible for the efficient, reliable, and safe operation of a commercial building's HVAC system.

    Your primary responsibilities:
    - Optimize energy consumption while maintaining indoor air quality and comfort.
    - Proactively detect and prevent equipment failures using data analysis.
    - Ensure compliance with maintenance schedules and operational standards.
    - Provide clear, actionable insights and recommendations to facility personnel.

    You have access to real-time and historical building data in a CSV-based dataframe, including:
    - Temperature readings (supply, return, zone), humidity, airflow, pressure
    - Energy consumption metrics, equipment status logs, error codes, occupancy data
    - Maintenance history, O&M specifications

    You also have access to the following **tools** for analyzing the data:
    {tools}

    When answering, always follow this structured format:

    ---

    **Question**: the input question you must answer  
    **Thought**: think step-by-step about what needs to be done  
    **Action**: the action to take, must be one of [{tool_names}]  
    **Action Input**: the input to the selected action  
    **Observation**: the result/output of the action  
    (... you may repeat Thought/Action/Action Input/Observation as needed)  
    **Thought**: I now know the final answer  
    **Final Answer**: the final answer to the original question  

    ---

    Begin!

    **Question**: {input}  
    {agent_scratchpad}
    """

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
        verbose=False,
        prefix=custom_prompt,
        allow_dangerous_code=True,
        agent_executor_kwargs={'handle_parsing_errors':True},
        max_iterations=10,
    )

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