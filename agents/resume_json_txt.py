import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain.prompts import PromptTemplate
from langchain_ibm import ChatWatsonx
from constants import *

def run(file_path):
    """
    Functionality:
        Convert an json file into resume.txt file
    Input:
        - file_path : path of the json file
    Output:
        retruns an resume response

    """
    with open(file_path,"r") as file:
        input_json = file.read()
    
    with open("resume/sample-resume.txt","r") as file:
        example_resume = file.read()

    model = ChatWatsonx(
        model_id=IBM_MODEL,
        project_id=WATSONX_PROJECT_ID,
        apikey=WATSONX_API_KEY,
        url=SERVER_URL,
        params=WASTSONX_PARAMS
    )

    json_to_resume_prompt = PromptTemplate(
        input_variables=["example_resume","input_json"],
        template="""
            You are a resume writing assistant. Convert the given structured JSON data into a clean, professional plain-text resume. 
            Make sure to format it neatly with headers and proper spacing. The output should resemble a real human-written resume.

            Use this format:

            ===========================
            {example_resume}
            ===========================

            Rules:
            - Do not include fields that are empty or missing in the JSON.
            - Keep it readable and professional.
            - Output only the resume text (no JSON or extra comments).

            Now convert this JSON into a text-based resume:
            {input_json}
            """
        )

    resume_chain = json_to_resume_prompt | model

    response = resume_chain.invoke({"example_resume":example_resume,"input_json":input_json}).content

    os.makedirs("output",exist_ok=True)

    with open(os.path.join("./output","output.txt"),"w+") as file:
        file.write(response)

    print("Resume.txt was created successfully !")
    
    return response


if __name__ == "__main__":
    bot = run(
        file_path="input/sample_resume1.json"
    )