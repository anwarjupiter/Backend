import os
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_ibm import ChatWatsonx
from constants import *

def run(file_path):

    with open(file_path,"r") as file:
        resume = file.read()
    
    with open("/Users/jb49/Library/CloudStorage/OneDrive-InnowellEngineeringInternationalPvt.Ltd/2025/Backend/assets/input/sample_resume.json","r") as file:
        example_json = file.read()

    model = ChatWatsonx(
        model_id=MODEL_GRANITE_8B,
        project_id=WATSONX_PROJECT_ID,
        apikey=WATSONX_API_KEY,
        url=SERVER_URL,
        params=WASTSONX_PARAMS
    )

    resume_to_json_prompt = PromptTemplate(
        input_variables=["resume_text","example_json"],
        template="""
            You are an intelligent assistant that converts unstructured resume text into a clean, valid JSON format. 
            You must always return a properly structured JSON object without errors.

            Follow this structure for the output JSON:
            {example_json}

            Now convert the following resume text to JSON:

            {resume_text}
        """
    )

    resume_chain = resume_to_json_prompt | model

    response = resume_chain.invoke({"resume_text":resume,"example_json":example_json}).content

    return response


if __name__ == "__main__":
    bot = run(
        file_path="resume/sample-resume.txt"
    )
    
    with open("output/output.json","w") as file:
        file.write(bot)
    
    print("Sample JSON Resume file was created successfully !")