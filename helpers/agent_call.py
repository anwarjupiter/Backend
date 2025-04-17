from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_ibm import WatsonxLLM
from constants import *
from langchain.tools import StructuredTool,Tool
from pydantic import BaseModel
import random
from agents import pdf

class PDFInput(BaseModel):
    question: str
    file_path: str
    database_path:str


def process_pdf_tool_structured(input: PDFInput) -> str:
    return pdf.run(question=input.question,pdf_path=input.file_path,vector_db=input.database_path)

def random_number(input:str) -> str:
    return random.randint(1,5)

pdf_tool = StructuredTool.from_function(
    func=process_pdf_tool_structured,
    name="PDFTool",
    description="Summarize PDF files based on a question and the file content."
)

random_tool = StructuredTool.from_function(
    func=random_number,
    name="RandomNumberTool",
    description="useful to generate a random number quickly"
)

tools = [
    pdf_tool,
    random_tool
]

llm = WatsonxLLM(
        model_id=MODEL_GRANITE_13B,
        project_id=WATSONX_PROJECT_ID,
        apikey=WATSONX_API_KEY,
        url=SERVER_URL,
        params=WASTSONX_PARAMS
    )

react_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# tool_input = {
#     "question": "what is assembly building ?",
#     "pdf_path": "input/tn.pdf"
# }


# result = react_agent.run(input=tool_input)
# print(result)