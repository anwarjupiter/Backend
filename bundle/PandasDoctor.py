from typing_extensions import TypedDict, Annotated
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain import hub
import pandas as pd
from constants import *
from pathlib import Path

# State
class State(TypedDict):
    question: str
    code: str
    result: str
    answer: str

# Output structure
class PandasCodeOutput(TypedDict):
    """Generated Pandas code."""
    code: Annotated[str, ..., "Valid Python code using Pandas to answer the question using the DataFrame named `df`."]

class PandasDoctor:

    def __init__(self):
        self.dataset = pd.DataFrame()
        self.python_tool = None
        self.llm = ChatGoogleGenerativeAI(
            model=MODEL_FLASH_2_0,
            api_key=GOOGLE_GEMINI_KEY,
            temperature=0,
            max_tokens=1000,
            top_k=100,
        )

    def _load_dataset(self,path):
        if Path(path).suffix == '.csv':
            self.dataset = pd.read_csv(path,encoding="ISO-8859-1")
        elif Path(path).suffix == '.xlsx' or '.xls':
            self.dataset = pd.read_excel(path)
        globals()["df"] = self.dataset 
        self.python_tool = PythonAstREPLTool(globals=globals())

    def write_pandas_code(self,state: State, prompt):
        system_prompt = prompt.invoke({
            "dialect": "pandas",          # <-- dummy value
            "top_k": 10,                  # <-- dummy value
            "table_info": f"Columns: {', '.join(self.dataset.columns)}; Example rows:\n{self.dataset.head().to_string(index=False)}",
            "input": state["question"]
        })
        structured_llm = self.llm.with_structured_output(PandasCodeOutput)
        result = structured_llm.invoke(system_prompt)
        return {"code": result["code"]}

    def execute_code(self,state: State):
        return {"result": self.python_tool.invoke(state["code"])}

    def generate_answer(self, state: State):
        with open("prompts/1.txt", "r") as file:
            prompt_template = file.read()
        print("Prompt Loaded")
        prompt = prompt_template.format(
            question=state["question"],
            code=state["code"],
            result=state["result"]
        )
        response = self.llm.invoke(prompt)
        return {"answer": response.content}

    def run(self,question="What is the average salary?"):
        pandas_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")  # reuse this for now
        code = self.write_pandas_code({"question": question}, prompt=pandas_prompt_template)
        result = self.execute_code({"code": code["code"]})
        answer = self.generate_answer({
            "question": question,
            "code": code["code"],
            "result": result["result"]
        })
        return answer["answer"]

if __name__ == "__main__":
    pandas_doc = PandasDoctor()
    pandas_doc._load_dataset("PandasDoctor/assets/live.csv")
    while True:
        try:
            user = input("You: ")
            if user.lower() in ("exit", "quit"):
                break
            bot = pandas_doc.run(question=user)
            print("Bot :\n",bot)
        except Exception as e:
            print(e)