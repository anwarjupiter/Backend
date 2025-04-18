from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import AgentExecutor, create_structured_chat_agent
from pydantic import BaseModel,Field
from typing import List, Dict, Any, Optional, Union
import json
from constants import *
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

# Tool Selection Model
class ToolSelection(BaseModel):
    tool_name: str = Field("tool_name", description="Name of the tool")
    reasoning: str = Field(description="Reasoning for selecting this tool")
    parameters: Dict[str, Any] = Field(description="Parameters to pass to the tool")

class PDFTool(BaseTool):
    name: str = Field("pdf_tool", description="Name of the tool")
    description: str = Field("Extracts information from PDF files based on questions", description="Description of the tool")
    
    def _run(self, file: str, vectorDB: str, question: str) -> str:
        # Implement PDF processing logic here
        return f"PDF Tool processed file '{file}' using vectorDB '{vectorDB}' for question: '{question}'"
    
    def run(self, file: str, vectorDB: str, question: str) -> str:
        # Ensure `run` method aligns with the expected arguments
        return self._run(file, vectorDB, question)
    
    @classmethod
    def args_schema(cls) -> type[BaseModel]:
        class PDFToolArgs(BaseModel):
            file: str = Field(..., description="Path to PDF file")
            vectorDB: str = Field(..., description="Vector database to use")
            question: str = Field(..., description="Question to answer from PDF")
        return PDFToolArgs

class CSVTool(BaseTool):
    name: str = Field("csv_tool", description="Name of the tool")
    description: str = Field("Analyzes CSV files and answers questions about the data", description="Description of the tool")
    
    def _run(self, file: str, question: str) -> str:
        # Implement CSV processing logic here
        return f"CSV Tool processed file '{file}' for question: '{question}'"
    
    def run(self, file: str, question: str) -> str:
        # Ensure `run` method aligns with the expected arguments
        return self._run(file, question)
    
    @classmethod
    def args_schema(cls) -> type[BaseModel]:
        class CSVToolArgs(BaseModel):
            file: str = Field(..., description="Path to CSV file")
            question: str = Field(..., description="Question to answer from CSV data")
        return CSVToolArgs

class MongoTool(BaseTool):
    name: str = Field("mongo_tool", description="Name of the tool")
    description: str = Field("Queries MongoDB databases to answer questions", description="Description of the tool")
    
    def _run(self, conn: str, db: str, question: str) -> str:
        # Implement MongoDB query logic here
        return f"Mongo Tool connected to '{conn}', queried database '{db}' for: '{question}'"
    
    def run(self, conn: str, db: str, question: str) -> str:
        # Ensure `run` method aligns with the expected arguments
        return self._run(conn, db, question)
    
    @classmethod
    def args_schema(cls) -> type[BaseModel]:
        class MongoToolArgs(BaseModel):
            conn: str = Field(..., description="MongoDB connection string")
            db: str = Field(..., description="Database name")
            question: str = Field(..., description="Question to answer from database")
        return MongoToolArgs


class DynamicToolRouter:
    def __init__(self, llm: BaseLanguageModel, tools: List[BaseTool]):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.tool_selector = self._setup_tool_selector()
        
    def _setup_tool_selector(self):
        """Create a tool selector using the LLM"""
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}\n  Required parameters: {list(tool.args_schema().__annotations__.keys())}"
            for tool in self.tools.values()
        ])
        
        selector_prompt = ChatPromptTemplate.from_template("""
        You are an intelligent tool router. Based on the user's question, select the most appropriate tool and provide the necessary parameters.
        
        Available tools:
        {tool_descriptions}
        
        User question: {question}
        File: {file}
        VectorDB: {vectorDb}
        
        Analyze the question carefully and determine which tool would be most effective for answering it.
        Then, extract or infer the necessary parameters from the question.
        
        Output your decision in the following format:
        
        {{
            "tool_name": "selected_tool_name",
            "reasoning": "explanation for why this tool was selected",
            "parameters": {{
                "param1": "value1",
                "param2": "value2",
                ...
            }}
        }}
        """)
        
        parser = PydanticOutputParser(pydantic_object=ToolSelection)
        
        chain = selector_prompt | self.llm | parser
        return chain
    
    async def route_question(self, input_data: Dict[str, str]) -> str:
        """Route a question to the appropriate tool and return the result"""
        question = input_data["question"]
        file = input_data.get("file", None)  # Optional file
        vectorDb = input_data.get("vectorDb", None)  # Optional vectorDB

        # Select the appropriate tool and parameters
        selection = await self.tool_selector.ainvoke({
            "tool_descriptions": "\n".join([
                f"- {tool.name}: {tool.description}\n  Required parameters: {list(tool.args_schema().__annotations__.keys())}"
                for tool in self.tools.values()
            ]),
            "question": question,
            "file": file,
            "vectorDb": vectorDb
        })
        
        # Validate tool exists
        if selection.tool_name not in self.tools:
            return f"Error: Tool '{selection.tool_name}' not found"
        
        # Get the selected tool
        tool = self.tools[selection.tool_name]
        
        # Execute the tool with the provided parameters
        try:
            result = tool.run(**selection.parameters)
            
            # Return the result along with the reasoning
            return {
                "tool": selection.tool_name,
                "reasoning": selection.reasoning,
                "parameters": selection.parameters,
                "result": result
            }
        except Exception as e:
            return f"Error executing tool '{selection.tool_name}': {str(e)}"


# Example usage
async def main():
    llm = ChatGoogleGenerativeAI(model=MODEL_FLASH_2_0,api_key=GOOGLE_GEMINI_KEY,temperature=0)
    # Create tools
    tools = [PDFTool(), CSVTool(), MongoTool()]
    
    # Create router
    router = DynamicToolRouter(llm=llm, tools=tools)
    
   # Example input
    input_data = {
        "question": "Can you analyze the sales_data.csv file and tell me the trend for Q1?",
        "file": "/path/to/sales_data.csv",  # You can provide the actual file path here
        "vectorDb": "sales_vector_db"  # Example vector database
    }

    # Call the method
    result = await router.route_question(input_data)

    # Output the result
    print(result)


# Alternative implementation using LangChain's built-in agent framework
def create_dynamic_agent(llm: BaseLanguageModel, tools: List[BaseTool]):
    """Create an agent that can use multiple tools with dynamic arguments"""
    agent = create_structured_chat_agent(llm, tools)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())