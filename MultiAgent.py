import random
from constants import *
from pydantic import BaseModel
from langchain.chains import LLMChain
from bundle.Pdfteacher import PDFQABot
from typing import TypedDict,Optional,Literal
from langgraph.graph import StateGraph, END
from bundle.PandasDoctor import PandasDoctor
from langchain.prompts import PromptTemplate
from langchain_ibm.chat_models import ChatWatsonx
from bundle.MongoTool import MongoAggregationTool
from langchain.output_parsers import PydanticOutputParser

# --- STATE MODEL ---
class AgentState(TypedDict):
    question: str
    file: Optional[str]
    route: Optional[str]
    result: Optional[str]
    mongo_uri:Optional[str]
    db_name:Optional[str]
    vectorDB:Optional[str]

class RouteOutput(BaseModel):
    route: Literal["pdf", "csv", "mongo", "random"]

#defining tools
def pdf_tool(state: AgentState) -> AgentState:
    pdfBot = PDFQABot()
    pdfBot._build_qa_chain(pdf_path=state['file'],vectorDB=state['vectorDB'])
    answer = pdfBot.ask(question=state['question'])
    return {**state, "result": answer}

def csv_tool(state: AgentState) -> AgentState:
    pandasBot = PandasDoctor()
    pandasBot._load_dataset(path=state['file'])
    answer = pandasBot.run(question=state['question'])
    return {**state, "result": answer}

def mongo_tool(state: AgentState) -> AgentState:
    mongoBot = MongoAggregationTool(connection_string=state['mongo_uri'],db_name=state["db_name"])
    answer = mongoBot.run(query=state['question'])
    return {**state, "result": answer}

def random_tool(state: AgentState) -> AgentState:
    number = random.randint(1, 100)
    result = f"🎲 Random Number Tool: Your number is {number}"
    return {**state, "result": result}

class AgentRouter:
    def __init__(self):
        self.route_parser = PydanticOutputParser(pydantic_object=RouteOutput)
        self.llm = ChatWatsonx(
            model_id=MODEL_GRANITE_8B,
            project_id=WATSONX_PROJECT_ID,
            apikey=WATSONX_API_KEY,
            url=SERVER_URL,
            params=WASTSONX_PARAMS
        )
        self.graph = None
        self.router_chain = None

    def build_chain(self):
        router_prompt = PromptTemplate(
            input_variables=["question", "file","mongo_uri","dbname"],
            template="""
            You are a tool router. Given a user question, an optional file name, and context, select the correct tool.

            Tools:
            - pdf: Use this if the question is about the content of a PDF document.
            - csv: Use this if the question is about tabular data or statistics in a CSV file.
            - mongo: Use this if the question is about a MongoDB database or involves structured document queries.
            - random: Use this for general tasks like generating a random number or fallback when unclear.

            Respond with JSON only:
            {format_instructions}

            Question: {question}
            File: {file}
            Mongo URL: {mongo_uri}
            Databse Name: {dbname}
            """.strip(),
            partial_variables={"format_instructions": self.route_parser.get_format_instructions()}
        )
        self.router_chain = LLMChain(llm=self.llm, prompt=router_prompt)

    def router_node(self,state: AgentState) -> AgentState:
        question = state["question"]
        file = state.get("file", "None")
        mongo_uri = state.get("mongo_uri","None")
        db_name = state.get("db_name","None")
        output = self.router_chain.run(question=question, file=file,mongo_uri=mongo_uri,dbname=db_name)

        try:
            parsed = self.route_parser.parse(output)
            route = parsed.route
        except Exception:
            print("⚠️ Routing fallback to 'random'")
            route = "random"
        return {**state, "route": route}
    
    def build(self):
        self.build_chain()
        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node("router", self.router_node)
        workflow.add_node("pdf_tool", pdf_tool)
        workflow.add_node("csv_tool", csv_tool)
        workflow.add_node("mongo_tool",mongo_tool)
        workflow.add_node("random_tool", random_tool)
        workflow.add_node("end", lambda x: x)

        # Entry & Routing
        workflow.set_entry_point("router")
        workflow.add_conditional_edges(
            "router",
            lambda state: state["route"],
            {
                "pdf": "pdf_tool",
                "csv": "csv_tool",
                "random": "random_tool",
                "mongo":"mongo_tool"
            }
        )
        workflow.add_edge("pdf_tool", "end")
        workflow.add_edge("csv_tool", "end")
        workflow.add_edge("mongo_tool", "end")
        workflow.add_edge("random_tool", "end")
        self.graph = workflow.compile()
        return self.graph

if __name__ == "__main__":
    agentRouter = AgentRouter()
    agent = agentRouter.build()
    response = agent.invoke({"question": "Can you give me top 10 tamil movies do you have ?","mongo_uri":MONGO_URL, "db_name":"sample_mflix"})
    print(response['result'])