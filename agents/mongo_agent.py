import sys,os,json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import *
from pymongo import MongoClient
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import initialize_agent,Tool,AgentType
from langchain_ibm.llms import WatsonxLLM
from langchain.prompts import PromptTemplate


DB_NAME = "sample_mflix"

COLLECTION_NAME = "movies"

client = MongoClient(MONGO_URL)

db = client.get_database(DB_NAME)

collection = db.get_collection(COLLECTION_NAME)

def infer_type_custom(value):
    if isinstance(value, str):
        return "string"
    elif isinstance(value, bool):
        return "boolean"
    elif isinstance(value, int):
        return "int"
    elif isinstance(value, float):
        return "float"
    elif isinstance(value, list):
        if not value:
            return "array of unknown"
        first_type = infer_type_custom(value[0])
        if isinstance(value[0], dict):
            return [infer_schema_custom(value[0], indent=2)]
        return f"array of {first_type}s"
    elif isinstance(value, dict):
        return infer_schema_custom(value, indent=2)
    elif value is None:
        return "null"
    else:
        return type(value).__name__

def infer_schema_custom(doc, indent=0):
    lines = []
    pad = " " * indent
    for key, value in doc.items():
        inferred = infer_type_custom(value)
        if isinstance(inferred, str):
            lines.append(f'{pad}"{key}": "{inferred}",')
        elif isinstance(inferred, list):
            lines.append(f'{pad}"{key}": [')
            lines.extend(inferred)
            lines.append(f"{pad}],")
        elif isinstance(inferred, dict):
            lines.append(f'{pad}"{key}": ')
            lines.extend(infer_schema_custom(value, indent + 2))
    return lines

def get_schema(db_name="sample_mflix", collection_name="movies"):
    try:
        client = MongoClient(MONGO_URL)
        collection = client[db_name][collection_name]
        sample_doc = collection.find_one()
        schema_lines = infer_schema_custom(sample_doc)
        return "\n".join(schema_lines)
    except Exception as e:
        return f"Error: {e}"

def retriever(mongo_query_str):
    try:
        # print(f"Generated Mongo Query : {mongo_query_str}")

        # Remove ```json and ``` if present
        if mongo_query_str.strip().startswith("```json"):
            mongo_query_str = mongo_query_str.replace("```json","").replace("```","").strip()

        data = list(collection.aggregate(json.loads(mongo_query_str)))
        return json.dumps(data, indent=2)
    except Exception as e:
        print(f"Error during query execution: {e}")
        return None

tools = [
    Tool(
        name="MongoDB Tool",
        description="Useful agent to create a mongodb pipelines/queries.",
        func=retriever
    )
]

def run(question):
    llm = GoogleGenerativeAI(
        model=MODEL_FLASH_2_0,
        api_key=GOOGLE_GEMINI_KEY,
        temperature=0,
        max_tokens=500,
        top_k=10
    )
    
    react_prompt_template = PromptTemplate(
        input_variables=['schema','question'],
        template="""
            You are a senior MongoDB engineer with deep expertise in writing aggregation pipelines.
            Your task is to answer questions by constructing and running MongoDB queries using aggregation pipelines, and then interpreting the results.

            Use this structured format:

            ---
            Question: {question}
            Thought: Think about the best way to query the data based on the schema.
            Action: MongoDB Tool
            Action Input: A valid MongoDB aggregation pipeline (in JSON list format).
            Observation: (The result returned from the MongoDB query.)
            ... (repeat Thought/Action/Observation if needed)
            Thought: Now interpret the Observation to produce the final result.
            Final Answer: The actual data or summary based on the Observation.
            ---

            Collection Schema: {schema}

            Begin!
        """
    )

    prompt_prefix = react_prompt_template.format(schema=get_schema(),question=question)

    mongo_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
        agent_kwargs={"prefix": prompt_prefix}
    )

    response = mongo_agent.run(input=question)
    return response

if __name__ == "__main__":
    while True:
        try:
            user = input("You :")
            if user in ('exit','quit'):
                break
            bot = run(question=user)
            print("Bot : \n",bot)
        except Exception as e:
            print(e)