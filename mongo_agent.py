import sys
import os
import json
from bson import ObjectId
from datetime import datetime
from pymongo import MongoClient
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import GoogleGenerativeAI
from constants import MONGO_URL, GOOGLE_GEMINI_KEY

# Define the prompt template
mongo_prompt_template = PromptTemplate.from_template("""
You are an expert in crafting NoSQL queries for MongoDB with 10 years of experience, particularly in MongoDB. 
I will provide you with the table_schema in a specified format. 
Your task is to read the user_question, which will adhere to certain guidelines or formats, and create a NOSQL MongoDb pipeline accordingly.

Table schema: {schema}

Input: {question}

The response should be in this format:

[
    {{
        "$match": {{"departments.employees": {{"$gt": 1000}}}}
    }},
    {{
        "$project": {{
            "departments": {{
                "$filter": {{
                    "input": "$departments",
                    "as": "dept",
                    "cond": {{"$gt": ["$$dept.employees", 1000]}}
                }}
            }}
        }}
    }}
]

Note: Just return the query, nothing else.
""")


react_prompt_template = PromptTemplate.from_template("""
You are a senior MongoDB engineer with deep expertise in writing aggregation pipelines.
Your job is to help answer questions by crafting precise MongoDB queries.

Use this structured format to think and act:

---
Question: {question}
Thought: Think about the best way to query the data based on the schema.
Action: generate_mongo_query
Action Input: Provide the aggregation pipeline based on the user's question and schema.
Observation: (The tool will fill this in with the MongoDB output.)
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: Analyze the result and decide the final answer.
Final Answer: (Summarize or return the answer for the user.)
---

Table Schema: {schema}

Begin!
""")


# MongoDB runner function
def run_mongo_query_from_natural_language(question, schema, db_name="hvac_system", collection_name="sensor_data"):
    # Initialize LLM
    llm = GoogleGenerativeAI(
        model="gemini-1.5-pro",
        api_key=GOOGLE_GEMINI_KEY,
        temperature=0,
        max_tokens=1024
    )

    # Step 1: Generate the Mongo aggregation pipeline
    chain = LLMChain(llm=llm, prompt=mongo_prompt_template)
    mongo_query_str = chain.run({"schema": schema, "question": question}).strip()

    try:
        action_input_start = mongo_query_str.find("Action Input:") + len("Action Input:")
        observation_start = mongo_query_str.find("Observation:")
        mongo_query_str = mongo_query_str[action_input_start:observation_start].strip()
        mongo_query = json.loads(mongo_query_str)

        # Step 2: Parse the output into a Python list
        # mongo_query = json.loads(mongo_query_str)

        # # Step 3: Connect to Mongo and run aggregation
        # client = MongoClient(MONGO_URL)
        # collection = client[db_name][collection_name]
        # results = list(collection.aggregate(mongo_query))

        client = MongoClient(MONGO_URL)
        collection = client[db_name][collection_name]
        results = list(collection.aggregate(mongo_query))

        # Step 4: Re-pass result to the LLM to generate final answer
        final_prompt = f"""{mongo_query_str[:observation_start]}
        Observation: {json.dumps(results[:3], indent=2)}  # Only show first 3 for readability
        Thought: I now know the final answer
        Final Answer:"""

        return final_prompt + final_answer.content

    except Exception as e:
        return f"Error: {str(e)}\n\nGenerated Query:\n{mongo_query_str}"

def infer_type(value):
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
            return ["unknown"]
        return [infer_type(value[0])]
    elif isinstance(value, dict):
        return infer_schema_from_document(value)
    elif value is None:
        return "null"
    else:
        return type(value).__name__

def infer_schema_from_document(doc):
    schema = {}
    for key, value in doc.items():
        schema[key] = infer_type(value)
    return schema

def get_schema(db_name="sample_mflix",collection_name="movies")->str:
    try:
        client = MongoClient(MONGO_URL)
        collection = client[db_name][collection_name]
        sample_doc = collection.find_one()
        return infer_schema_from_document(sample_doc)
    except Exception as e:
        return "{}"

# CLI for user input
if __name__ == "__main__":
    schema = get_schema()
    while True:
        user = input("You: ")
        if user.lower() in ("exit", "quit"):
            break
        output = run_mongo_query_from_natural_language(user, schema)
        print("Bot:\n", output)
