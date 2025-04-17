import json
import os
from typing import List, Dict, Any
from constants import *
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_community.document_loaders import JSONLoader
from pydantic import BaseModel, Field


# Define Pydantic model for the dynamic schema (initially empty, will be populated)
class DynamicSchema(BaseModel):
    pass

def load_training_examples(file_path: str) -> List[Dict[str, Any]]:
    loader = JSONLoader(
        file_path=file_path,
        jq_schema='.[ ]',
        content_key=None,
        text_content=False,
        metadata_func=lambda x, _: {"source": "training_example"}  # fixed here
    )
    documents = loader.load()
    return [doc.page_content for doc in documents]


def generate_dynamic_schema(raw_text: str, training_examples, model_name: str = "gemini-pro") -> Dict[str, Any]:
    """Generates a dynamic schema based on raw text and training examples.

    Args:
        raw_text: The raw text to analyze.
        training_examples: A list of example schemas as dictionaries.
        model_name: The name of the Google Generative AI model to use.

    Returns:
        A dictionary representing the dynamically generated schema (JSON format).
    """
    llm = ChatGoogleGenerativeAI(model=MODEL_FLASH_2_0,api_key=GOOGLE_GEMINI_KEY)
    output_parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Analyze the following raw text and the provided examples of JSON structures. 
                     Based on the patterns and fields observed in the examples, infer a dynamic schema 
                     that could be used to represent the information present in the raw text. 
                     Return the schema as a JSON object.

                     **Raw Text:**
                     {raw_text}

                     **Example JSON Structures:**
                     {examples}

                     Consider the types of information present in the raw text and how they map to the fields in the examples. 
                     Aim to create a schema that is both comprehensive enough to capture the key information and consistent 
                     with the structure demonstrated in the examples."""),
        ("human", "Generate the dynamic schema in JSON format.")
    ])

    formatted_examples = json.dumps(training_examples, indent=2)

    chain = prompt | llm | output_parser
    schema = chain.invoke({"raw_text": raw_text, "examples": formatted_examples})
    return schema

def transform_text_with_schema(raw_text: str, schema: Dict[str, Any], model_name: str = "gemini-pro") -> Dict[str, Any]:
    """Transforms the raw text into a structured format based on the provided schema.

    Args:
        raw_text: The raw text to transform.
        schema: The dynamic schema (as a dictionary) to use for transformation.
        model_name: The name of the Google Generative AI model to use.

    Returns:
        A dictionary representing the structured information extracted from the raw text.
    """
    llm = ChatGoogleGenerativeAI(model=MODEL_FLASH_2_0,api_key=GOOGLE_GEMINI_KEY)
    output_parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Analyze the following raw text and extract the relevant information based on the provided JSON schema. 
                     Structure the extracted information as a JSON object that conforms to the schema.

                     **Raw Text:**
                     {raw_text}

                     **Schema:**
                     {schema}

                     Identify the key pieces of information in the raw text and map them to the corresponding fields defined in the schema. 
                     Ensure that the output JSON object is a valid instance of the provided schema."""),
        ("human", "Extract information and structure it as a JSON object according to the schema.")
    ])

    formatted_schema = json.dumps(schema, indent=2)

    chain = prompt | llm | output_parser
    structured_data = chain.invoke({"raw_text": raw_text, "schema": formatted_schema})
    return structured_data

if __name__ == "__main__":
    # 1. Prepare your training JSON file
    training_file_path = "schema.json"

    # 2. Load training examples
    training_examples = load_training_examples(training_file_path)
    print("Loaded Training Examples:")
    for example in training_examples:
        print(json.dumps(example, indent=2))
    print("-" * 30)

    # 3. Input raw text
    with open("raw.txt") as file:
        raw_text_input = file.read()

    # 4. Generate dynamic schema
    dynamic_schema = generate_dynamic_schema(raw_text_input, training_examples)
    print("Generated Dynamic Schema:")
    print(json.dumps(dynamic_schema, indent=2))
    print("-" * 30)

    # 5. Transform raw text using the generated schema
    if dynamic_schema:
        transformed_data = transform_text_with_schema(raw_text_input, dynamic_schema)
        print("Transformed Data:")
        output_data =json.dumps(transformed_data, indent=2)
        with open("output.json","w") as file:
            file.write(output_data)
    else:
        print("Failed to generate a dynamic schema.")

    # # Clean up the dummy training file
    # os.remove(training_file_path)