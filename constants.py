import os
from dotenv import load_dotenv
MODEL_GRANITE_13B = "ibm/granite-13b-sft"

load_dotenv()
SERVER_URL = os.getenv("SERVER_URL")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")

# WatsonX model constants
MODEL_GRANITE_13B = "ibm/granite-13b-instruct-v2"
MODEL_GRANITE_8B = "ibm/granite-3-8b-instruct"
WASTSONX_PARAMS = {
    "temperature": 0,
    "max_new_tokens": 500,
    "min_new_tokens":10,
    'top_k':3,
}

#WatsonX Embedding Models
IBM_SLATE_125M_ENGLISH_RTRVR ="ibm/slate-125m-english-rtrvr-v2"

#HuggingFace Embedding Model
IBM_GRANITE_125M_ENGLISH = "ibm-granite/granite-embedding-125m-english"
