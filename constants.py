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

OPENAI_KEY = "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"

GOOGLE_GEMINI_KEY = "AIzaSyCqgpJTOLeA-BIk2lrHw2YojZA37NRBTJo"
MONGO_URL= "mongodb+srv://anwarmydheenk:xcwSgYCDarOKZzrq@cluster0.3t32wd8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# MONGO_URL=mongodb://localhost:27017/