import os,logging,warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

load_dotenv()
SERVER_URL = os.getenv("SERVER_URL")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")

SUPPORTED_IBM_MODELS = [
    'ibm/granite-3-2b-instruct', 
    'ibm/granite-3-8b-instruct', 
    'ibm/granite-guardian-3-2b', 
    'ibm/granite-guardian-3-8b', 
    'meta-llama/llama-3-1-70b-instruct',
    'meta-llama/llama-3-2-11b-vision-instruct',
    'meta-llama/llama-3-2-90b-vision-instruct', 
    'meta-llama/llama-guard-3-11b-vision', 
    'mistralai/mistral-large'
]

IBM_MODEL = SUPPORTED_IBM_MODELS[0]

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

MODEL_FLASH_2_0 = "gemini-2.0-flash"
GOOGLE_GEMINI_KEY = "AIzaSyCqgpJTOLeA-BIk2lrHw2YojZA37NRBTJo"

MONGO_URL= "mongodb+srv://anwarmydheenk:xcwSgYCDarOKZzrq@cluster0.3t32wd8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# MONGO_URL=mongodb://localhost:27017/

logging.basicConfig(
    level=logging.INFO,
    # filename="log.txt",
    # filemode="w+",
    format='%(asctime)s - %(levelname)s - %(message)s'
)

