from google.api_core.exceptions import ResourceExhausted
from tqdm import tqdm
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from constants import *
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_ibm.chat_models import ChatWatsonx
from langchain_ibm.embeddings import WatsonxEmbeddings


class PDFQABot():

    def __init__(self):
        self.pdf_path = None
        self.llm = self._init_llm()
        self.custom_prompt = self._build_prompt()
        # self.qa_chain = self._build_qa_chain()

    # def _init_llm(self):
    #     return ChatGoogleGenerativeAI(
    #         model=MODEL_FLASH_2_0,
    #         api_key=GOOGLE_GEMINI_KEY,
    #         temperature=0,
    #         max_tokens=500,
    #         top_k=50,
    #     )
    
    def _init_llm(self):
        logging.info(f"Model : Gemini Flash")
        genai = ChatGoogleGenerativeAI(
            model=MODEL_FLASH_2_0,
            api_key=GOOGLE_GEMINI_KEY,
            temperature=0,
            max_tokens=500,
            top_k=50,
        )
        # watsonx = ChatWatsonx(
        #     model_id=MODEL_GRANITE_8B,
        #     project_id=WATSONX_PROJECT_ID,
        #     apikey=WATSONX_API_KEY,
        #     url=SERVER_URL,
        #     params=WASTSONX_PARAMS
        # )
        return genai

    def _load_pdf(self,pdf_path):
        loader = PyPDFLoader(file_path=pdf_path, extraction_mode="plain")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        return docs

    def _get_retriever(self, documents,vectorDB):
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_GEMINI_KEY
        )
        # embeddings = WatsonxEmbeddings(
        #     model_id=IBM_SLATE_125M_ENGLISH_RTRVR,
        #     apikey=WATSONX_API_KEY,
        #     project_id=WATSONX_PROJECT_ID,
        #     url=SERVER_URL
        # )
        if os.path.exists(f'store/{vectorDB}'):
            vectorstore = FAISS.load_local(f'store/{vectorDB}',embeddings=embeddings,allow_dangerous_deserialization=True)
            logging.info("VectorStore Loaded Successfully")
        else:
            vectorstore = FAISS.from_documents(documents[:1], embeddings)
            for doc in tqdm(documents[1:],desc="Creating Vector Store",unit='doc'):
                vectorstore.add_documents([doc])
            vectorstore.save_local(f'store/{vectorDB}')
            logging.info(f"Vector Database Created Successfully at store/{vectorDB}")
        return vectorstore.as_retriever()

    def _build_prompt(self):
        return PromptTemplate(
            input_variables=["context", "question"],
           template = """
You are a highly intelligent and helpful PDF tutor. Your role is to read the context extracted from a PDF document and provide thoughtful, clear, and educational answers. 

Based on the context, do the following:
- Understand the intent behind the user's question.
- Provide a well-reasoned explanation, even if the answer requires interpreting the content.
- Use examples from the context if applicable.
- If the topic allows, teach the concept in a simple yet informative way, as if you're explaining it to someone learning for the first time.
- Be accurate, concise, and context-aware.

Context:
{context}

Question:
{question}

Answer:
""")

    def _build_qa_chain(self,pdf_path,vectorDB):
        documents = self._load_pdf(pdf_path)
        logging.info("Pdf Loaded Successfully !")
        retriever = self._get_retriever(documents,vectorDB=vectorDB)
        logging.info("VectoreStore Created Successfully !")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.custom_prompt},
            return_source_documents=False,
            verbose=True
        )
        logging.info("Retrieval Chain Builid Successfully !")

    def ask(self, question: str) -> str:
        result = self.qa_chain.invoke({"query": question})
        logging.info("Finalizing Natural Answer !")
        return result["result"]
