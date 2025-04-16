import time,os,tempfile
import pandas as pd
from google.api_core.exceptions import ResourceExhausted
from tqdm import tqdm
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from constants import *
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings


class PDFQABot():

    def __init__(self):
        self.pdf_path = None
        self.llm = self._init_llm()
        self.custom_prompt = self._build_prompt()
        # self.qa_chain = self._build_qa_chain()

    def _init_llm(self):
        return ChatGoogleGenerativeAI(
            model=MODEL_FLASH_2_0,
            api_key=GOOGLE_GEMINI_KEY,
            temperature=0,
            max_tokens=500,
            top_k=50,
        )
    
    def _load_pdf(self,pdf_path):
        loader = PyPDFLoader(file_path=pdf_path, extraction_mode="plain")
        return loader.load()

    def _get_retriever(self, documents):
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_GEMINI_KEY
        )
        vectorstore = FAISS.from_documents(documents[:1], embeddings)
        for doc in tqdm(documents[1:],desc="Creating Vector Store",unit='doc'):
            vectorstore.add_documents([doc])
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

    def _build_qa_chain(self,pdf_path):
        documents = self._load_pdf(pdf_path)
        logging.info("Pdf Loaded Successfully !")
        retriever = self._get_retriever(documents)
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

if __name__ == "__main__":
    pdf_path = "pdfteacher/assets/pdf/1.pdf"
    bot = PDFQABot()
    # while True:
    #     user_input = input("You: ")
    #     if user_input.lower() in ('exit', 'quit'):
    #         print("Exiting chatbot. Goodbye!")
    #         break
    #     response = bot.ask(user_input)
    #     print("Bot:\n", response)