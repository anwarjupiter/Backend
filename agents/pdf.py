import os
from tqdm import tqdm
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ibm import ChatWatsonx
from langchain_community.document_loaders import PyPDFLoader
from langchain_ibm import WatsonxEmbeddings
from constants import *

retriever = None

def process_pdf(pdf_path,vector_db)->FAISS:
    """Loads, splits, and stores PDF content in a retriever."""

    print("--- Step 1 --- Loading Pdf Pages")
    loader = PyPDFLoader(pdf_path,extraction_mode="layout")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    print("--- Step 2 --- Generating Embddings for Pages")
    embeddings = WatsonxEmbeddings(
        model_id=IBM_SLATE_125M_ENGLISH_RTRVR,
        apikey=WATSONX_API_KEY,
        project_id=WATSONX_PROJECT_ID,
        url=SERVER_URL
    )

    # embeddings = HuggingFaceEmbeddings(model_name=IBM_GRANITE_125M_ENGLISH)

    if os.path.exists(vector_db):
        vectorstore = FAISS.load_local(vector_db,embeddings,allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents([docs[0]],embedding=embeddings)
        for doc in tqdm(docs[1:],desc="Generating Embeddings",unit="Doc"):
            vectorstore.add_documents([doc])
        vectorstore.save_local(vector_db)
        print(f"Vectorstore saved at: {vector_db}")
    return vectorstore

def query_vectorstore(vectorstore:FAISS, query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieved_documents = retriever.invoke(query)

    if not retrieved_documents:
        print("No relevant documents found.")
        return ""

    return " ".join([doc.page_content for doc in retrieved_documents])

def run(question,pdf_path,vector_db):
    print("--- Step 3 --- Prompting with instruction")
    with open("input/instruction.txt","r") as file:
        instruction = file.read()
    
    model = ChatWatsonx(
        model_id=MODEL_GRANITE_8B,
        project_id=WATSONX_PROJECT_ID,
        apikey=WATSONX_API_KEY,
        url=SERVER_URL,
        params=WASTSONX_PARAMS
    )

    qa_template = """
    You are an expert in Context Analysis. Your role is to provide accurate, clear, and concise answers based on the following context given below.
    Note you must follow the below instruction:

    {instruction}
    
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    
    <context>
    {context}
    </context>
    <question>
    {question}
    </question>
    """
    
    vectore_store = process_pdf(pdf_path=pdf_path,vector_db=vector_db)

    context = query_vectorstore(vectorstore=vectore_store,query=question)

    prompt = PromptTemplate.from_template(template=qa_template).format(instruction=instruction,context=context,question=question)

    print("--- Step 4 --- Invoke Question Chain with instruction")
    response = model.invoke(input=prompt)
    return response.content

if __name__ == "__main__":
    bot = run(
        pdf_path="input/tn.pdf",
        question="what is assembly building ?",
        vector_db="store/tn_ibm_embeddings_slate_125m_english_rtvr"
    )
    print(bot)