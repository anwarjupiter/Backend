import time,os,tempfile
import pandas as pd
from google.api_core.exceptions import ResourceExhausted
from tqdm import tqdm
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from constants import MODEL_FLASH_2_0,GOOGLE_GEMINI_KEY
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
        retriever = self._get_retriever(documents)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.custom_prompt},
            return_source_documents=False,
            verbose=True
        )

    def ask(self, question: str) -> str:
        result = self.qa_chain.invoke({"query": question})
        return result["result"]

    # def prepare(self, topic: str = "overall", num_questions: int = 5) -> pd.DataFrame:
    #     prompt = f"""
    #     Generate {num_questions} multiple choice questions from the PDF content.
    #     Format the output as a JSON list of dictionaries with the following keys:
    #     - question
    #     - option1
    #     - option2
    #     - option3
    #     - option4
    #     - answer (must match one of the options)

    #     Ensure questions are clear and based only on the context provided.
    #     Topic: {topic}
    #     Only return JSON — no extra explanation.
    #     """
    #     response = self.llm.invoke(prompt)
        
    #     try:
    #         data = response.content.replace('```json','').replace('```','').strip()
    #         questions_data = json.loads(data)
    #         df = pd.DataFrame(questions_data)
    #         return df
    #     except json.JSONDecodeError:
    #         print("Failed to parse quiz JSON. Raw response:")
    #         print(response.content)
    #         return pd.DataFrame()

    def prepare(self,topic:str,num_questions:int=5)->pd.DataFrame:
        format_instructions = self.quiz_parser.get_format_instructions()
        prompt = f"""
        Generate {num_questions} multiple choice questions from the PDF content.

        Ensure questions are clear and based only on the context provided.
        Topic: {topic}

        {format_instructions}
        """
        response = self.llm.invoke(prompt)
        formated_response = self.quiz_parser.parse(response.content)
        df = pd.DataFrame([question.model_dump() for question in formated_response.questions])
        return df

    def generate_quiz(self,pdf_path):
        documents = self._load_pdf(pdf_path=pdf_path)
        for doc in tqdm(documents,desc="Preparing Quiz ",unit='page'):
            try:
                df = self.prepare(topic=doc,num_questions=5)
                self.questions = pd.concat([self.questions,df], ignore_index=True)
            except ResourceExhausted as e:
                print("oops resource exhausted ! Please wait 60 seconds ")
                break
            except Exception as e:
                print(e)
        self.questions.to_csv(f'pdfteacher/quiz/{time.time()}.csv')
        quiz_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w', encoding='utf-8')
        self.questions.to_csv(quiz_file.name, index=False)
        quiz_file.close()
        return quiz_file.name

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