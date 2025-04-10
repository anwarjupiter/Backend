import pandas as pd
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_ibm import ChatWatsonx
from langchain_ibm import WatsonxEmbeddings
from constants import *
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def create_retriever(csv_file="input/civil.csv"):
    """
    Function:
    Load the csv file into documents and save into vector database
    
    Input:
    csv_file - Path of the uplaoded csv file
    
    Ouput:
    Return the vectore database retriever object
    """
    filename_without_ext = os.path.basename(csv_file).replace(".csv","")
    # vector_db = f"store/{filename_without_ext}"
    vector_db = "store/civil"

    print("--- Step 1 --- Loading Documents")
    df = pd.read_csv(csv_file)

    docs = "\n".join(df.astype(str).apply(lambda row: ", ".join(row), axis=1)) 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    texts = text_splitter.split_text(docs)

    print(len(texts))
    print(texts[0])

    embeddings = WatsonxEmbeddings(
        model_id=IBM_SLATE_125M_ENGLISH_RTRVR,
        apikey=WATSONX_API_KEY,
        project_id=WATSONX_PROJECT_ID,
        url=SERVER_URL
    )

    if os.path.exists(vector_db):
        print("--- Step 2 --- Loading Vector Store")
        vectorstore = FAISS.load_local(vector_db,embeddings,allow_dangerous_deserialization=True)
    else:
        print("--- Step 2 --- Generating Embddings for Rows")
        vectorstore = FAISS.from_texts([texts[0]],embedding=embeddings)
        for doc in tqdm(texts[1:],desc="Generating Embeddings",unit="Doc"):
            vectorstore.add_texts([doc])
        vectorstore.save_local(vector_db)
        print(f"Vectorstore saved at: {vector_db}")
    return vectorstore.as_retriever()

def run(query,csv_file=""):
    model = ChatWatsonx(
        model_id=MODEL_GRANITE_8B,
        project_id=WATSONX_PROJECT_ID,
        apikey=WATSONX_API_KEY,
        url=SERVER_URL,
        params=WASTSONX_PARAMS
    )
    retriever = create_retriever()
    print("--- Step 3 --- Asking Question via Retrieval Chain")
    
    # prompt1 = PromptTemplate(
    #     input_variables=["context", "question"],
    #         template="""
    #     You are an AI Facility Manager expert, named "FacilityAI," responsible for overseeing the efficient and safe operation of a commercial building's Heating, Ventilation, and Air Conditioning (HVAC) system.
    #     Your primary goals are to: 
    #         * **Optimize energy consumption** while maintaining comfortable and healthy indoor environmental quality.
    #         * **Proactively identify potential equipment failures** to minimize downtime and costly repairs. 
    #         * **Ensure compliance** with operational guidelines and maintenance schedules.
    #         * **Provide actionable insights and recommendations** to human facility staff. You have access to the following information (which will be provided separately): 
    #         * **Building Data:** [Specify the types of building data you will provide, e.g., floor plans, occupancy schedules, zoning information, building automation system (BAS) architecture].
    #         * **HVAC Equipment Data:** [Specify the types of HVAC equipment data you will provide, e.g., equipment lists, specifications (make, model, capacity), installation dates].
    #         * **Operation & Maintenance Manuals:** [Specify that you will provide access to or excerpts from O&M manuals for the HVAC equipment].
    #         * **Live Operating Data:** [Specify the types of live data you will provide, e.g., temperature readings (zone, supply, return), humidity levels, airflow rates, pressure readings, energy consumption (overall, HVAC-specific), equipment status (on/off, error codes), sensor data]. 
    #     Based on this information, please perform the following tasks and respond to my queries: 
    #         1. **Analyze the current operating conditions** and identify any deviations from optimal performance or setpoints. 
    #         2. **Predict potential equipment failures** based on historical data, current readings, and O&M guidelines (if applicable). 
    #         3. **Recommend specific actions** to improve energy efficiency, address potential issues, and optimize system performance. Prioritize recommendations based on urgency and impact. 
    #         4. **Answer my questions** related to the building's HVAC system operation, maintenance, and troubleshooting. 
    #         5. **Generate reports** summarizing key performance indicators (KPIs), identified issues, and recommended actions (upon request). 
    #         **Example Scenarios/Questions you should be prepared to address:** 
    #         * "What is the current energy consumption of the HVAC system, and how does it compare to historical data for this time of year?" 
    #         * "Are there any zones that are experiencing significant temperature fluctuations or discomfort?" 
    #         * "Based on the live data, is there any equipment that shows signs of potential malfunction?" 
    #         * "What are the recommended maintenance tasks for the air handling unit (AHU) scheduled for next month, according to its O&M manual?" 
    #         * "How can we optimize the cooling schedule for the third floor based on the current occupancy?" 
    #         * "What are the potential causes of the high humidity levels reported in the server room?" 

    #         Context:
    #         {context}

    #         Question:
    #         {question}

    #         Answer:
    #         """)
    
    prompt2 = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an AI Facility Manager expert, named "FacilityAI," responsible for overseeing the efficient and safe operation of a commercial building's Heating, Ventilation, and Air Conditioning (HVAC) system.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=retriever,
        chain_type="stuff",  # Most common choice
        chain_type_kwargs={"prompt": prompt2},
        # verbose=True
    )
    
    response = qa_chain.invoke(query)
    return response['result']

if __name__ == "__main__":
    while True: 
        try:
            user = input("You :")
            if user in ('exit','quit'):
                break
            bot = run(csv_file="input/civil.csv",query=user)
            print("Bot :\n",bot)
        except Exception as e:
            print(e)