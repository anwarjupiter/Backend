import tempfile,logging
from pathlib import Path
from typing import Optional
from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form
from slowapi import Limiter, _rate_limit_exceeded_handler
from agents import pdf,resume_json_txt,pdf_to_csv,csv_google,mongo_agent
from MultiAgent import AgentRouter

app = FastAPI()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Register the exception handler
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# checking for server Hello !
@app.get("/hello")
def hello():
    return {"message": "Hello, FastAPI!"}

@app.post("/agent")
async def ask_to_agent(agent:str = Form(default="any"),question:str=Form(...),file: Optional[UploadFile] = File(default=None),mongo_uri:Optional[str] = Form(default=None),db_name:Optional[str] = Form(default=None)):
    try:
        temp_file_path = None
        agentRouter = AgentRouter()
        agent = agentRouter.build()
        
        if file:
            ext = Path(file.filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(await file.read())
                temp_file_path = tmp.name

        if mongo_uri and db_name:
            response = agent.invoke({"question": question,"mongo_uri":mongo_uri, "db_name":db_name})
            answer = response['result']
        
        if temp_file_path:
            response = agent.invoke({"question":question,"file":temp_file_path})
            answer = response['result']

        return JSONResponse(status_code=200,content={"answer":answer})
    except Exception as e:
        print(e)
        return JSONResponse(status_code=500,content={'error':str(e)})

# Aksing question with pdf file
@app.post("/ask-pdf")
async def ask_pdf(question: str = Form(...),pdf_file: UploadFile = File(...),vector_db_path:str = Form(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await pdf_file.read())
            temp_pdf_path = tmp.name

        # Run your existing pipeline
        answer = pdf.run(question=question, pdf_path=temp_pdf_path,vector_db=vector_db_path)

        return JSONResponse(content={"answer": answer})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Converting json to resume.txt file   
@app.post("/json-to-resume")
async def json_to_resume(json_file: UploadFile = File(...)):
    try:
        # Save uploaded JSON to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_json:
            tmp_json.write(await json_file.read())
            temp_json_path = tmp_json.name

        # Process to get resume text
        resume_text = resume_json_txt.run(file_path=temp_json_path)

        # Save the result to a temp .txt file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w") as tmp_txt:
            tmp_txt.write(resume_text)
            resume_txt_path = tmp_txt.name

        return JSONResponse(status_code=200,content={"file_path":resume_txt_path})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Extracting tables from the pdf and saved into csv    
@app.post("/pdf-to-table")
async def pdf_to_table(pdf_file:UploadFile = File(...)):
    try:
        # Save uploaded pdf to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(await pdf_file.read())
            temp_pdf_path = tmp_pdf.name
        
        response = pdf_to_csv.run(pdf_file=temp_pdf_path,output=f"output/{pdf_file.filename.replace('.pdf','')}")

        return FileResponse(
            path=response,
            filename=pdf_file.filename.replace('.pdf','')+'.zip',
            media_type="application/zip"
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Asking Question With CSV     
@app.post("/ask-to-civil")
@limiter.limit("5/minute")
async def ask_to_civil(request: Request, question: str = Form(...)):
    try:
        if not question:
            return JSONResponse(status_code=400, content={"error": "Question Required"})
        answer = csv_google.run(query=question)
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# For Front End Changes    
@app.post("/ask-to-dummy")
async def ask_to_dummy(request: Request, question: str = Form(...)):
    try:
        if not question:
            return JSONResponse(status_code=400, content={"error": "Question Required"})
        answer = "This is your answers goes here ..."
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Asking Question with Uploaded CSV File    
@app.post("/ask-to-csv")
async def ask_to_csv(question: str = Form(...),csv_file: UploadFile = File(...)):
    try:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(await csv_file.read())
            temp_csv_path = tmp.name

        # pandasBot = PandasDoctor()
        # pandasBot._load_dataset(path=temp_csv_path)
        # answer = pandasBot.run(question=question)
        answer = csv_google.run(query=question,file_path=temp_csv_path)
        # answer = csv_llm.run(csv_file=temp_csv_path,query=question)

        return JSONResponse(content={"answer": answer})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.post("/ask-to-mongodb")
async def ask_to_mongodb(question:str = Form(...)):
    try:
        answer = mongo_agent.run(question=question)
        return JSONResponse(status_code=200,content={'answer':answer})
    except Exception as e:
        return JSONResponse(status_code=500,content={"error":str(e)})