from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile
import logging
from typing import Optional
from AgentCall import AgentRouter
from CustomTools import *

app = FastAPI()

@app.post("/agent")
async def ask_to_agent(
    question: str = Form(...),
    file: Optional[UploadFile] = File(default=None),
    mongo_uri: Optional[str] = Form(default=None),
    db_name: Optional[str] = Form(default=None),
    vectorDB: Optional[str] = Form(default=None)
):
    try:
        context = {}
        temp_file_path = None
        if file:
            ext = Path(file.filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(await file.read())
                temp_file_path = tmp.name
            logging.info(f"Uploaded file saved to: {temp_file_path}")
        
        if temp_file_path is not None and Path(temp_file_path).suffix == ".pdf":
            context['file'] = temp_file_path
            context['vectorDB'] = vectorDB
        
        if temp_file_path is not None and Path(temp_file_path).suffix == ".csv":
            context["file"] = temp_file_path
        
        if mongo_uri and db_name:
            context['mongo_uri'] = mongo_uri
            context['db_name'] = db_name
        
        agent = AgentRouter(tools=[mongo_tool,pdf_tool,csv_tool,get_joke,ed_tool])
        response = agent.run(question=question,kwargs=context)

        return JSONResponse(status_code=200, content={"answer": str(response)})

    except Exception as e:
        logging.error(f"Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
