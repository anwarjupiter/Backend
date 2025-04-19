from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile
import logging,json
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
        
        context['api_metadata'] = json.dumps([
            {
                "name": "Generate Random Number",
                "description": "Generates a random number between a start and end value.",
                "body":{
                    "start":"number",
                    "end":"number"
                },
                "default":{
                    "start":0,
                    "end":5
                },
                "url": "http://192.168.10.124:3500/edchatbot/generaterandom",
                "method": "POST"
            },
            {
                "name": "Get Panel Properties",
                "description": "Return all panel properties",
                "url": "http://192.168.10.124:3500/edchatbot/getpanelproperties",
                "method": "GET"
            }
        ])
        
        agent = AgentRouter(tools=[mongo_tool,pdf_tool,csv_tool,get_joke,ed_tool])
        response = agent.run(question=question,kwargs=context)

        return JSONResponse(status_code=200, content={"answer": str(response)})

    except Exception as e:
        logging.error(f"Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/edchatbot")
async def ask_to_ed(question:str = Form(...)):
    try:
        api_metadata = [
            {
                "name": "Generate Random Number",
                "description": "Generates a random number between a start and end value.",
                "body":{
                    "start":"number",
                    "end":"number"
                },
                "default":{
                    "start":0,
                    "end":5
                },
                "url": "http://192.168.10.124:3500/edchatbot/generaterandom",
                "method": "POST"
            },
            {
                "name": "Get Panel Properties",
                "description": "Return all panel properties",
                "url": "http://192.168.10.124:3500/edchatbot/getpanelproperties",
                "method": "GET"
            }
        ]
        agent = EDAgent(routes=api_metadata)
        edBot = agent.build()
        answer = edBot.invoke(input=question)['output']
        return JSONResponse(status_code=200,content={"answer":answer})
    except Exception as e:
        logging.error(str(e))
        return JSONResponse(status_code=500,content={'error':str(e)})