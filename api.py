from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents.router_agent import RouterAgent
from agents.knowledge_agent import KnowledgeAgent
import os
import shutil

app = FastAPI()
router = RouterAgent()

# Enable CORS so browser frontend can talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === API Models ===
class QueryInput(BaseModel):
    query: str
    channel: str = "chat"

# === API Routes ===

@app.post("/route")
def route_query(data: QueryInput):
    response = router.route(data.query, channel=data.channel)
    return {"response": response}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Save the uploaded file
    upload_dir = "uploaded_docs"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Rebuild KnowledgeAgent index
    try:
        KnowledgeAgent.rebuild_index_from_folder(upload_dir)
        return {"message": f"✅ {file.filename} uploaded and indexed."}
    except Exception as e:
        return {"message": f"❌ Failed to index: {str(e)}"}
        
@app.get("/files")
def list_uploaded_files():
    folder = "uploaded_docs"
    os.makedirs(folder, exist_ok=True)
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    return {"files": sorted(files)}