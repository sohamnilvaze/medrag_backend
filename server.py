import os
import shutil
import subprocess
import json
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
from langchain_community.vectorstores import Chroma
import uuid
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv       

app = FastAPI(title="MEDRAG - Real-Time Analysis")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PERSIST_DIR = "rag_slm/vectordb"
COLLECTION = "medrag"

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = OllamaLLM(model="gemma3:1b", temperature=0.2)

# Global VectorDB placeholder
vectordb = None

# Variable to check whether new document is uploaded or not
new_docs_uploaded = False

# --- Pydantic Models ---
class AskRequest(BaseModel):
    question: str
    session_id: str
    top_k: int = 5

class Chunk(BaseModel):
    content: str
    source: str
    test_name: Optional[str] = None
    score: float

class AskResponse(BaseModel):
    answer: str
    chunks: List[Chunk]

from datetime import datetime, timedelta

load_dotenv()

MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("❌ MONGO_URI environment variable not set")

mongo_client = MongoClient(MONGO_URI)

def auto_cleanup_expired_sessions():
    # Find sessions older than 24 hours
    expiry_time = datetime.utcnow() - timedelta(hours=24)
    expired_sessions = mongo_client["medrag"]["sessions"].find({"created_at": {"$lt": expiry_time}})
    
    for sess in expired_sessions:
        # Call the same logic as end_session(sess['session_id'])
        # This ensures your server doesn't keep data forever
        pass

# --- Routes ---
@app.get("/")
def root():
    return {"status": "MEDRAG backend running"} 

@app.post("/session/end/{session_id}")
async def end_session(session_id: str):
    try:
        # 1. Delete the specific collection from ChromaDB
        # We initialize it just to call delete_collection
        db_instance = Chroma(
            persist_directory=PERSIST_DIR, 
            embedding_function=embeddings, 
            collection_name=f"sess_{session_id}"
        )
        db_instance.delete_collection()
        print(f"🗑️ Deleted Chroma collection: sess_{session_id}")

        # 2. Find and delete file from MongoDB GridFS
        session_record = mongo_client["medrag"]["sessions"].find_one({"session_id": session_id})
        if session_record:
            import gridfs
            fs = gridfs.GridFS(mongo_client["medrag"])
            fs.delete(session_record["file_id"])
            
            # 3. Remove session record
            mongo_client["medrag"]["sessions"].delete_one({"session_id": session_id})
            print(f"🗑️ Deleted Mongo file and session record for: {session_id}")

        return {"status": "success", "message": f"Session {session_id} wiped."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # Always reload to ensure latest data is queried
    collection_name = f"sess_{req.session_id}"
    db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings, collection_name=collection_name)
    results = db.similarity_search_with_score(req.question, k=req.top_k)

    chunks = [
        Chunk(
            content=doc.page_content,
            source=doc.metadata.get("source", "unknown"),
            test_name=doc.metadata.get("test_name"),
            score=float(score)
        ) for doc, score in results
    ]

    context = "\n\n".join([c.content for c in chunks])
    prompt = f"Using this context, answer the question accurately.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{req.question}\n\nANSWER:"
    
    answer = llm.invoke(prompt).strip()
    return AskResponse(answer=answer, chunks=chunks)

# @app.post("/upload")
# async def upload_report(file: UploadFile = File(...)):
#     temp_path = f"temp_{file.filename}"
#     with open(temp_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
    
#     try:
#         # 1. OCR -> 2. Parse -> 3. Ingest
#         subprocess.run(["python", "rag_slm/ocr.py", "--input", temp_path, "--out", "ocr_out.json"], check=True)
#         subprocess.run(["python", "rag_slm/parse.py", "--input", "ocr_out.json", "--out", "structured.json"], check=True)
#         # subprocess.run(["python", "ingest.py", "--json", "structured.json"], check=True)

#         with open("ocr_out.json", "r") as f:
#             ocr_result = json.load(f)
#             mongo_file_id = ocr_result.get("mongo_file_id")

#         # 4. Map the Session to the File in MongoDB
#         # THIS IS THE NEW PART
#         session_data = {
#             "session_id": session_id,
#             "file_id": mongo_file_id, 
#             "filename": file.filename,
#             "created_at": datetime.utcnow()
#         }
#         # Assuming 'db' is your pymongo database instance
#         mongo_client["medrag"]["sessions"].insert_one(session_data)

#         subprocess.run(["python", "rag_slm/ingest.py", "--mongo"], check=True)

#         # 4. Read the parsed result for Real-Time UI update
#         with open("structured.json", "r") as f:
#             analysis_data = json.load(f)
        
#         return {
#             "status": "success",
#             "session_id": session_id,    
#             "analysis": analysis_data 
#         }
#     except Exception as e:
#         return {"status": "error", "message": str(e)}
#     finally:
#         if os.path.exists(temp_path): os.remove(temp_path)


@app.post("/upload")
async def upload_report(file: UploadFile = File(...)):
    # 1. DEFINE IT FIRST: Generate the ID immediately
    session_id = str(uuid.uuid4()) 
    
    temp_path = f"temp_{session_id}_{file.filename}" # Good practice: make temp file unique too
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 2. Pass the ID or use it in the subprocess calls if needed
        # We use it specifically to track the results
        subprocess.run(["python", "rag_slm/ocr.py", "--input", temp_path, "--out", f"ocr_{session_id}.json"], check=True)
        subprocess.run(["python", "rag_slm/parse.py", "--input", f"ocr_{session_id}.json", "--out", f"structured_{session_id}.json"], check=True)
        
        # 3. Read the OCR result to get the mongo_file_id
        with open(f"ocr_{session_id}.json", "r") as f:
            ocr_result = json.load(f)
            mongo_report_id = ocr_result.get("ocr_doc_id")

        # 4. Use 'session_id' here - it is now defined!
        session_data = {
            "session_id": session_id,
            "file_id": mongo_file_id, 
            "filename": file.filename,
            "created_at": datetime.utcnow()
        }
        # Ensure your mongo_client is defined globally
        mongo_client["medrag"]["sessions"].insert_one(session_data)

        # 5. Run Ingest with the collection name tied to the session_id
        subprocess.run([
            "python", "rag_slm/ingest.py", 
            "--mongo", 
            "--collection", f"sess_{session_id}",
            "--report_id", mongo_report_id  # NEW ARGUMENT
        ], check=True)

        with open(f"structured_{session_id}.json", "r") as f:
            analysis_data = json.load(f)
        
        return {
            "status": "success",    
            "session_id": session_id,
            "mongo report id": mongo_report_id, 
            "analysis": analysis_data
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        # Cleanup temp files
        for f in [temp_path, f"ocr_{session_id}.json", f"structured_{session_id}.json"]:
            if os.path.exists(f): os.remove(f)