from pymongo import MongoClient
import gridfs
import os
from datetime import datetime
from urllib.parse import quote_plus

username = quote_plus("sohamnilvaze739")
password = quote_plus("Soham@2004")

# MONGO_URI = (
#     f"mongodb+srv://{username}:{password}"
#     "@med-rag.hnapijj.mongodb.net/?appName=Med-Rag"
# )

MONGO_URI = os.getenv("MONGO_URI")

_client = MongoClient(MONGO_URI)

DB_NAME = "medrag"

_db = _client[DB_NAME]
_fs = gridfs.GridFS(_db)

def save_file_to_mongo(
    file_path: str,
    content_type: str,
    metadata: dict | None = None
) -> str:
    if metadata is None:
        metadata = {}

    metadata["content_type"] = content_type

    with open(file_path, "rb") as f:
        file_id = _fs.put(
            f,
            filename=os.path.basename(file_path),
            contentType=content_type,
            metadata=metadata
        )

    return str(file_id)

def save_ocr_json(report_json: dict) -> str:
    report_json["created_at"] = datetime.utcnow()
    result = _db.ocr_reports.insert_one(report_json)
    return str(result.inserted_id)

def get_db():
    return _db
