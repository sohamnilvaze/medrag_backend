import os
import json
import argparse
from dotenv import load_dotenv
from typing import Dict, Any, List
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

from utils_docs import build_documents
from pymongo import MongoClient

load_dotenv()
MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("❌ MONGO_URI environment variable not set")

MONGO_DB = "medrag"
MONGO_COLLECTION = "ocr_reports"

def load_reports_from_mongo(limit: int | None = None) -> List[Dict[str, Any]]:
    client = MongoClient(MONGO_URI)
    col = client[MONGO_DB][MONGO_COLLECTION]

    # cursor = col.find({"status": "OCR_COMPLETED"})
    cursor = col.find({})
    if limit:
        cursor = cursor.limit(limit)

    reports = list(cursor)
    client.close()

    if not reports:
        raise RuntimeError("❌ No OCR reports found in MongoDB")

    return reports



def load_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_json_inputs(path: str) -> List[str]:
    if os.path.isfile(path) and path.lower().endswith(".json"):
        return [path]

    if os.path.isdir(path):
        files = []
        for root, _, fs in os.walk(path):
            for fn in fs:
                if fn.lower().endswith(".json"):
                    files.append(os.path.join(root, fn))
        return sorted(files)

    raise FileNotFoundError(f"Invalid JSON path: {path}")


def chunk_documents(docs: List[Document], size: int, overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--json", required=True, help="Structured JSON file or directory")
    ap.add_argument("--json", help="Structured JSON file or directory")
    ap.add_argument("--persist_dir", default="rag_slm/vectordb")
    ap.add_argument("--collection", required=True, help="Session-specific collection name")
    # ap.add_argument("--collection", default="medrag")
    ap.add_argument("--embed_model", default="nomic-embed-text")
    ap.add_argument("--chunk_size", type=int, default=900)
    ap.add_argument("--chunk_overlap", type=int, default=120)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--mongo", action="store_true", help="Load OCR JSONs from MongoDB")
    ap.add_argument("--mongo_limit", type=int, default=None, help="Limit MongoDB docs")
    ap.add_argument("--report_id", help="The specific MongoDB Object ID to ingest")
    args = ap.parse_args()
    if args.mongo and args.json:
        raise RuntimeError("❌ Use either --mongo or --json, not both")

    if not args.mongo and not args.json:
        raise RuntimeError("❌ Provide either --mongo or --json")


    if args.reset and os.path.exists(args.persist_dir):
        import shutil
        shutil.rmtree(args.persist_dir, ignore_errors=True)
        print(f"🧹 Deleted old vectordb: {args.persist_dir}")

    # json_files = discover_json_inputs(args.json)
    # print(f"📄 Found {len(json_files)} JSON file(s)")

    all_docs: List[Document] = []

    # for jf in json_files:
    #     report = load_json_file(jf)
    #     report_id = report.get("source") or jf
    #     docs = build_documents(report, report_id=report_id)
    #     all_docs.extend(docs)
    #     print(f"  ✅ {os.path.basename(jf)} → {len(docs)} docs")

    if args.mongo:
        print("📦 Loading OCR reports from MongoDB...")
        client = MongoClient(MONGO_URI)
        col = client[MONGO_DB][MONGO_COLLECTION]

        if args.report_id:
            from bson import ObjectId
            query = {"_id": ObjectId(args.report_id)}
        else:
            query = {}
        
        reports = list(col.find(query))
        client.close()

        if not reports:
            raise RuntimeError(f"❌ No OCR report found for ID: {args.report_id}")

        for r in reports:
            r["_id"] = str(r["_id"])
            docs = build_documents(r, report_id=r["_id"])
            all_docs.extend(docs)

        #reports = load_reports_from_mongo(args.mongo_limit)

        # for r in reports:
        #     r["_id"] = str(r["_id"])
        #     report_id = r["_id"]
        #     docs = build_documents(r, report_id=report_id)
        #     all_docs.extend(docs)


    else:
        json_files = discover_json_inputs(args.json)
        print(f"📄 Found {len(json_files)} JSON file(s)")

        for jf in json_files:
            report = load_json_file(jf)
            report_id = report.get("mongo_file_id") or jf
            docs = build_documents(report, report_id=report_id)
            all_docs.extend(docs)

        if not all_docs:
            raise RuntimeError("❌ No documents generated. Check utils_docs.build_documents().")

    chunks = chunk_documents(all_docs, args.chunk_size, args.chunk_overlap)
    print(f"✂️  Chunks created: {len(chunks)}")

    embeddings = OllamaEmbeddings(model=args.embed_model)

    Chroma(
        collection_name=args.collection,
        persist_directory=args.persist_dir,
        embedding_function=embeddings,
    ).add_documents(chunks)

    print("\n✅ INGEST COMPLETE")
    print(f"📦 Chunks stored: {len(chunks)}")
    print(f"📚 Collection: {args.collection}")
    print(f"💾 Persist dir: {args.persist_dir}")


if __name__ == "__main__":
    main()
    