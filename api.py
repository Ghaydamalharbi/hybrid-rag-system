from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from datetime import datetime
import csv
import os
import gc
import traceback

from rag.ingestion.pdf_loader import load_pdf
from rag.processing.text_splitter import split_documents
from rag.retrieval.vector_store import create_vector_store
from rag.engine.chat_engine import answer_question

app = FastAPI(title="RAG SaaS API")

vectorstore = None

# =========================
# Request Schema
# =========================
class QuestionRequest(BaseModel):
    question: str


# =========================
# CSV Logger
# =========================
LOG_FILE = "metrics.csv"

def log_metrics(question, result):
    try:
        file_exists = os.path.isfile(LOG_FILE)

        with open(LOG_FILE, "a", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "question",
                    "model",
                    "latency",
                    "hallucination",
                    "context_length",
                    "docs_used"
                ])

            perf = result.get("performance", {})

            writer.writerow([
                datetime.now().isoformat(),
                question,
                result.get("model"),
                perf.get("latency"),
                perf.get("hallucination"),
                perf.get("context_length"),
                perf.get("docs_used")
            ])
    except Exception as e:
        print("[LOG ERROR]:", e)


# =========================
# Upload PDF
# =========================
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore

    try:
        print("\n=== NEW UPLOAD ===")

        # 🔥 تفريغ الذاكرة
        vectorstore = None
        gc.collect()

        # =========================
        # Save file
        # =========================
        file_path = "temp.pdf"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        print("[DEBUG] File saved")

        # =========================
        # Load PDF
        # =========================
        docs = load_pdf(file_path)
        print(f"[DEBUG] Pages loaded: {len(docs)}")

        if not docs:
            return {"error": "PDF NOT LOADED"}

        # =========================
        # Split
        # =========================
        chunks = split_documents(docs)
        print(f"[DEBUG] Chunks created: {len(chunks)}")

        if not chunks:
            return {"error": "NO CHUNKS CREATED"}

        # =========================
        # Vector Store
        # =========================
        vectorstore = create_vector_store(chunks)

        if not vectorstore:
            return {"error": "VECTOR STORE FAILED"}

        print("[DEBUG] Vector store ready")

        return {
            "status": "PDF processed",
            "pages": len(docs),
            "chunks": len(chunks)
        }

    except Exception as e:
        print("[UPLOAD ERROR]:")
        traceback.print_exc()

        return {
            "error": "UPLOAD FAILED",
            "details": str(e)
        }


# =========================
# Ask Question
# =========================
@app.post("/ask")
async def ask_question_api(req: QuestionRequest):
    global vectorstore

    try:
        if not vectorstore:
            return {"error": "Upload PDF first"}

        print("\n=== NEW QUESTION ===")
        print("Q:", req.question)

        result = answer_question(vectorstore, req.question)

        if isinstance(result, dict):
            log_metrics(req.question, result)

        return result

    except Exception as e:
        print("[ASK ERROR]:")
        traceback.print_exc()

        return {
            "error": "ASK FAILED",
            "details": str(e)
        }


# =========================
# Health Check
# =========================
@app.get("/")
def home():
    return {"message": "RAG API is running"}