"""
Dealer RAG API — Production-grade FastAPI application.

Refactored from the original chatbot_fast.py:
- Upload endpoint is now async (returns job_id, processes via worker)
- Ask endpoint has Redis caching + Redis-backed chat history
- No global mutable state
- No collection deletion (multi-document support)
- LangSmith tracing preserved via load_dotenv()
"""

import os
import uuid

from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

from config import llm, get_vector_store, UPLOAD_DIR, ensure_collection
from database import create_tables, get_db
from models import IngestionJob, JobStatus
from redis_client import (
    get_cached_response,
    set_cached_response,
    get_chat_history,
    append_chat_history,
    get_job_queue,
)

# ── App setup ────────────────────────────────────────────────────────
app = FastAPI(title="Dealer RAG API")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.on_event("startup")
def startup():
    """Run once when the API starts."""
    create_tables()
    ensure_collection()


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


# ── Request models ───────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"


# =============================================================================
# UPLOAD ENDPOINT — Async job submission
# =============================================================================

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Save uploaded PDF to the shared volume and enqueue an ingestion job.
    Returns immediately with a job_id for polling.
    """
    # 1. Generate unique job ID
    job_id = str(uuid.uuid4())

    # 2. Save file to shared volume (accessible by worker container)
    file_ext = os.path.splitext(file.filename)[1]
    safe_filename = f"{job_id}{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # 3. Create job record in PostgreSQL
    job = IngestionJob(
        id=job_id,
        filename=file.filename,
        status=JobStatus.PENDING,
    )
    db.add(job)
    db.commit()

    # 4. Enqueue ingestion task to Redis queue
    queue = get_job_queue()
    queue.enqueue(
        "worker.process_ingestion",
        job_id,
        file_path,
        file.filename,
        job_timeout="30m",  # OCR can be slow for large PDFs
    )

    return JSONResponse(
        content={"job_id": job_id, "status": "PENDING"},
        status_code=202,
    )


# =============================================================================
# JOB STATUS ENDPOINT — Poll for ingestion progress
# =============================================================================

@app.get("/job/{job_id}")
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Poll the status of an ingestion job."""
    job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
    if not job:
        return JSONResponse(
            content={"error": "Job not found"}, status_code=404
        )
    return JSONResponse(content=job.to_dict())


# =============================================================================
# ASK ENDPOINT — With Redis caching + Redis-backed chat history
# =============================================================================

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Answer a question using retrieved document context.
    - Checks Redis cache first
    - Uses Redis-backed per-session chat history
    - Streams the response
    """
    session_id = request.session_id

    # 1. Check Redis cache
    cached = get_cached_response(request.question)
    if cached:
        return StreamingResponse(
            iter([cached]), media_type="text/plain"
        )

    # 2. Rebuild chat history from Redis
    history_data = get_chat_history(session_id)
    chat_history = []
    for entry in history_data:
        if entry["role"] == "human":
            chat_history.append(HumanMessage(content=entry["content"]))
        else:
            chat_history.append(AIMessage(content=entry["content"]))

    # 3. Retrieve relevant documents (existing logic preserved)
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    contextualize_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Rephrase the user question into standalone form if needed.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    retrieved_docs = history_aware_retriever.invoke({
        "input": request.question,
        "chat_history": chat_history,
    })

    # 4. Generate answer (existing logic preserved)
    system_prompt = (
        "You are a document analysis assistant.\n\n"
        "Guidelines:\n"
        "1. Use only the provided document context.\n"
        "2. If the answer is not present, say: 'Not mentioned in the document.'\n"
        "3. Be concise but informative.\n"
        "4. When appropriate, summarize key points clearly.\n"
        "5. Do not add external knowledge.\n"
        "6. If the question is vague, infer intent from context but do not hallucinate.\n\n"
        "Document Context:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 5. Stream response + cache + save history
    def stream_response():
        full_response = ""

        for chunk in question_answer_chain.stream({
            "input": request.question,
            "chat_history": chat_history,
            "context": retrieved_docs,
        }):
            full_response += chunk
            yield chunk

        # Cache the full response
        set_cached_response(request.question, full_response)

        # Persist chat history to Redis
        append_chat_history(session_id, request.question, full_response)

    return StreamingResponse(stream_response(), media_type="text/plain")
