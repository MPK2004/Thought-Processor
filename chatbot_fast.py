import os
import uuid

from fastapi import FastAPI, UploadFile, File, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

from config import (
    llm, get_vector_store, UPLOAD_DIR, ensure_collection,
    qdrant_client, QDRANT_COLLECTION, REDIS_URL, DATABASE_URL,
)
from database import create_tables, get_db, engine
from models import IngestionJob, JobStatus
from redis_client import (
    get_cached_response,
    set_cached_response,
    get_chat_history,
    append_chat_history,
    get_job_queue,
    get_redis,
)
from logger import get_logger, request_id_var, new_request_id

log = get_logger("api")

app = FastAPI(title="Dealer RAG API")
app.mount("/static", StaticFiles(directory="static"), name="static")

os.makedirs(UPLOAD_DIR, exist_ok=True)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID", new_request_id())
        request_id_var.set(rid)
        log.info(f"{request.method} {request.url.path}")
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        log.info(f"{request.method} {request.url.path} -> {response.status_code}")
        return response


app.add_middleware(RequestIDMiddleware)


@app.on_event("startup")
def startup():
    create_tables()
    ensure_collection()
    log.info("API started, tables and collection ready")


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    """Liveness probe - confirms the process is running."""
    return {"status": "ok"}


@app.get("/ready")
def readiness():
    """Readiness probe - checks Redis, Postgres, and Qdrant connectivity."""
    checks = {}

    try:
        r = get_redis()
        r.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"

    try:
        conn = engine.connect()
        conn.close()
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {e}"

    try:
        qdrant_client.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:
        checks["qdrant"] = f"error: {e}"

    all_ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        content={"status": "ready" if all_ok else "degraded", "checks": checks},
        status_code=200 if all_ok else 503,
    )

class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    job_id = str(uuid.uuid4())
    log.info(f"Upload received: {file.filename}, job_id={job_id}")

    file_ext = os.path.splitext(file.filename)[1]
    safe_filename = f"{job_id}{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    job = IngestionJob(
        id=job_id,
        filename=file.filename,
        status=JobStatus.PENDING,
    )
    db.add(job)
    db.commit()
    log.info(f"Job {job_id} created as PENDING")

    queue = get_job_queue()
    queue.enqueue(
        "worker.process_ingestion",
        job_id,
        file_path,
        file.filename,
        job_timeout="30m",
    )
    log.info(f"Job {job_id} enqueued to Redis")

    return JSONResponse(
        content={"job_id": job_id, "status": "PENDING"},
        status_code=202,
    )


@app.get("/job/{job_id}")
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
    if not job:
        return JSONResponse(
            content={"error": "Job not found"}, status_code=404
        )
    return JSONResponse(content=job.to_dict())


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    session_id = request.session_id
    log.info(f"Question from session={session_id}: {request.question[:80]}")

    cached = get_cached_response(request.question)
    if cached:
        log.info("Cache hit, returning cached response")
        return StreamingResponse(
            iter([cached]), media_type="text/plain"
        )

    history_data = get_chat_history(session_id)
    chat_history = []
    for entry in history_data:
        if entry["role"] == "human":
            chat_history.append(HumanMessage(content=entry["content"]))
        else:
            chat_history.append(AIMessage(content=entry["content"]))

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
    log.info(f"Retrieved {len(retrieved_docs)} docs for context")

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

    def stream_response():
        full_response = ""

        for chunk in question_answer_chain.stream({
            "input": request.question,
            "chat_history": chat_history,
            "context": retrieved_docs,
        }):
            full_response += chunk
            yield chunk

        set_cached_response(request.question, full_response)
        append_chat_history(session_id, request.question, full_response)
        log.info(f"Response streamed, length={len(full_response)}")

    return StreamingResponse(stream_response(), media_type="text/plain")
