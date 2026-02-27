"""
Shared configuration — single source of truth for env vars and shared objects.
Both the API (chatbot_fast.py) and the worker (worker.py) import from here.
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore

load_dotenv()

# ── Env vars ─────────────────────────────────────────────────────────
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "dealer_docs")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@postgres:5432/rag")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/app/uploads")

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# ── Shared objects ───────────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.4)
llm_summary = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)

# ── Qdrant client + collection bootstrap ─────────────────────────────
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def ensure_collection():
    """Create the Qdrant collection if it doesn't exist yet."""
    existing = [c.name for c in qdrant_client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )


def get_vector_store() -> QdrantVectorStore:
    """Return a LangChain vector store bound to the shared Qdrant collection."""
    ensure_collection()
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
    )
