import os
import time
import shutil
import gc
from typing import List

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pypdf import PdfReader, PdfWriter
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from qdrant_client.models import VectorParams, Distance
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

load_dotenv()

app = FastAPI(title="Dealer RAG API")

QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.4)
llm_summary = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)



# First ensure collection exists
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

collections = [c.name for c in client.get_collections().collections]

if QDRANT_COLLECTION not in collections:
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    )

# Then connect via LangChain
vector_store = QdrantVectorStore(
    client=client,
    collection_name=QDRANT_COLLECTION,
    embedding=embeddings,
)
chat_history: List = []

class QuestionRequest(BaseModel):
    question: str

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    global vector_store
    global chat_history

    temp_dir = "temp_pages"
    os.makedirs(temp_dir, exist_ok=True)

    file_path = os.path.join(temp_dir, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # 🔥 Reset collection (demo mode)
    client.delete_collection(QDRANT_COLLECTION)

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    )

    # Recreate vector store
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
    )

    # Clear chat history
    chat_history.clear()

    # Configure OCR + Table extraction
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    doc_converted = converter.convert(file_path)
    md_content = doc_converted.document.export_to_markdown()

    langchain_doc = Document(
        page_content=md_content,
        metadata={"source": file.filename}
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )

    splits = text_splitter.split_documents([langchain_doc])

    vector_store.add_documents(splits)

    shutil.rmtree(temp_dir)

    return {"message": "PDF processed with OCR and stored successfully."}
# =========================
# ASK ENDPOINT
# =========================

@app.post("/ask")
async def ask_question(request: QuestionRequest):

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Rephrase the user question into standalone form if needed."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_prompt
    )

    retrieved_docs = history_aware_retriever.invoke({
        "input": request.question,
        "chat_history": chat_history
    })

    system_prompt = (
    "You are a document analysis assistant.\n\n"
    "Guidelines:\n"
    "1. Use only the provided document context.\n"
    "2. If the answer is not present, say: 'Not mentioned in the document.'\n"
    "3. Be concise but informative.\n"
    "4. When appropriate, summarize key points clearly.\n"
    "5. Do not add external knowledge.\n\n"
    "6. If the question is vague, infer intent from context but do not hallucinate."
    "Document Context:\n{context}"
    
)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    def stream_response():
        full_response = ""

        for chunk in question_answer_chain.stream({
            "input": request.question,
            "chat_history": chat_history,
            "context": retrieved_docs
        }):
            full_response += chunk
            yield chunk

        chat_history.extend([
            HumanMessage(content=request.question),
            AIMessage(content=full_response)
        ])

    return StreamingResponse(stream_response(), media_type="text/plain")