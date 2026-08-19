"""
FastAPI application defining REST API endpoints for RAG pipeline.
"""

import os
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.utils.helpers import settings, get_logger, validate_file_path
from src.ingestion.loader import PDFLoader
from src.chunking.chunker import TextChunker
from src.embeddings.embedder import Embedder
from src.vectordb.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.prompts.prompt_templates import format_prompt
from src.llm.llm_client import LLMClient

logger = get_logger("api.routes")

# Initialize Pipeline Components
loader = PDFLoader(validate_paths=True)
chunker = TextChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
embedder = Embedder()
vector_store = VectorStore()
retriever = Retriever(embedder=embedder, vector_store=vector_store)
llm_client = LLMClient()

# Initialize FastAPI App
app = FastAPI(
    title=settings.APP_NAME,
    description="Production grade Retrieval-Augmented Generation API for PDF document Q&A",
    version=settings.APP_VERSION
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request and Response Models
class QuestionRequest(BaseModel):
    question: str = Field(..., example="What are the key conclusions in this report?")
    top_k: Optional[int] = Field(default=3, description="Number of context passages to retrieve")


class SourceCitation(BaseModel):
    page: int
    content: str
    similarity_score: float


class QuestionResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceCitation]
    execution_time_ms: float


class UploadResponse(BaseModel):
    status: str
    filename: str
    total_chunks: int
    ingestion_time_ms: float


class HealthResponse(BaseModel):
    status: str
    vector_store_loaded: bool
    total_indexed_chunks: int
    api_key_configured: bool


@app.get("/", tags=["General"])
def read_root():
    """Root endpoint returning API metadata."""
    return {
        "message": f"Welcome to {settings.APP_NAME} API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """Health check endpoint returning system status and component readiness."""
    total_chunks = vector_store.total_vectors()
    return {
        "status": "healthy",
        "vector_store_loaded": total_chunks > 0,
        "total_indexed_chunks": total_chunks,
        "api_key_configured": llm_client.is_configured()
    }


@app.post("/upload", response_model=UploadResponse, tags=["Ingestion"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Ingest a PDF document, extract text, partition into chunks, and store vector embeddings.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files (.pdf) are supported."
        )

    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)

    if not validate_file_path(file_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or unauthorized file path."
        )

    start_time = time.time()
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Ingestion phase
        pages = loader.load(file_path)
        if not pages:
            raise ValueError("No extractable text found in PDF document.")

        # Chunking phase
        chunks = chunker.chunk_pages(pages)
        if not chunks:
            raise ValueError("Failed to create text chunks from document.")

        # Embedding & Vector Database phase
        corpus_texts = [c.page_content for c in chunks]
        embeddings = embedder.embed_documents(corpus_texts)
        vector_store.add_documents(chunks, embeddings)

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Successfully processed {file.filename} in {elapsed_ms} ms")

        return {
            "status": "success",
            "filename": file.filename,
            "total_chunks": len(chunks),
            "ingestion_time_ms": elapsed_ms
        }
    except Exception as e:
        logger.error(f"Error processing PDF upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process PDF file: {str(e)}"
        )
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as cleanup_err:
            logger.warning(f"Could not remove temporary file {file_path}: {cleanup_err}")


@app.post("/ask", response_model=QuestionResponse, tags=["Retrieval & Generation"])
def ask_question(request: QuestionRequest):
    """
    Retrieve top context passages for a query and synthesize grounded answer with citations.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty."
        )

    start_time = time.time()

    if vector_store.total_vectors() == 0:
        return {
            "query": request.question,
            "answer": "No PDF document has been processed yet. Please upload a PDF first.",
            "sources": [],
            "execution_time_ms": round((time.time() - start_time) * 1000, 2)
        }

    try:
        # Retrieval phase
        retrieval_res = retriever.retrieve(request.question, top_k=request.top_k)
        context = retrieval_res["context"]
        sources = retrieval_res["sources"]

        # Prompt formatting phase
        prompt = format_prompt(context=context, question=request.question)

        # Generation phase
        answer = llm_client.generate(prompt)
        if not answer:
            answer = llm_client.generate_fallback_summary(sources)

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "query": request.question,
            "answer": answer,
            "sources": sources,
            "execution_time_ms": elapsed_ms
        }
    except Exception as e:
        logger.error(f"Error during question answering: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while answering question: {str(e)}"
        )
