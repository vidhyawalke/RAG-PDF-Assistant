"""
================================================================================
FastAPI Backend Application Module
--------------------------------------------------------------------------------
References & Documentation Sources:
- FastAPI Official Tutorial: https://fastapi.tiangolo.com/tutorial/first-steps/
- FastAPI Request Files & Uploads: https://fastapi.tiangolo.com/tutorial/request-files/
- FastAPI CORS Middleware Guide: https://fastapi.tiangolo.com/tutorial/cors/
- Pydantic BaseModel Data Validation: https://docs.pydantic.dev/latest/
================================================================================
"""

import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from backend.config import settings
from backend.rag_chain import rag_pipeline

# Initialize FastAPI Application
# Source: https://fastapi.tiangolo.com/tutorial/first-steps/
app = FastAPI(
    title="RAG PDF Assistant API",
    description="Production-ready Retrieval-Augmented Generation API for PDF document Q&A",
    version="1.0.0"
)

# Enable CORS middleware to allow cross-origin requests from Streamlit UI or web clients
# Source: https://fastapi.tiangolo.com/tutorial/cors/
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# Pydantic Schemas for Data Request & Response Validation
# Source: https://docs.pydantic.dev/latest/concepts/models/
# ------------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    question: str = Field(..., example="What is the primary role described in this document?")
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
    api_key_configured: bool


# ------------------------------------------------------------------------------
# REST API Endpoint Route Definitions
# Source: https://fastapi.tiangolo.com/tutorial/
# ------------------------------------------------------------------------------

@app.get("/", tags=["General"])
def read_root():
    """Root Endpoint returning API metadata."""
    return {
        "message": "Welcome to RAG PDF Assistant API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """
    Health Check Endpoint verifying API state, vector index, and API keys.
    Source Pattern: https://fastapi.tiangolo.com/tutorial/
    """
    vector_store_loaded = len(rag_pipeline.chunks) > 0 or rag_pipeline.faiss_index is not None or rag_pipeline.tfidf_matrix is not None
    api_key_configured = bool(settings.GOOGLE_API_KEY and settings.GOOGLE_API_KEY != "your_gemini_api_key_here")
    return {
        "status": "healthy",
        "vector_store_loaded": vector_store_loaded,
        "api_key_configured": api_key_configured
    }


@app.post("/upload", response_model=UploadResponse, tags=["Data Preparation"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Data Preparation Endpoint:
    Upload a PDF file, extract text, build overlapping chunks, and store in FAISS vector store.
    Source (FastAPI UploadFile): https://fastapi.tiangolo.com/tutorial/request-files/
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files (.pdf) are supported."
        )

    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    
    # Validate file path to prevent path traversal attacks
    if not settings.validate_file_path(file_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path."
        )
    
    try:
        # Save file to disk using standard shutil stream
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = rag_pipeline.process_pdf(file_path)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process PDF file: {str(e)}"
        )
    finally:
        # Clean up uploaded file after processing to prevent disk exhaustion
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as cleanup_error:
            # Log but do not fail the request if cleanup fails
            print(f"Warning: Could not clean up {file_path}: {cleanup_error}")


@app.post("/ask", response_model=QuestionResponse, tags=["Model Integration"])
def ask_question(request: QuestionRequest):
    """
    Model Integration Endpoint:
    Query the PDF vector index, retrieve top-k context passages, and invoke Gemini LLM API.
    Source Pattern: Standard REST POST Handler.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question prompt cannot be empty."
        )

    try:
        response = rag_pipeline.answer_question(request.question, top_k=request.top_k)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while answering question: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    # Launch Uvicorn Development Server
    # Source: https://www.uvicorn.org/
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=True)
