"""
================================================================================
RAG Engine & Vector Search Pipeline
--------------------------------------------------------------------------------
References & Documentation Sources:
- Google Gemini REST API Guide: https://ai.google.dev/api/generate-content
- PyPDF PDF Text Extraction: https://pypdf.readthedocs.io/en/latest/user/reading-pdf.html
- Meta FAISS Vector Indexing & Search: https://github.com/facebookresearch/faiss/wiki/Getting-started
- SentenceTransformers Embeddings Guide: https://www.sbert.net/docs/pretrained_models.html
- Scikit-Learn TF-IDF Vectorizer API: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
================================================================================
"""

import os
# Workaround for Google Protobuf descriptor compatibility on Windows
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import time
import logging
import requests
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader
from backend.config import settings

# Configure logger output formatting
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_chain")

# Enhanced System Prompt Template requiring structured Markdown response
SYSTEM_PROMPT_TEMPLATE = """You are an expert AI assistant answering questions based strictly on the provided context retrieved from a PDF document.

Context:
{context}

Question: {question}

Instructions for Structured Output:
1. Format your response into clean sections with bold headers and bullet points.
2. Synthesize key concepts concisely and highlight core terms in bold.
3. Every main point MUST include its source page citation in brackets, e.g., [Page X].
4. Do NOT output unformatted plain text blocks.
5. If the context does not contain enough information, state: "I cannot find the answer to this question in the uploaded document."

Structured Response:"""


class DocumentChunk:
    """
    Data Structure representing a single text chunk with preserved page metadata.
    """
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.page_content = content
        self.metadata = metadata


class RAGPipeline:
    """
    Retrieval-Augmented Generation Pipeline combining Data Preparation,
    FAISS Vector Indexing, and Google Gemini REST API Generation.
    """
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.embedding_model = None
        self.faiss_index = None
        self._init_vector_engine()

    def _init_vector_engine(self):
        """
        Initialize SentenceTransformers embedding model with fallback to Scikit-Learn TF-IDF.
        Uses timeout to prevent indefinite hangs on model loading.
        Source: https://www.sbert.net/
        """
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
            import signal
            
            has_alarm = hasattr(signal, "SIGALRM") and hasattr(signal, "alarm")
            if has_alarm:
                def timeout_handler(signum, frame):
                    raise TimeoutError("Model loading timed out after 60 seconds")
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)
            
            try:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("SentenceTransformer initialized successfully.")
            finally:
                if has_alarm:
                    signal.alarm(0)
        except Exception as e:
            logger.warning(f"SentenceTransformer unavailable ({e}). Using Scikit-Learn TF-IDF vector engine.")
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))

    def load_and_split_pdf(self, file_path: str) -> List[DocumentChunk]:
        """
        Data Preparation Phase: Extract page text and split into overlapping chunks.
        Validates file path before processing.
        Source (PyPDF): https://pypdf.readthedocs.io/en/latest/user/reading-pdf.html
        """
        # Validate file path to prevent path traversal attacks
        if not settings.validate_file_path(file_path):
            raise ValueError(f"Invalid file path: {file_path}")
        
        logger.info(f"Loading PDF file: {file_path}")
        reader = PdfReader(file_path)
        chunks: List[DocumentChunk] = []
        filename = Path(file_path).name

        chunk_counter = 0
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            page_text = " ".join(page_text.split())
            
            if not page_text.strip():
                continue

            page_chunks = self._recursive_split_text(
                page_text, 
                chunk_size=settings.CHUNK_SIZE, 
                overlap=settings.CHUNK_OVERLAP
            )

            for text_snippet in page_chunks:
                metadata = {
                    "chunk_id": chunk_counter,
                    "source_filename": filename,
                    "page_number": page_idx + 1
                }
                chunks.append(DocumentChunk(content=text_snippet, metadata=metadata))
                chunk_counter += 1

        logger.info(f"Parsed {len(reader.pages)} pages into {len(chunks)} text chunks.")
        return chunks

    def _recursive_split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Splits text into chunks of specified maximum character size with overlap.
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            if end < len(text):
                last_space = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
                if last_space > start + (chunk_size // 2):
                    end = last_space + 1

            chunk_str = text[start:end].strip()
            if chunk_str:
                chunks.append(chunk_str)

            if end >= len(text):
                break

            start = end - overlap
        return chunks

    def process_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Build FAISS inner-product vector index or TF-IDF matrix from PDF chunks.
        Source (FAISS Tutorial): https://github.com/facebookresearch/faiss/wiki/Getting-started
        """
        start_time = time.time()
        self.chunks = self.load_and_split_pdf(file_path)

        if not self.chunks:
            raise ValueError("No extractable text found in PDF document.")

        corpus_texts = [c.page_content for c in self.chunks]

        if self.embedding_model is not None:
            try:
                import faiss
                embeddings = self.embedding_model.encode(corpus_texts, convert_to_numpy=True, show_progress_bar=False)
                faiss.normalize_L2(embeddings)
                
                dimension = embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dimension)
                self.faiss_index.add(embeddings)
                logger.info(f"FAISS vector index created with {self.faiss_index.ntotal} vectors.")
            except Exception as e:
                logger.warning(f"FAISS indexing failed ({e}). Falling back to TF-IDF matrix.")
                self.faiss_index = None

        if self.faiss_index is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)
            logger.info("Scikit-Learn TF-IDF vector matrix created successfully.")

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "status": "success",
            "filename": Path(file_path).name,
            "total_chunks": len(self.chunks),
            "ingestion_time_ms": elapsed_ms
        }

    def _call_gemini_api(self, prompt: str) -> str:
        """
        Generate structured answer using Google Gemini REST API.
        Uses only documented, stable models with proper error handling.
        Source: https://ai.google.dev/api/generate-content
        """
        api_key = settings.GOOGLE_API_KEY
        if not api_key or api_key == "your_gemini_api_key_here":
            return ""

        # Use stable and latest Gemini models
        models_to_try = [
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro",
            "gemini-1.0-pro"
        ]

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1000
            }
        }

        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            try:
                res = requests.post(url, json=payload, headers=headers, timeout=15)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        for part in parts:
                            if "text" in part and part["text"].strip():
                                text_ans = part["text"].strip()
                                logger.info(f"Successfully generated response using model: {model_name}")
                                return text_ans
                elif res.status_code == 429:
                    logger.warning(f"Rate limit 429 on model {model_name}. Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    logger.warning(f"Model {model_name} HTTP {res.status_code}: {res.text}")
            except Exception as e:
                logger.error(f"Exception calling model {model_name}: {e}")

        return ""

    def answer_question(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Perform similarity search, retrieve top-k context passages, and generate answer via Gemini REST API.
        """
        start_time = time.time()

        if not self.chunks:
            return {
                "query": query,
                "answer": "No PDF document has been processed yet. Please upload a PDF first.",
                "sources": [],
                "execution_time_ms": round((time.time() - start_time) * 1000, 2)
            }

        sources = []
        context_snippets = []

        # Vector Retrieval Phase
        if self.faiss_index is not None and self.embedding_model is not None:
            import faiss
            query_vec = self.embedding_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_vec)
            scores, indices = self.faiss_index.search(query_vec, top_k)
            
            for idx, score in zip(indices[0], scores[0]):
                if 0 <= idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    page_num = chunk.metadata.get("page_number", 1)
                    sources.append({
                        "page": page_num,
                        "content": chunk.page_content,
                        "similarity_score": round(float(score), 4)
                    })
                    context_snippets.append(f"[Page {page_num}]: {chunk.page_content}")
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            query_vec = self.vectorizer.transform([query])
            sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            top_indices = np.argsort(sim_scores)[::-1][:top_k]

            for idx in top_indices:
                score = sim_scores[idx]
                chunk = self.chunks[idx]
                page_num = chunk.metadata.get("page_number", 1)
                sources.append({
                    "page": page_num,
                    "content": chunk.page_content,
                    "similarity_score": round(float(score), 4)
                })
                context_snippets.append(f"[Page {page_num}]: {chunk.page_content}")

        context_str = "\n\n".join(context_snippets)
        prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context_str, question=query)

        # Call Gemini REST API
        answer = self._call_gemini_api(prompt)

        # Beautifully structured fallback if API call returned empty or failed
        if not answer:
            formatted_fallback = "**Summary of Relevant Document Passages:**\n\n"
            for src in sources:
                formatted_fallback += f"* **Page {src['page']}**: {src['content']}\n\n"
            answer = formatted_fallback

        execution_time_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "execution_time_ms": execution_time_ms
        }

# Global singleton RAG pipeline instance
rag_pipeline = RAGPipeline()
