"""
Vector store implementation using FAISS with fallback to sparse TF-IDF cosine matrix.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from src.chunking.chunker import DocumentChunk
from src.utils.helpers import settings, get_logger

logger = get_logger("vectordb.vector_store")


class VectorStore:
    """Manages document vector indexing and nearest neighbor similarity queries."""

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or settings.VECTOR_STORE_DIR
        self.chunks: List[DocumentChunk] = []
        self.faiss_index = None
        self.tfidf_matrix = None

    def clear(self):
        """Reset index and stored chunks."""
        self.chunks = []
        self.faiss_index = None
        self.tfidf_matrix = None
        logger.info("Vector store reset and cleared.")

    def total_vectors(self) -> int:
        """Return the count of indexed document chunks."""
        return len(self.chunks)

    def add_documents(self, chunks: List[DocumentChunk], embeddings_or_matrix: Any):
        """
        Populate vector database with new document chunks and corresponding embeddings.
        """
        self.chunks = chunks

        if not chunks:
            self.clear()
            return

        # Check if FAISS dense embeddings were provided
        if isinstance(embeddings_or_matrix, np.ndarray) and embeddings_or_matrix.size > 0:
            try:
                import faiss
                norm_embeddings = embeddings_or_matrix.copy()
                faiss.normalize_L2(norm_embeddings)

                dimension = norm_embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dimension)
                self.faiss_index.add(norm_embeddings)
                logger.info(f"FAISS index built successfully with {self.faiss_index.ntotal} vectors.")
                return
            except Exception as e:
                logger.warning(f"FAISS indexing failed ({e}). Falling back to sparse matrix indexing.")

        # Fallback to TF-IDF matrix
        self.tfidf_matrix = embeddings_or_matrix
        logger.info(f"TF-IDF matrix indexed with {len(chunks)} passages.")

    def search(self, query_vec: Any, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the top k most relevant document passages for a given query vector.
        """
        if not self.chunks:
            return []

        results: List[Dict[str, Any]] = []
        k_val = min(top_k, len(self.chunks))

        if self.faiss_index is not None and isinstance(query_vec, np.ndarray):
            import faiss
            norm_q = query_vec.copy()
            faiss.normalize_L2(norm_q)
            scores, indices = self.faiss_index.search(norm_q, k_val)

            for idx, score in zip(indices[0], scores[0]):
                if 0 <= idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    page_num = chunk.metadata.get("page_number", 1)
                    results.append({
                        "page": page_num,
                        "content": chunk.page_content,
                        "similarity_score": round(float(score), 4),
                        "metadata": chunk.metadata
                    })
        elif self.tfidf_matrix is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            top_indices = np.argsort(sim_scores)[::-1][:k_val]

            for idx in top_indices:
                score = sim_scores[idx]
                chunk = self.chunks[idx]
                page_num = chunk.metadata.get("page_number", 1)
                results.append({
                    "page": page_num,
                    "content": chunk.page_content,
                    "similarity_score": round(float(score), 4),
                    "metadata": chunk.metadata
                })

        return results
