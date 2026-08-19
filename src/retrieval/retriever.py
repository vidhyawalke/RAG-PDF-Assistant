"""
Context retriever module orchestrating query vectorization and similarity retrieval.
"""

from typing import List, Dict, Any
from src.embeddings.embedder import Embedder
from src.vectordb.vector_store import VectorStore
from src.utils.helpers import settings, get_logger

logger = get_logger("retrieval.retriever")


class Retriever:
    """Retrieves the most semantically relevant document passages for a query."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """
        Execute similarity search against indexed vector database and compile context text.
        """
        k = top_k if top_k is not None else settings.TOP_K

        if self.vector_store.total_vectors() == 0:
            logger.warning("Retrieval attempted on empty vector store.")
            return {
                "context": "",
                "sources": []
            }

        query_vec = self.embedder.embed_query(query)
        sources: List[Dict[str, Any]] = self.vector_store.search(query_vec, top_k=k)

        context_snippets = []
        for src in sources:
            page_num = src.get("page", 1)
            content = src.get("content", "")
            context_snippets.append(f"[Page {page_num}]: {content}")

        context_str = "\n\n".join(context_snippets)
        logger.info(f"Retrieved {len(sources)} context passages for query.")

        return {
            "context": context_str,
            "sources": sources
        }
