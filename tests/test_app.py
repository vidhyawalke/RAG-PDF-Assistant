"""
Unit and integration tests for RAG PDF Assistant components.
"""

import unittest
from fastapi.testclient import TestClient

from src.chunking.chunker import TextChunker, DocumentChunk
from src.prompts.prompt_templates import format_prompt, SYSTEM_PROMPT_TEMPLATE
from src.vectordb.vector_store import VectorStore
from src.embeddings.embedder import Embedder
from src.api.routes import app


class TestRAGApplication(unittest.TestCase):
    """Test suite for RAG core modules and FastAPI endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_chunker_split_text(self):
        """Verify that TextChunker splits large text into overlapping windows."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        sample_text = (
            "Retrieval Augmented Generation enhances large language models by retrieving "
            "relevant knowledge from external authoritative document sources. This reduces hallucinations "
            "and keeps answers factually grounded and verifiable."
        )
        chunks = chunker.split_text(sample_text)
        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 120)

    def test_chunker_empty_input(self):
        """Verify chunker behavior on empty input strings."""
        chunker = TextChunker()
        self.assertEqual(chunker.split_text(""), [])
        self.assertEqual(chunker.split_text("   "), [])

    def test_chunk_pages_metadata(self):
        """Verify chunk_pages preserves page number and filename metadata."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        pages = [
            {"page_number": 1, "text": "Page one content about data ingestion pipelines.", "source_filename": "doc.pdf"},
            {"page_number": 2, "text": "Page two content detailing vector database indexing.", "source_filename": "doc.pdf"}
        ]
        doc_chunks = chunker.chunk_pages(pages)
        self.assertGreaterEqual(len(doc_chunks), 2)
        self.assertEqual(doc_chunks[0].metadata["page_number"], 1)
        self.assertEqual(doc_chunks[0].metadata["source_filename"], "doc.pdf")

    def test_prompt_template_formatting(self):
        """Verify format_prompt injects context and query correctly."""
        context = "[Page 1]: Machine learning systems rely on quality training data."
        question = "What do machine learning systems rely on?"
        formatted = format_prompt(context, question)

        self.assertIn(context, formatted)
        self.assertIn(question, formatted)
        self.assertIn("Instructions for Structured Output:", formatted)

    def test_vector_store_indexing_and_search(self):
        """Verify VectorStore stores chunks and returns relevant passages."""
        embedder = Embedder()
        store = VectorStore()

        chunks = [
            DocumentChunk("FastAPI is a modern high performance web framework for building APIs.", {"page_number": 1}),
            DocumentChunk("FAISS is a library for efficient similarity search of dense vectors.", {"page_number": 2}),
            DocumentChunk("Streamlit lets developers turn data scripts into shareable web applications.", {"page_number": 3})
        ]

        corpus = [c.page_content for c in chunks]
        embeddings = embedder.embed_documents(corpus)
        store.add_documents(chunks, embeddings)

        self.assertEqual(store.total_vectors(), 3)

        query_vec = embedder.embed_query("Tell me about FAISS vector indexing")
        results = store.search(query_vec, top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["page"], 2)
        self.assertIn("FAISS", results[0]["content"])

    def test_api_root_endpoint(self):
        """Verify GET / returns root welcome message."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("docs", data)

    def test_api_health_endpoint(self):
        """Verify GET /health returns valid health status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("vector_store_loaded", data)

    def test_api_ask_validation_empty_query(self):
        """Verify POST /ask returns 400 when question is empty."""
        response = self.client.post("/ask", json={"question": "   ", "top_k": 3})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
