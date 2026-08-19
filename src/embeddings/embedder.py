"""
Embeddings engine for converting text chunks into dense vector representations.
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from typing import List, Union, Any
import numpy as np

from src.utils.helpers import settings, get_logger

logger = get_logger("embeddings.embedder")


class Embedder:
    """Computes vector embeddings using SentenceTransformers with TF-IDF fallback."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.model = None
        self.vectorizer = None
        self._init_engine()

    def _init_engine(self):
        """Initialize SentenceTransformer or fallback to Scikit-Learn TfidfVectorizer."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("SentenceTransformer model initialized successfully.")
        except Exception as e:
            logger.warning(f"SentenceTransformer unavailable ({e}). Using Scikit-Learn TF-IDF vectorizer.")
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))

    def is_dense(self) -> bool:
        """Return True if dense neural embeddings model is loaded."""
        return self.model is not None

    def embed_documents(self, texts: List[str]) -> Any:
        """Generate embeddings or fit vectorizer for a list of document strings."""
        if not texts:
            return np.array([])

        if self.is_dense():
            raw_embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return raw_embeddings.astype("float32")
        else:
            if self.vectorizer is None:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            return self.vectorizer.fit_transform(texts)

    def embed_query(self, query: str) -> Any:
        """Generate embedding vector for a single query string."""
        if self.is_dense():
            raw_vec = self.model.encode([query], convert_to_numpy=True)
            return raw_vec.astype("float32")
        else:
            if self.vectorizer is None:
                raise ValueError("TF-IDF vectorizer has not been fitted on any document corpus yet.")
            return self.vectorizer.transform([query])
