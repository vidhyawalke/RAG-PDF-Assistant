"""
Chunking module for segmenting extracted document text into overlapping passages.
"""

from typing import List, Dict, Any
from src.utils.helpers import settings, get_logger

logger = get_logger("chunking.chunker")


class DocumentChunk:
    """Represents an individual text chunk with associated document metadata."""

    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.page_content = content
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk into a dictionary representation."""
        return {
            "page_content": self.page_content,
            "metadata": self.metadata
        }


class TextChunker:
    """Splits full page text into bounded character chunks with configured overlap."""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP

    def split_text(self, text: str) -> List[str]:
        """
        Split a string into chunks respecting natural sentence and word boundaries.
        """
        if not text or not text.strip():
            return []

        if len(text) <= self.chunk_size:
            return [text.strip()]

        chunks: List[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            if end < text_len:
                # Seek natural break point such as sentence end or word space
                last_break = max(
                    text.rfind(". ", start, end),
                    text.rfind("\n", start, end),
                    text.rfind(" ", start, end)
                )
                if last_break > start + (self.chunk_size // 2):
                    end = last_break + 1

            snippet = text[start:end].strip()
            if snippet:
                chunks.append(snippet)

            if end >= text_len:
                break

            start = max(start + 1, end - self.chunk_overlap)

        return chunks

    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """
        Transform a list of page text dicts into DocumentChunk instances with metadata.
        """
        document_chunks: List[DocumentChunk] = []
        chunk_id = 0

        for page_info in pages:
            page_text = page_info.get("text", "")
            page_num = page_info.get("page_number", 1)
            filename = page_info.get("source_filename", "document.pdf")

            snippets = self.split_text(page_text)
            for snippet in snippets:
                meta = {
                    "chunk_id": chunk_id,
                    "source_filename": filename,
                    "page_number": page_num,
                    "char_count": len(snippet)
                }
                document_chunks.append(DocumentChunk(content=snippet, metadata=meta))
                chunk_id += 1

        logger.info(f"Chunked {len(pages)} pages into {len(document_chunks)} document chunks.")
        return document_chunks
