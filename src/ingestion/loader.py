"""
Document ingestion loader for extracting text from PDF files.
"""

from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader

from src.utils.helpers import get_logger, validate_file_path

logger = get_logger("ingestion.loader")


class PDFLoader:
    """Extracts raw text and page level metadata from PDF files."""

    def __init__(self, validate_paths: bool = True):
        self.validate_paths = validate_paths

    def load(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Read a PDF file and extract text per page with 1-indexed page numbering.
        """
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        if self.validate_paths and not validate_file_path(file_path):
            raise ValueError(f"Unauthorized or invalid file path: {file_path}")

        logger.info(f"Ingesting PDF: {path_obj.name}")
        pages_data: List[Dict[str, Any]] = []

        with open(str(path_obj), "rb") as f:
            reader = PdfReader(f)
            for page_idx, page in enumerate(reader.pages):
                raw_text = page.extract_text() or ""
                cleaned_text = " ".join(raw_text.split())

                if not cleaned_text.strip():
                    continue

                pages_data.append({
                    "page_number": page_idx + 1,
                    "text": cleaned_text,
                    "source_filename": path_obj.name
                })

        logger.info(f"Successfully extracted {len(pages_data)} pages from {path_obj.name}")
        return pages_data
