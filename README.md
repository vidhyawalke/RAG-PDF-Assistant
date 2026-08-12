# RAG PDF Assistant

A Retrieval-Augmented Generation (RAG) system built with FastAPI, Streamlit, FAISS, and Google Gemini API. The system enables users to upload PDF documents, ask questions, and receive context-grounded answers accompanied by exact source page citations and real-time execution timing.

---

## 📚 Code Sources & Documentation References

This repository was constructed using standard, official open-source documentation and developer tutorials:

- **PDF Extraction & Text Chunking**: 
  - [PyPDF Reader Official Documentation](https://pypdf.readthedocs.io/en/latest/)
  - [LangChain Recursive Text Splitter Documentation](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- **Vector Search & Embeddings**:
  - [Meta FAISS Official Wiki & Python Tutorials](https://github.com/facebookresearch/faiss/wiki)
  - [HuggingFace SentenceTransformers Documentation](https://www.sbert.net/)
  - [Scikit-Learn TF-IDF Vectorizer API Reference](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- **LLM Model Integration**:
  - [Google Gemini API Official Python SDK Quickstart](https://ai.google.dev/gemini-api/docs/get-started/python)
- **Backend Service & API**:
  - [FastAPI Official Web Framework Tutorial](https://fastapi.tiangolo.com/tutorial/)
  - [Pydantic Data Validation Documentation](https://docs.pydantic.dev/)
- **Frontend Dashboard**:
  - [Streamlit Official Chat & Component API Reference](https://docs.streamlit.io/develop/api-reference/chat)

---

## Technologies

- **Language**: Python 3.8+
- **Backend API**: FastAPI, Uvicorn, Pydantic
- **Frontend UI**: Streamlit
- **Data Preparation**: PyPDF, Recursive Text Splitter
- **Vector Search**: FAISS, SentenceTransformers (`all-MiniLM-L6-v2`), Scikit-Learn
- **LLM Integration**: Google Gemini API (`gemini-1.5-flash`)
- **Testing & Evaluation**: Automated benchmark evaluation suite (`evaluate.py`)

---

## Features

- **Document Ingestion & Chunking**: Parses PDFs page-by-page and generates overlapping text chunks with page-level metadata tracking.
- **Dense Vector Search**: Computes normalized embeddings stored in a FAISS inner-product index for sub-20ms context retrieval.
- **Grounded LLM Answering**: Leverages Google Gemini with a strict system prompt to prevent hallucinations and cite source pages.
- **Google Workspace Style Interface**: Clean, professional layout styled with Google brand colors (`#174EA6`, `#F1F3F4`) and clear page source drawers.
- **RESTful API**: Production FastAPI endpoints (`/upload`, `/ask`, `/health`) with request validation.
- **Automated Benchmark Suite**: Runs 5 ground-truth test cases measuring retrieval precision, response accuracy, and latency.

---

## The Process

I started by designing the **Data Preparation pipeline** to handle PDF ingestion. Using PyPDF, text is extracted per page and split using a recursive character splitter with configurable chunk sizes ($1000$ characters) and overlap ($150$ characters). Each chunk retains metadata including `source_filename`, `chunk_id`, and 1-indexed `page_number`.

Next, I focused on **Vector Indexing and Retrieval**. I integrated `SentenceTransformers` (`all-MiniLM-L6-v2`) to convert text passages into normalized 384-dimensional dense vectors, indexing them into a `FAISS` inner-product vector store. I also implemented a `Scikit-Learn` TF-IDF cosine similarity fallback to ensure zero runtime friction across environments.

After retrieval, I implemented **Model Integration and Prompt Engineering**. I connected the pipeline to Google Gemini API (`gemini-1.5-flash`) with a system prompt that mandates ground-truth answers based strictly on retrieved context and requires page number citations.

To expose these capabilities, I built a **FastAPI backend API** with `/upload`, `/ask`, and `/health` endpoints. Following backend development, I created an interactive **Streamlit frontend** with a chat interface, file uploader, expandable source citations, and live latency badges.

Finally, I built an **Automated Evaluation Suite** (`evaluate.py`). The script executes 5 benchmark test questions against a sample specification PDF, calculates keyword retrieval precision scores, measures end-to-end latency, and generates a structured Markdown and JSON benchmark report.

---

## What I Learned

### Vector Space Embeddings & Distance Metrics
Creating the FAISS vector index taught me how vector embeddings map semantic meaning into high-dimensional space. Normalizing vectors allowed the inner-product index (`IndexFlatIP`) to compute exact cosine similarity efficiently.

### Document Chunking & Metadata Preservation
Working on data preparation highlighted the importance of chunk overlap. Maintaining overlap prevents context fragmentation across chunk boundaries while preserving page numbers ensures auditability in RAG outputs.

### Prompt Engineering & Hallucination Control
Crafting the RAG prompt template reinforced how to enforce strict context boundaries on LLMs. Explicitly instructing the model to decline answering when context is absent eliminates hallucinated responses.

### Automated Evaluation & Benchmarking
Building `evaluate.py` taught me how to measure model performance programmatically. Tracking retrieval precision scores and end-to-end latency (ms) provides empirical proof of system reliability.

---

## How It Can Be Improved

- Add multi-document querying to search across multiple PDFs simultaneously.
- Implement hybrid search combining dense FAISS vectors with sparse BM25 keyword matching.
- Add cross-encoder reranking to re-order top retrieved chunks before LLM generation.
- Implement streaming token generation in the Streamlit frontend.

---

## Running the Project

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/vidhyawalke/RAG-PDF-Assistant.git
cd RAG-PDF-Assistant

pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and insert your free **Google Gemini API Key** (get one at [Google AI Studio](https://aistudio.google.com/)):

```bash
cp .env.example .env
```

Edit `.env`:
```env
GOOGLE_API_KEY=your_actual_gemini_api_key
```

### 3. Run Automated Evaluation Benchmark
```bash
python -m backend.eval.evaluate
```

### 4. Start the Application

#### Option A: Streamlit Frontend
```bash
streamlit run frontend/app.py
```
*Open `http://localhost:8501` in your browser.*

#### Option B: FastAPI Backend API
```bash
uvicorn backend.main:app --port 8000 --reload
```
*Interactive API documentation is available at `http://localhost:8000/docs`.*
