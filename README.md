# 📄 RAG PDF Assistant

A Retrieval-Augmented Generation (RAG) system built with FastAPI, Streamlit, FAISS, and Google Gemini API. The system enables users to upload PDF documents, ask natural language questions, and receive context-grounded answers accompanied by exact source page citations and real-time execution timing.

---

## 📦 Technologies

- **FastAPI**: Production-grade RESTful API framework
- **Streamlit**: Interactive chat UI dashboard
- **Python 3.8+**: Core programming language
- **Meta FAISS**: Sub-20ms dense vector similarity search
- **SentenceTransformers**: `all-MiniLM-L6-v2` 384-dimensional dense text embeddings
- **Google Gemini API**: `gemini-1.5-flash` LLM for grounded answer generation
- **PyPDF**: Document parsing & page-level text extraction
- **Scikit-Learn**: TF-IDF cosine similarity fallback retriever
- **Docker & Docker Compose**: Multi-container containerization with Supervisor

---

## 🦄 Features

Here's what you can do with RAG PDF Assistant:

- **Upload PDF Documents**: Drag and drop any PDF file. The application parses text page-by-page, tracks metadata, and chunks text with configurable overlap ($1000$ chunk size, $150$ overlap).
- **Sub-20ms Dense Vector Search**: Converts text passages into normalized 384-dimensional dense vectors stored in a FAISS inner-product index for fast context retrieval.
- **Grounded LLM Answering**: Queries Google Gemini API (`gemini-1.5-flash`) with a strict system prompt to deliver verified, context-grounded answers and eliminate hallucinations.
- **Expandable Page Source Citations**: Every generated answer includes an expandable drawer showing exact source page numbers and raw context passages for full auditability.
- **Google Workspace Style Interface**: Interactive chat UI styled with Google brand colors (`#174EA6`, `#F1F3F4`), latency badges (`⚡ Response generated in X.XXs`), dark mode support, and session history management.
- **Automated Benchmark Suite**: Programmatically benchmark retrieval precision, model accuracy, and latency using `python -m backend.eval.evaluate`.

---

## 👩🏽🍳 The Process

I started by designing the **Data Preparation pipeline** to handle PDF ingestion. Using PyPDF, text is extracted page-by-page and split using a recursive character text splitter with a chunk size of 1000 characters and 150 overlap. Each chunk retains metadata including `source_filename`, `chunk_id`, and 1-indexed `page_number`.

Next, I focused on **Vector Indexing and Retrieval**. I integrated SentenceTransformers (`all-MiniLM-L6-v2`) to convert text passages into normalized 384-dimensional dense vectors, indexing them into a FAISS inner-product vector store for sub-20ms similarity search. I also implemented a Scikit-Learn TF-IDF cosine similarity fallback to ensure zero runtime friction across environments.

After retrieval, I implemented **Model Integration and Prompt Engineering**. I connected the pipeline to Google Gemini API (`gemini-1.5-flash`) with a system prompt that mandates ground-truth answers based strictly on retrieved context and requires page number citations.

To expose these capabilities, I built a **FastAPI backend API** with `/upload`, `/ask`, and `/health` endpoints with strict Pydantic data validation and file path traversal safeguards. Following backend development, I created an interactive **Streamlit frontend** with a chat interface, file uploader, expandable source citations, and live latency badges.

Finally, I built an **Automated Evaluation Suite** (`evaluate.py`). The script executes 5 benchmark test questions against a sample specification PDF, calculates keyword retrieval precision scores, measures end-to-end latency, and generates structured Markdown and JSON benchmark reports.

---

## 📚 What I Learned

During this project, I picked up important skills in vector search, prompt engineering, and full-stack AI application design:

### 🧠 Vector Space Embeddings & Distance Metrics:
Creating the FAISS vector index taught me how high-dimensional vector embeddings map semantic meaning. Normalizing vectors allowed the inner-product index (`IndexFlatIP`) to compute exact cosine similarity efficiently.

### 📏 Document Chunking & Metadata Preservation:
Working on data preparation highlighted the importance of chunk overlap. Maintaining overlap prevents context fragmentation across chunk boundaries while preserving page numbers ensures auditability in RAG outputs.

### 🎨 Prompt Engineering & Hallucination Control:
Crafting the RAG prompt template reinforced how to enforce strict context boundaries on LLMs. Explicitly instructing the model to decline answering when context is absent eliminates hallucinated responses.

### 🔍 RESTful API Design & Containerization:
Building FastAPI endpoints with Pydantic models taught me clean API layering, file path security validation, and Docker container orchestration with supervisor multi-process management.

### 📈 Automated Evaluation & Benchmarking:
Building `evaluate.py` taught me how to measure model performance programmatically. Tracking retrieval precision scores and end-to-end latency provides empirical proof of system reliability.

---

## 💭 How can it be improved?

- Add multi-document querying to search across multiple PDFs simultaneously.
- Implement hybrid search combining dense FAISS vectors with sparse BM25 keyword matching.
- Add cross-encoder reranking to re-order top retrieved chunks before LLM generation.
- Implement streaming token generation in the Streamlit frontend.
- Add OCR (Tesseract / EasyOCR) support for scanned PDF documents.

---

## 🚦 Running the Project

To run the project in your local environment, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/vidhyawalke/RAG-PDF-Assistant.git
   cd RAG-PDF-Assistant
   ```

2. **Set up virtual environment & install dependencies**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   Copy `.env.example` to `.env` and insert your free **Google Gemini API Key** (get one at [Google AI Studio](https://aistudio.google.com/)):
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```env
   GOOGLE_API_KEY=your_actual_gemini_api_key
   ```

4. **Start the application**:
   - **Streamlit Frontend**: `streamlit run frontend/app.py` (open `http://localhost:8501`)
   - **FastAPI Backend API**: `uvicorn backend.main:app --port 8000 --reload` (Swagger UI at `http://localhost:8000/docs`)
   - **Docker Compose**: `docker-compose up --build`

---

## 🍿 Video

*(Demo video walkthrough link coming soon)*
