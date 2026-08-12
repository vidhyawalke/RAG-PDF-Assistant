# 📄 RAG PDF Assistant

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-red.svg)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/FAISS-1.7.4-orange.svg)](https://github.com/facebookresearch/faiss)
[![Gemini API](https://img.shields.io/badge/Google_Gemini-1.5_Flash-purple.svg)](https://aistudio.google.com/)

A high-performance Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **Streamlit**, **FAISS**, and **Google Gemini API**. The assistant enables users to upload PDF documents, query them in natural language, and receive precise, context-grounded answers with exact source page citations and real-time execution timing.

---

## ✨ Features

- 📄 **PDF Document Ingestion**: Efficient page-by-page text extraction and chunking with metadata tracking.
- ⚡ **Sub-20ms Dense Vector Search**: High-dimensional text embeddings powered by `SentenceTransformers` (`all-MiniLM-L6-v2`) and indexed via Meta `FAISS`.
- 🤖 **Grounded LLM Generation**: Integrates Google Gemini API (`gemini-1.5-flash`) with strict system prompt enforcement to eliminate hallucinations.
- 🔍 **Source Citation Drawers**: Every answer includes expandable page source citations for complete auditability.
- 🎨 **Google Workspace UI**: Modern Streamlit chat dashboard featuring clean color palettes, dark mode support, and live response timing badges.
- 🔌 **RESTful API**: Production FastAPI endpoints (`/upload`, `/ask`, `/health`) with input validation.
- 📊 **Automated Evaluation Suite**: Pre-built benchmark evaluation framework (`evaluate.py`) measuring retrieval precision and latency.
- 🐳 **Docker Ready**: Complete Docker containerization support with supervisor process management.

---

## 🏗️ Project Architecture

```text
RAG-PDF-Assistant/
├── backend/
│   ├── main.py           # FastAPI REST API application & routing
│   ├── rag_chain.py      # PDF loader, chunking, FAISS indexer, and Gemini LLM chain
│   ├── config.py         # App configuration & path security validation
│   └── eval/
│       ├── evaluate.py   # Benchmark evaluation script
│       └── EVALUATION_REPORT.md
├── frontend/
│   └── app.py            # Streamlit interactive chat UI & state management
├── .streamlit/
│   └── config.toml       # Streamlit theme settings
├── uploads/              # Temporary PDF storage (git-ignored)
├── vector_store/         # FAISS vector database store (git-ignored)
├── .env.example          # Template for environment variables
├── Dockerfile            # Container definition
├── docker-compose.yml    # Multi-container orchestration
└── requirements.txt      # Python dependencies
```

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/vidhyawalke/RAG-PDF-Assistant.git
cd RAG-PDF-Assistant
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure API Key
Copy `.env.example` to `.env` and insert your free **Google Gemini API Key** (get one at [Google AI Studio](https://aistudio.google.com/)):

```bash
cp .env.example .env
```

Edit `.env`:
```env
GOOGLE_API_KEY=your_actual_gemini_api_key
```

---

## 🚀 Running the Application

### Option A: Streamlit Web Dashboard
```bash
streamlit run frontend/app.py
```
Open **`http://localhost:8501`** in your browser.

### Option B: FastAPI Backend Server
```bash
uvicorn backend.main:app --port 8000 --reload
```
Interactive API documentation (Swagger UI) is available at **`http://localhost:8000/docs`**.

### Option C: Docker Compose
```bash
docker-compose up --build
```

---

## 📊 Running Automated Benchmarks

To run the automated evaluation suite against test documents:

```bash
python -m backend.eval.evaluate
```
