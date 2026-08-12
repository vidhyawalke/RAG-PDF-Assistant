# RAG PDF Assistant

A Retrieval-Augmented Generation (RAG) system built with FastAPI, Streamlit, FAISS, and Google Gemini API. It lets users upload PDF documents, ask questions about the contents, and get back answers with exact page citations and response timings.

## Technologies

* FastAPI: Backend API framework
* Streamlit: Interactive chat frontend
* Python 3.8+: Core language
* FAISS: Vector similarity search index
* SentenceTransformers: `all-MiniLM-L6-v2` dense text embeddings
* Google Gemini API: `gemini-1.5-flash` model for answering
* PyPDF: Extracting text from PDF pages
* Scikit-Learn: Cosine similarity fallback retriever
* Docker & Docker Compose: Containerization setup

## Features

Here is what you can do with RAG PDF Assistant:

* Upload PDF Documents: Drag and drop a PDF file. The app reads page text, tracks page numbers, and splits text into chunks of 1000 characters with 150 character overlap.
* Fast Vector Search: Converts text chunks into 384-dimensional dense vectors stored in a FAISS index for quick context retrieval.
* Context-Grounded Answers: Sends retrieved text chunks to Google Gemini API (`gemini-1.5-flash`) with instructions to answer based only on the uploaded document.
* Expandable Source Citations: Each answer includes a drawer showing the exact source page numbers and matching text snippets.
* Clean Web Interface: Streamlit chat UI with timing badges, dark mode support, and session clearing.
* Automated Evaluation Suite: Benchmark retrieval precision and latency by running `python -m backend.eval.evaluate`.

## The Process

I started by building the document ingestion pipeline. Using PyPDF, text is extracted page by page and split using a recursive text splitter with a chunk size of 1000 characters and 150 overlap. Each chunk stores metadata like file name, chunk index, and page number.

Next, I worked on vector indexing and retrieval. I used SentenceTransformers (`all-MiniLM-L6-v2`) to turn text passages into 384-dimensional dense vectors and saved them into a FAISS vector index. I also added a Scikit-Learn TF-IDF fallback to handle environments where vector models might fail.

After setup, I integrated the Google Gemini API (`gemini-1.5-flash`). I wrote a system prompt that tells the model to answer strictly using the provided context and cite the exact page numbers used.

To connect everything together, I built a FastAPI backend with `/upload`, `/ask`, and `/health` endpoints with input validation and file path security checks. Then I built the Streamlit frontend with file upload buttons, chat boxes, and expandable source drawers.

Finally, I wrote an evaluation script (`evaluate.py`) that runs test questions against a sample PDF, checks keyword precision scores, measures latency, and exports JSON reports.

## What I Learned

Building this project helped me understand how vector search, prompt constraints, and API design fit together:

### Vector Embeddings and Search
Creating the FAISS vector index showed me how high-dimensional vectors represent semantic similarity. Normalizing vectors helped compute cosine similarity accurately.

### Chunking and Metadata Tracking
Working on PDF processing showed why chunk overlap matters. Keeping overlaps prevents context from getting cut off at chunk boundaries, while tracking page numbers keeps answers verifiable.

### Prompting and Context Enforcing
Writing the prompt template taught me how to keep LLMs focused. Explicitly telling the model to decline answering when information is missing helps avoid made-up responses.

### API Architecture and Containerization
Building FastAPI routes with Pydantic models gave me practice with request validation, safe file handling, and Docker container setup with Supervisor.

### Benchmarking and Evaluation
Building `evaluate.py` showed me how to measure system accuracy automatically instead of relying on manual testing.

## How can it be improved?

* Support asking questions across multiple uploaded PDFs at once.
* Combine dense FAISS vectors with sparse BM25 search for hybrid retrieval.
* Add a cross-encoder model to rerank retrieved chunks before sending them to the LLM.
* Add streaming text output to the Streamlit chat UI.
* Add OCR support for scanned PDF files.

## Running the Project

To run the project on your local machine, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/vidhyawalke/RAG-PDF-Assistant.git
   cd RAG-PDF-Assistant
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate

   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Copy `.env.example` to `.env` and add your Google Gemini API key:
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```env
   GOOGLE_API_KEY=your_actual_gemini_api_key
   ```

4. Start the application:
   * Streamlit Frontend: `streamlit run frontend/app.py` (open http://localhost:8501)
   * FastAPI Backend API: `uvicorn backend.main:app --port 8000 --reload` (open http://localhost:8000/docs)
   * Docker Compose: `docker-compose up --build`

## Video

*(Demo video walkthrough link coming soon)*
