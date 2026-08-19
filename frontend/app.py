"""
Streamlit Frontend User Interface Module.
"""

import os
import sys
import time
import requests
import streamlit as st
from pathlib import Path

# Add project root directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.helpers import settings
from src.ingestion.loader import PDFLoader
from src.chunking.chunker import TextChunker
from src.embeddings.embedder import Embedder
from src.vectordb.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.prompts.prompt_templates import format_prompt
from src.llm.llm_client import LLMClient

# Page Configuration Setup
st.set_page_config(
    page_title="RAG PDF Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = settings.API_URL
SESSION_TIMEOUT_SECONDS = 900

# Session State Initialization
if "initialized" not in st.session_state:
    st.session_state["initialized"] = True
    st.session_state["last_activity_time"] = time.time()
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Welcome to RAG PDF Assistant. Upload a PDF document in the left panel, then ask any question."
        }
    ]
    st.session_state["document_processed"] = False
    st.session_state["document_name"] = None
    st.session_state["doc_chunks"] = 0
else:
    current_time = time.time()
    if current_time - st.session_state["last_activity_time"] > SESSION_TIMEOUT_SECONDS:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Session timed out due to inactivity. Started a new chat session. Upload a PDF to begin."
            }
        ]
        st.session_state["last_activity_time"] = current_time
    st.session_state["last_activity_time"] = time.time()

if "document_processed" not in st.session_state:
    st.session_state["document_processed"] = False
if "document_name" not in st.session_state:
    st.session_state["document_name"] = None
if "doc_chunks" not in st.session_state:
    st.session_state["doc_chunks"] = 0

# Sidebar Document Upload
with st.sidebar:
    st.header("Document Upload")
    st.caption("Upload a PDF document from your computer.")

    uploaded_file = st.file_uploader("Select PDF File", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Process Document", type="primary", use_container_width=True):
            with st.spinner("Processing PDF document..."):
                try:
                    temp_path = os.path.join(settings.UPLOAD_DIR, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Prepare local processing fallback
                    loader = PDFLoader(validate_paths=False)
                    chunker = TextChunker()
                    embedder = Embedder()
                    vstore = VectorStore()

                    pages = loader.load(temp_path)
                    chunks = chunker.chunk_pages(pages)
                    corpus = [c.page_content for c in chunks]
                    embs = embedder.embed_documents(corpus)
                    vstore.add_documents(chunks, embs)

                    st.session_state["local_vstore"] = vstore
                    st.session_state["local_embedder"] = embedder

                    # Send to backend API
                    api_request_success = False
                    total_chunks = len(chunks)
                    try:
                        with open(temp_path, "rb") as f_upload:
                            res = requests.post(
                                f"{API_URL}/upload",
                                files={"file": (uploaded_file.name, f_upload, "application/pdf")},
                                timeout=30
                            )
                        if res.status_code == 200:
                            data = res.json()
                            total_chunks = data["total_chunks"]
                            api_request_success = True
                    except requests.exceptions.RequestException:
                        pass

                    st.session_state["document_processed"] = True
                    st.session_state["document_name"] = uploaded_file.name
                    st.session_state["doc_chunks"] = total_chunks

                    st.session_state["messages"] = [
                        {
                            "role": "assistant",
                            "content": f"Successfully indexed **{uploaded_file.name}** with {total_chunks} passages. Ask any question below."
                        }
                    ]

                    if api_request_success:
                        st.success("Document processed successfully via backend API.")
                    else:
                        st.success("Document processed successfully in local mode.")

                except Exception as e:
                    st.error(f"Failed to process PDF: {e}")

    if st.session_state["document_processed"]:
        st.divider()
        st.markdown(f"**Active File:** `{st.session_state['document_name']}`")
        st.markdown(f"**Indexed Passages:** `{st.session_state['doc_chunks']} chunks`")

    st.divider()

    if st.button("Reset Chat", use_container_width=True):
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Chat reset. Ask any question about your document."
            }
        ]
        st.session_state["last_activity_time"] = time.time()
        st.rerun()

# Main Header
st.title("RAG PDF Assistant")
st.caption("Ask questions about your uploaded PDF and receive grounded answers with exact page citations.")

st.divider()

m1, m2, m3 = st.columns(3)
with m1:
    st.metric(label="Active Document", value=st.session_state["document_name"] or "No PDF Uploaded")
with m2:
    st.metric(label="Indexed Passages", value=f"{st.session_state['doc_chunks']} Chunks")
with m3:
    status_text = "Ready to Answer" if st.session_state["document_processed"] else "Upload PDF in Sidebar"
    st.metric(label="Status", value=status_text)

st.divider()

# Render Chat History
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("View Page Sources"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Source Snippet {idx} (Page {src['page']})**")
                    st.info(src["content"])
        if "execution_time_ms" in msg:
            st.caption(f"Response Speed: {msg['execution_time_ms']} ms")

# Chat Input
if user_input := st.chat_input("Type any question about your document here"):
    st.session_state["last_activity_time"] = time.time()

    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching document context..."):
            try:
                api_request_success = False
                answer = ""
                sources = []
                latency = 0.0

                try:
                    res = requests.post(f"{API_URL}/ask", json={"question": user_input, "top_k": 3}, timeout=30)
                    if res.status_code == 200:
                        data = res.json()
                        answer = data["answer"]
                        sources = data["sources"]
                        latency = data["execution_time_ms"]
                        api_request_success = True
                except requests.exceptions.RequestException:
                    pass

                if not api_request_success:
                    start_t = time.time()
                    vstore = st.session_state.get("local_vstore")
                    embedder = st.session_state.get("local_embedder")

                    if vstore and embedder:
                        retriever = Retriever(embedder=embedder, vector_store=vstore)
                        ret_res = retriever.retrieve(user_input, top_k=3)
                        sources = ret_res["sources"]
                        prompt = format_prompt(context=ret_res["context"], question=user_input)
                        llm = LLMClient()
                        answer = llm.generate(prompt)
                        if not answer:
                            answer = llm.generate_fallback_summary(sources)
                    else:
                        answer = "Please upload and process a PDF document first."
                        sources = []

                    latency = round((time.time() - start_t) * 1000, 2)

                st.markdown(answer)

                if sources:
                    with st.expander("View Page Sources"):
                        for idx, src in enumerate(sources, 1):
                            st.markdown(f"**Source Snippet {idx} (Page {src['page']})**")
                            st.info(src["content"])

                st.caption(f"Response Speed: {latency} ms")

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "execution_time_ms": latency
                })

            except Exception as e:
                err_msg = f"Error processing question: {e}"
                st.error(err_msg)
                st.session_state["messages"].append({"role": "assistant", "content": err_msg})
