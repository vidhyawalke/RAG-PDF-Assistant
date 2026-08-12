"""
================================================================================
Streamlit Frontend User Interface Module
--------------------------------------------------------------------------------
References & Documentation Sources:
- Streamlit Session State & Timeout Management: https://docs.streamlit.io/develop/concepts/architecture/session-state
- Streamlit Chat Interface API: https://docs.streamlit.io/develop/api-reference/chat
- Python Requests HTTP Library: https://requests.readthedocs.io/en/latest/
================================================================================
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

from backend.config import settings

# Page Configuration Setup
# Source: https://docs.streamlit.io/develop/api-reference/configuration/st.set_page_config
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Use API_URL from settings; for Docker, this should be the service name
API_URL = settings.API_URL
SESSION_TIMEOUT_SECONDS = 900  # 15 minutes session timeout

# Session State Initialization & Inactivity Timeout Handler
# Source: https://docs.streamlit.io/develop/concepts/architecture/session-state
# Initialize session state on first load only
if "initialized" not in st.session_state:
    st.session_state["initialized"] = True
    st.session_state["last_activity_time"] = time.time()
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Welcome to **RAG PDF Assistant**. Upload a PDF document in the left panel, then type any question below."
        }
    ]
    st.session_state["document_processed"] = False
    st.session_state["document_name"] = None
    st.session_state["doc_chunks"] = 0
else:
    # Check for session timeout only (15 minutes of inactivity)
    current_time = time.time()
    if current_time - st.session_state["last_activity_time"] > SESSION_TIMEOUT_SECONDS:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Session timed out due to 15 minutes of inactivity. Started a new chat session. Upload a PDF to begin!"
            }
        ]
        st.session_state["last_activity_time"] = current_time
    
    # Update activity timestamp on each interaction (after timeout check)
    st.session_state["last_activity_time"] = time.time()

if "document_processed" not in st.session_state:
    st.session_state["document_processed"] = False
if "document_name" not in st.session_state:
    st.session_state["document_name"] = None
if "doc_chunks" not in st.session_state:
    st.session_state["doc_chunks"] = 0

# Sidebar Document Upload & Chat Reset Interface
# Source: https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
with st.sidebar:
    st.header("📄 Document Upload")
    st.caption("Upload any PDF document from your computer.")
    
    uploaded_file = st.file_uploader("Select PDF File", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Process Document", type="primary", use_container_width=True):
            with st.spinner("Processing PDF document..."):
                try:
                    temp_path = os.path.join(settings.UPLOAD_DIR, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Send to FastAPI backend endpoint with error handling
                    api_request_success = False
                    try:
                        with open(temp_path, "rb") as f_upload:
                            res = requests.post(
                                f"{API_URL}/upload", 
                                files={"file": (uploaded_file.name, f_upload, "application/pdf")},
                                timeout=30
                            )
                        if res.status_code == 200:
                            data = res.json()
                            st.session_state["document_processed"] = True
                            st.session_state["document_name"] = uploaded_file.name
                            st.session_state["doc_chunks"] = data["total_chunks"]
                            
                            # Auto-refresh chat history when a NEW document is uploaded
                            st.session_state["messages"] = [
                                {
                                    "role": "assistant",
                                    "content": f"I have read **{uploaded_file.name}** ({data['total_chunks']} passages indexed). Ask me any question about this document below!"
                                }
                            ]
                            api_request_success = True
                            st.success("Document processed and chat refreshed.")
                        else:
                            st.warning(f"Backend returned status {res.status_code}. Attempting local processing...")
                    except requests.exceptions.RequestException as req_error:
                        st.warning(f"Backend unavailable ({req_error}). Attempting local processing...")
                    
                    # Fallback to local processing if API call failed
                    if not api_request_success:
                        try:
                            from backend.rag_chain import rag_pipeline
                            data = rag_pipeline.process_pdf(temp_path)
                            st.session_state["document_processed"] = True
                            st.session_state["document_name"] = uploaded_file.name
                            st.session_state["doc_chunks"] = data["total_chunks"]
                            
                            # Auto-refresh chat history when a NEW document is uploaded
                            st.session_state["messages"] = [
                                {
                                    "role": "assistant",
                                    "content": f"I have read **{uploaded_file.name}** ({data['total_chunks']} passages indexed). Ask me any question about this document below!"
                                }
                            ]
                            st.success("Document processed and chat refreshed (local mode).")
                        except Exception as local_error:
                            st.error(f"Both API and local processing failed: {local_error}")

                except Exception as e:
                    st.error(f"Failed to process PDF: {e}")

    if st.session_state["document_processed"]:
        st.divider()
        st.markdown(f"**Active File:** `{st.session_state['document_name']}`")
        st.markdown(f"**Indexed Passages:** `{st.session_state['doc_chunks']} chunks`")

    st.divider()
    
    # Manual Reset / New Chat Button
    if st.button("🔄 New Chat / Reset", use_container_width=True):
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Chat reset. Ask any question about your document!"
            }
        ]
        st.session_state["last_activity_time"] = time.time()
        st.rerun()

# Main Header Section
st.title("RAG PDF Assistant")
st.caption("Ask any question about your uploaded PDF and get accurate answers with exact page number citations.")

st.divider()

# Overview Status Cards
m1, m2, m3 = st.columns(3)
with m1:
    st.metric(label="Active Document", value=st.session_state["document_name"] or "No PDF Uploaded")
with m2:
    st.metric(label="Indexed Passages", value=f"{st.session_state['doc_chunks']} Chunks")
with m3:
    status_text = "Ready to Answer" if st.session_state["document_processed"] else "Upload PDF in Sidebar"
    st.metric(label="Status", value=status_text)

st.divider()

# Render Chat Trajectory
# Source: https://docs.streamlit.io/develop/api-reference/chat/st.chat_message
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📖 View Page Sources"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Source Snippet {idx} (Page {src['page']})**")
                    st.info(src["content"])
        if "execution_time_ms" in msg:
            st.caption(f"⚡ Response Speed: {msg['execution_time_ms']} ms")

# Open Chat Input Field (User types any question manually)
# Source: https://docs.streamlit.io/develop/api-reference/chat/st.chat_input
if user_input := st.chat_input("Type any question about your document here..."):
    # Update activity timestamp
    st.session_state["last_activity_time"] = time.time()
    
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching document context..."):
            try:
                api_request_success = False
                try:
                    res = requests.post(f"{API_URL}/ask", json={"question": user_input, "top_k": 3}, timeout=15)
                    if res.status_code == 200:
                        data = res.json()
                        answer = data["answer"]
                        sources = data["sources"]
                        latency = data["execution_time_ms"]
                        api_request_success = True
                    else:
                        st.warning(f"Backend error: {res.status_code}. Attempting local processing...")
                except requests.exceptions.RequestException as req_error:
                    st.warning(f"Backend unavailable. Attempting local processing...")

                # Fallback to local processing if API call failed
                if not api_request_success:
                    try:
                        from backend.rag_chain import rag_pipeline
                        data = rag_pipeline.answer_question(user_input, top_k=3)
                        answer = data["answer"]
                        sources = data["sources"]
                        latency = data["execution_time_ms"]
                    except Exception as local_error:
                        st.error(f"Error processing question: {local_error}")
                        answer = f"Error: {local_error}"
                        sources = []
                        latency = 0

                st.markdown(answer)

                if sources:
                    with st.expander("📖 View Page Sources"):
                        for idx, src in enumerate(sources, 1):
                            st.markdown(f"**Source Snippet {idx} (Page {src['page']})**")
                            st.info(src["content"])

                st.caption(f"⚡ Response Speed: {latency} ms")

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
