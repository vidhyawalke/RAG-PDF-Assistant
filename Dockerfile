# Production Dockerfile for RAG PDF Assistant
FROM python:3.10-slim

WORKDIR /app

# Install minimal system dependencies (removed build-essential - not needed for CPU PyTorch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install CPU-optimized packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application files
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY .streamlit/ ./.streamlit/
COPY .env.example .

# Create directories for supervisor logs
RUN mkdir -p /var/log/supervisor

# Copy supervisor configuration for proper process management
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose ports for Streamlit UI (8501) as primary, and FastAPI (8000)
EXPOSE 8501
EXPOSE 8000

# Use supervisor to manage both FastAPI and Streamlit processes
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
