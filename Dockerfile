# Production Dockerfile for RAG PDF Assistant
FROM python:3.10-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install CPU-optimized packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application files
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY tests/ ./tests/
COPY main.py .
COPY config.yaml .
COPY .streamlit/ ./.streamlit/
COPY .env.example .

# Create directories for supervisor and application logs
RUN mkdir -p /var/log/supervisor /app/logs /app/uploads /app/vector_store

# Copy supervisor configuration for process management
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose port 8000 for Streamlit UI and port 8001 for FastAPI
EXPOSE 8000
EXPOSE 8001

# Use supervisor to manage both FastAPI and Streamlit processes
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
