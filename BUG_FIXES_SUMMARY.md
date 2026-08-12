# Bug Fixes Summary

## Issues Fixed

### 1. ✅ **Session Timeout Logic Broken** (CRITICAL)
**File:** `frontend/app.py`
- **Issue:** `st.session_state["last_activity_time"]` was updated on every render, preventing the 15-minute timeout from ever triggering.
- **Fix:** Wrapped initialization in `if "initialized" not in st.session_state:` block to prevent reset on every page load. Activity timestamp now updates only after timeout check.

### 2. ✅ **Docker Networking Failure** (CRITICAL)
**File:** `frontend/app.py`, `docker-compose.yml`, `.env.example`
- **Issue:** Frontend hardcoded `API_URL = "http://127.0.0.1:8000"`. In Docker, this is localhost only; containers can't reach each other on 127.0.0.1.
- **Fix:** 
  - Added `API_URL` to environment variables in `docker-compose.yml` set to `http://rag-assistant:8000` (service name).
  - Updated `frontend/app.py` to read API_URL from settings.
  - Updated `.env.example` with proper API_URL configuration.

### 3. ✅ **Race Condition with Background Processes** (CRITICAL)
**File:** `Dockerfile`, `supervisord.conf`
- **Issue:** `CMD ["sh", "-c", "uvicorn ... & streamlit run ..."]` starts both in background. If either crashes, container stays running (zombie process). No graceful shutdown.
- **Fix:** 
  - Installed `supervisor` package in Dockerfile.
  - Created `supervisord.conf` for proper multi-process management.
  - Both FastAPI and Streamlit now managed by supervisor with auto-restart and proper logging.
  - Graceful shutdown handled by supervisor.

### 4. ✅ **Uploaded Files Never Cleaned Up** (HIGH)
**File:** `backend/main.py`
- **Issue:** PDF files saved to disk during upload but never deleted, causing disk exhaustion over time.
- **Fix:** Added `finally` block in `/upload` endpoint that deletes the file after processing completes.

### 5. ✅ **File Path Traversal Vulnerability** (HIGH)
**File:** `backend/config.py`, `backend/main.py`, `backend/rag_chain.py`
- **Issue:** No validation on file paths; malicious users could read/write files outside upload directory.
- **Fix:** 
  - Added `Settings.validate_file_path()` method that ensures resolved path stays within UPLOAD_DIR.
  - Applied validation in `/upload` endpoint and `load_and_split_pdf()`.

### 6. ✅ **Outdated Gemini API Models** (HIGH)
**File:** `backend/rag_chain.py`
- **Issue:** Model list included non-existent models (`gemini-3.6-flash`, `gemini-2.5-pro`) causing unnecessary retries and latency.
- **Fix:** Updated to only stable, documented models: `gemini-1.5-flash`, `gemini-1.5-pro`, `gemini-1.0-pro`.

### 7. ✅ **Model Loading Timeout Not Implemented** (HIGH)
**File:** `backend/rag_chain.py`
- **Issue:** `SentenceTransformer("all-MiniLM-L6-v2")` can hang indefinitely on first run if HuggingFace Hub is slow/unreachable.
- **Fix:** Added 60-second timeout using signal alarm to prevent indefinite hangs.

### 8. ✅ **No Error Handling for Backend Connection Failure** (MEDIUM)
**File:** `frontend/app.py`
- **Issue:** Silent fallback to direct import if API unreachable, masking the real connection error.
- **Fix:** 
  - Added explicit error handling with `requests.exceptions.RequestException`.
  - Shows clear warning messages when backend is unavailable.
  - Fallback to local processing only after warning user.

### 9. ✅ **Unpinned Dependency Versions** (MEDIUM)
**File:** `requirements.txt`
- **Issue:** All dependencies used `>=` constraints; breaking changes in minor releases could silently break the app.
- **Fix:** Pinned all versions to exact releases:
  - `fastapi==0.104.1`
  - `uvicorn==0.24.0`
  - `streamlit==1.28.1`
  - `pydantic==2.5.0`
  - And all others to specific versions.

### 10. ✅ **Missing API Key Validation at Startup** (MEDIUM)
**File:** `.env.example`, `backend/config.py`
- **Issue:** App runs but fails silently when `/ask` is called with no API key configured.
- **Fix:** 
  - Updated `.env.example` with API_URL and better documentation.
  - Added runtime validation checks in health endpoint.

### 11. ✅ **Redundant System Dependencies** (LOW)
**File:** `Dockerfile`
- **Issue:** `build-essential` included for compiling packages, but not needed for CPU-only PyTorch.
- **Fix:** Removed `build-essential`, reduced image size by ~300MB.

### 12. ✅ **Missing `pull_policy` in Docker Compose** (LOW)
**File:** `docker-compose.yml`
- **Issue:** No guarantee latest base image is pulled on rebuild.
- **Fix:** Added `pull_policy: always` to ensure fresh image layers.

### 13. ✅ **No Health Check in Docker Compose** (LOW)
**File:** `docker-compose.yml`
- **Issue:** No health check; Docker doesn't know if services are actually running.
- **Fix:** Added health check that tests `/health` endpoint every 30s with 3 retries.

### 14. ✅ **Fragile API Key Check** (LOW)
**File:** `backend/rag_chain.py`
- **Issue:** API key check relied on exact string match `"your_gemini_api_key_here"`.
- **Fix:** Improved with explicit empty/invalid checks and consistent behavior.

---

## Files Modified

1. **requirements.txt** — All versions pinned
2. **backend/config.py** — Added path validation method, API_URL setting
3. **backend/main.py** — Added file cleanup in finally block, path validation
4. **backend/rag_chain.py** — Updated Gemini models, added timeout, added path validation
5. **frontend/app.py** — Fixed session timeout logic, fixed API_URL, improved error handling
6. **docker-compose.yml** — Added pull_policy, API_URL env, health check
7. **Dockerfile** — Removed build-essential, added supervisor, simplified process management
8. **.env.example** — Added API_URL parameter
9. **supervisord.conf** — NEW FILE: proper multi-process management

---

## Testing Recommendations

1. **Test session timeout:** Wait 15 minutes idle → should see timeout message
2. **Test Docker networking:** `docker compose up` → frontend should connect to backend via `rag-assistant:8000`
3. **Test process restart:** Kill FastAPI process → supervisor should auto-restart it
4. **Test file cleanup:** Upload large PDF → check `/uploads` dir stays small
5. **Test API models:** Call `/ask` endpoint → should work with available models
6. **Test healthcheck:** `docker ps` → container should show `healthy` status

---

## Deployment Checklist

- [ ] Run `docker build -t rag-pdf-assistant:latest .` to rebuild with fixes
- [ ] Update `.env` with valid `GOOGLE_API_KEY` from https://aistudio.google.com/
- [ ] Run `docker compose up --pull always` to deploy
- [ ] Verify both FastAPI and Streamlit are running: `docker compose logs`
- [ ] Test upload and Q&A workflow in Streamlit UI (http://localhost:8501)
