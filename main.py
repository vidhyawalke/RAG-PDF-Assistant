"""
Main entry point for the RAG PDF Assistant application.
"""

import uvicorn
from src.utils.helpers import settings, get_logger, ensure_directories

logger = get_logger("main")

if __name__ == "__main__":
    ensure_directories()
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION} on {settings.HOST}:{settings.FASTAPI_PORT}")
    uvicorn.run(
        "src.api.routes:app",
        host=settings.HOST,
        port=settings.FASTAPI_PORT,
        reload=False
    )
