"""
Helper utilities for configuration loading, logging, and file safety.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict
import yaml
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables
env_file = ROOT_DIR / ".env"
env_example = ROOT_DIR / ".env.example"
if env_file.exists():
    load_dotenv(dotenv_path=env_file)
elif env_example.exists():
    load_dotenv(dotenv_path=env_example)
else:
    load_dotenv()


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


class Settings:
    """Unified application settings combining config.yaml and environment variables."""

    def __init__(self):
        config_data = load_yaml_config(ROOT_DIR / "config.yaml")

        server_cfg = config_data.get("server", {})
        storage_cfg = config_data.get("storage", {})
        chunking_cfg = config_data.get("chunking", {})
        embedding_cfg = config_data.get("embedding", {})
        vector_cfg = config_data.get("vector_store", {})
        llm_cfg = config_data.get("llm", {})
        logging_cfg = config_data.get("logging", {})

        self.APP_NAME: str = config_data.get("app", {}).get("name", "RAG PDF Assistant")
        self.APP_VERSION: str = config_data.get("app", {}).get("version", "1.0.0")

        self.GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
        self.HOST: str = os.getenv("HOST", server_cfg.get("host", "0.0.0.0"))
        self.FASTAPI_PORT: int = int(os.getenv("FASTAPI_PORT", server_cfg.get("port", 8001)))
        self.PORT: int = int(os.getenv("PORT", 8000))
        self.API_URL: str = os.getenv("API_URL", server_cfg.get("api_url", "http://127.0.0.1:8001"))

        self.CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", chunking_cfg.get("chunk_size", 1000)))
        self.CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", chunking_cfg.get("chunk_overlap", 150)))

        self.EMBEDDING_MODEL_NAME: str = embedding_cfg.get("model_name", "all-MiniLM-L6-v2")
        self.EMBEDDING_TIMEOUT: int = embedding_cfg.get("timeout_seconds", 60)

        self.TOP_K: int = int(vector_cfg.get("top_k", 3))
        self.VECTOR_STORE_DIR: str = str(ROOT_DIR / storage_cfg.get("vector_store_dir", "vector_store"))
        self.UPLOAD_DIR: str = str(ROOT_DIR / storage_cfg.get("upload_dir", "uploads"))

        self.LLM_TEMPERATURE: float = float(llm_cfg.get("temperature", 0.2))
        self.LLM_MAX_TOKENS: int = int(llm_cfg.get("max_output_tokens", 1000))
        self.LLM_MODELS: list = llm_cfg.get("models", [
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-flash-latest",
            "gemini-3.1-flash-lite",
            "gemini-1.5-flash"
        ])

        self.LOG_FILE: str = str(ROOT_DIR / logging_cfg.get("log_file", "logs/app.log"))
        self.LOG_LEVEL: str = logging_cfg.get("log_level", "INFO")


settings = Settings()


def ensure_directories():
    """Create essential operational directories if they do not exist."""
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)


ensure_directories()


def get_logger(name: str) -> logging.Logger:
    """Initialize and configure a logger instance writing to console and log file."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        try:
            file_handler = logging.FileHandler(settings.LOG_FILE, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            pass

    return logger


def validate_file_path(file_path: str, base_dir: str = settings.UPLOAD_DIR) -> bool:
    """Validate that a target path resides within the authorized base directory."""
    try:
        resolved_target = Path(file_path).resolve()
        resolved_base = Path(base_dir).resolve()
        return str(resolved_target).startswith(str(resolved_base))
    except Exception:
        return False
