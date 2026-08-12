"""
================================================================================
Application Configuration Module
--------------------------------------------------------------------------------
References & Documentation Sources:
- python-dotenv Official Docs: https://github.com/theskumar/python-dotenv
- Python pathlib Official Guide: https://docs.python.org/3/library/pathlib.html
- Pydantic Settings Conventions: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
================================================================================
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# Load environment variables from local .env or .env.example configuration file.
# Source Pattern: https://github.com/theskumar/python-dotenv#getting-started
# ------------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
env_path = ROOT_DIR / ".env"
env_example_path = ROOT_DIR / ".env.example"

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
elif env_example_path.exists():
    load_dotenv(dotenv_path=env_example_path)
else:
    load_dotenv()


class Settings:
    """
    Application Settings Class storing environment variables and default parameters.
    Source Pattern: Standard Python Object Configuration pattern.
    """
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    VECTOR_STORE_DIR: str = str(ROOT_DIR / os.getenv("VECTOR_STORE_DIR", "vector_store"))
    UPLOAD_DIR: str = str(ROOT_DIR / "uploads")
    API_URL: str = os.getenv("API_URL", "http://127.0.0.1:8000")
    
    @staticmethod
    def validate_file_path(file_path: str) -> bool:
        """
        Validate that file path is safe and does not contain path traversal attacks.
        Ensures the resolved path is within UPLOAD_DIR.
        """
        try:
            resolved_path = Path(file_path).resolve()
            upload_dir = Path(Settings.UPLOAD_DIR).resolve()
            return str(resolved_path).startswith(str(upload_dir))
        except Exception:
            return False


# Instantiate global settings object
settings = Settings()

# Ensure required working directories exist on filesystem
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.VECTOR_STORE_DIR, exist_ok=True)
