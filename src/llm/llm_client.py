"""
Language Model client integrating with Google Gemini REST API.
"""

import time
import requests
from typing import List, Dict, Any, Optional

from src.utils.helpers import settings, get_logger

logger = get_logger("llm.llm_client")


class LLMClient:
    """Client for generating completions using Google Gemini REST API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        models: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        self.api_key = api_key if api_key is not None else settings.GOOGLE_API_KEY
        self.models = models or settings.LLM_MODELS
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS

    def is_configured(self) -> bool:
        """Check if a valid Google Gemini API key is configured."""
        return bool(self.api_key and self.api_key.strip() and self.api_key != "your_gemini_api_key_here")

    def generate(self, prompt: str) -> str:
        """
        Send prompt to Google Gemini REST API and return the synthesized response text.
        """
        if not self.is_configured():
            logger.warning("Google API Key not configured. Skipping remote LLM call.")
            return ""

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens
            }
        }

        for model_name in self.models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
            try:
                logger.info(f"Invoking Gemini model: {model_name}")
                response = requests.post(url, json=payload, headers=headers, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        for part in parts:
                            text_snippet = part.get("text", "").strip()
                            if text_snippet:
                                logger.info(f"Model {model_name} generated response successfully.")
                                return text_snippet
                elif response.status_code == 429:
                    logger.warning(f"Rate limit 429 received on model {model_name}. Retrying next model.")
                    time.sleep(1)
                else:
                    logger.warning(f"Model {model_name} returned HTTP {response.status_code}: {response.text[:200]}")
            except Exception as e:
                logger.error(f"Error calling model {model_name}: {e}")

        return ""

    def generate_fallback_summary(self, sources: List[Dict[str, Any]]) -> str:
        """
        Provide clean structured summary from retrieved passages when API is not available.
        """
        if not sources:
            return "No relevant context passages found in document."

        summary_lines = ["**Summary of Relevant Document Passages:**\n"]
        for src in sources:
            page_num = src.get("page", 1)
            content = src.get("content", "")
            summary_lines.append(f"* **Page {page_num}**: {content}\n")

        return "\n".join(summary_lines)
