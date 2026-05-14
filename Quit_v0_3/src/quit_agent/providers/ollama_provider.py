from __future__ import annotations

from quit_agent.providers.config import LLMProviderConfig
from quit_agent.providers.http import post_json


class OllamaClient:
    """Client for a local Ollama server."""

    def __init__(self, config: LLMProviderConfig) -> None:
        self.config = config
        self.base_url = (config.base_url or "http://localhost:11434").rstrip("/")

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.config.model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        response = post_json(
            url=f"{self.base_url}/api/chat",
            payload=payload,
            timeout_seconds=self.config.timeout_seconds,
        )
        return response.get("message", {}).get("content", "")
