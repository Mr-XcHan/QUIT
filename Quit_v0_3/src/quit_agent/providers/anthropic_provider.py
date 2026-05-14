from __future__ import annotations

import base64
import os
from pathlib import Path

from quit_agent.providers.config import LLMProviderConfig
from quit_agent.providers.http import post_json


class AnthropicClient:
    """Client for the Anthropic Messages API using stdlib HTTP.

    Covers claude-opus-*, claude-sonnet-*, claude-haiku-*, and any future
    claude-* models.  Use provider="anthropic" or provider="claude-code".
    """

    def __init__(self, config: LLMProviderConfig) -> None:
        self.config = config
        self.base_url = (config.base_url or "https://api.anthropic.com").rstrip("/")
        self.api_key = config.api_key or os.environ.get(config.api_key_env or "ANTHROPIC_API_KEY")

    def complete(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("Anthropic provider requires an API key or ANTHROPIC_API_KEY env var.")
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        response = post_json(
            url=f"{self.base_url}/v1/messages",
            payload=payload,
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            timeout_seconds=self.config.timeout_seconds,
        )
        parts = response.get("content", [])
        return "".join(part.get("text", "") for part in parts if part.get("type") == "text")

    def complete_with_files(self, prompt: str, file_paths: list[str]) -> str:
        if not self.api_key:
            raise RuntimeError("Anthropic provider requires an API key or ANTHROPIC_API_KEY env var.")
        content = []
        for raw_path in file_paths:
            path = Path(raw_path)
            if path.suffix.lower() != ".pdf":
                raise ValueError(f"Anthropic file input currently supports PDF files only: {path}")
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": data,
                    },
                }
            )
        content.append({"type": "text", "text": prompt})
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        response = post_json(
            url=f"{self.base_url}/v1/messages",
            payload=payload,
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            timeout_seconds=self.config.timeout_seconds,
        )
        parts = response.get("content", [])
        return "".join(part.get("text", "") for part in parts if part.get("type") == "text")
