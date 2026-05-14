from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class LLMProviderConfig:
    """Configuration for selecting and calling an LLM provider."""

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.0
    max_tokens: int = 8192
    timeout_seconds: int = 600
    stream: bool | None = None
    extra_body: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data.get("api_key"):
            data["api_key"] = "***"
        return data
