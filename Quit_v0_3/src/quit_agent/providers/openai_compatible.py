from __future__ import annotations

import os

from quit_agent.providers.config import LLMProviderConfig
from quit_agent.providers.http import post_json


class OpenAICompatibleClient:
    """Client for any provider that exposes the OpenAI chat-completions API.

    Covers: openai, codex, vllm, lmstudio, localai, groq, together, deepseek,
    openrouter, siliconflow, qwen (Alibaba DashScope), moonshot, glm (智谱AI),
    and any other OpenAI-compatible endpoint.
    """

    def __init__(self, config: LLMProviderConfig) -> None:
        self.config = config
        self.base_url = (config.base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = config.api_key or (os.environ.get(config.api_key_env) if config.api_key_env else None)

    def complete(self, prompt: str) -> str:
        is_vllm = self.config.provider.lower().strip() in {"vllm", "local-vllm"}
        # vLLM defaults to non-streaming. Some hosted proxies ignore
        # stream=False and only deliver content via SSE chunks, so hosted
        # OpenAI-compatible providers default to streaming.
        use_stream = self.config.stream if self.config.stream is not None else not is_vllm
        payload: dict = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": use_stream,
        }
        if self.config.extra_body:
            payload.update(self.config.extra_body)
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = post_json(
            url=f"{self.base_url}/chat/completions",
            payload=payload,
            headers=headers,
            timeout_seconds=self.config.timeout_seconds,
        )
        return response["choices"][0]["message"].get("content") or ""
