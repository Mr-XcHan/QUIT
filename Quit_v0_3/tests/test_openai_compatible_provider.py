from __future__ import annotations

from quit_agent.providers.config import LLMProviderConfig
from quit_agent.providers.openai_compatible import OpenAICompatibleClient


def test_vllm_does_not_send_extra_body_by_default(monkeypatch):
    seen: dict[str, object] = {}

    def fake_post_json(**kwargs):
        seen.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr("quit_agent.providers.openai_compatible.post_json", fake_post_json)

    client = OpenAICompatibleClient(
        LLMProviderConfig(provider="vllm", model="Mistral-24B", base_url="http://localhost:8000/v1")
    )
    assert client.complete("hello") == "ok"

    payload = seen["payload"]
    assert isinstance(payload, dict)
    assert payload["stream"] is False
    assert "chat_template_kwargs" not in payload


def test_openai_compatible_sends_configured_extra_body(monkeypatch):
    seen: dict[str, object] = {}

    def fake_post_json(**kwargs):
        seen.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr("quit_agent.providers.openai_compatible.post_json", fake_post_json)

    client = OpenAICompatibleClient(
        LLMProviderConfig(
            provider="vllm",
            model="Qwen3.5-9B",
            base_url="http://localhost:8000/v1",
            stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}, "top_p": 0.9},
        )
    )
    assert client.complete("hello") == "ok"

    payload = seen["payload"]
    assert isinstance(payload, dict)
    assert payload["stream"] is True
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["top_p"] == 0.9
