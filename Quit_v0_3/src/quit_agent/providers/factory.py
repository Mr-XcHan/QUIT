from __future__ import annotations

from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from quit_agent.providers.anthropic_provider import AnthropicClient
from quit_agent.providers.config import LLMProviderConfig
from quit_agent.providers.ollama_provider import OllamaClient
from quit_agent.providers.openai_compatible import OpenAICompatibleClient
from quit_agent.providers.transformers_provider import TransformersClient


# (default_base_url, api_key_env)
OPENAI_COMPATIBLE_PROVIDERS: dict[str, tuple[str | None, str | None]] = {
    # OpenAI and Codex (same API, model name selects behaviour)
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "codex": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    # Generic OpenAI-compatible catch-all
    "openai-compatible": (None, None),
    # Local / self-hosted
    "vllm": ("http://localhost:8000/v1", None),
    "local-vllm": ("http://localhost:8000/v1", None),
    "lmstudio": ("http://localhost:1234/v1", None),
    "localai": ("http://localhost:8080/v1", None),
    # Hosted inference APIs
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "siliconflow": ("https://api.siliconflow.cn/v1", "SILICONFLOW_API_KEY"),
    # Chinese LLM providers
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "moonshot": ("https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
    "glm": ("https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY"),
}

KNOWN_HOST_PROVIDER_MAP: dict[str, tuple[str, str | None, str | None]] = {
    "api.openai.com": ("openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "api.anthropic.com": ("anthropic", "https://api.anthropic.com", "ANTHROPIC_API_KEY"),
    "api.groq.com": ("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "api.together.xyz": ("together", "https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "api.deepseek.com": ("deepseek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "openrouter.ai": ("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "api.siliconflow.cn": ("siliconflow", "https://api.siliconflow.cn/v1", "SILICONFLOW_API_KEY"),
    "dashscope.aliyuncs.com": ("qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "api.moonshot.cn": ("moonshot", "https://api.moonshot.cn/v1", "MOONSHOT_API_KEY"),
    "open.bigmodel.cn": ("glm", "https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY"),
}


def resolve_auto_provider_config(config: LLMProviderConfig) -> LLMProviderConfig:
    """Infer a concrete provider from a base URL.

    Detection order:
    1. Port 11434 or /api/tags endpoint → ollama
    2. Known hosted domain → mapped provider
    3. /v1/models endpoint → openai-compatible
    """

    if config.provider.lower().strip() != "auto":
        return config
    if not config.base_url:
        raise ValueError("provider='auto' requires base_url to be set.")

    base_url = config.base_url.rstrip("/")
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    host_with_port = f"{host}:{parsed.port}" if parsed.port else host

    if host_with_port.endswith(":11434") or _endpoint_exists(f"{base_url}/api/tags", config.timeout_seconds):
        return _clone(config, provider="ollama", base_url=base_url)

    for known_host, (provider, default_base_url, key_env) in KNOWN_HOST_PROVIDER_MAP.items():
        if host == known_host or host.endswith(f".{known_host}"):
            return _clone(config, provider=provider, base_url=config.base_url or default_base_url, api_key_env=key_env)

    if base_url.endswith("/v1") and _endpoint_exists(f"{base_url}/models", config.timeout_seconds):
        return _clone(config, provider="openai-compatible", base_url=base_url)
    if _endpoint_exists(f"{base_url}/v1/models", config.timeout_seconds):
        return _clone(config, provider="openai-compatible", base_url=f"{base_url}/v1")
    if _endpoint_exists(f"{base_url}/models", config.timeout_seconds):
        return _clone(config, provider="openai-compatible", base_url=base_url)

    raise ValueError(
        f"Could not auto-detect LLM provider from base_url={config.base_url!r}. "
        "Use an explicit provider name."
    )


def create_llm_client(config: LLMProviderConfig) -> AnthropicClient | OpenAICompatibleClient | OllamaClient | TransformersClient:
    """Instantiate the appropriate LLM client from a provider config."""

    provider = config.provider.lower().strip()
    if not provider:
        raise ValueError("No LLM provider configured. Set llm.provider in config.json.")

    if provider == "auto":
        config = resolve_auto_provider_config(config)
        provider = config.provider.lower().strip()

    # Anthropic / Claude Code
    if provider in {"anthropic", "claude-code", "claude_code"}:
        return AnthropicClient(config)

    # Ollama
    if provider == "ollama":
        return OllamaClient(config)

    # Local HuggingFace Transformers
    if provider in {"transformers", "hf", "huggingface"}:
        return TransformersClient(config)

    # OpenAI-compatible family
    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        default_base_url, default_key_env = OPENAI_COMPATIBLE_PROVIDERS[provider]
        resolved = LLMProviderConfig(
            provider=config.provider,
            model=config.model,
            base_url=config.base_url or default_base_url,
            api_key=config.api_key,
            api_key_env=config.api_key_env or default_key_env,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_seconds=config.timeout_seconds,
            stream=config.stream,
            extra_body=config.extra_body,
        )
        if not resolved.base_url:
            raise ValueError(f"provider='{provider}' requires base_url to be set.")
        return OpenAICompatibleClient(resolved)

    supported = sorted(
        ["auto", "anthropic", "claude-code", "ollama", "transformers", *OPENAI_COMPATIBLE_PROVIDERS.keys()]
    )
    raise ValueError(f"Unknown LLM provider '{config.provider}'. Supported: {', '.join(supported)}")


# ── internal helpers ──────────────────────────────────────────────────────────

def _endpoint_exists(url: str, timeout_seconds: int) -> bool:
    try:
        with urlopen(url, timeout=min(timeout_seconds, 5)) as response:
            return 200 <= response.status < 500
    except HTTPError as exc:
        return exc.code in {400, 401, 403, 405}
    except URLError:
        return False


def _clone(
    config: LLMProviderConfig,
    *,
    provider: str,
    base_url: str | None,
    api_key_env: str | None = None,
) -> LLMProviderConfig:
    return LLMProviderConfig(
        provider=provider,
        model=config.model,
        base_url=base_url,
        api_key=config.api_key,
        api_key_env=api_key_env or config.api_key_env,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout_seconds=config.timeout_seconds,
        stream=config.stream,
        extra_body=config.extra_body,
    )
