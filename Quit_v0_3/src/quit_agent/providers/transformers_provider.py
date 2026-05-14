from __future__ import annotations

from quit_agent.providers.config import LLMProviderConfig


class TransformersClient:
    """Optional local Hugging Face Transformers text-generation client.

    Requires: pip install transformers torch
    """

    def __init__(self, config: LLMProviderConfig) -> None:
        self.config = config
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Transformers provider requires optional dependencies. "
                "Install with: pip install transformers torch"
            ) from exc
        self._pipeline = pipeline("text-generation", model=config.model)

    def complete(self, prompt: str) -> str:
        outputs = self._pipeline(
            prompt,
            max_new_tokens=self.config.max_tokens,
            temperature=max(self.config.temperature, 1e-5),
            do_sample=self.config.temperature > 0,
        )
        text: str = outputs[0]["generated_text"]
        return text[len(prompt):].strip()
