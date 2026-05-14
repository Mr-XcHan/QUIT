from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str:
        """Return raw model text for a prompt."""


class FileLLMClient(Protocol):
    def complete_with_files(self, prompt: str, file_paths: list[str]) -> str:
        """Return raw model text for a prompt plus local file inputs."""
