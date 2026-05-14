from __future__ import annotations

from typing import Protocol


class Retriever(Protocol):
    def search(self, query: str, max_results: int) -> list[dict]:
        """Return candidate paper dictionaries for a query."""
