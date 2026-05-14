from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, TypeVar

from pydantic import BaseModel

from .io import read_json, read_jsonl, write_json, write_jsonl
from .paths import RunPaths

T = TypeVar("T", bound=BaseModel)


class ArtifactManager:
    """Creates run directories and persists JSON, JSONL, and Markdown artifacts."""

    def __init__(self, runs_root: str | Path = "runs", run_id: str = "demo") -> None:
        self.paths = RunPaths(Path(runs_root), run_id)
        self.paths.run_dir.mkdir(parents=True, exist_ok=True)
        self.paths.trace_dir.mkdir(parents=True, exist_ok=True)
        self.paths.llm_dir.mkdir(parents=True, exist_ok=True)

    @property
    def run_dir(self) -> Path:
        return self.paths.run_dir

    def path(self, name: str) -> Path:
        return self.run_dir / name

    def write_json(self, name: str, payload: Any) -> Path:
        if isinstance(payload, BaseModel):
            payload = payload.model_dump(mode="json")
        return write_json(self.path(name), payload)

    def read_json(self, name: str) -> Any:
        return read_json(self.path(name))

    def load_model(self, name: str, model_type: type[T]) -> T:
        return model_type.model_validate(self.read_json(name))

    def write_jsonl(self, name: str, records: Iterable[Any]) -> Path:
        normalized = [item.model_dump(mode="json") if isinstance(item, BaseModel) else item for item in records]
        return write_jsonl(self.path(name), normalized)

    def read_jsonl(self, name: str) -> list[dict[str, Any]]:
        return read_jsonl(self.path(name))

    def load_jsonl_models(self, name: str, model_type: type[T]) -> list[T]:
        return [model_type.model_validate(item) for item in self.read_jsonl(name)]

    def write_markdown(self, name: str, text: str) -> Path:
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path
