from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    root: Path
    run_id: str

    @property
    def run_dir(self) -> Path:
        return self.root / self.run_id

    @property
    def trace_dir(self) -> Path:
        return self.run_dir / "trace"

    @property
    def llm_dir(self) -> Path:
        return self.run_dir / "llm"

    @property
    def extracts_dir(self) -> Path:
        return self.root.parent / "extracts"
