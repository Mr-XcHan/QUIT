from __future__ import annotations

from pydantic import Field

from .research_brief import JsonModel


class CodeRunReport(JsonModel):
    status: str
    code_dir: str
    generation_mode: str = ""
    fallback_used: bool = False
    repair_attempted: bool = False
    repair_succeeded: bool = False
    outputs: list[str] = Field(default_factory=list)
    executed: bool = False
    returncode: int | None = None
    errors: list[str] = Field(default_factory=list)
