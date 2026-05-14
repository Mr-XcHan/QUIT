from __future__ import annotations

from typing import Literal

from pydantic import Field

from .research_brief import JsonModel


class PaperSections(JsonModel):
    abstract: str = ""
    introduction: str = ""
    method: str = ""
    experiments: str = ""
    limitations: str = ""
    conclusion: str = ""


class PaperText(JsonModel):
    paper_id: str = Field(min_length=1)
    title: str = ""
    abstract: str = ""
    full_text: str = ""
    sections: PaperSections = Field(default_factory=PaperSections)
    source_path: str = ""
    extraction_status: Literal["success", "partial", "failed"] = "partial"
    errors: list[str] = Field(default_factory=list)
