from __future__ import annotations

from pydantic import Field

from .enums import PaperStatus
from .research_brief import JsonModel


class PaperCard(JsonModel):
    paper_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    authors: list[str] = Field(default_factory=list)
    year: int = Field(ge=1900, le=2100)
    venue: str = ""
    abstract: str = ""
    source: str = "mock"
    paper_url: str = ""
    pdf_url: str = ""
    code_url: str = ""
    query_source: str = ""
    status: PaperStatus = PaperStatus.FOUND
    local_pdf_path: str = ""
    retrieval_score: float = Field(default=0.0, ge=0.0)
