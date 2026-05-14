from __future__ import annotations

from pydantic import Field

from .research_brief import JsonModel


class EvidenceCard(JsonModel):
    evidence_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    task: str = Field(min_length=1)
    method: str = Field(min_length=1)
    setting: str = ""
    claims: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    transferable_idea_seeds: list[str] = Field(default_factory=list)


class EvidenceValidationReport(JsonModel):
    status: str
    errors: list[str] = Field(default_factory=list)
    missing_paper_ids: list[str] = Field(default_factory=list)
    malformed_count: int = 0
