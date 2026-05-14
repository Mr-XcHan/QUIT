from __future__ import annotations

from pydantic import Field

from .research_brief import JsonModel


class IdeaCard(JsonModel):
    idea_id: str = Field(min_length=1)
    target_task: str = Field(min_length=1)
    novelty_claim: str = Field(min_length=1)
    supporting_evidence_ids: list[str] = Field(min_length=1)
    expected_gain: str = Field(min_length=1)
    priority_score: float = Field(default=0.0, ge=0.0)


class IdeaGenerationReport(JsonModel):
    status: str
    idea_count: int
    errors: list[str] = Field(default_factory=list)
