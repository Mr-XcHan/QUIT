from __future__ import annotations

from typing import Optional

from pydantic import Field

from .enums import FallbackTarget, IdeaDecisionType
from .research_brief import JsonModel


class LLMPaperReview(JsonModel):
    soundness: int = 0
    presentation: int = 0
    contribution: int = 0
    novelty: int = 0
    overall: int = 0
    confidence: int = 0
    summary: str = ""
    strengths: str = ""
    weaknesses: str = ""
    questions: str = ""
    improvement_hints: list[str] = Field(default_factory=list)


class IdeaDecision(JsonModel):
    idea_id: str
    decision: IdeaDecisionType
    reason: str
    fallback_target: FallbackTarget
    violations: list[str] = Field(default_factory=list)
    required_changes: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)


class ExperimentAudit(JsonModel):
    status: str
    failures: list[str] = Field(default_factory=list)
    fallback_target: FallbackTarget = FallbackTarget.CODE


class PaperReview(JsonModel):
    status: str
    failures: list[str] = Field(default_factory=list)
    fallback_target: FallbackTarget = FallbackTarget.WRITE
    llm_review: Optional[LLMPaperReview] = None
