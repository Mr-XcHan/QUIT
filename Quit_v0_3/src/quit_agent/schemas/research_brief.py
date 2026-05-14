from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .enums import ValidationStatus


class JsonModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json_text(self) -> str:
        return self.model_dump_json(indent=2)


class SearchBudget(JsonModel):
    max_queries: int = Field(default=3, ge=1)
    max_papers_screened: int = Field(default=20, ge=1)
    max_papers_selected: int = Field(default=5, ge=1)
    max_repo_checked: int = Field(default=3, ge=0)
    stop_if_no_new_signal_rounds: int = Field(default=2, ge=1)


class BuildBudget(JsonModel):
    max_code_iterations: int = Field(default=2, ge=1)
    max_experiments: int = Field(default=2, ge=1)
    max_review_revisions: int = Field(default=2, ge=1)


class FallbackPolicy(JsonModel):
    supervise: str = "emit artifact and stop after repeated failure"
    code_fail: str = "return to BUILD_SPEC"
    write_fail: str = "return to WRITE"


class ResearchBrief(JsonModel):
    topic: str = Field(min_length=3)
    objective: str = ""
    search_keywords: list[str] = Field(default_factory=list)
    domain: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    deliverable: list[str] = Field(default_factory=lambda: ["ranked idea"])
    search_budget: SearchBudget = Field(default_factory=SearchBudget)
    build_budget: BuildBudget = Field(default_factory=BuildBudget)
    red_lines: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    fallback_policy: FallbackPolicy = Field(default_factory=FallbackPolicy)


class ResearchBriefValidationReport(JsonModel):
    status: ValidationStatus
    errors: list[str] = Field(default_factory=list)
    repair_attempted: bool = False
    repaired: bool = False
    fallback_target: Literal["PLAN", "RETRIEVE"] | None = None
