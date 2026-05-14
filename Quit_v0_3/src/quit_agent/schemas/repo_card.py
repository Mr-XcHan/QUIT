from __future__ import annotations

from pydantic import Field

from .research_brief import JsonModel


class RepoCard(JsonModel):
    """Repository metadata discovered during RETRIEVE.

    RepoCard is intentionally small: it captures enough environment signal for
    BUILD_SPEC and CODE without making later stages read full repository history.
    """

    repo_id: str = Field(min_length=1)
    repo_url: str = Field(min_length=1)
    source_paper_id: str = ""
    source_title: str = ""
    local_repo_path: str = ""
    env_files: list[str] = Field(default_factory=list)
    language: str = ""
    framework: str = ""
    status: str = "found"
    relevance_score: float = Field(default=0.0, ge=0.0)
    errors: list[str] = Field(default_factory=list)
