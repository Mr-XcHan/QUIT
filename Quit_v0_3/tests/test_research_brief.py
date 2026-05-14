from __future__ import annotations

from quit_agent.artifacts.manager import ArtifactManager
from quit_agent.schemas.enums import ValidationStatus
from quit_agent.validators.research_brief_validator import ResearchBriefValidator


def test_valid_brief_passes(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    payload = {
        "topic": "artifact driven research agents",
        "objective": "produce high quality code and paper artifacts",
        "domain": ["agents"],
        "constraints": ["local-first"],
        "deliverable": ["idea"],
    }
    brief, report = ResearchBriefValidator(artifacts).validate_or_repair(payload)
    assert brief is not None
    assert report.status == ValidationStatus.PASS
    assert brief.objective == "produce high quality code and paper artifacts"
    assert artifacts.path("ResearchBrief.json").exists()


def test_malformed_brief_repairs_defaults_and_types(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    payload = {
        "topic": "artifact driven research agents",
        "domain": "agents",
        "constraints": "local-first",
        "deliverable": "idea",
        "search_budget": {"max_queries": "2"},
    }
    brief, report = ResearchBriefValidator(artifacts).validate_or_repair(payload)
    assert brief is not None
    assert report.status == ValidationStatus.REPAIR
    assert brief.domain == ["agents"]
    assert brief.search_budget.max_queries == 2
    assert brief.build_budget.max_code_iterations == 2


def test_unrecoverable_brief_writes_failure_report(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    brief, report = ResearchBriefValidator(artifacts).validate_or_repair("{bad json")
    assert brief is None
    assert report.status == ValidationStatus.FAIL
    assert artifacts.path("ResearchBriefValidationReport.json").exists()


def test_brief_validator_extracts_json_from_model_reasoning(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    raw = """
Thinking Process:
I will now produce JSON.

{
  "topic": "offline rl policy generalization",
  "domain": ["offline reinforcement learning"],
  "constraints": "local-first",
  "deliverable": "ranked ideas"
}
"""
    brief, report = ResearchBriefValidator(artifacts).validate_or_repair(raw)
    assert brief is not None
    assert report.status == ValidationStatus.REPAIR
    assert brief.constraints == ["local-first"]
