from __future__ import annotations

import json

from quit_agent.agents.builder_agent import BuilderAgent
from quit_agent.agents.planner_agent import PlannerAgent
from quit_agent.agents.research_agent import ResearchAgent
from quit_agent.agents.reviewer_agent import ReviewerAgent
from quit_agent.artifacts.manager import ArtifactManager
from quit_agent.artifacts.trace import TraceManager
from quit_agent.orchestrator.state_machine import StateMachine
from quit_agent.schemas.enums import WorkflowState
from quit_agent.tools.retrievers import MockRetriever


class EchoPromptLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps(
            {
                "topic": "artifact driven research planning",
                "objective": "preserve user quality requirements",
                "search_keywords": ["research agent", "artifact workflow", "local planning"],
                "domain": ["research automation"],
                "constraints": ["local-first"],
                "deliverable": ["ResearchBrief"],
                "search_budget": {
                    "max_queries": 4,
                    "max_papers_screened": 20,
                    "max_papers_selected": 5,
                    "max_repo_checked": 3,
                    "stop_if_no_new_signal_rounds": 2,
                },
                "build_budget": {
                    "max_code_iterations": 2,
                    "max_experiments": 2,
                    "max_review_revisions": 2,
                },
                "red_lines": ["unsupported claims"],
                "acceptance_criteria": ["valid structured plan"],
                "fallback_policy": {
                    "supervise": "emit artifact and stop after repeated failure",
                    "code_fail": "return to BUILD_SPEC",
                    "write_fail": "return to WRITE",
                },
            }
        )


class BadThenGoodLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        if self.calls == 1:
            return "{bad json"
        return json.dumps(
            {
                "topic": "replanned artifact research agent",
                "domain": ["research automation"],
                "constraints": ["local-first"],
                "deliverable": ["ResearchBrief"],
            }
        )


class ValidPlanningLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        return json.dumps(
            {
                "topic": "retrieval recovery research agent",
                "domain": ["research automation"],
                "constraints": ["local-first"],
                "deliverable": ["ResearchBrief"],
                "search_budget": {
                    "max_queries": 1,
                    "max_papers_screened": 5,
                    "max_papers_selected": 10,
                    "max_repo_checked": 1,
                    "stop_if_no_new_signal_rounds": 1,
                },
                "build_budget": {
                    "max_code_iterations": 1,
                    "max_experiments": 1,
                    "max_review_revisions": 1,
                },
                "red_lines": ["unsupported claims"],
                "acceptance_criteria": ["enough retrieved papers"],
                "fallback_policy": {
                    "supervise": "emit artifact and stop after repeated failure",
                    "code_fail": "return to BUILD_SPEC",
                    "write_fail": "return to WRITE",
                },
            }
        )


class EmptyRetriever:
    def search(self, query: str, max_results: int) -> list[dict]:
        return []


def test_first_plan_uses_plan_skill_template(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    llm = EchoPromptLLM()
    planner = PlannerAgent(llm, artifacts)

    prompt, response, artifact_name = planner.generate_brief("build a local research agent")

    assert artifact_name == "ResearchBrief.raw.json"
    assert response == artifacts.path("ResearchBrief.raw.json").read_text(encoding="utf-8")
    assert prompt == llm.prompts[0]
    assert "You are a research planning assistant" in prompt
    assert "User request:\nbuild a local research agent" in prompt
    assert '"max_queries": 4' in prompt
    assert '"objective"' in prompt
    assert "{{max_queries}}" not in prompt
    assert "{{user_request}}" not in prompt


def test_validation_failure_uses_replan_skill(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    llm = BadThenGoodLLM()
    machine = StateMachine(
        artifacts=artifacts,
        trace=TraceManager(artifacts.paths),
        planner=PlannerAgent(llm, artifacts),
        researcher=ResearchAgent(MockRetriever(), artifacts),
        reviewer=ReviewerAgent(artifacts),
        builder=BuilderAgent(artifacts),
    )

    assert machine.step(WorkflowState.PLAN, "test request") == WorkflowState.VALIDATE_BRIEF
    assert machine.step(WorkflowState.VALIDATE_BRIEF, "test request") == WorkflowState.PLAN
    assert machine.step(WorkflowState.PLAN, "test request") == WorkflowState.VALIDATE_BRIEF

    summary = artifacts.read_json("run_trace.json")
    assert summary["steps"][0]["skill"] == "plan_research_brief"
    assert summary["steps"][2]["skill"] == "replan_research_brief"
    assert '"search_keywords"' in llm.prompts[1]
    assert '"max_queries": 4' in llm.prompts[1]
    assert "{{max_queries}}" not in llm.prompts[1]


def test_retrieval_failure_uses_replan_skill(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    llm = ValidPlanningLLM()
    machine = StateMachine(
        artifacts=artifacts,
        trace=TraceManager(artifacts.paths),
        planner=PlannerAgent(llm, artifacts),
        researcher=ResearchAgent(EmptyRetriever(), artifacts),
        reviewer=ReviewerAgent(artifacts),
        builder=BuilderAgent(artifacts),
    )

    assert machine.step(WorkflowState.PLAN, "test request") == WorkflowState.VALIDATE_BRIEF
    assert machine.step(WorkflowState.VALIDATE_BRIEF, "test request") == WorkflowState.RETRIEVE
    assert machine.step(WorkflowState.RETRIEVE, "test request") == WorkflowState.PLAN
    assert machine.step(WorkflowState.PLAN, "test request") == WorkflowState.VALIDATE_BRIEF

    summary = artifacts.read_json("run_trace.json")
    replan_trace = artifacts.read_json("trace/0004_plan.json")
    assert artifacts.read_json("RetrievalReport.json")["status"] == "FAIL"
    assert summary["steps"][3]["skill"] == "replan_research_brief"
    assert replan_trace["input_artifacts"] == ["ResearchBrief.raw.json", "RetrievalReport.json"]
    assert '"search_keywords"' in llm.prompts[1]
    assert '"max_queries": 4' in llm.prompts[1]
    assert "{{max_queries}}" not in llm.prompts[1]
