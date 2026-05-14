from __future__ import annotations

from quit_agent.agents.builder_agent import BuilderAgent
from quit_agent.agents.planner_agent import PlannerAgent
from quit_agent.agents.research_agent import ResearchAgent
from quit_agent.agents.reviewer_agent import ReviewerAgent
from quit_agent.artifacts.manager import ArtifactManager
from quit_agent.artifacts.trace import TraceManager
from quit_agent.orchestrator.state_machine import StateMachine
from quit_agent.schemas.enums import WorkflowState
from quit_agent.tools.retrievers import MockRetriever


class StubLLMClient:
    def complete(self, prompt: str) -> str:
        return """
{
  "topic": "artifact-driven research agents",
  "domain": ["research automation"],
  "constraints": ["local artifacts"],
  "deliverable": ["ranked idea"],
  "search_budget": {
    "max_queries": 3,
    "max_papers_screened": 20,
    "max_papers_selected": 5,
    "max_repo_checked": 3,
    "stop_if_no_new_signal_rounds": 2
  },
  "build_budget": {
    "max_code_iterations": 2,
    "max_experiments": 2,
    "max_review_revisions": 2
  },
  "red_lines": ["fabricated results"],
  "acceptance_criteria": ["evidence-backed idea"],
  "fallback_policy": {
    "supervise": "emit artifact and stop after repeated failure",
    "code_fail": "return to BUILD_SPEC",
    "write_fail": "return to WRITE"
  }
}
"""


def test_stop_after_executes_requested_state(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    machine = StateMachine(
        artifacts=artifacts,
        trace=TraceManager(artifacts.paths),
        planner=PlannerAgent(StubLLMClient(), artifacts),
        researcher=ResearchAgent(MockRetriever(), artifacts),
        reviewer=ReviewerAgent(artifacts),
        builder=BuilderAgent(artifacts),
    )

    result = machine.run("artifact-driven research agents", stop_after=WorkflowState.VALIDATE_BRIEF)

    assert result.final_state == WorkflowState.VALIDATE_BRIEF
    assert result.next_state == WorkflowState.RETRIEVE
    assert artifacts.path("ResearchBrief.raw.json").exists()
    assert artifacts.path("ResearchBrief.json").exists()
    assert artifacts.path("ResearchBriefValidationReport.json").exists()
    assert not artifacts.path("PaperCards.jsonl").exists()


def test_start_at_retrieve_uses_existing_brief(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    machine = StateMachine(
        artifacts=artifacts,
        trace=TraceManager(artifacts.paths),
        planner=PlannerAgent(StubLLMClient(), artifacts),
        researcher=ResearchAgent(MockRetriever(), artifacts),
        reviewer=ReviewerAgent(artifacts),
        builder=BuilderAgent(artifacts),
    )
    machine.run("artifact-driven research agents", stop_after=WorkflowState.VALIDATE_BRIEF)

    resumed = StateMachine(
        artifacts=artifacts,
        trace=TraceManager(artifacts.paths),
        planner=PlannerAgent(StubLLMClient(), artifacts),
        researcher=ResearchAgent(MockRetriever(), artifacts),
        reviewer=ReviewerAgent(artifacts),
        builder=BuilderAgent(artifacts),
    )
    result = resumed.run(
        "artifact-driven research agents",
        start_at=WorkflowState.RETRIEVE,
        stop_after=WorkflowState.RETRIEVE,
    )

    assert result.final_state == WorkflowState.RETRIEVE
    assert result.next_state == WorkflowState.READ
    assert artifacts.path("PaperCards.jsonl").exists()
    summary = artifacts.read_json("run_trace.json")
    assert summary["steps"][-1]["step_id"] == "0003"
    assert summary["steps"][-1]["state"] == "RETRIEVE"


def test_code_failure_routes_to_code_repair(tmp_path):
    from quit_agent.schemas.build_spec import BuildSpec
    from quit_agent.schemas.code_artifacts import CodeRunReport

    class FailingBuilder:
        def code(self, spec, evidence=None):
            report = CodeRunReport(
                status="FAIL",
                code_dir="code",
                outputs=["CodeRunReport.json"],
                executed=True,
                returncode=1,
                errors=["boom"],
            )
            artifacts.write_json("CodeRunReport.json", report)
            return report, "code prompt", "code failed"

    artifacts = ArtifactManager(tmp_path, "run")
    artifacts.write_json(
        "BuildSpec.json",
        BuildSpec(
            build_id="build-1",
            idea_id="idea-1",
            target_task="generic task",
            problem_statement="problem",
            method_summary="method",
        ),
    )
    machine = StateMachine(
        artifacts=artifacts,
        trace=TraceManager(artifacts.paths),
        planner=PlannerAgent(StubLLMClient(), artifacts),
        researcher=ResearchAgent(MockRetriever(), artifacts),
        reviewer=ReviewerAgent(artifacts),
        builder=FailingBuilder(),
    )

    next_state = machine.step(WorkflowState.CODE, "request")

    assert next_state == WorkflowState.CODE
    trace = artifacts.read_json("run_trace.json")
    assert trace["steps"][-1]["state"] == "CODE"
    assert trace["steps"][-1]["next_state"] == "CODE"
    assert trace["steps"][-1]["fallback_decision"] == "CODE_REPAIR"


def test_progress_reporter_marks_state_and_code_failure(tmp_path):
    from quit_agent.schemas.build_spec import BuildSpec
    from quit_agent.schemas.code_artifacts import CodeRunReport

    messages = []

    class FailingBuilder:
        def code(self, spec, evidence=None):
            report = CodeRunReport(
                status="FAIL",
                code_dir="code",
                outputs=["CodeRunReport.json"],
                executed=True,
                returncode=1,
                errors=["Traceback\nRuntimeError: boom"],
            )
            artifacts.write_json("CodeRunReport.json", report)
            return report, "code prompt", "code failed"

    artifacts = ArtifactManager(tmp_path, "run")
    artifacts.write_json(
        "BuildSpec.json",
        BuildSpec(
            build_id="build-1",
            idea_id="idea-1",
            target_task="generic task",
            problem_statement="problem",
            method_summary="method",
        ),
    )
    machine = StateMachine(
        artifacts=artifacts,
        trace=TraceManager(artifacts.paths),
        planner=PlannerAgent(StubLLMClient(), artifacts),
        researcher=ResearchAgent(MockRetriever(), artifacts),
        reviewer=ReviewerAgent(artifacts),
        builder=FailingBuilder(),
        reporter=messages.append,
    )

    machine.run("request", start_at=WorkflowState.CODE, stop_after=WorkflowState.CODE)

    joined = "\n".join(messages)
    assert "START CODE" in joined
    assert "END   CODE -> CODE" in joined
    assert "CODE needs attention" in joined
    assert "RuntimeError: boom" in joined


def test_write_failure_stops_without_write_eval(tmp_path):
    from quit_agent.schemas.build_spec import BuildSpec

    class FailingWriteBuilder:
        def write(self, spec, evidence, papers):
            report = {
                "status": "FAIL",
                "reason": "writer LLM request failed: HTTP provider request failed: 524",
                "outputs": ["paper_gene/references.bib"],
            }
            artifacts.write_json("WriteReport.json", report)
            return report, "write prompt", ""

    artifacts = ArtifactManager(tmp_path, "run")
    artifacts.write_json(
        "BuildSpec.json",
        BuildSpec(
            build_id="build-1",
            idea_id="idea-1",
            target_task="generic task",
            problem_statement="problem",
            method_summary="method",
        ),
    )
    machine = StateMachine(
        artifacts=artifacts,
        trace=TraceManager(artifacts.paths),
        planner=PlannerAgent(StubLLMClient(), artifacts),
        researcher=ResearchAgent(MockRetriever(), artifacts),
        reviewer=ReviewerAgent(artifacts),
        builder=FailingWriteBuilder(),
    )

    next_state = machine.step(WorkflowState.WRITE, "request")

    assert next_state == WorkflowState.STOP
    trace = artifacts.read_json("run_trace.json")
    assert trace["steps"][-1]["state"] == "WRITE"
    assert trace["steps"][-1]["next_state"] == "STOP"
    assert trace["steps"][-1]["fallback_decision"] == "WRITE_FAILED"


def test_write_compile_failure_routes_through_write_eval_back_to_write(tmp_path):
    from quit_agent.schemas.build_spec import BuildSpec

    class CompileFailWriteBuilder:
        def write(self, spec, evidence, papers):
            artifacts.path("paper_gene").mkdir(parents=True)
            artifacts.path("paper_gene/main.tex").write_text("\\documentclass{article}\\begin{document}broken\\end{document}\n", encoding="utf-8")
            report = {
                "status": "FAIL",
                "outputs": ["paper_gene/main.tex"],
                "compile": {"status": "FAIL", "reason": "Undefined control sequence"},
            }
            artifacts.write_json("WriteReport.json", report)
            return report, "write prompt", "bad tex"

    artifacts = ArtifactManager(tmp_path, "run")
    artifacts.write_json(
        "BuildSpec.json",
        BuildSpec(
            build_id="build-1",
            idea_id="idea-1",
            target_task="generic task",
            problem_statement="problem",
            method_summary="method",
        ),
    )
    machine = StateMachine(
        artifacts=artifacts,
        trace=TraceManager(artifacts.paths),
        planner=PlannerAgent(StubLLMClient(), artifacts),
        researcher=ResearchAgent(MockRetriever(), artifacts),
        reviewer=ReviewerAgent(artifacts),
        builder=CompileFailWriteBuilder(),
    )

    next_state = machine.step(WorkflowState.WRITE, "request")
    assert next_state == WorkflowState.WRITE_EVAL

    next_state = machine.step(WorkflowState.WRITE_EVAL, "request")
    assert next_state == WorkflowState.WRITE
    review = artifacts.read_json("PaperReview.json")
    assert review["status"] == "FAIL"
    assert any("Undefined control sequence" in item for item in review["failures"])
