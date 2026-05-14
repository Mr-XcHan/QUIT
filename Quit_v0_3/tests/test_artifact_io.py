from __future__ import annotations

from quit_agent.artifacts.manager import ArtifactManager
from quit_agent.artifacts.trace import TraceManager


def test_json_and_jsonl_roundtrip(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    artifacts.write_json("x.json", {"a": 1})
    artifacts.write_jsonl("items.jsonl", [{"i": 1}, {"i": 2}])
    assert artifacts.read_json("x.json") == {"a": 1}
    assert artifacts.read_jsonl("items.jsonl") == [{"i": 1}, {"i": 2}]


def test_trace_manager_writes_step_and_summary_files(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    trace = TraceManager(artifacts.paths)
    trace_path = trace.record_step(
        step_id="0001",
        state="PLAN",
        agent="PlannerAgent",
        skill="plan_research_brief",
        input_artifacts=[],
        output_artifacts=["ResearchBrief.raw.json"],
        prompt_text="prompt",
        response_text="response",
        parsed_output_preview={"x": 1},
        validation_result="RAW",
        next_state="VALIDATE_BRIEF",
        fallback_decision=None,
    )
    assert trace_path.exists()
    assert artifacts.path("llm/0001_plan_prompt.txt").exists()
    assert artifacts.path("llm/0001_plan_response.txt").exists()
    summary = artifacts.read_json("run_trace.json")
    assert summary["steps"][0]["state"] == "PLAN"
