from __future__ import annotations

import struct
import zlib

from quit_agent.agents.reviewer_agent import ReviewerAgent
from quit_agent.artifacts.manager import ArtifactManager
from quit_agent.schemas.build_spec import BuildSpec, ExperimentPlotSpec
from quit_agent.schemas.enums import FallbackTarget, IdeaDecisionType
from quit_agent.schemas.idea_card import IdeaCard
from quit_agent.schemas.research_brief import ResearchBrief


class ReviewLLM:
    def complete(self, prompt: str) -> str:
        return """{
          "idea_id": "idea-1",
          "decision": "PASS",
          "reason": "evidence backed and specific",
          "fallback_target": "BUILD_SPEC",
          "violations": [],
          "required_changes": [],
          "missing_evidence": []
        }"""


class BadReviewLLM:
    def complete(self, prompt: str) -> str:
        return "not json"


def test_reviewer_agent_uses_llm_decision(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    reviewer = ReviewerAgent(artifacts, llm=ReviewLLM())
    brief = ResearchBrief(topic="offline RL")
    ideas = [
        IdeaCard(
            idea_id="idea-1",
            target_task="offline RL",
            novelty_claim="specific evidence backed idea",
            supporting_evidence_ids=["ev-1"],
            expected_gain="better return",
        )
    ]

    decision, prompt, response = reviewer.evaluate_idea(brief, ideas)

    assert "ResearchBrief" in prompt
    assert "idea-1" in response
    assert decision.decision == IdeaDecisionType.PASS
    assert decision.fallback_target == FallbackTarget.BUILD_SPEC
    assert artifacts.path("IdeaDecision.json").exists()


def test_reviewer_agent_falls_back_on_bad_llm_output(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    reviewer = ReviewerAgent(artifacts, llm=BadReviewLLM())
    brief = ResearchBrief(topic="offline RL")
    ideas = [
        IdeaCard(
            idea_id="idea-1",
            target_task="offline RL",
            novelty_claim="specific evidence backed idea",
            supporting_evidence_ids=["ev-1"],
            expected_gain="better return",
        )
    ]

    decision, _, _ = reviewer.evaluate_idea(brief, ideas)

    assert decision.decision == IdeaDecisionType.PASS
    assert artifacts.path("IdeaDecision.raw.txt").exists()


def test_reviewer_code_eval_uses_quality_tool(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    artifacts.write_json(
        "BuildSpec.json",
        BuildSpec(
            build_id="build-1",
            idea_id="idea-1",
            target_task="3D pattern synthesis",
            problem_statement="problem",
            method_summary="method",
            baselines=["baseline"],
            metrics=["micro reconstruction IoU", "Chamfer distance"],
        ),
    )
    artifacts.write_json(
        "CodeRunReport.json",
        {
            "status": "PASS",
            "code_dir": "code",
            "outputs": ["results/metrics.json", "results/results_table.csv"],
            "executed": True,
            "returncode": 0,
            "errors": [],
        },
    )
    artifacts.path("results").mkdir(parents=True)
    artifacts.path("results/metrics.json").write_text('{"summary": {"validation_loss": 0.1}}\n', encoding="utf-8")
    artifacts.path("results/results_table.csv").write_text(
        "method,validation_loss\nproposed,0.1\nbaseline,0.2\n",
        encoding="utf-8",
    )
    artifacts.path("results/progress_log.jsonl").write_text(
        '{"epoch": 1, "train_loss": 0.2, "eval_loss": 0.1}\n',
        encoding="utf-8",
    )

    audit = ReviewerAgent(artifacts).evaluate_code()

    assert audit.status == "FAIL"
    assert any("missing_build_spec_metrics" in failure for failure in audit.failures)
    assert artifacts.path("CodeEvalQualityReport.json").exists()


def test_reviewer_code_eval_counts_build_spec_metric_columns_as_numeric_results(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    artifacts.write_json(
        "BuildSpec.json",
        BuildSpec(
            build_id="build-1",
            idea_id="idea-1",
            target_task="offline RL",
            problem_statement="problem",
            method_summary="method",
            baselines=["CQL"],
            metrics=["1-Wasserstein distance", "Eval Reward"],
            plots=[],
        ),
    )
    artifacts.write_json(
        "CodeRunReport.json",
        {
            "status": "PASS",
            "code_dir": "code",
            "outputs": ["results/metrics.json", "results/results_table.csv"],
            "executed": True,
            "returncode": 0,
            "errors": [],
        },
    )
    artifacts.path("results").mkdir(parents=True)
    artifacts.path("results/metrics.json").write_text(
        '{"1-Wasserstein distance": 1.2, "Eval Reward": 0.7}\n',
        encoding="utf-8",
    )
    artifacts.path("results/results_table.csv").write_text(
        "method,source,seed,evaluation_source,1-Wasserstein distance,Eval Reward\n"
        "PRFM,proposed,0,offline_eval,1.2,0.7\n"
        "CQL,baseline,0,offline_eval,1.5,0.5\n"
        "CQL,paper_reported,,paper,N/A,N/A\n",
        encoding="utf-8",
    )
    artifacts.path("results/summary_table.csv").write_text(
        "Method,1-Wasserstein distance,Eval Reward\nPRFM,1.2,0.7\nCQL,1.5,0.5\n",
        encoding="utf-8",
    )
    artifacts.path("results/progress_log.jsonl").write_text(
        '{"epoch": 1, "1-Wasserstein distance": 1.3}\n{"epoch": 2, "1-Wasserstein distance": 1.2}\n',
        encoding="utf-8",
    )
    _write_reviewer_test_png(artifacts.path("results/progress_curve.png"), width=900, height=520)
    _write_reviewer_test_png(artifacts.path("results/eval_curve.png"), width=900, height=520)

    audit = ReviewerAgent(artifacts).evaluate_code()
    quality = artifacts.read_json("CodeEvalQualityReport.json")

    assert audit.status == "PASS"
    assert quality["generated_numeric_row_count"] == 2
    assert quality["generated_baseline_row_count"] == 1
    assert quality["method_names"] == ["cql", "prfm"]


def _write_reviewer_test_png(path, width=900, height=520):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            row.extend(((x * 3 + y) % 256, (x + y * 5) % 256, (x * 7 + y * 11) % 256))
        rows.append(b"\x00" + bytes(row))
    raw = b"".join(rows)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    payload = b"\x89PNG\r\n\x1a\n"
    payload += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += chunk(b"IDAT", zlib.compress(raw, 1))
    payload += chunk(b"IEND", b"")
    path.write_bytes(payload)


def test_reviewer_code_eval_rejects_zero_placeholder_baselines(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    artifacts.write_json(
        "BuildSpec.json",
        BuildSpec(
            build_id="build-1",
            idea_id="idea-1",
            target_task="offline RL",
            problem_statement="problem",
            method_summary="method",
            baselines=["CQL"],
            metrics=["Normalized score", "average return"],
            plots=[],
        ),
    )
    artifacts.write_json(
        "CodeRunReport.json",
        {
            "status": "PASS",
            "code_dir": "code",
            "outputs": ["results/metrics.json", "results/results_table.csv"],
            "executed": True,
            "returncode": 0,
            "errors": [],
        },
    )
    artifacts.path("results").mkdir(parents=True)
    artifacts.path("results/metrics.json").write_text(
        '{"Normalized score": 0.7, "average return": 70.0}\n',
        encoding="utf-8",
    )
    artifacts.path("results/results_table.csv").write_text(
        "method,source,Normalized score,average return\n"
        "Proposed,proposed,0.7,70.0\n"
        "CQL,baseline,0.0,0.0\n",
        encoding="utf-8",
    )

    audit = ReviewerAgent(artifacts).evaluate_code()
    quality = artifacts.read_json("CodeEvalQualityReport.json")
    rules = {failure["rule"] for failure in quality["failures"]}

    assert audit.status == "FAIL"
    assert "zero_placeholder_baseline_results" in rules
    assert "missing_baseline_evaluation_source" in rules


def test_reviewer_write_eval_checks_paper_against_build_spec_reporting(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    artifacts.write_json(
        "BuildSpec.json",
        BuildSpec(
            build_id="build-1",
            idea_id="idea-1",
            target_task="robot dispatch",
            problem_statement="problem",
            method_summary="method",
            metrics=["operator profit", "EV satisfaction rate"],
            plots=[
                ExperimentPlotSpec(
                    path="results/eval_curve.png",
                    title="Final operator profit comparison",
                    source="results_table",
                    x="method",
                    y="operator profit",
                    kind="bar",
                )
            ],
        ),
    )
    artifacts.write_json("WriteReport.json", {"status": "PASS"})
    artifacts.path("results").mkdir(parents=True)
    artifacts.path("results/eval_curve.png").write_bytes(b"png")
    artifacts.path("results/secondary_metric_curve.png").write_bytes(b"png")
    artifacts.path("results/results_table.csv").write_text(
        "method,operator_profit,ev_satisfaction_rate\nproposed,1.0,0.9\nbaseline,0.8,0.7\n",
        encoding="utf-8",
    )
    artifacts.path("paper_gene").mkdir(parents=True)
    artifacts.path("paper_gene/main.tex").write_text(
        r"""
        \section{Experiments}
        We report a generic score only.
        """,
        encoding="utf-8",
    )

    review = ReviewerAgent(artifacts).evaluate_write()

    assert review.status == "FAIL"
    assert any("paper_missing_build_spec_metrics" in failure for failure in review.failures)
    assert any("paper_missing_build_spec_plots" in failure for failure in review.failures)
    assert any("paper_missing_result_pngs" in failure for failure in review.failures)
    assert any("paper_missing_csv_latex_tables" in failure for failure in review.failures)


def test_reviewer_write_eval_direct_fails_below_seven_pages(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    artifacts.write_json(
        "BuildSpec.json",
        BuildSpec(
            build_id="build-1",
            idea_id="idea-1",
            target_task="robot dispatch",
            problem_statement="problem",
            method_summary="method",
            metrics=["operator profit"],
        ),
    )
    artifacts.write_json(
        "WriteReport.json",
        {
            "status": "FAIL",
            "page_validation": {
                "status": "FAIL",
                "expected_pages": 7,
                "target_pages": 7,
                "hard_fail_below_pages": 7,
                "actual_pages": 5,
                "hard_fail": True,
                "failures": [
                    {
                        "expected_pages": 7,
                        "target_pages": 7,
                        "hard_fail_below_pages": 7,
                        "actual_pages": 5,
                        "reason": "compiled PDF is below configured expected_main_pages=7",
                    }
                ],
            },
        },
    )
    artifacts.path("paper_gene").mkdir(parents=True)
    artifacts.path("paper_gene/main.tex").write_text(
        r"\section{Experiments} operator profit",
        encoding="utf-8",
    )

    review = ReviewerAgent(artifacts).evaluate_write()

    assert review.status == "FAIL"
    assert any("direct FAIL" in failure and "configured expected page threshold 7" in failure for failure in review.failures)
