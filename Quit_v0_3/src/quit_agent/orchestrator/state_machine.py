from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import re
from typing import Any, Callable

from quit_agent.agents.builder_agent import BuilderAgent
from quit_agent.agents.planner_agent import PlannerAgent
from quit_agent.agents.research_agent import ResearchAgent
from quit_agent.agents.reviewer_agent import ReviewerAgent
from quit_agent.artifacts.manager import ArtifactManager
from quit_agent.artifacts.trace import TraceManager
from quit_agent.orchestrator.transitions import next_after_idea_decision
from quit_agent.schemas.enums import FallbackTarget, IdeaDecisionType, WorkflowState
from quit_agent.schemas.evidence_card import EvidenceCard
from quit_agent.schemas.idea_card import IdeaCard
from quit_agent.schemas.paper_card import PaperCard
from quit_agent.schemas.repo_card import RepoCard
from quit_agent.schemas.build_spec import BuildSpec
from quit_agent.schemas.research_brief import ResearchBrief
from quit_agent.schemas.review_artifacts import IdeaDecision
from quit_agent.validators.research_brief_validator import ResearchBriefValidator


@dataclass
class RunResult:
    run_id: str
    final_state: WorkflowState
    next_state: WorkflowState
    run_dir: str


class StateMachine:
    """Artifact-driven state machine for the research workflow."""

    def __init__(
        self,
        *,
        artifacts: ArtifactManager,
        trace: TraceManager,
        planner: PlannerAgent,
        researcher: ResearchAgent,
        reviewer: ReviewerAgent,
        builder: BuilderAgent,
        max_plan_attempts: int = 2,
        reporter: Callable[[str], None] | None = None,
    ) -> None:
        self.artifacts = artifacts
        self.trace = trace
        self.planner = planner
        self.researcher = researcher
        self.reviewer = reviewer
        self.builder = builder
        self.step_counter = self._existing_step_count()
        self.plan_attempts = 0
        self.max_plan_attempts = max_plan_attempts
        self.reporter = reporter

    def run(
        self,
        user_request: str,
        start_at: WorkflowState = WorkflowState.PLAN,
        stop_after: WorkflowState = WorkflowState.BUILD_SPEC,
        max_steps: int = 50,
    ) -> RunResult:
        state = start_at
        next_state = state
        steps = 0
        while state != WorkflowState.STOP and steps < max_steps:
            display_step = f"{self.step_counter + 1:04d}"
            self._progress(f"[PIPELINE] step {display_step} START {state.value}")
            next_state = self.step(state, user_request)
            self._progress_state_end(display_step, state, next_state)
            steps += 1
            if state == stop_after and _stop_after_satisfied(state, next_state):
                break
            state = next_state
        return RunResult(
            run_id=self.artifacts.paths.run_id,
            final_state=state,
            next_state=next_state,
            run_dir=str(self.artifacts.run_dir),
        )

    def _progress(self, message: str) -> None:
        if self.reporter is not None:
            self.reporter(message)

    def _progress_state_end(self, step_id: str, state: WorkflowState, next_state: WorkflowState) -> None:
        self._progress(f"[PIPELINE] step {step_id} END   {state.value} -> {next_state.value}")
        if next_state == state or next_state == WorkflowState.STOP:
            summary = self._failure_summary_for_state(state)
            if summary:
                self._progress(summary)

    def _failure_summary_for_state(self, state: WorkflowState) -> str:
        if state == WorkflowState.CODE and self.artifacts.path("CodeRunReport.json").exists():
            report = self.artifacts.read_json("CodeRunReport.json")
            if report.get("status") == "PASS":
                return ""
            errors = report.get("errors") or []
            tail = str(errors[0] if errors else "No error text recorded.").strip().splitlines()[-6:]
            return (
                "[PIPELINE] CODE needs attention: CodeRunReport.json, "
                "CodeSyntaxReport.json, CodeRepairReport.json, EXPERIMENT_LOG.md\n"
                + "\n".join(f"[PIPELINE]   {line}" for line in tail)
            )
        if state == WorkflowState.CODE_EVAL and self.artifacts.path("ExperimentAudit.json").exists():
            audit = self.artifacts.read_json("ExperimentAudit.json")
            if audit.get("status") == "PASS":
                return ""
            failures = audit.get("failures") or []
            tail = str(failures[0] if failures else "No failure text recorded.").strip().splitlines()[-6:]
            return (
                "[PIPELINE] CODE_EVAL needs experiment/code revision: ExperimentAudit.json, "
                "CodeEvalQualityReport.json\n"
                + "\n".join(f"[PIPELINE]   {line}" for line in tail)
            )
        if state == WorkflowState.VALIDATE_BRIEF and self.artifacts.path("ResearchBriefValidationReport.json").exists():
            report = self.artifacts.read_json("ResearchBriefValidationReport.json")
            if report.get("status") == "PASS":
                return ""
            return f"[PIPELINE] VALIDATE_BRIEF failed: ResearchBriefValidationReport.json status={report.get('status')}"
        if state == WorkflowState.RETRIEVE and self.artifacts.path("RetrievalReport.json").exists():
            report = self.artifacts.read_json("RetrievalReport.json")
            if report.get("status") == "PASS":
                return ""
            return f"[PIPELINE] RETRIEVE failed: RetrievalReport.json status={report.get('status')}"
        return ""

    def _existing_step_count(self) -> int:
        if not self.artifacts.paths.trace_dir.exists():
            return 0
        highest = 0
        for path in self.artifacts.paths.trace_dir.glob("*.json"):
            match = re.match(r"(\d+)_", path.name)
            if match:
                highest = max(highest, int(match.group(1)))
        return highest

    def step(self, state: WorkflowState, user_request: str) -> WorkflowState:
        self.step_counter += 1
        step_id = f"{self.step_counter:04d}"
        started_at = datetime.now(UTC).isoformat()
        if state == WorkflowState.PLAN:
            self.plan_attempts += 1
            replan_context = self._replan_context()
            if replan_context:
                if self.plan_attempts > self.max_plan_attempts:
                    failure = {
                        "status": "FAIL",
                        "reason": "ResearchBrief planning/replanning exceeded retry budget",
                        "max_plan_attempts": self.max_plan_attempts,
                    }
                    self.artifacts.write_json("PlanFailure.json", failure)
                    next_state = WorkflowState.STOP
                    self._trace(
                        step_id,
                        state,
                        "PlannerAgent",
                        "plan_retry_budget_exceeded",
                        ["ResearchBrief.raw.json", "ResearchBriefValidationReport.json"],
                        ["PlanFailure.json"],
                        "Stop after repeated invalid ResearchBrief outputs.",
                        repr(failure),
                        failure,
                        failure,
                        next_state,
                        "STOP",
                        started_at,
                    )
                    return next_state
                report_name, recovery_report = replan_context
                previous_raw = self.artifacts.path("ResearchBrief.raw.json").read_text(encoding="utf-8")
                prompt, raw, raw_name = self.planner.regenerate_brief_after_failure(
                    user_request=user_request,
                    previous_raw_response=previous_raw,
                    validation_report=recovery_report,
                )
                skill = "replan_research_brief"
                input_artifacts = ["ResearchBrief.raw.json", report_name]
            else:
                prompt, raw, raw_name = self.planner.generate_brief(user_request)
                skill = "plan_research_brief"
                input_artifacts = []
            next_state = WorkflowState.VALIDATE_BRIEF
            self._trace(step_id, state, "PlannerAgent", skill, input_artifacts, [raw_name], prompt, raw, raw[:500], "RAW", next_state, None, started_at)
            return next_state

        if state == WorkflowState.VALIDATE_BRIEF:
            raw = self.artifacts.path("ResearchBrief.raw.json").read_text(encoding="utf-8")
            brief, report = ResearchBriefValidator(self.artifacts).validate_or_repair(raw)
            next_state = WorkflowState.RETRIEVE if brief else WorkflowState.PLAN
            self._trace(
                step_id,
                state,
                "PlannerAgent",
                "validate_and_repair_brief",
                ["ResearchBrief.raw.json"],
                ["ResearchBrief.json", "ResearchBriefValidationReport.json"] if brief else ["ResearchBriefValidationReport.json"],
                "Validate and repair ResearchBrief.raw.json",
                report.to_json_text(),
                brief.to_dict() if brief else None,
                report.to_dict(),
                next_state,
                report.fallback_target,
                started_at,
            )
            return next_state

        if state == WorkflowState.RETRIEVE:
            brief = self.artifacts.load_model("ResearchBrief.json", ResearchBrief)
            papers, report, prompt, response = self.researcher.retrieve(brief)
            passed = report.get("status") == "PASS"
            # download_fail: brief is fine, just retry retrieval without replanning
            if not passed and report.get("download_fail"):
                next_state = WorkflowState.RETRIEVE
                fallback_label = "RETRIEVE"
            else:
                next_state = WorkflowState.READ if passed else WorkflowState.PLAN
                fallback_label = None if passed else "PLAN"
            output_artifacts = ["PaperCards.jsonl", "RepoCards.jsonl", "RepoRetrievalReport.json", "RetrievalReport.json"]
            if not passed:
                output_artifacts.append("RetrievalFailure.json")
            self._trace(step_id, state, "ResearchAgent", "retrieve_papers", ["ResearchBrief.json"], output_artifacts, prompt, response, [p.to_dict() for p in papers[:2]], report, next_state, fallback_label, started_at)
            return next_state

        if state == WorkflowState.READ:
            brief = self.artifacts.load_model("ResearchBrief.json", ResearchBrief)
            papers = self.artifacts.load_jsonl_models("PaperCards.jsonl", PaperCard)
            evidence, report, prompt, response = self.researcher.read(brief, papers)
            next_state = WorkflowState.IDEATE if report["status"] == "PASS" else WorkflowState.RETRIEVE
            self._trace(step_id, state, "ResearchAgent", "read_papers", ["ResearchBrief.json", "PaperCards.jsonl"], ["PaperTexts.jsonl", "EvidenceCards.jsonl", "EvidenceValidationReport.json", "ReadReport.json"], prompt, response, [e.to_dict() for e in evidence[:2]], report, next_state, None if report["status"] == "PASS" else "RETRIEVE", started_at)
            return next_state

        if state == WorkflowState.IDEATE:
            brief = self.artifacts.load_model("ResearchBrief.json", ResearchBrief)
            evidence = self.artifacts.load_jsonl_models("EvidenceCards.jsonl", EvidenceCard)
            revision_decision = None
            input_artifacts = ["ResearchBrief.json", "EvidenceCards.jsonl"]
            if self.artifacts.path("IdeaDecision.json").exists():
                prior_decision = self.artifacts.load_model("IdeaDecision.json", IdeaDecision)
                if prior_decision.decision != IdeaDecisionType.PASS and prior_decision.fallback_target == FallbackTarget.IDEATE:
                    revision_decision = prior_decision
                    input_artifacts.append("IdeaDecision.json")
            ideas, report, prompt, response = self.researcher.ideate(brief, evidence, revision_decision)
            next_state = WorkflowState.IDEA_EVAL if report["status"] == "PASS" else WorkflowState.READ
            self._trace(step_id, state, "ResearchAgent", "ideate_from_evidence", input_artifacts, ["IdeaClusters.json", "IdeaLibrary.jsonl", "IdeaGenerationReport.json"], prompt, response, [i.to_dict() for i in ideas[:2]], report, next_state, None if report["status"] == "PASS" else "READ", started_at)
            return next_state

        if state == WorkflowState.IDEA_EVAL:
            brief = self.artifacts.load_model("ResearchBrief.json", ResearchBrief)
            ideas = self.artifacts.load_jsonl_models("IdeaLibrary.jsonl", IdeaCard)
            decision, prompt, response = self.reviewer.evaluate_idea(brief, ideas)
            next_state = next_after_idea_decision(decision)
            self._trace(step_id, state, "ReviewerAgent", "isolated_idea_review", ["ResearchBrief.json", "IdeaLibrary.jsonl"], ["IdeaDecision.json"], prompt, response, decision.to_dict(), decision.to_dict(), next_state, decision.fallback_target, started_at)
            return next_state

        return self._placeholder_state(step_id, state, started_at)

    def _replan_context(self) -> tuple[str, dict[str, Any]] | None:
        report_path = self.artifacts.path("ResearchBriefValidationReport.json")
        raw_path = self.artifacts.path("ResearchBrief.raw.json")
        if not raw_path.exists():
            return None
        if report_path.exists():
            report = self.artifacts.read_json("ResearchBriefValidationReport.json")
            if report.get("status") == "FAIL":
                return "ResearchBriefValidationReport.json", report
        retrieval_report_path = self.artifacts.path("RetrievalReport.json")
        if retrieval_report_path.exists():
            report = self.artifacts.read_json("RetrievalReport.json")
            if report.get("status") == "FAIL":
                return "RetrievalReport.json", report
        return None

    def _placeholder_state(self, step_id: str, state: WorkflowState, started_at: str) -> WorkflowState:
        if state == WorkflowState.BUILD_SPEC:
            brief = self.artifacts.load_model("ResearchBrief.json", ResearchBrief)
            decision = self.artifacts.load_model("IdeaDecision.json", IdeaDecision)
            ideas = self.artifacts.load_jsonl_models("IdeaLibrary.jsonl", IdeaCard)
            evidence = self.artifacts.load_jsonl_models("EvidenceCards.jsonl", EvidenceCard)
            repos = self.artifacts.load_jsonl_models("RepoCards.jsonl", RepoCard) if self.artifacts.path("RepoCards.jsonl").exists() else []
            idea = self._approved_idea(decision, ideas)
            supporting_evidence = [card for card in evidence if card.evidence_id in set(idea.supporting_evidence_ids)]
            spec, prompt, response = self.builder.build_spec(
                brief=brief,
                idea=idea,
                decision=decision,
                evidence=supporting_evidence,
                repos=repos,
            )
            next_state = WorkflowState.CODE
            input_artifacts = ["ResearchBrief.json", "IdeaDecision.json", "IdeaLibrary.jsonl", "EvidenceCards.jsonl"]
            if repos:
                input_artifacts.append("RepoCards.jsonl")
            output_artifacts = ["BuildSpec.json"]
            if self.artifacts.path("RepoCloneReport.json").exists():
                output_artifacts.append("RepoCloneReport.json")
            self._trace(step_id, state, "BuilderAgent", "build_spec", input_artifacts, output_artifacts, prompt, response, spec.to_dict(), "PASS", next_state, None, started_at)
            return next_state
        if state == WorkflowState.CODE:
            spec = self.artifacts.load_model("BuildSpec.json", BuildSpec)
            evidence = self.artifacts.load_jsonl_models("EvidenceCards.jsonl", EvidenceCard) if self.artifacts.path("EvidenceCards.jsonl").exists() else []
            report, prompt, response = self.builder.code(spec, evidence=evidence)
            passed = report.status == "PASS"
            next_state = WorkflowState.CODE_EVAL if passed else WorkflowState.CODE
            inputs = ["BuildSpec.json"]
            if evidence:
                inputs.append("EvidenceCards.jsonl")
            self._trace(
                step_id,
                state,
                "BuilderAgent",
                "code_from_build_spec",
                inputs,
                ["code/", "results/", "ImplementationContract.json", "CodeStageGenerationReport.json", "EnvironmentResolutionReport.json", "ConfigNormalizationReport.json", "DatasetAcquisitionReport.json", "DeviceReport.json", "CodeSyntaxReport.json", "DependencyInstallReport.json", "RuntimePatchReport.json", "code/EXPERIMENT_METRICS.md", "EXPERIMENT_LOG.md", "CodeRunReport.json", "CodeRepairReport.json"],
                prompt,
                response,
                report.to_dict(),
                report.to_dict(),
                next_state,
                None if passed else "CODE_REPAIR",
                started_at,
            )
            return next_state
        if state == WorkflowState.CODE_EVAL:
            audit = self.reviewer.evaluate_code()
            passed = audit.status == "PASS"
            next_state = WorkflowState.CODE_LLM_EVAL if passed else WorkflowState.CODE
            self._trace(
                step_id,
                state,
                "ReviewerAgent",
                "code_quality_audit",
                ["BuildSpec.json", "CodeRunReport.json", "EXPERIMENT_LOG.md", "results/"],
                ["ExperimentAudit.json", "CodeEvalQualityReport.json"],
                "Hard rule-based audit of code artifacts against BuildSpec",
                audit.to_json_text(),
                audit.to_dict(),
                audit.to_dict(),
                next_state,
                None if passed else "CODE_REPAIR",
                started_at,
            )
            return next_state
        if state == WorkflowState.CODE_LLM_EVAL:
            perf_result = self.reviewer.evaluate_code_performance()
            passed = perf_result is None or perf_result.get("verdict") != "FAIL"
            next_state = WorkflowState.WRITE if passed else WorkflowState.CODE_REVISE
            summary = perf_result if perf_result is not None else {"status": "skipped", "reason": "no LLM or missing results"}
            self._trace(
                step_id,
                state,
                "ReviewerAgent",
                "code_performance_eval",
                ["results/metrics.json", "results/results_table.csv", "BuildSpec.json"],
                ["CodePerformanceEval.json"],
                "LLM judgment: does the proposed method outperform baselines overall?",
                json.dumps(summary, indent=2),
                summary,
                summary,
                next_state,
                None if passed else "CODE_REPAIR",
                started_at,
            )
            return next_state
        if state == WorkflowState.CODE_REVISE:
            spec = self.artifacts.load_model("BuildSpec.json", BuildSpec)
            report, prompt, response = self.builder.revise_code_performance(spec)
            passed = report.get("run_report", {}).get("status") == "PASS"
            next_state = WorkflowState.CODE_EVAL
            self._trace(
                step_id,
                state,
                "BuilderAgent",
                "revise_code_performance",
                ["CodePerformanceEval.json", "BuildSpec.json", "code/src/method.py", "code/src/train.py", "code/configs/experiment_config.json"],
                ["code/src/method.py", "code/src/train.py", "code/configs/experiment_config.json", "CodeReviseReport.json", "CodeRunReport.json", "EXPERIMENT_LOG.md"],
                prompt,
                response,
                report,
                report,
                next_state,
                None,
                started_at,
            )
            return next_state
        if state == WorkflowState.WRITE:
            spec = self.artifacts.load_model("BuildSpec.json", BuildSpec)
            evidence = self.artifacts.load_jsonl_models("EvidenceCards.jsonl", EvidenceCard) if self.artifacts.path("EvidenceCards.jsonl").exists() else []
            papers = self.artifacts.load_jsonl_models("PaperCards.jsonl", PaperCard) if self.artifacts.path("PaperCards.jsonl").exists() else []
            report, prompt, response = self.builder.write(spec, evidence, papers)
            passed = report.get("status") == "PASS"
            has_draft = self.artifacts.path("paper_gene/main.tex").exists()
            next_state = WorkflowState.WRITE_EVAL if passed or has_draft else WorkflowState.STOP
            inputs = ["BuildSpec.json"]
            if evidence:
                inputs.append("EvidenceCards.jsonl")
            if papers:
                inputs.append("PaperCards.jsonl")
            self._trace(
                step_id,
                state,
                "BuilderAgent",
                "write_from_build_spec",
                inputs,
                ["paper_gene/", *report.get("outputs", []), "WriteReport.json"],
                prompt,
                response,
                report,
                report,
                next_state,
                None if passed else ("WRITE_EVAL_FEEDBACK" if has_draft else "WRITE_FAILED"),
                started_at,
            )
            return next_state
        if state == WorkflowState.WRITE_EVAL:
            review = self.reviewer.evaluate_write()
            passed = review.status == "PASS"
            next_state = WorkflowState.WRITE_LLM_EVAL if passed else WorkflowState.WRITE
            self._trace(
                step_id,
                state,
                "ReviewerAgent",
                "paper_review_hard",
                ["paper_gene/main.tex", "WriteReport.json"],
                ["PaperReview.json"],
                "Hard rule-based review of paper and write-stage report",
                review.to_json_text(),
                review.to_dict(),
                review.to_dict(),
                next_state,
                None if passed else "WRITE_REPAIR",
                started_at,
            )
            return next_state
        if state == WorkflowState.WRITE_LLM_EVAL:
            llm_review = self.reviewer.evaluate_write_llm()
            review_summary = llm_review.to_dict() if llm_review else {"status": "skipped", "reason": "no LLM configured"}
            self._trace(
                step_id,
                state,
                "ReviewerAgent",
                "paper_review_llm",
                ["paper_gene/main.tex", "BuildSpec.json"],
                ["PaperReview.json"],
                "Soft LLM review; improvement_hints fed into next WRITE_REVISE",
                str(review_summary),
                review_summary,
                review_summary,
                WorkflowState.WRITE_REVISE,
                None,
                started_at,
            )
            return WorkflowState.WRITE_REVISE
        if state == WorkflowState.WRITE_REVISE:
            spec = self.artifacts.load_model("BuildSpec.json", BuildSpec)
            report, prompt, response = self.builder.revise_paper(spec)
            passed = report.get("status") == "PASS"
            has_draft = self.artifacts.path("paper_gene/main.tex").exists()
            next_state = WorkflowState.WRITE_FINAL_EVAL if passed or has_draft else WorkflowState.STOP
            self._trace(
                step_id,
                state,
                "BuilderAgent",
                "revise_paper",
                ["paper_gene/main.tex", "PaperReview.json", "BuildSpec.json"],
                ["paper_gene/main.tex", *report.get("outputs", []), "WriteReport.json"],
                prompt,
                response,
                report,
                report,
                next_state,
                None if passed else ("WRITE_FINAL_EVAL_FEEDBACK" if has_draft else "WRITE_REVISE_FAILED"),
                started_at,
            )
            return next_state
        if state == WorkflowState.WRITE_FINAL_EVAL:
            review = self.reviewer.evaluate_write()
            passed = review.status == "PASS"
            next_state = WorkflowState.EXTRACT if passed else WorkflowState.WRITE_REVISE
            self._trace(
                step_id,
                state,
                "ReviewerAgent",
                "paper_review_hard",
                ["paper_gene/main.tex", "WriteReport.json"],
                ["PaperReview.json"],
                "Final hard rule-based review after LLM-guided revision",
                review.to_json_text(),
                review.to_dict(),
                review.to_dict(),
                next_state,
                None if passed else "WRITE_REVISE",
                started_at,
            )
            return next_state
        if state == WorkflowState.EXTRACT:
            entry = self.reviewer.extract_summary()
            result = entry if entry is not None else {"status": "skipped", "reason": "no LLM or no paper draft"}
            next_state = WorkflowState.STOP
            self._trace(
                step_id,
                state,
                "ReviewerAgent",
                "extract_research_summary",
                ["paper_gene/main.tex", "PaperReview.json", "BuildSpec.json"],
                [],
                "Extract structured research summary to per-topic knowledge queue",
                json.dumps(result, indent=2),
                result,
                {"status": "PASS" if entry is not None else "SKIP"},
                next_state,
                None,
                started_at,
            )
            return next_state
        return WorkflowState.STOP

    def _approved_idea(self, decision: IdeaDecision, ideas: list[IdeaCard]) -> IdeaCard:
        for idea in ideas:
            if idea.idea_id == decision.idea_id:
                return idea
        raise ValueError(f"approved idea not found in IdeaLibrary: {decision.idea_id}")

    def _trace(
        self,
        step_id: str,
        state: WorkflowState,
        agent: str,
        skill: str,
        input_artifacts: list[str],
        output_artifacts: list[str],
        prompt_text: str,
        response_text: str,
        parsed_output_preview: Any,
        validation_result: Any,
        next_state: WorkflowState,
        fallback_decision: Any,
        started_at: str,
    ) -> None:
        self.trace.record_step(
            step_id=step_id,
            state=state.value,
            agent=agent,
            skill=skill,
            input_artifacts=input_artifacts,
            output_artifacts=output_artifacts,
            prompt_text=prompt_text,
            response_text=response_text,
            parsed_output_preview=parsed_output_preview,
            validation_result=validation_result,
            next_state=next_state.value,
            fallback_decision=str(fallback_decision) if fallback_decision else None,
            started_at=started_at,
        )


# Eval states that loop back on failure — stop_after should only trigger when
# they succeed (i.e. next_state is NOT the retry target).
_EVAL_RETRY_MAP: dict[WorkflowState, WorkflowState] = {
    WorkflowState.WRITE_EVAL:       WorkflowState.WRITE,
    WorkflowState.WRITE_FINAL_EVAL: WorkflowState.WRITE_REVISE,
    WorkflowState.CODE_EVAL:        WorkflowState.CODE,
    WorkflowState.CODE_LLM_EVAL:    WorkflowState.CODE_REVISE,
    WorkflowState.IDEA_EVAL:        WorkflowState.IDEATE,
}


def _stop_after_satisfied(state: WorkflowState, next_state: WorkflowState) -> bool:
    """Return True only when it is safe to stop after executing `state`.

    For eval states that can loop back on failure, we only stop when the eval
    passed (next_state is not the failure retry target).  For all other states
    we stop unconditionally.
    """
    retry_target = _EVAL_RETRY_MAP.get(state)
    if retry_target is not None:
        return next_state != retry_target
    return True
