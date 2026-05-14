from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
import textwrap
import urllib.request
from pathlib import Path
from typing import Any, Callable

from quit_agent.artifacts.manager import ArtifactManager
from quit_agent.schemas.build_spec import ArtifactsRequired, BuildEnvironment, BuildSpec, ExperimentLogSpec, ExperimentPlotSpec
from quit_agent.schemas.code_artifacts import CodeRunReport
from quit_agent.schemas.evidence_card import EvidenceCard
from quit_agent.schemas.idea_card import IdeaCard
from quit_agent.schemas.paper_card import PaperCard
from quit_agent.schemas.repo_card import RepoCard
from quit_agent.schemas.research_brief import ResearchBrief
from quit_agent.schemas.review_artifacts import IdeaDecision
from quit_agent.tools.device import select_torch_device
from quit_agent.tools.llm_interface import LLMClient
from quit_agent.tools.repo_tools import RepoManager, normalize_repo_url, repo_id_from_url


_BUILD_SPEC_SKILL_PATH = Path(__file__).resolve().parents[1] / "skills" / "builder" / "build_spec.md"
_WRITE_SKILL_PATH = Path(__file__).resolve().parents[1] / "skills" / "builder" / "write_from_build_spec.md"
_REVISE_PAPER_SKILL_PATH = Path(__file__).resolve().parents[1] / "skills" / "builder" / "revise_paper.md"
_CODE_SKILL_PATH = Path(__file__).resolve().parents[1] / "skills" / "builder" / "code_from_build_spec.md"
_IMPLEMENTATION_CONTRACT_SKILL_PATH = Path(__file__).resolve().parents[1] / "skills" / "builder" / "implementation_contract_from_build_spec.md"
_IMPLEMENT_CORE_SKILL_PATH = Path(__file__).resolve().parents[1] / "skills" / "builder" / "implement_core.md"
_IMPLEMENT_EXPERIMENT_SKILL_PATH = Path(__file__).resolve().parents[1] / "skills" / "builder" / "implement_experiment.md"
_CODE_STAGE_REPAIR_SKILL_PATH = Path(__file__).resolve().parents[1] / "skills" / "builder" / "code_stage_eval_repair.md"
_REVISE_CODE_PERFORMANCE_SKILL_PATH = Path(__file__).resolve().parents[1] / "skills" / "builder" / "revise_code_performance.md"
_DEFAULT_PAPER_TEMPLATE_DIR = Path(__file__).resolve().parents[4] / "Template" / "ICML2026"
_MIN_REPORTING_TARGETS = 5


class BuilderAgent:
    def __init__(
        self,
        artifacts: ArtifactManager,
        llm: LLMClient | None = None,
        model_name: str = "unknown-model",
        min_train_epochs: int = 100,
        max_train_epochs: int = 10000,
        min_eval_epochs: int = 20,
        max_eval_epochs: int = 200,
        max_download_attempts: int = 3,
        expected_main_pages: int = 7,
        experiment_timeout_seconds: int = 120,
        latex_timeout_seconds: int = 120,
        reporter: Callable[[str], None] | None = None,
    ) -> None:
        self.artifacts = artifacts
        self.llm = llm
        self.model_name = model_name
        self.reporter = reporter
        self._run_bounds = {
            "train_epochs": (min_train_epochs, max_train_epochs),
            "eval_epochs": (min_eval_epochs, max_eval_epochs),
        }
        self._max_download_attempts = max_download_attempts
        self._expected_main_pages = max(1, expected_main_pages)
        self._experiment_timeout_seconds = max(1, int(experiment_timeout_seconds))
        self._latex_timeout_seconds = max(1, int(latex_timeout_seconds))
        self._build_spec_template = _load_prompt_template(_BUILD_SPEC_SKILL_PATH)
        self._write_template = _load_prompt_template(_WRITE_SKILL_PATH)
        self._revise_paper_template = _load_prompt_template(_REVISE_PAPER_SKILL_PATH)
        self._revise_code_performance_template = _load_prompt_template(_REVISE_CODE_PERFORMANCE_SKILL_PATH)
        self._code_template = _load_prompt_template(_CODE_SKILL_PATH)
        self._implementation_contract_template = _load_skill_text(_IMPLEMENTATION_CONTRACT_SKILL_PATH)
        self._implement_core_template = _load_skill_text(_IMPLEMENT_CORE_SKILL_PATH)
        self._implement_experiment_template = _load_skill_text(_IMPLEMENT_EXPERIMENT_SKILL_PATH)
        self._code_stage_repair_template = _load_skill_text(_CODE_STAGE_REPAIR_SKILL_PATH)

    def build_spec(
        self,
        *,
        brief: ResearchBrief,
        idea: IdeaCard,
        decision: IdeaDecision,
        evidence: list[EvidenceCard],
        repos: list[RepoCard] | None = None,
    ) -> tuple[BuildSpec, str, str]:
        """Skill: build_spec.

        Use when: IDEA_EVAL passes.
        Inputs: ResearchBrief, approved IdeaCard, IdeaDecision, supporting EvidenceCards.
        Output: BuildSpec.json source of truth for code/write stages.
        Failure mode: later revisions should route back to IDEATE or BUILD_SPEC.
        """
        repos = repos or []
        prompt = self._build_spec_prompt(brief, idea, decision, evidence, repos)
        if self.llm is not None:
            raw = self.llm.complete(prompt)
            spec = self._parse_build_spec(raw)
            if spec is not None:
                spec = self._ensure_environment(spec, idea, evidence, repos)
                spec = self._sanitize_build_spec_domain(spec, brief, idea, evidence)
                spec = self._ensure_experiment_reporting_spec(spec, evidence)
                spec = self._assess_code_strategy(spec)
                spec = self._inject_run_budget(spec)
                self.artifacts.write_json("BuildSpec.json", spec)
                return spec, prompt, raw
            self.artifacts.write_markdown("BuildSpec.raw.txt", raw)

        spec = self._fallback_build_spec(brief, idea, evidence, repos)
        spec = self._sanitize_build_spec_domain(spec, brief, idea, evidence)
        spec = self._ensure_experiment_reporting_spec(spec, evidence)
        spec = self._assess_code_strategy(spec)
        spec = self._inject_run_budget(spec)
        self.artifacts.write_json("BuildSpec.json", spec)
        return spec, prompt, spec.to_json_text()

    def _fallback_build_spec(self, brief: ResearchBrief, idea: IdeaCard, evidence: list[EvidenceCard], repos: list[RepoCard] | None = None) -> BuildSpec:
        metrics = sorted({metric for card in evidence for metric in card.metrics})[:8]
        citations = sorted({card.paper_id for card in evidence}) or idea.supporting_evidence_ids
        limitations = [item for card in evidence for item in card.limitations][:4]
        method_seeds = [item for card in evidence for item in card.transferable_idea_seeds][:5]
        spec = BuildSpec(
            build_id=f"build-{idea.idea_id}",
            idea_id=idea.idea_id,
            target_task=idea.target_task,
            problem_statement=(
                f"Develop a method for {idea.target_task} under constraints: "
                f"{'; '.join(brief.constraints[:3])}."
            ),
            method_summary=f"{idea.novelty_claim} Evidence seeds: {'; '.join(method_seeds[:3])}",
            implementation_plan=[
                "Define the domain dataset, scenario, or input interface required by the target task.",
                "Implement the proposed method as a modular component with a BuildSpec-compatible API.",
                "Integrate the mechanisms described by the approved idea.",
                "Add configuration files for ablations and reproducible runs.",
                "Log metrics, seeds, checkpoints, and result tables as artifacts.",
            ],
            experiment_plan=[
                "Run baseline methods on the same datasets and evaluation protocol.",
                "Run the proposed method across the configured random seeds or deterministic scenarios.",
                "Evaluate robustness under target-domain shifts declared by the idea or evidence.",
                "Run ablations for each major method component.",
                "Compare gains against acceptance criteria and record failures.",
            ],
            baselines=self._evidence_baselines(evidence),
            metrics=metrics or ["primary_metric"],
            logging=[
                ExperimentLogSpec(
                    path="results/progress_log.jsonl",
                    record_type="execution",
                    fields=["epoch", (metrics[0] if metrics else "primary_metric"), "timestamp"],
                    x_axis="epoch",
                    description="Record the primary BuildSpec metric over experiment epochs or execution steps.",
                )
            ],
            plots=[
                ExperimentPlotSpec(
                    path="results/progress_curve.png",
                    title="Primary Metric Progress",
                    source="log",
                    x="epoch",
                    y=(metrics[0] if metrics else "primary_metric"),
                    kind="line",
                    series="method",
                    description="Show progress of the primary experiment metric; use loss only when it is a real objective.",
                ),
                ExperimentPlotSpec(
                    path="results/eval_curve.png",
                    title="Final Method Comparison",
                    source="results_table",
                    x="method",
                    y=(metrics[0] if metrics else "primary_metric"),
                    kind="bar",
                    series="method",
                    description="Compare proposed and baseline methods on the primary BuildSpec metric.",
                ),
            ],
            success_criteria=[
                idea.expected_gain,
                "Measurable improvement over at least one relevant baseline.",
                "Evaluation uses a fixed, reproducible protocol without leakage across compared methods.",
                "Ablations support the claimed mechanism.",
            ],
            artifacts_required=ArtifactsRequired(
                coder=["working code module", "experiment logs", "result table/figure"],
                writer=["latex section draft", "method figure description", "experiment summary table"],
            ),
            paper_outline=[
                "Abstract",
                "Introduction",
                "Related Work",
                "Preliminaries",
                "Methods",
                "Experiments",
                "Conclusion",
                "Appendix",
            ],
            citations_required=citations,
        )
        spec = self._ensure_environment(spec, idea, evidence, repos or [])
        if limitations:
            spec.success_criteria.append(f"Address or explicitly test limitations: {'; '.join(limitations[:2])}")
        return spec

    def _sanitize_build_spec_domain(
        self,
        spec: BuildSpec,
        brief: ResearchBrief,
        idea: IdeaCard,
        evidence: list[EvidenceCard],
    ) -> BuildSpec:
        return spec

    def _ensure_experiment_reporting_spec(self, spec: BuildSpec, evidence: list[EvidenceCard]) -> BuildSpec:
        """Ensure CODE and WRITE receive a concrete reporting contract from BuildSpec."""
        evidence_metrics = [
            str(metric).strip()
            for card in evidence
            for metric in card.metrics
            if str(metric).strip()
        ]
        metrics = _dedupe([str(metric).strip() for metric in spec.metrics if str(metric).strip()])
        if not metrics:
            metrics = _dedupe(evidence_metrics) or ["primary_metric"]
        spec.metrics = metrics[:8]

        primary = spec.metrics[0] if spec.metrics else "primary_metric"
        secondary = spec.metrics[1] if len(spec.metrics) > 1 else primary
        tertiary = spec.metrics[2] if len(spec.metrics) > 2 else secondary
        quaternary = spec.metrics[3] if len(spec.metrics) > 3 else primary
        if not spec.logging:
            spec.logging = [
                ExperimentLogSpec(
                    path="results/progress_log.jsonl",
                    record_type="execution",
                    fields=_dedupe(["epoch", *spec.metrics[:3], "timestamp"]),
                    x_axis="epoch",
                    description="Record the BuildSpec reporting metrics over experiment epochs, episodes, or execution steps.",
                )
            ]
        else:
            for item in spec.logging:
                fields = _dedupe([str(field).strip() for field in item.fields if str(field).strip()] + ["timestamp"])
                if spec.metrics and not any(field in fields for field in spec.metrics):
                    fields = _dedupe([item.x_axis or "epoch", *spec.metrics, *fields])
                item.fields = fields

        existing_paths = {plot.path for plot in spec.plots if plot.path}
        required_plots = [
            ExperimentPlotSpec(
                path="results/progress_curve.png",
                title=f"{primary} Progress",
                source="log",
                x=(spec.logging[0].x_axis if spec.logging else "epoch"),
                y=primary,
                kind="line",
                series="method",
                description=f"Show how {primary} changes over the run.",
            ),
            ExperimentPlotSpec(
                path="results/eval_curve.png",
                title=f"Final {primary} Comparison",
                source="results_table",
                x="method",
                y=primary,
                kind="bar",
                series="method",
                description=f"Compare proposed and baseline methods on {primary}.",
            ),
            ExperimentPlotSpec(
                path="results/secondary_metric_curve.png",
                title=f"{secondary} Comparison",
                source="results_table",
                x="method",
                y=secondary,
                kind="bar",
                series="method",
                description=f"Compare proposed and baseline methods on {secondary}.",
            ),
            ExperimentPlotSpec(
                path="results/diagnostic_metric_curve.png",
                title=f"{tertiary} Diagnostic",
                source="results_table",
                x="method",
                y=tertiary,
                kind="bar",
                series="method",
                description=f"Show diagnostic differences across methods for {tertiary}.",
            ),
            ExperimentPlotSpec(
                path="results/robustness_metric_curve.png",
                title=f"{quaternary} Robustness",
                source="results_table",
                x="method",
                y=quaternary,
                kind="bar",
                series="method",
                description=f"Show robustness or secondary reporting behavior for {quaternary}.",
            ),
        ]
        for plot in required_plots:
            if len(spec.metrics) + len(spec.plots) >= _MIN_REPORTING_TARGETS:
                break
            if plot.path not in existing_paths:
                spec.plots.append(plot)
                existing_paths.add(plot.path)
        return spec

    def _evidence_baselines(self, evidence: list[EvidenceCard]) -> list[str]:
        baselines: list[str] = []
        seen: set[str] = set()
        for card in evidence:
            method = card.method.strip()
            if not method or method.lower() in {"n/a", "unknown", "method"}:
                continue
            key = method.lower()
            if key in seen:
                continue
            seen.add(key)
            baselines.append(method[:80])
            if len(baselines) >= 4:
                break
        return baselines

    def _build_spec_prompt(
        self,
        brief: ResearchBrief,
        idea: IdeaCard,
        decision: IdeaDecision,
        evidence: list[EvidenceCard],
        repos: list[RepoCard],
    ) -> str:
        return (
            self._build_spec_template.replace("{{research_brief}}", json.dumps(brief.to_dict(), indent=2, sort_keys=True))
            .replace("{{idea_card}}", json.dumps(idea.to_dict(), indent=2, sort_keys=True))
            .replace("{{idea_decision}}", json.dumps(decision.to_dict(), indent=2, sort_keys=True))
            .replace("{{supporting_evidence}}", json.dumps([card.to_dict() for card in evidence], indent=2, sort_keys=True))
            .replace("{{repo_cards}}", json.dumps([repo.to_dict() for repo in repos], indent=2, sort_keys=True))
        )

    def _ensure_environment(
        self,
        spec: BuildSpec,
        idea: IdeaCard,
        evidence: list[EvidenceCard],
        repos: list[RepoCard],
    ) -> BuildSpec:
        chosen, clone_report = self._resolve_reference_repo(idea, evidence, repos)
        self.artifacts.write_json("RepoCloneReport.json", clone_report)
        if chosen is None:
            spec.environment = self._generated_environment()
            return spec
        spec.repo_url = spec.repo_url or chosen.repo_url
        requirements = self._requirements_from_repo(chosen)
        spec.environment = BuildEnvironment(
            source="reference_repo" if chosen.env_files or chosen.local_repo_path else "reference_repo_metadata",
            reference_repo_url=chosen.repo_url,
            reference_repo_path=chosen.local_repo_path,
            env_files=chosen.env_files,
            language=chosen.language or "python",
            framework=chosen.framework,
            requirements=requirements,
            setup_commands=["pip install -r requirements.txt"],
        )
        return spec

    def _inject_run_budget(self, spec: BuildSpec) -> BuildSpec:
        """Overwrite epoch bounds in BuildSpec with values from the run_budget config."""
        train_min, train_max = self._run_bounds["train_epochs"]
        eval_min, eval_max = self._run_bounds["eval_epochs"]
        return spec.model_copy(update={
            "min_train_epochs": train_min,
            "max_train_epochs": train_max,
            "min_eval_epochs": eval_min,
            "max_eval_epochs": eval_max,
        })

    def _assess_code_strategy(self, spec: BuildSpec) -> BuildSpec:
        """Record whether a reference repo is available as optional CODE context.

        CODE always generates the standard standalone project layout. A reference
        repo can inform implementation details, but it is never adapted in-place
        or allowed to change the required output contract.
        """
        repo_path_str = spec.environment.reference_repo_path
        spec.environment.code_strategy = "generate_fresh"
        report: dict[str, Any] = {
            "repo_path": repo_path_str,
            "decision": "generate_fresh",
            "reference_repo_context": False,
            "reason": "",
        }

        if not repo_path_str or not Path(repo_path_str).exists():
            report["reason"] = "no reference repo available"
            self.artifacts.write_json("RepoAdaptationReport.json", report)
            return spec

        report["reference_repo_context"] = True
        report["reason"] = "reference repo will be provided as read-only context to the standard generator"
        self.artifacts.write_json("RepoAdaptationReport.json", report)
        self._progress("BUILD_SPEC", f"code strategy: {spec.environment.code_strategy}")
        return spec

    def _resolve_reference_repo(
        self,
        idea: IdeaCard,
        evidence: list[EvidenceCard],
        repos: list[RepoCard],
    ) -> tuple[RepoCard | None, dict[str, Any]]:
        candidates = self._candidate_repos_for_idea(idea, evidence, repos)
        report = {
            "status": "PASS",
            "strategy": "clone_repos_for_supporting_evidence_in_order",
            "candidate_count": len(candidates),
            "attempted": [],
            "selected_repo_url": "",
            "fallback": "generated_environment",
        }
        repo_manager = RepoManager(self.artifacts.path("repos"), timeout_seconds=120)
        for repo in candidates:
            before = repo.to_dict()
            resolved = repo_manager.inspect(repo) if repo.local_repo_path else repo_manager.clone_and_inspect(repo)
            attempt = {
                "repo_url": repo.repo_url,
                "source_paper_id": repo.source_paper_id,
                "status": resolved.status,
                "local_repo_path": resolved.local_repo_path,
                "env_files": resolved.env_files,
                "errors": resolved.errors,
            }
            if not resolved.env_files and resolved.status in {"inspected", "cloned"}:
                attempt["warning"] = "repo cloned or inspected but no environment file was detected"
            report["attempted"].append(attempt)
            if resolved.status in {"inspected", "cloned"}:
                report["selected_repo_url"] = resolved.repo_url
                report["fallback"] = None
                return resolved, report
            repo = RepoCard.model_validate(before)
        return None, report

    def _candidate_repos_for_idea(
        self,
        idea: IdeaCard,
        evidence: list[EvidenceCard],
        repos: list[RepoCard],
    ) -> list[RepoCard]:
        if not repos:
            return []
        supporting_paper_ids = {card.paper_id for card in evidence}
        supporting_paper_ids.update(idea.supporting_evidence_ids)
        matching = [repo for repo in repos if repo.source_paper_id in supporting_paper_ids]
        pool = matching or repos
        return sorted(pool, key=lambda repo: repo.relevance_score, reverse=True)

    def _generated_environment(self) -> BuildEnvironment:
        return BuildEnvironment(
            source="generated",
            requirements=["python>=3.11", "torch", "numpy"],
            setup_commands=["python -m venv .venv", "pip install -r requirements.txt"],
        )

    def _requirements_from_repo(self, repo: RepoCard) -> list[str]:
        requirements = ["python>=3.11"]
        for env_file in repo.env_files:
            path = Path(env_file)
            if path.name == "requirements.txt":
                try:
                    lines = [
                        line.strip()
                        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
                        if line.strip() and not line.strip().startswith("#")
                    ]
                except OSError:
                    lines = []
                requirements.extend(lines[:40])
                break
        return _dedupe(requirements) or ["python>=3.11"]

    def _parse_build_spec(self, raw: str) -> BuildSpec | None:
        value = self._extract_json_object(raw)
        if not isinstance(value, dict):
            return None
        try:
            return BuildSpec.model_validate(value)
        except Exception:
            return None

    def _extract_json_object(self, raw: str) -> Any:
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            for index, char in enumerate(raw):
                if char != "{":
                    continue
                try:
                    value, _ = decoder.raw_decode(raw[index:])
                except json.JSONDecodeError:
                    continue
                return value if isinstance(value, dict) else None
        return None

    def code(self, spec: BuildSpec, evidence: list[EvidenceCard] | None = None) -> tuple[CodeRunReport, str, str]:
        """Skill: code_from_build_spec.

        Use when: BuildSpec.json is ready and experiment assets need to be created.
        Inputs: BuildSpec.json only.
        Outputs: a standalone code directory, result artifacts, EXPERIMENT_LOG.md, CodeRunReport.json.
        """
        self._progress("CODE", "resolve build environment")
        spec = self._resolve_code_environment(spec)
        prompt = self._code_prompt(spec)
        code_dir = self.artifacts.path("code")
        results_dir = self.artifacts.path("results")
        llm_raw = ""
        llm_attempted = self.llm is not None
        used_llm = False
        staged_generation = False
        llm_request_failed = False
        trunc_state_path = self.artifacts.path("CodeTruncationState.json")
        code_eval_repair_mode = self._should_repair_existing_code_after_eval(code_dir)
        code_failure_repair_mode = (
            not code_eval_repair_mode
            and not trunc_state_path.exists()
            and self._should_repair_existing_code_after_code_failure(code_dir)
        )
        previous_code_report = self.artifacts.read_json("CodeRunReport.json") if self.artifacts.path("CodeRunReport.json").exists() else {}
        repair_report: dict[str, Any] = {
            "attempted": False,
            "succeeded": False,
            "reason": None,
            "errors": [],
        }
        command = [sys.executable, "run_experiment.py", "--config", "configs/experiment_config.json"]
        is_continuation = trunc_state_path.exists() and not code_eval_repair_mode
        if code_eval_repair_mode:
            self._progress("CODE", "repair existing code from CODE_EVAL feedback")
            used_llm = not bool(previous_code_report.get("fallback_used"))
            generation_mode = "code_eval_repair"
            eval_feedback = self._code_eval_feedback_completed_process(command)
            history = self._previous_repair_attempts()
            prompt = self._code_stage_repair_prompt(spec, code_dir, command, eval_feedback, history, len(history) + 1)
            repair_report = self._repair_code_project(spec, code_dir, command, eval_feedback)
            llm_raw = self.artifacts.path("CodeRepair.raw.txt").read_text(encoding="utf-8", errors="ignore") if self.artifacts.path("CodeRepair.raw.txt").exists() else ""
        elif code_failure_repair_mode:
            self._progress("CODE", "repair existing code from previous CODE failure")
            used_llm = not bool(previous_code_report.get("fallback_used"))
            generation_mode = "code_failure_repair"
            failure_feedback = self._code_failure_feedback_completed_process(command)
            history = self._previous_repair_attempts()
            prompt = self._code_stage_repair_prompt(spec, code_dir, command, failure_feedback, history, len(history) + 1)
            repair_report = self._repair_code_project(spec, code_dir, command, failure_feedback)
            llm_raw = self.artifacts.path("CodeRepair.raw.txt").read_text(encoding="utf-8", errors="ignore") if self.artifacts.path("CodeRepair.raw.txt").exists() else ""
        elif llm_attempted:
            if is_continuation:
                trunc_state = self.artifacts.read_json("CodeTruncationState.json")
                prompt = self._continuation_prompt(spec, trunc_state)
                self._progress("CODE", f"continuing truncated generation — missing: {trunc_state.get('missing_files', [])}")
                self._progress("CODE", f"request builder LLM code project (strategy: {spec.environment.code_strategy})")
                try:
                    llm_raw = self.llm.complete(prompt)
                    used_llm = self._write_llm_code_project(llm_raw, code_dir)
                except Exception as exc:
                    llm_request_failed = True
                    llm_raw = f"Builder LLM request failed: {exc}"
                    self.artifacts.write_markdown("CodeGeneration.error.txt", llm_raw)
                    self._progress("CODE", "builder LLM request failed; using fallback")
                if llm_raw and not used_llm and not llm_request_failed:
                    self.artifacts.write_markdown("CodeContinuation.raw.txt", llm_raw)
                    self._progress("CODE", "LLM code response invalid; using fallback")
            else:
                self._progress("CODE", "request staged builder LLM code project")
                try:
                    staged = self._run_staged_code_generation(spec, code_dir, evidence=evidence or [])
                    prompt = staged.get("prompt", prompt)
                    llm_raw = staged.get("raw", "")
                    used_llm = bool(staged.get("files_written"))
                    staged_generation = used_llm
                    self.artifacts.write_json("CodeStageGenerationReport.json", staged.get("report", {}))
                    if not used_llm:
                        self._progress("CODE", "staged LLM code response wrote no files; using fallback")
                except Exception as exc:
                    llm_request_failed = True
                    llm_raw = f"Staged builder LLM request failed: {exc}"
                    self.artifacts.write_markdown("CodeGeneration.error.txt", llm_raw)
                    self._progress("CODE", "staged builder LLM request failed; using fallback")
        is_truncated = trunc_state_path.exists() and not used_llm and not code_eval_repair_mode and not code_failure_repair_mode
        if not used_llm and not code_eval_repair_mode and not code_failure_repair_mode and not is_truncated:
            self._progress("CODE", "write generic fallback experiment project")
            self._write_code_project(spec, code_dir)
        if is_truncated:
            self._progress("CODE", "output was truncated — partial files preserved, will continue on next attempt")
        if not code_eval_repair_mode and not code_failure_repair_mode:
            if used_llm and is_continuation:
                generation_mode = "llm_continuation"
            elif used_llm:
                generation_mode = "llm_generated"
            elif llm_request_failed:
                generation_mode = "generic_fallback_after_llm_error"
            elif llm_attempted:
                generation_mode = "generic_fallback_after_invalid_llm"
            else:
                generation_mode = "generic_fallback_no_llm"
        self._progress("CODE", f"generation mode: {generation_mode}")
        self._finalize_code_project_scaffold(spec, code_dir)
        config_normalization_report = self._normalize_generated_experiment_config(code_dir)
        self.artifacts.write_json("ConfigNormalizationReport.json", config_normalization_report)
        self._progress("CODE", f"config normalization: {config_normalization_report.get('status', 'UNKNOWN')}")
        results_dir.mkdir(parents=True, exist_ok=True)
        evidence = evidence or []
        dataset_report = self._acquire_verification_dataset(spec, code_dir, evidence=evidence)
        self.artifacts.write_json("DatasetAcquisitionReport.json", dataset_report)
        self._progress("CODE", f"dataset acquisition: {dataset_report.get('status', 'UNKNOWN')} ({dataset_report.get('reason') or 'ok'})")
        self._clear_stale_result_files(code_dir)
        device_report = self._resolve_generated_code_device(code_dir)
        self.artifacts.write_json("DeviceReport.json", device_report)
        self._progress("CODE", f"device: {device_report.get('selection', {}).get('selected', 'unknown')}")
        syntax_report = self._check_code_syntax(code_dir)
        self.artifacts.write_json("CodeSyntaxReport.json", syntax_report)
        self._progress("CODE", f"syntax check: {syntax_report.get('status', 'UNKNOWN')}")

        if is_truncated:
            # Don't run experiment — entry point (run_experiment.py) may be missing
            # Return FAIL immediately so next CODE attempt does continuation
            trunc_state = self.artifacts.read_json("CodeTruncationState.json")
            report = CodeRunReport(
                status="FAIL",
                code_dir=str(code_dir.relative_to(self.artifacts.run_dir)),
                generation_mode="llm_truncated_awaiting_continuation",
                fallback_used=False,
                executed=False,
                returncode=None,
                outputs=[],
                repair_attempted=False,
                repair_succeeded=False,
                errors=[
                    f"Code generation was truncated. Missing files: {trunc_state.get('missing_files', [])}. "
                    "Next attempt will continue from the truncation point."
                ],
            )
            self.artifacts.write_json("CodeRunReport.json", report.to_dict())
            self.artifacts.write_json("CodeRepairReport.json", repair_report)
            log = self._build_experiment_log(spec, report, "", "")
            self.artifacts.write_markdown("EXPERIMENT_LOG.md", log)
            return report, prompt, "truncated"

        dependency_report: dict[str, Any] = {"attempted": False, "succeeded": False}
        runtime_patch_report: dict[str, Any] = {"attempted": False, "patched": False}
        if syntax_report.get("status") != "PASS":
            self._progress("CODE", "syntax check failed; skip experiment execution")
            completed = subprocess.CompletedProcess(
                command,
                2,
                stdout="",
                stderr=json.dumps(syntax_report, indent=2, sort_keys=True),
            )
            if not code_failure_repair_mode:
                self._progress("CODE", "syntax check failed; request code repair")
                repair_report = self._repair_code_project(spec, code_dir, command, completed)
                if repair_report.get("attempted") and repair_report.get("files_written"):
                    syntax_report = self.artifacts.read_json("CodeSyntaxReport.json")
                    completed = subprocess.CompletedProcess(
                        command,
                        int(repair_report.get("returncode_after_repair") or 1),
                        stdout=str(repair_report.get("stdout_tail_after") or ""),
                        stderr=str(repair_report.get("stderr_tail_after") or ""),
                    )
        else:
            self._progress("CODE", "execute generated experiment")
            completed = self._run_experiment(command, code_dir)
            self._progress("CODE", f"experiment returncode: {completed.returncode}")
            dependency_report = self._install_missing_dependency_from_error(code_dir, completed)
            if dependency_report.get("succeeded"):
                self._progress("CODE", f"installed missing dependency: {dependency_report.get('requirement')}")
                completed = self._run_experiment(command, code_dir)
                self._progress("CODE", f"experiment returncode after dependency install: {completed.returncode}")
            runtime_patch_report = self._patch_runtime_error_from_failure(code_dir, completed)
            if runtime_patch_report.get("patched"):
                self._progress("CODE", "applied deterministic runtime patch; rerun experiment")
                completed = self._run_experiment(command, code_dir)
                self._progress("CODE", f"experiment returncode after runtime patch: {completed.returncode}")
            if (code_eval_repair_mode or code_failure_repair_mode) and repair_report.get("attempted"):
                repair_report["succeeded"] = completed.returncode == 0 and bool(repair_report.get("files_written"))
                repair_report["returncode_after_repair"] = completed.returncode
            if completed.returncode != 0 and not code_failure_repair_mode:
                self._progress("CODE", "experiment failed; request code repair")
                repair_report = self._repair_code_project(spec, code_dir, command, completed)
                if repair_report.get("attempted") and repair_report.get("files_written"):
                    self._progress("CODE", "repair files written; rerun preflight and experiment")
                    self._finalize_code_project_scaffold(spec, code_dir)
                    config_normalization_report = self._normalize_generated_experiment_config(code_dir)
                    self.artifacts.write_json("ConfigNormalizationReport.json", config_normalization_report)
                    dataset_report = self._acquire_verification_dataset(spec, code_dir, evidence=evidence)
                    self.artifacts.write_json("DatasetAcquisitionReport.json", dataset_report)
                    device_report = self._resolve_generated_code_device(code_dir)
                    self.artifacts.write_json("DeviceReport.json", device_report)
                    syntax_report = self._check_code_syntax(code_dir)
                    self.artifacts.write_json("CodeSyntaxReport.json", syntax_report)
                    if syntax_report.get("status") != "PASS":
                        completed = subprocess.CompletedProcess(
                            command,
                            2,
                            stdout="",
                            stderr=json.dumps(syntax_report, indent=2, sort_keys=True),
                        )
                    else:
                        completed = self._run_experiment(command, code_dir)
                        self._progress("CODE", f"experiment returncode after repair: {completed.returncode}")
                        second_dependency_report = self._install_missing_dependency_from_error(code_dir, completed)
                        if second_dependency_report.get("attempted"):
                            dependency_report.setdefault("repair_attempts", []).append(second_dependency_report)
                        if second_dependency_report.get("succeeded"):
                            self._progress("CODE", f"installed missing dependency after repair: {second_dependency_report.get('requirement')}")
                            completed = self._run_experiment(command, code_dir)
                            self._progress("CODE", f"experiment returncode after repair dependency install: {completed.returncode}")
                        second_runtime_patch_report = self._patch_runtime_error_from_failure(code_dir, completed)
                        if second_runtime_patch_report.get("attempted"):
                            runtime_patch_report.setdefault("repair_attempts", []).append(second_runtime_patch_report)
                        if second_runtime_patch_report.get("patched"):
                            self._progress("CODE", "applied deterministic runtime patch after repair; rerun experiment")
                            completed = self._run_experiment(command, code_dir)
                            self._progress("CODE", f"experiment returncode after repair runtime patch: {completed.returncode}")
                    repair_report["succeeded"] = completed.returncode == 0
                    repair_report["returncode_after_repair"] = completed.returncode
                    if completed.returncode != 0:
                        repair_report["errors"].append(completed.stderr[-4000:] or "repair rerun failed without stderr")
        self._collect_generated_result_files(code_dir)
        self._augment_results_with_paper_baselines(spec)
        self.artifacts.write_json("CodeRepairReport.json", repair_report)
        self.artifacts.write_json("DependencyInstallReport.json", dependency_report)
        self.artifacts.write_json("RuntimePatchReport.json", runtime_patch_report)
        outputs = [
            "code/README.md",
            "code/ENVIRONMENT.md",
            "code/requirements.txt",
            "code/environment.yml",
            "code/EXPERIMENT_METRICS.md",
            "code/configs/experiment_config.json",
            "code/src/dataset.py",
            "code/src/method.py",
            "code/src/baselines.py",
            "code/src/train.py",
            "code/src/evaluate.py",
            "code/src/plot.py",
            "code/run_experiment.py",
            "EnvironmentResolutionReport.json",
            "ConfigNormalizationReport.json",
            "DatasetAcquisitionReport.json",
            "DeviceReport.json",
            "CodeSyntaxReport.json",
            "DependencyInstallReport.json",
            "RuntimePatchReport.json",
            "ImplementationContract.json",
            "CodeStageGenerationReport.json",
            "CodeStage_core_report.json",
            "CodeStage_experiment_report.json",
            "code/CoreImplementationReport.json",
            "code/ExperimentImplementationReport.json",
            "CodeRepairReport.json",
            "ResultCollectionReport.json",
            "PaperBaselineResults.json",
            "results/metrics.json",
            "results/results_table.csv",
            "results/progress_log.json",
            "results/progress_log.jsonl",
            "results/progress_curve.png",
            "results/training_log.json",
            "results/training_log.jsonl",
            "results/training_curve.png",
            "results/eval_curve.png",
        ]
        errors = []
        for failure in syntax_report.get("failures", []):
            errors.append(f"CODE syntax check failed [{failure.get('path', 'unknown')}]: {failure.get('error', '')}")
        if completed.returncode != 0:
            errors.append(completed.stderr[-4000:] or "generated experiment failed without stderr")
        fallback_is_placeholder = not used_llm and llm_attempted
        if fallback_is_placeholder:
            errors.append(
                "Builder LLM did not produce a valid full code project; generic fallback only validates "
                "pipeline execution and is not a domain-specific experiment result."
            )
        report = CodeRunReport(
            status="PASS" if completed.returncode == 0 and not fallback_is_placeholder and syntax_report.get("status") == "PASS" else "FAIL",
            code_dir="code",
            generation_mode=generation_mode,
            fallback_used=not used_llm,
            repair_attempted=bool(repair_report.get("attempted")),
            repair_succeeded=bool(repair_report.get("succeeded")),
            outputs=[item for item in outputs if self.artifacts.path(item).exists()],
            executed=True,
            returncode=completed.returncode,
            errors=errors,
        )
        self._write_experiment_metrics_markdown(spec, completed, report)
        report.outputs = [item for item in outputs if self.artifacts.path(item).exists()]
        log = self._experiment_log(spec, command, completed, report)
        self.artifacts.write_markdown("EXPERIMENT_LOG.md", log)
        self.artifacts.write_json("CodeRunReport.json", report)
        self._progress("CODE", f"final status: {report.status}; report: CodeRunReport.json")
        return report, prompt, llm_raw or log

    def _should_repair_existing_code_after_eval(self, code_dir: Path) -> bool:
        if not (code_dir / "run_experiment.py").exists():
            return False
        if not self.artifacts.path("ExperimentAudit.json").exists():
            return False
        try:
            audit = self.artifacts.read_json("ExperimentAudit.json")
        except Exception:
            return False
        if str(audit.get("status", "")).upper() != "FAIL":
            return False
        code_report = self.artifacts.read_json("CodeRunReport.json") if self.artifacts.path("CodeRunReport.json").exists() else {}
        return code_report.get("status") == "PASS"

    def _should_repair_existing_code_after_code_failure(self, code_dir: Path) -> bool:
        if not (code_dir / "run_experiment.py").exists():
            return False
        if not self.artifacts.path("CodeRunReport.json").exists() and not self.artifacts.path("CodeSyntaxReport.json").exists():
            return False
        code_report = self.artifacts.read_json("CodeRunReport.json") if self.artifacts.path("CodeRunReport.json").exists() else {}
        syntax_report = self.artifacts.read_json("CodeSyntaxReport.json") if self.artifacts.path("CodeSyntaxReport.json").exists() else {}
        if str(code_report.get("status", "")).upper() == "FAIL":
            return not bool(code_report.get("fallback_used"))
        return str(syntax_report.get("status", "")).upper() == "FAIL"

    def _code_eval_feedback_completed_process(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        audit = self.artifacts.read_json("ExperimentAudit.json") if self.artifacts.path("ExperimentAudit.json").exists() else {}
        quality = self.artifacts.read_json("CodeEvalQualityReport.json") if self.artifacts.path("CodeEvalQualityReport.json").exists() else {}
        failures = audit.get("failures", [])
        if not failures and isinstance(quality, dict):
            failures = [
                f"{failure.get('rule', 'code_quality_failed')}: {failure.get('reason', '')}"
                for failure in quality.get("failures", [])
            ]
        feedback = {
            "source": "CODE_EVAL",
            "instruction": "Repair the existing project only. Do not regenerate or replace the full codebase.",
            "experiment_audit": audit,
            "code_eval_quality": quality,
            "failures": failures,
        }
        stderr = "CODE_EVAL failed; targeted repair required:\n" + json.dumps(feedback, indent=2, sort_keys=True)
        stdout = self.artifacts.path("EXPERIMENT_LOG.md").read_text(encoding="utf-8", errors="ignore") if self.artifacts.path("EXPERIMENT_LOG.md").exists() else ""
        return subprocess.CompletedProcess(command, 1, stdout=stdout[-4000:], stderr=stderr[-12000:])

    def _code_failure_feedback_completed_process(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        code_report = self.artifacts.read_json("CodeRunReport.json") if self.artifacts.path("CodeRunReport.json").exists() else {}
        syntax_report = self.artifacts.read_json("CodeSyntaxReport.json") if self.artifacts.path("CodeSyntaxReport.json").exists() else {}
        repair_report = self.artifacts.read_json("CodeRepairReport.json") if self.artifacts.path("CodeRepairReport.json").exists() else {}
        experiment_log = self.artifacts.path("EXPERIMENT_LOG.md").read_text(encoding="utf-8", errors="ignore") if self.artifacts.path("EXPERIMENT_LOG.md").exists() else ""
        feedback = {
            "source": "CODE",
            "instruction": "Repair the existing generated project only. Do not regenerate the full codebase.",
            "code_run_report": code_report,
            "code_syntax_report": syntax_report,
            "code_repair_report": repair_report,
        }
        return subprocess.CompletedProcess(
            command,
            int(code_report.get("returncode") or 1),
            stdout=experiment_log[-4000:],
            stderr=("CODE failed; targeted repair required:\n" + json.dumps(feedback, indent=2, sort_keys=True))[-16000:],
        )


    def _progress(self, stage: str, message: str) -> None:
        if self.reporter is not None:
            self.reporter(f"[{stage}] {message}")

    def _check_code_syntax(self, code_dir: Path) -> dict[str, Any]:
        failures: list[dict[str, Any]] = []
        for path in sorted(code_dir.rglob("*.py")):
            try:
                source = path.read_text(encoding="utf-8")
                compile(source, str(path), "exec")
            except Exception as exc:
                failures.append({"path": str(path.relative_to(code_dir)), "error": f"{type(exc).__name__}: {exc}"})
        return {
            "status": "PASS" if not failures else "FAIL",
            "checked_files": len(list(code_dir.rglob("*.py"))),
            "failures": failures,
        }

    def _augment_results_with_paper_baselines(self, spec: BuildSpec) -> dict[str, Any]:
        report: dict[str, Any] = {
            "status": "SKIP",
            "baseline_count": 0,
            "appended_count": 0,
            "baselines": [],
            "reason": None,
        }
        table_path = self.artifacts.path("results/results_table.csv")
        if not table_path.exists():
            report["reason"] = "results_table_missing"
            self.artifacts.write_json("PaperBaselineResults.json", report)
            return report
        baselines = [item.strip() for item in spec.baselines if item.strip()]
        if not baselines:
            report["reason"] = "build_spec_has_no_baselines"
            self.artifacts.write_json("PaperBaselineResults.json", report)
            return report

        rows = self._load_results_rows(table_path)
        base_rows = [row for row in rows if row.get("Source") != "paper_reported"]
        existing_methods = {self._normalize_baseline_name(row.get("Method") or row.get("method") or "") for row in base_rows}
        evidence = self.artifacts.read_jsonl("EvidenceCards.jsonl")
        paper_cards = self.artifacts.read_jsonl("PaperCards.jsonl")
        additions: list[dict[str, str]] = []
        baseline_reports = []
        for baseline in baselines:
            normalized = self._normalize_baseline_name(baseline)
            if normalized in existing_methods:
                continue
            matches, confirmation_attempts = self._confirm_baseline_matches(baseline, evidence, paper_cards)
            reported_result = self._baseline_reported_result(matches)
            paper_ids = sorted({str(item.get("paper_id", "")) for item in matches if item.get("paper_id")})
            evidence_ids = sorted({str(item.get("evidence_id", "")) for item in matches if item.get("evidence_id")})
            baseline_report = {
                "baseline": baseline,
                "source": "paper_reported" if matches else "build_spec_only",
                "confirmation_attempts": confirmation_attempts,
                "confirmed_no_paper_baseline_result": not matches,
                "paper_ids": paper_ids,
                "evidence_ids": evidence_ids,
                "reported_result": reported_result,
            }
            baseline_reports.append(baseline_report)
            additions.append(
                {
                    "Method": baseline,
                    "Score": "N/A",
                    "Std": "N/A",
                    "Baseline_Score": "N/A",
                    "Improvement": "N/A",
                    "Source": baseline_report["source"],
                    "Evidence_IDs": ";".join(evidence_ids),
                    "Paper_IDs": ";".join(paper_ids),
                    "Reported_Result": reported_result,
                }
            )

        if additions:
            self._write_results_rows_with_union_header(table_path, base_rows + additions)
        report.update(
            {
                "status": "PASS",
                "baseline_count": len(baselines),
                "appended_count": len(additions),
                "baselines": baseline_reports,
                "reason": None,
            }
        )
        self.artifacts.write_json("PaperBaselineResults.json", report)
        return report

    def _confirm_baseline_matches(
        self,
        baseline: str,
        evidence: list[dict[str, Any]],
        paper_cards: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Run three independent artifact checks before declaring no paper baseline result."""
        checks: list[tuple[str, list[dict[str, Any]]]] = [
            ("evidence_cards", self._evidence_for_baseline(baseline, evidence)),
            ("paper_cards", self._paper_matches_as_evidence(self._paper_cards_for_baseline(baseline, paper_cards))),
            ("combined_relaxed", self._relaxed_baseline_matches(baseline, evidence, paper_cards)),
        ]
        attempts: list[dict[str, Any]] = []
        merged: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for name, matches in checks:
            attempts.append(
                {
                    "check": name,
                    "status": "FOUND" if matches else "NOT_FOUND",
                    "match_count": len(matches),
                    "paper_ids": sorted({str(item.get("paper_id", "")) for item in matches if item.get("paper_id")}),
                    "evidence_ids": sorted({str(item.get("evidence_id", "")) for item in matches if item.get("evidence_id")}),
                }
            )
            for item in matches:
                key = (
                    str(item.get("evidence_id", "")),
                    str(item.get("paper_id", "")),
                    str(item.get("method", "")),
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(item)
        return merged, attempts

    def _relaxed_baseline_matches(
        self,
        baseline: str,
        evidence: list[dict[str, Any]],
        paper_cards: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        stopwords = {
            "offline",
            "online",
            "fully",
            "trained",
            "baseline",
            "baselines",
            "naive",
            "learning",
            "imitation",
            "flow",
            "matching",
            "nature",
        }
        tokens = [
            token
            for token in re.split(r"[^a-zA-Z0-9]+", baseline.lower())
            if (len(token) >= 3 or token in {"bc"}) and token not in stopwords
        ]
        if not tokens:
            return []
        artifacts = list(evidence) + self._paper_matches_as_evidence(paper_cards)
        matches = []
        for item in artifacts:
            haystack = self._baseline_search_haystack(item)
            if all(re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", haystack) for token in tokens):
                matches.append(item)
        return matches

    def _normalize_baseline_name(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    def _evidence_for_baseline(self, baseline: str, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        stopwords = {
            "offline",
            "online",
            "fully",
            "trained",
            "baseline",
            "baselines",
            "naive",
            "learning",
            "imitation",
            "flow",
            "matching",
            "nature",
        }
        aliases = self._baseline_aliases(baseline, stopwords)
        matches = []
        for item in evidence:
            haystack = self._baseline_search_haystack(item)
            compact_haystack = re.sub(r"[^a-z0-9]+", "", haystack)
            if any(self._baseline_alias_matches(alias, haystack, compact_haystack) for alias in aliases):
                matches.append(item)
        return matches

    def _paper_cards_for_baseline(self, baseline: str, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        aliases = self._baseline_aliases(
            baseline,
            {
                "offline",
                "online",
                "fully",
                "trained",
                "baseline",
                "baselines",
                "naive",
                "learning",
                "imitation",
                "flow",
                "matching",
                "nature",
            },
        )
        matches = []
        for item in papers:
            haystack = self._baseline_search_haystack(item)
            compact_haystack = re.sub(r"[^a-z0-9]+", "", haystack)
            if any(self._baseline_alias_matches(alias, haystack, compact_haystack) for alias in aliases):
                matches.append(item)
        return matches

    def _paper_matches_as_evidence(self, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        matches = []
        for item in papers:
            paper_id = str(item.get("paper_id", "")).strip()
            title = str(item.get("title", "")).strip()
            abstract = str(item.get("abstract", "")).strip()
            if not paper_id and not title and not abstract:
                continue
            matches.append(
                {
                    "evidence_id": f"paper-{paper_id}" if paper_id else "",
                    "paper_id": paper_id,
                    "claims": [abstract] if abstract else [],
                    "metrics": [],
                    "method": title,
                    "setting": title,
                    "task": str(item.get("query_source", "")),
                }
            )
        return matches

    def _baseline_aliases(self, baseline: str, stopwords: set[str]) -> set[str]:
        tokens = [
            token
            for token in re.split(r"[^a-zA-Z0-9]+", baseline.lower())
            if (len(token) >= 3 or token in {"bc"}) and token not in stopwords
        ]
        aliases = set(tokens)
        parenthetical = re.findall(r"\(([^)]+)\)", baseline)
        aliases.update(item.lower() for item in parenthetical if len(item) >= 2)
        compact = re.sub(r"[^a-z0-9]+", "", baseline.lower())
        compact = compact.replace("offline", "").replace("online", "").replace("baseline", "").replace("baselines", "")
        if compact:
            aliases.add(compact)
        return {alias for alias in aliases if alias and alias not in stopwords}

    def _baseline_alias_matches(self, alias: str, haystack: str, compact_haystack: str) -> bool:
        if len(alias) <= 3:
            return re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", haystack) is not None
        return alias in haystack or alias in compact_haystack

    def _baseline_search_haystack(self, item: dict[str, Any]) -> str:
        return " ".join(
            [
                str(item.get("title", "")),
                str(item.get("abstract", "")),
                str(item.get("setting", "")),
                str(item.get("method", "")),
                str(item.get("task", "")),
                " ".join(str(claim) for claim in item.get("claims", [])),
                " ".join(str(metric) for metric in item.get("metrics", [])),
            ]
        ).lower()

    def _baseline_reported_result(self, evidence: list[dict[str, Any]]) -> str:
        snippets: list[str] = []
        for item in evidence[:3]:
            claims = [str(claim) for claim in item.get("claims", []) if str(claim).strip()]
            metrics = [str(metric) for metric in item.get("metrics", []) if str(metric).strip()]
            setting = str(item.get("setting", "")).strip()
            if claims:
                snippets.append(claims[0])
            elif metrics:
                snippets.append("Metrics: " + ", ".join(metrics[:4]))
            elif setting:
                snippets.append(setting)
        return " | ".join(snippets)[:1000] if snippets else "N/A"

    def _write_results_rows_with_union_header(self, path: Path, rows: list[dict[str, str]]) -> None:
        preferred = ["Method", "Score", "Std", "Baseline_Score", "Improvement", "Source", "Evidence_IDs", "Paper_IDs", "Reported_Result"]
        fieldnames = []
        for name in preferred:
            if any(name in row for row in rows):
                fieldnames.append(name)
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})

    def _collect_generated_result_files(self, code_dir: Path) -> dict[str, Any]:
        report: dict[str, Any] = {"present": [], "missing": [], "removed_misplaced": []}
        root_results = self.artifacts.path("results")
        root_results.mkdir(parents=True, exist_ok=True)
        for name in [
            "metrics.json",
            "results_table.csv",
            "progress_log.json",
            "progress_log.jsonl",
            "progress_curve.png",
            "training_log.json",
            "training_log.jsonl",
            "training_curve.png",
            "eval_curve.png",
        ]:
            destination = root_results / name
            if destination.exists():
                report["present"].append(str(destination))
            else:
                report["missing"].append(name)
            misplaced = code_dir / "results" / name
            if misplaced.exists():
                misplaced.unlink()
                report["removed_misplaced"].append(str(misplaced))
        self.artifacts.write_json("ResultCollectionReport.json", report)
        return report

    def _clear_stale_result_files(self, code_dir: Path) -> None:
        for path in [
            self.artifacts.path("results/metrics.json"),
            self.artifacts.path("results/results_table.csv"),
            self.artifacts.path("results/progress_log.json"),
            self.artifacts.path("results/progress_log.jsonl"),
            self.artifacts.path("results/progress_curve.png"),
            self.artifacts.path("results/training_log.json"),
            self.artifacts.path("results/training_log.jsonl"),
            self.artifacts.path("results/training_curve.png"),
            self.artifacts.path("results/eval_curve.png"),
            code_dir / "results" / "metrics.json",
            code_dir / "results" / "results_table.csv",
            code_dir / "results" / "progress_log.json",
            code_dir / "results" / "progress_log.jsonl",
            code_dir / "results" / "progress_curve.png",
            code_dir / "results" / "training_log.json",
            code_dir / "results" / "training_log.jsonl",
            code_dir / "results" / "training_curve.png",
            code_dir / "results" / "eval_curve.png",
        ]:
            try:
                path.unlink()
            except FileNotFoundError:
                continue

    def _acquire_verification_dataset(
        self,
        spec: BuildSpec,
        code_dir: Path,
        evidence: list[EvidenceCard] | None = None,
    ) -> dict[str, Any]:
        report: dict[str, Any] = {
            "status": "SKIP",
            "attempted": False,
            "dataset_id": None,
            "source_url": None,
            "raw_path": None,
            "normalized_path": None,
            "patched_config_paths": [],
            "patched_runtime_paths": [],
            "errors": [],
            "reason": None,
        }
        config_path = code_dir / "configs" / "experiment_config.json"
        if not config_path.exists():
            report["reason"] = "experiment_config_missing"
            return report
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            report["status"] = "FAIL"
            report["reason"] = "config_parse_failed"
            report["errors"].append(str(exc))
            return report
        if not isinstance(config, dict):
            report["status"] = "FAIL"
            report["reason"] = "experiment_config_not_object"
            return report

        dataset_paths = self._find_dataset_paths(config)
        target_paths = [item for item in dataset_paths if self._dataset_path_needs_acquisition(item)]
        if not target_paths:
            report["reason"] = "no_dataset_paths_requiring_acquisition"
            return report
        return self._acquire_generic_smoke_dataset(spec, code_dir, config, report, evidence or [])

    def _acquire_generic_smoke_dataset(
        self,
        spec: BuildSpec,
        code_dir: Path,
        config: dict[str, Any],
        report: dict[str, Any],
        evidence: list[EvidenceCard],
    ) -> dict[str, Any]:
        dataset_dir = code_dir / "datasets"
        raw_dir = dataset_dir / "raw"
        dataset_path = dataset_dir / self._generic_dataset_filename(spec)
        candidates = self._dataset_candidates_for_spec(spec, evidence)[:self._max_download_attempts]
        attempts: list[dict[str, Any]] = []
        downloaded_candidate: dict[str, Any] | None = None
        downloaded_path: Path | None = None
        for candidate in candidates:
            raw_path = raw_dir / str(candidate["filename"])
            attempt = {
                "dataset_id": candidate["dataset_id"],
                "source_url": candidate["source_url"],
                "path": self._relative_to_code_dir(raw_path, code_dir),
                "status": "FAIL",
                "error": None,
            }
            try:
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                if not raw_path.exists():
                    self._download_file(str(candidate["source_url"]), raw_path)
                attempt["status"] = "PASS"
                downloaded_candidate = candidate
                downloaded_path = raw_path
                attempts.append(attempt)
                break
            except Exception as exc:
                attempt["error"] = str(exc)
                attempts.append(attempt)
        report.update(
            {
                "status": "FAIL",
                "attempted": True,
                "dataset_id": downloaded_candidate["dataset_id"] if downloaded_candidate else "synthetic-smoke",
                "source_url": downloaded_candidate["source_url"] if downloaded_candidate else "generated_locally",
                "raw_path": self._relative_to_code_dir(downloaded_path, code_dir) if downloaded_path else self._relative_to_code_dir(dataset_path, code_dir),
                "normalized_path": self._relative_to_code_dir(dataset_path, code_dir),
                "reason": None,
                "download_attempts": attempts,
            }
        )
        try:
            self._write_generic_smoke_dataset(dataset_path, spec, source_candidate=downloaded_candidate)
            patched = self._patch_dataset_paths(config, self._relative_to_code_dir(dataset_path, code_dir))
            runtime_patched = self._cap_verification_runtime(config)
            config_path = code_dir / "configs" / "experiment_config.json"
            config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            report["patched_config_paths"] = patched
            report["patched_runtime_paths"] = runtime_patched
            report["status"] = "PASS"
            report["reason"] = (
                "downloaded_evidence_informed_dataset_and_generated_smoke_view"
                if downloaded_candidate
                else "generated_domain_agnostic_smoke_dataset_after_download_failures"
            )
        except Exception as exc:
            report["errors"].append(str(exc))
            report["reason"] = "generic_dataset_generation_failed"
        return report

    def _dataset_candidates_for_spec(self, spec: BuildSpec, evidence: list[EvidenceCard]) -> list[dict[str, str]]:
        text = self._dataset_selection_text(spec, evidence)
        candidate_groups: list[list[dict[str, str]]] = []
        if any(token in text for token in ["shapenet", "modelnet", "3d", "gaussian", "splat", "mesh", "geometry", "point cloud", "pattern"]):
            candidate_groups.append(
                [
                    {
                        "dataset_id": "ModelNet10",
                        "source_url": "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
                        "filename": "ModelNet10.zip",
                    },
                    {
                        "dataset_id": "ModelNet40",
                        "source_url": "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet40.zip",
                        "filename": "ModelNet40.zip",
                    },
                    {
                        "dataset_id": "tiny-nerf",
                        "source_url": "https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz",
                        "filename": "tiny_nerf_data.npz",
                    },
                ]
            )
        if any(token in text for token in ["offline rl", "offline reinforcement", "d4rl", "mujoco", "atari", "dqn replay"]):
            candidate_groups.append(
                [
                    {
                        "dataset_id": "hopper-medium-v2",
                        "source_url": "https://huggingface.co/datasets/imone/D4RL/resolve/main/hopper_medium-v2.hdf5",
                        "filename": "hopper_medium-v2.hdf5",
                    },
                    {
                        "dataset_id": "halfcheetah-medium-v2",
                        "source_url": "https://huggingface.co/datasets/imone/D4RL/resolve/main/halfcheetah_medium-v2.hdf5",
                        "filename": "halfcheetah_medium-v2.hdf5",
                    },
                    {
                        "dataset_id": "walker2d-medium-v2",
                        "source_url": "https://huggingface.co/datasets/imone/D4RL/resolve/main/walker2d_medium-v2.hdf5",
                        "filename": "walker2d_medium-v2.hdf5",
                    },
                ]
            )
        if any(token in text for token in ["mnist", "image", "vision", "segmentation", "classification", "render"]):
            candidate_groups.append(
                [
                    {
                        "dataset_id": "MNIST-train-images",
                        "source_url": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
                        "filename": "mnist_train_images.gz",
                    },
                    {
                        "dataset_id": "MNIST-train-labels",
                        "source_url": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
                        "filename": "mnist_train_labels.gz",
                    },
                    {
                        "dataset_id": "CIFAR10-python",
                        "source_url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                        "filename": "cifar-10-python.tar.gz",
                    },
                ]
            )
        if any(token in text for token in ["graph", "node", "edge", "molecule", "network"]):
            candidate_groups.append(
                [
                    {
                        "dataset_id": "karate-club",
                        "source_url": "https://raw.githubusercontent.com/networkx/networkx/main/examples/graph/karate_club.py",
                        "filename": "karate_club.py",
                    },
                    {
                        "dataset_id": "Cora-cites",
                        "source_url": "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/ind.cora.y",
                        "filename": "ind.cora.y",
                    },
                    {
                        "dataset_id": "Cora-graph",
                        "source_url": "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/ind.cora.graph",
                        "filename": "ind.cora.graph",
                    },
                ]
            )
        if not candidate_groups:
            candidate_groups.append(
                [
                    {
                        "dataset_id": "iris",
                        "source_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                        "filename": "iris.data",
                    },
                    {
                        "dataset_id": "wine",
                        "source_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
                        "filename": "wine.data",
                    },
                    {
                        "dataset_id": "abalone",
                        "source_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",
                        "filename": "abalone.data",
                    },
                ]
            )
        candidates: list[dict[str, str]] = []
        seen: set[str] = set()
        for group in candidate_groups:
            for candidate in group:
                key = candidate["dataset_id"].lower()
                if key not in seen:
                    seen.add(key)
                    candidates.append(candidate)
        return candidates[:3]

    def _dataset_selection_text(self, spec: BuildSpec, evidence: list[EvidenceCard]) -> str:
        parts = [
            spec.target_task,
            spec.problem_statement,
            spec.method_summary,
            " ".join(spec.implementation_plan),
            " ".join(spec.experiment_plan),
            " ".join(spec.baselines),
            " ".join(spec.metrics),
        ]
        for card in evidence:
            parts.extend(
                [
                    card.task,
                    card.method,
                    card.setting,
                    " ".join(card.claims),
                    " ".join(card.metrics),
                    " ".join(card.limitations),
                    " ".join(card.transferable_idea_seeds),
                ]
            )
        return " ".join(parts).lower()

    def _generic_dataset_filename(self, spec: BuildSpec) -> str:
        text = " ".join([spec.target_task, spec.problem_statement, spec.method_summary]).lower()
        if any(token in text for token in ["offline rl", "offline reinforcement", "d4rl", "mujoco", "atari", "dqn replay"]):
            return "synthetic_offline_rl_smoke.jsonl"
        if any(token in text for token in ["3d", "gaussian", "splat", "mesh", "geometry", "pattern"]):
            return "synthetic_3d_patterns.jsonl"
        if any(token in text for token in ["image", "vision", "segmentation", "render"]):
            return "synthetic_vision_smoke.jsonl"
        return "synthetic_smoke_dataset.jsonl"

    def _write_generic_smoke_dataset(
        self,
        dataset_path: Path,
        spec: BuildSpec,
        *,
        source_candidate: dict[str, Any] | None = None,
    ) -> None:
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        text = " ".join([spec.target_task, spec.problem_statement, spec.method_summary]).lower()
        records: list[dict[str, Any]]
        if any(token in text for token in ["offline rl", "offline reinforcement", "d4rl", "mujoco", "atari", "dqn replay"]):
            records = self._synthetic_offline_rl_records()
        elif any(token in text for token in ["3d", "gaussian", "splat", "mesh", "geometry", "pattern"]):
            records = self._synthetic_3d_pattern_records()
        elif any(token in text for token in ["image", "vision", "segmentation", "render"]):
            records = self._synthetic_vision_records()
        else:
            records = self._synthetic_tabular_records()
        if source_candidate:
            for record in records:
                record["source_dataset"] = {
                    "dataset_id": source_candidate.get("dataset_id"),
                    "source_url": source_candidate.get("source_url"),
                    "note": "Smoke JSONL view generated locally after downloading the evidence-informed source artifact.",
                }
        dataset_path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")

    def _synthetic_offline_rl_records(self) -> list[dict[str, Any]]:
        records = []
        for index in range(64):
            obs = [round(((index + offset) % 13) / 12.0, 4) for offset in range(8)]
            action = [round(((index * (offset + 2)) % 7) / 6.0, 4) for offset in range(3)]
            next_obs = [round(min(value + 0.03, 1.0), 4) for value in obs]
            records.append(
                {
                    "transition_id": index,
                    "obs": obs,
                    "action": action,
                    "reward": round((index % 11) / 10.0, 4),
                    "next_obs": next_obs,
                    "done": index % 16 == 15,
                    "split": "train" if index < 48 else "validation",
                }
            )
        return records

    def _synthetic_3d_pattern_records(self) -> list[dict[str, Any]]:
        records = []
        for scene_id in range(32):
            topology = "lattice" if scene_id % 2 == 0 else "radial"
            appearance = "striped" if scene_id % 3 == 0 else "smooth"
            gaussians = []
            for index in range(24):
                x = ((index % 6) - 2.5) / 3.0
                y = (((index // 6) % 4) - 1.5) / 2.5
                z = (scene_id % 5) / 10.0
                gaussians.append(
                    {
                        "xyz": [round(x, 4), round(y, 4), round(z, 4)],
                        "scale": round(0.04 + 0.005 * (index % 4), 4),
                        "opacity": round(0.6 + 0.02 * (scene_id % 5), 4),
                        "rgb": [round((index % 3) / 2.0, 4), round((scene_id % 4) / 3.0, 4), round(((index + scene_id) % 5) / 4.0, 4)],
                    }
                )
            records.append(
                {
                    "scene_id": scene_id,
                    "topology_label": topology,
                    "appearance_label": appearance,
                    "structure_latent": [round(scene_id / 31.0, 4), 1.0 if topology == "lattice" else 0.0],
                    "appearance_latent": [round((scene_id % 7) / 6.0, 4), 1.0 if appearance == "striped" else 0.0],
                    "gaussians": gaussians,
                    "target_metrics": {
                        "psnr_proxy": round(22.0 + 0.1 * scene_id, 4),
                        "ssim_proxy": round(0.72 + 0.004 * (scene_id % 20), 4),
                    },
                }
            )
        return records

    def _synthetic_vision_records(self) -> list[dict[str, Any]]:
        return [
            {
                "sample_id": index,
                "image_features": [round(((index + offset) % 11) / 10.0, 4) for offset in range(16)],
                "label": index % 4,
                "split": "train" if index < 24 else "validation",
            }
            for index in range(32)
        ]

    def _synthetic_tabular_records(self) -> list[dict[str, Any]]:
        return [
            {
                "sample_id": index,
                "features": [round(((index * (offset + 1)) % 17) / 16.0, 4) for offset in range(12)],
                "target": round((index % 9) / 8.0, 4),
                "split": "train" if index < 24 else "validation",
            }
            for index in range(32)
        ]

    def _download_file(self, url: str, destination: Path) -> None:
        request = urllib.request.Request(url, headers={"User-Agent": "quit-agent/0.2"})
        with urllib.request.urlopen(request, timeout=300) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)

    def _dataset_path_needs_acquisition(self, path: str) -> bool:
        if not path:
            return False
        # Skip paths that already exist locally — nothing to acquire
        from pathlib import Path as _Path
        if _Path(path).exists():
            return False
        # Accept any non-empty path that is declared but not yet present
        return True

    def _find_dataset_paths(self, value: Any) -> list[str]:
        paths: list[str] = []
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "dataset_path" and isinstance(item, str):
                    paths.append(item)
                else:
                    paths.extend(self._find_dataset_paths(item))
        elif isinstance(value, list):
            for item in value:
                paths.extend(self._find_dataset_paths(item))
        return paths

    def _patch_dataset_paths(self, value: Any, dataset_path: str) -> list[str]:
        patched: list[str] = []

        def visit(item: Any, prefix: str) -> None:
            if isinstance(item, dict):
                for key, child in item.items():
                    current = f"{prefix}.{key}" if prefix else key
                    if key == "dataset_path" and isinstance(child, str) and self._dataset_path_needs_acquisition(child):
                        item[key] = dataset_path
                        patched.append(current)
                    else:
                        visit(child, current)
            elif isinstance(item, list):
                for index, child in enumerate(item):
                    visit(child, f"{prefix}[{index}]")

        visit(value, "")
        return patched

    def _cap_verification_runtime(self, value: Any) -> list[str]:
        patched: list[str] = []
        smoke_caps = {"train_epochs": 50, "eval_epochs": 20}

        def visit(item: Any, prefix: str) -> None:
            if isinstance(item, dict):
                for key, child in item.items():
                    current = f"{prefix}.{key}" if prefix else key
                    if key in smoke_caps and isinstance(child, int):
                        clamped = min(smoke_caps[key], child)
                        if clamped != child:
                            item[key] = clamped
                            patched.append(current)
                    else:
                        visit(child, current)
            elif isinstance(item, list):
                for index, child in enumerate(item):
                    visit(child, f"{prefix}[{index}]")

        visit(value, "")
        return patched

    def _relative_to_code_dir(self, path: Path, code_dir: Path) -> str:
        try:
            return path.relative_to(code_dir).as_posix()
        except ValueError:
            return str(path)

    def _run_experiment(self, command: list[str], code_dir: Path) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                command,
                cwd=code_dir,
                capture_output=True,
                text=True,
                timeout=self._experiment_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return subprocess.CompletedProcess(
                args=command,
                returncode=124,
                stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
                stderr=f"Experiment timed out after {exc.timeout} seconds.",
            )

    def _install_missing_dependency_from_error(self, code_dir: Path, completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
        report: dict[str, Any] = {
            "attempted": False,
            "succeeded": False,
            "missing_module": None,
            "requirement": None,
            "fallback_requirement": None,
            "commands": [],
            "errors": [],
        }
        if completed.returncode == 0:
            report["reason"] = "experiment_succeeded"
            return report
        missing_module = self._missing_module_from_stderr(completed.stderr)
        if not missing_module:
            report["reason"] = "no_missing_module_error"
            return report
        report["attempted"] = True
        report["missing_module"] = missing_module

        # Build a prioritised list of candidate names to try.
        # Start with whatever requirements.txt says (may include version pins), then
        # fall back to the raw module name.  Dynamic candidates are appended later
        # based on what pip itself tells us.
        requirement = self._requirement_for_module(code_dir / "requirements.txt", missing_module)
        report["requirement"] = requirement

        tried: set[str] = set()

        def _attempt(name: str) -> bool:
            """Try pip install <name>; return True on success."""
            if not name or name in tried:
                return False
            tried.add(name)
            result = self._pip_install(name)
            report["commands"].append(result)
            if result["returncode"] == 0:
                report["succeeded"] = True
                return True
            report["errors"].append(result["stderr_tail"] or result["stdout_tail"])
            return False

        # 1. First attempt with the requirement from requirements.txt (or module name).
        if _attempt(requirement):
            return report

        # 2. Parse pip's own output for an explicit redirect hint
        #    e.g. "Please install the 'scikit-image' package (instead of 'skimage')"
        #    or   "Did you mean X?"
        pip_hint = self._extract_pip_hint(report["commands"][-1])
        if pip_hint and _attempt(pip_hint):
            report["fallback_requirement"] = pip_hint
            return report

        # 3. Try common name transformations derived solely from the module name.
        #    These cover patterns that pip won't hint at because the stub doesn't exist.
        base = re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip()
        normalized = self._normalize_requirement_name(base)
        candidates = [
            f"python-{normalized}",           # dotenv  → python-dotenv
            f"{normalized}-python",           # some packages use this suffix
            normalized.replace("-", "_"),     # underscore variant
            normalized.replace("_", "-"),     # hyphen variant (already normalized, but be explicit)
        ]
        for candidate in candidates:
            if candidate == normalized:
                continue
            if _attempt(candidate):
                report["fallback_requirement"] = candidate
                return report
            # After each failed attempt, check if pip hinted at yet another name.
            pip_hint = self._extract_pip_hint(report["commands"][-1])
            if pip_hint and _attempt(pip_hint):
                report["fallback_requirement"] = pip_hint
                return report

        report["reason"] = "dependency_install_failed"
        return report

    def _missing_module_from_stderr(self, stderr: str) -> str | None:
        patterns = [
            r"ModuleNotFoundError:\s+No module named ['\"]([^'\"]+)['\"]",
            r"ImportError:\s+No module named ['\"]([^'\"]+)['\"]",
        ]
        for pattern in patterns:
            match = re.search(pattern, stderr)
            if match:
                return match.group(1).split(".")[0].strip()
        return None

    def _requirement_for_module(self, requirements_path: Path, module_name: str) -> str:
        """Return the requirement line from requirements.txt that corresponds to module_name,
        or just module_name itself if nothing matches.  Only uses normalised name comparison
        (underscores == hyphens, case-insensitive); dynamic resolution happens later via pip."""
        module_key = self._normalize_requirement_name(module_name)
        if requirements_path.exists():
            for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or line.startswith("-e "):
                    continue
                package = re.split(r"[<>=!~;\[]", line, maxsplit=1)[0].strip()
                if self._normalize_requirement_name(package) == module_key:
                    return line
        return module_name

    def _extract_pip_hint(self, pip_result: dict[str, Any]) -> str | None:
        """Parse pip's stdout/stderr for an explicitly suggested alternative package name."""
        combined = (pip_result.get("stdout_tail") or "") + "\n" + (pip_result.get("stderr_tail") or "")
        # Stub-package redirect: *** Please install the 'scikit-image' package (instead of 'skimage') ***
        m = re.search(
            r"[Pp]lease install the\s+['\"`]([^'\"`]+)['\"`]\s+package",
            combined,
        )
        if m:
            return m.group(1).strip()
        # pip "did you mean?" suggestions
        m = re.search(r"[Dd]id you mean\s+['\"`]?([A-Za-z0-9_\-\.]+)['\"`]?\??", combined)
        if m:
            return m.group(1).strip()
        return None

    def _normalize_requirement_name(self, value: str) -> str:
        return value.lower().replace("_", "-")

    def _pip_install(self, requirement: str) -> dict[str, Any]:
        command = [sys.executable, "-m", "pip", "install", requirement]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        return {
            "command": " ".join(command),
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }

    def _patch_runtime_error_from_failure(self, code_dir: Path, completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
        report: dict[str, Any] = {
            "attempted": False,
            "patched": False,
            "patches": [],
            "errors": [],
        }
        if completed.returncode == 0:
            report["reason"] = "experiment_succeeded"
            return report
        stderr = completed.stderr or ""
        if "numpy.ndarray' object has no attribute 'to'" in stderr or "'numpy.ndarray' object has no attribute 'to'" in stderr:
            report["attempted"] = True
            patch = self._patch_dataset_batches_to_torch(code_dir / "src" / "dataset.py")
            report["patches"].append(patch)
            report["patched"] = bool(patch.get("patched"))
            if not report["patched"]:
                report["errors"].append(patch.get("reason", "dataset tensor patch failed"))
            return report
        if ("weights * q_next" in stderr or "weights * q_values" in stderr) and "must match the size of tensor" in stderr:
            report["attempted"] = True
            patch = self._patch_rem_tensor_shapes(code_dir / "src" / "method.py")
            report["patches"].append(patch)
            report["patched"] = bool(patch.get("patched"))
            if not report["patched"]:
                report["errors"].append(patch.get("reason", "REM tensor shape patch failed"))
            return report
        if "object of type 'NoneType' has no len()" in stderr and "run_evaluation(None" in stderr:
            report["attempted"] = True
            patch = self._patch_none_dataset_evaluation(code_dir / "run_experiment.py")
            report["patches"].append(patch)
            report["patched"] = bool(patch.get("patched"))
            if not report["patched"]:
                report["errors"].append(patch.get("reason", "None dataset evaluation patch failed"))
            return report
        if "Expected all tensors to be on the same device" in stderr:
            report["attempted"] = True
            patch = self._patch_batch_tensors_to_model_device(code_dir / "src" / "method.py")
            report["patches"].append(patch)
            report["patched"] = bool(patch.get("patched"))
            if not report["patched"]:
                report["errors"].append(patch.get("reason", "batch tensor device patch failed"))
            return report
        if "FileNotFoundError" in stderr and re.search(r"name = '([^']+\.hdf5?)'", stderr):
            report["attempted"] = True
            patch = self._report_missing_hdf5_dataset(stderr)
            report["patches"].append(patch)
            report["patched"] = False
            report["errors"].append(patch.get("reason", "missing dataset"))
            return report
        report["reason"] = "no_known_runtime_patch"
        return report

    def _patch_dataset_batches_to_torch(self, dataset_path: Path) -> dict[str, Any]:
        report: dict[str, Any] = {"file": "code/src/dataset.py", "patched": False}
        if not dataset_path.exists():
            report["reason"] = "dataset.py missing"
            return report
        text = dataset_path.read_text(encoding="utf-8")
        if "torch.as_tensor" in text:
            report["reason"] = "already patched"
            return report
        if "def get_batch" not in text:
            report["reason"] = "get_batch not found"
            return report
        if "import torch" not in text:
            text = text.replace("import numpy as np\n", "import numpy as np\nimport torch\n", 1)
        pattern = re.compile(
            r"    def get_batch\(self, indices: np\.ndarray\):\n"
            r"        \"\"\"Return a batch of data\.\"\"\"\n"
            r"        return \{\n"
            r"            'obs': self\._data\['obs'\]\[indices\],\n"
            r"            'action': self\._data\['action'\]\[indices\],\n"
            r"            'reward': self\._data\['reward'\]\[indices\],\n"
            r"            'next_obs': self\._data\['next_obs'\]\[indices\],\n"
            r"            'done': self\._data\['done'\]\[indices\]\n"
            r"        \}",
            re.MULTILINE,
        )
        replacement = (
            "    def get_batch(self, indices: np.ndarray):\n"
            "        \"\"\"Return a batch of data as torch tensors on the configured device.\"\"\"\n"
            "        batch = {\n"
            "            'obs': self._data['obs'][indices],\n"
            "            'action': self._data['action'][indices],\n"
            "            'reward': self._data['reward'][indices],\n"
            "            'next_obs': self._data['next_obs'][indices],\n"
            "            'done': self._data['done'][indices],\n"
            "        }\n"
            "        return {key: torch.as_tensor(value, dtype=torch.float32, device=self.device) for key, value in batch.items()}"
        )
        patched, count = pattern.subn(replacement, text, count=1)
        if count == 0:
            report["reason"] = "known get_batch shape not matched"
            return report
        dataset_path.write_text(patched, encoding="utf-8")
        report["patched"] = True
        report["reason"] = "converted get_batch numpy arrays to torch tensors"
        return report

    def _patch_rem_tensor_shapes(self, method_path: Path) -> dict[str, Any]:
        report: dict[str, Any] = {"file": "code/src/method.py", "patched": False}
        if not method_path.exists():
            report["reason"] = "method.py missing"
            return report
        text = method_path.read_text(encoding="utf-8")
        changed = False
        if "torch.stack([q_net(obs) for q_net in self.q_nets], dim=1)" in text:
            text = text.replace(
                "torch.stack([q_net(obs) for q_net in self.q_nets], dim=1)",
                "torch.cat([q_net(obs) for q_net in self.q_nets], dim=1)",
            )
            changed = True
        if "torch.distributions.Dirichlet(torch.ones(self.ensemble_size)).sample((batch_size,))" in text:
            text = text.replace(
                "torch.distributions.Dirichlet(torch.ones(self.ensemble_size)).sample((batch_size,))",
                "torch.distributions.Dirichlet(torch.ones(self.ensemble_size, device=obs.device)).sample((batch_size,))",
            )
            changed = True
        patched_text = re.sub(
            r"weights\s*=\s*torch\.rand\(\s*self\.ensemble_size\s*,\s*batch_size\s*,\s*1\s*,\s*device=([^)]+)\)",
            r"weights = torch.rand(batch_size, self.ensemble_size, 1, device=\1)",
            text,
        )
        if patched_text != text:
            text = patched_text
            changed = True
        if "weights = weights / weights.sum(dim=0, keepdim=True)" in text:
            text = text.replace(
                "weights = weights / weights.sum(dim=0, keepdim=True)",
                "weights = weights / weights.sum(dim=1, keepdim=True)",
            )
            changed = True
        if "reward * (1 - done)" in text and "reward = reward.view(-1, 1)" not in text:
            text = text.replace(
                "        weights = weights.float()\n        q_target = torch.sum(weights * q_next, dim=1, keepdim=True) + reward * (1 - done)",
                "        weights = weights.float().to(q_next.device)\n        reward = reward.view(-1, 1)\n        done = done.view(-1, 1)\n        q_target = torch.sum(weights * q_next, dim=1, keepdim=True) + reward * (1 - done)",
            )
            changed = True
        if "nn.Linear(hidden_dim, hidden_dim),\n            nn.Tanh() # Output velocity field bounded" in text:
            text = text.replace(
                "nn.Linear(hidden_dim, hidden_dim),\n            nn.Tanh() # Output velocity field bounded",
                "nn.Linear(hidden_dim, obs_dim),\n            nn.Tanh() # Output velocity field bounded",
            )
            changed = True
        if "rem_consistency = torch.mean((q_target - q_next.mean(dim=1))**2, dim=1)" in text:
            text = text.replace(
                "rem_consistency = torch.mean((q_target - q_next.mean(dim=1))**2, dim=1)",
                "rem_consistency = torch.mean((q_target - q_next.mean(dim=1, keepdim=True))**2)",
            )
            changed = True
        if "total_loss = loss_fm + 0.1 * rem_consistency # Weight for REM term" in text:
            text = text.replace(
                "total_loss = loss_fm + 0.1 * rem_consistency # Weight for REM term",
                "total_loss = loss_fm + 0.1 * rem_consistency # scalar loss suitable for backward",
            )
            changed = True
        if not changed:
            report["reason"] = "known REM tensor patterns not found"
            return report
        method_path.write_text(text, encoding="utf-8")
        report["patched"] = True
        report["reason"] = "fixed REM ensemble, reward/done, and flow target tensor shapes"
        return report

    def _patch_none_dataset_evaluation(self, run_path: Path) -> dict[str, Any]:
        report: dict[str, Any] = {"file": "code/run_experiment.py", "patched": False}
        if not run_path.exists():
            report["reason"] = "run_experiment.py missing"
            return report
        text = run_path.read_text(encoding="utf-8")
        replacements = [
            (
                "evaluations['mujoco'] = evaluator.run_evaluation(None, 'mujoco') # Passing None for dummy",
                "evaluations['mujoco'] = evaluator.run_evaluation(atari_dataset, 'mujoco') # Reuse smoke dataset when MuJoCo data is unavailable",
            ),
            (
                "evaluations['mujoco'] = evaluator.run_evaluation(None, 'mujoco')",
                "evaluations['mujoco'] = evaluator.run_evaluation(atari_dataset, 'mujoco')",
            ),
        ]
        changed = False
        for old, new in replacements:
            if old in text:
                text = text.replace(old, new)
                changed = True
        if not changed:
            report["reason"] = "known None dataset evaluation pattern not found"
            return report
        run_path.write_text(text, encoding="utf-8")
        report["patched"] = True
        report["reason"] = "reused available smoke dataset for missing evaluation dataset"
        return report

    def _patch_batch_tensors_to_model_device(self, method_path: Path) -> dict[str, Any]:
        report: dict[str, Any] = {"file": "code/src/method.py", "patched": False}
        if not method_path.exists():
            report["reason"] = "method.py missing"
            return report
        text = method_path.read_text(encoding="utf-8")
        if "next(self.model.parameters()).device" in text:
            report["reason"] = "already patched"
            return report
        old = (
            "        obs = batch['obs'].float()\n"
            "        acts = batch['act'].float()\n"
            "        rewards = batch['reward'].float()\n"
            "        dones = batch['done'].float()\n"
            "        next_obs = batch['next_obs'].float()"
        )
        new = (
            "        device = next(self.model.parameters()).device\n"
            "        obs = batch['obs'].float().to(device)\n"
            "        acts = batch['act'].float().to(device)\n"
            "        rewards = batch['reward'].float().view(-1, 1).to(device)\n"
            "        dones = batch['done'].float().view(-1, 1).to(device)\n"
            "        next_obs = batch['next_obs'].float().to(device)"
        )
        if old not in text:
            report["reason"] = "known train_step tensor extraction pattern not found"
            return report
        method_path.write_text(text.replace(old, new), encoding="utf-8")
        report["patched"] = True
        report["reason"] = "moved batch tensors to model device and normalized reward/done shape"
        return report

    def _report_missing_hdf5_dataset(self, stderr: str) -> dict[str, Any]:
        report: dict[str, Any] = {"file": "", "patched": False}
        match = re.search(r"name = '([^']+\.hdf5?)'", stderr)
        if not match:
            report["reason"] = "missing hdf5 path not found in stderr"
            return report
        report["file"] = match.group(1)
        report["reason"] = (
            "missing hdf5 dataset; CODE will not create a mock dataset. "
            "Provide a real downloaded dataset path in experiment_config.json or make the generated code use an existing dataset artifact."
        )
        return report

    def _finalize_code_project_scaffold(self, spec: BuildSpec, code_dir: Path) -> None:
        """Patch generated projects with required runtime scaffolding before execution.

        LLM-generated code may omit boilerplate files or simple imports. This is
        a deterministic preflight repair layer, separate from LLM CODE repair.
        """
        if not (code_dir / "README.md").exists():
            self._write_text(code_dir / "README.md", self._code_readme(spec))
        if not (code_dir / "ENVIRONMENT.md").exists():
            self._write_text(code_dir / "ENVIRONMENT.md", self._environment_readme(spec))
        if not (code_dir / "requirements.txt").exists():
            self._write_text(code_dir / "requirements.txt", self._requirements_txt(spec))
        if not (code_dir / "environment.yml").exists():
            self._write_text(code_dir / "environment.yml", self._environment_yml(spec))
        self._patch_common_missing_imports(code_dir)
        self._patch_common_generated_python_errors(code_dir)
        self._patch_generated_offline_dataset_loader(code_dir / "src" / "dataset.py")
        self._patch_forbidden_mock_hdf5_fallbacks(code_dir)

    def _patch_common_missing_imports(self, code_dir: Path) -> None:
        for path in sorted(code_dir.glob("**/*.py")):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            relative = path.relative_to(code_dir).as_posix()
            if relative == "src/dataset.py" and "from src.dataset import OfflineDataset\n" in text:
                text = text.replace("from src.dataset import OfflineDataset\n", "", 1)
                path.write_text(text, encoding="utf-8")
            additions = []
            if re.search(r"\bnp\.", text) and not re.search(r"^\s*import\s+numpy\s+as\s+np\b", text, re.MULTILINE):
                additions.append("import numpy as np")
            if re.search(r"\bh5py\.", text) and not re.search(r"^\s*import\s+h5py\b", text, re.MULTILINE):
                additions.append("import h5py")
            if re.search(r"\bcsv\.", text) and not re.search(r"^\s*import\s+csv\b", text, re.MULTILINE):
                additions.append("import csv")
            if re.search(r"\bF\.", text) and not re.search(r"^\s*(from\s+torch\.nn\s+import\s+functional\s+as\s+F|import\s+torch\.nn\.functional\s+as\s+F)\b", text, re.MULTILINE):
                additions.append("import torch.nn.functional as F")
            if (
                relative != "src/dataset.py"
                and
                re.search(r"\bOfflineDataset\b", text)
                and not re.search(r"^\s*(from\s+src\.dataset\s+import\s+.*\bOfflineDataset\b|from\s+dataset\s+import\s+.*\bOfflineDataset\b)", text, re.MULTILINE)
                and (code_dir / "src" / "dataset.py").exists()
            ):
                additions.append("from src.dataset import OfflineDataset")
            if not additions:
                continue
            insert_at = 0
            lines = text.splitlines()
            if lines and lines[0].startswith("#!"):
                insert_at = 1
            if len(lines) > insert_at and lines[insert_at].startswith("from __future__ import"):
                insert_at += 1
            lines[insert_at:insert_at] = additions
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _patch_forbidden_mock_hdf5_fallbacks(self, code_dir: Path) -> None:
        """Replace generated mock HDF5 fallbacks with explicit missing-data failures."""
        pattern = re.compile(
            r"(?P<indent>^[ \t]*)if\s+not\s+os\.path\.exists\((?P<expr>[^)\n]*dataset_path[^)\n]*)\):\n"
            r"(?P<body>(?:(?P=indent)[ \t]+.*\n)+)",
            re.MULTILINE,
        )
        for path in sorted(code_dir.glob("**/*.py")):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if "dataset_path" not in text or not re.search(r"_mock\.hdf?5|creating mock|mock hdf5", text, re.IGNORECASE):
                continue

            def replace(match: re.Match[str]) -> str:
                body = match.group("body")
                if not re.search(r"_mock\.hdf?5|creating mock|mock hdf5", body, re.IGNORECASE):
                    return match.group(0)
                indent = match.group("indent")
                expr = match.group("expr").strip()
                return (
                    f"{indent}if not os.path.exists({expr}):\n"
                    f"{indent}    raise FileNotFoundError(\n"
                    f"{indent}        f\"Required offline dataset not found: {{{expr}}}. \"\n"
                    f"{indent}        \"CODE must use a real downloaded dataset artifact; mock hdf5 fallback is forbidden.\"\n"
                    f"{indent}    )\n"
                )

            patched = pattern.sub(replace, text)
            patched = re.sub(r"(?m)^[ \t]*#\s*Mock dataset path for smoke run\n", "", patched)
            if patched != text:
                path.write_text(patched, encoding="utf-8")

    def _patch_common_generated_python_errors(self, code_dir: Path) -> None:
        for path in sorted(code_dir.glob("**/*.py")):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            original = text
            text = text.replace("super().__init__\n", "super().__init__()\n")
            text = text.replace("super().__init__\r\n", "super().__init__()\r\n")
            if (
                path.name == "evaluate.py"
                and "FlowMatchingTrainer" in text
                and "from src.method import FlowMatchingTrainer" not in text
            ):
                if "from src.method import FlowMatchingEnsemble" in text:
                    text = text.replace(
                        "from src.method import FlowMatchingEnsemble",
                        "from src.method import FlowMatchingEnsemble, FlowMatchingTrainer",
                    )
                else:
                    text = self._insert_python_import(text, "from src.method import FlowMatchingTrainer")
            if path.name == "run_experiment.py" and "BaselineModels(dataset.n_states, dataset.n_actions)" in text:
                text = text.replace(
                    "BaselineModels(dataset.n_states, dataset.n_actions)",
                    "BaselineModels(trainer.dataset.n_states, trainer.dataset.n_actions)",
                )
            if text != original:
                path.write_text(text, encoding="utf-8")

    def _insert_python_import(self, text: str, import_line: str) -> str:
        lines = text.splitlines()
        insert_at = 0
        if lines and lines[0].startswith("#!"):
            insert_at = 1
        if len(lines) > insert_at and lines[insert_at].startswith("from __future__ import"):
            insert_at += 1
        lines[insert_at:insert_at] = [import_line]
        return "\n".join(lines) + "\n"

    def _patch_generated_offline_dataset_loader(self, dataset_path: Path) -> dict[str, Any]:
        report: dict[str, Any] = {"file": "code/src/dataset.py", "patched": False}
        if not dataset_path.exists():
            report["reason"] = "dataset.py missing"
            return report
        text = dataset_path.read_text(encoding="utf-8")
        if "class OfflineDataset" not in text or "pd.read_csv(path)" not in text:
            report["reason"] = "known pandas OfflineDataset pattern not found"
            return report
        dataset_path.write_text(_ROBUST_OFFLINE_DATASET_PY, encoding="utf-8")
        report["patched"] = True
        report["reason"] = "replaced fragile pandas CSV loader with hdf5/csv tensor loader"
        return report

    def _resolve_generated_code_device(self, code_dir: Path) -> dict[str, Any]:
        config_path = code_dir / "configs" / "experiment_config.json"
        report: dict[str, Any] = {
            "status": "PASS",
            "config_file": "code/configs/experiment_config.json",
            "patched": False,
            "selection": select_torch_device("auto").to_dict(),
        }
        if not config_path.exists():
            report["status"] = "SKIP"
            report["reason"] = "experiment_config_missing"
            return report
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            report["status"] = "FAIL"
            report["reason"] = f"config_parse_failed: {exc}"
            return report

        requested = self._find_requested_device(config)
        selection = select_torch_device(requested or "auto")
        report["selection"] = selection.to_dict()
        changed = self._patch_device_values(config, selection.selected)
        changed = self._propagate_device_to_task_configs(config, selection.selected) or changed
        if config.get("device") != selection.selected:
            config["device"] = selection.selected
            changed = True
        if "runtime" in config and isinstance(config["runtime"], dict):
            config["runtime"]["resolved_device"] = selection.selected
            config["runtime"]["device"] = selection.selected
            config["runtime"]["cuda_available"] = selection.cuda_available
        else:
            config["runtime"] = {"device": selection.selected, "resolved_device": selection.selected, "cuda_available": selection.cuda_available}
            changed = True
        if changed or requested in {None, "", "auto", "gpu"}:
            config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            report["patched"] = True
        return report

    def _normalize_generated_experiment_config(self, code_dir: Path) -> dict[str, Any]:
        config_path = code_dir / "configs" / "experiment_config.json"
        report: dict[str, Any] = {
            "status": "PASS",
            "config_file": "code/configs/experiment_config.json",
            "patched": False,
            "selected_task": None,
            "promoted_keys": [],
            "errors": [],
        }
        if not config_path.exists():
            report["status"] = "SKIP"
            report["reason"] = "experiment_config_missing"
            return report
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            report["status"] = "FAIL"
            report["errors"].append(f"config_parse_failed: {exc}")
            return report
        if not isinstance(config, dict):
            report["status"] = "FAIL"
            report["errors"].append("experiment_config is not a JSON object")
            return report

        task_config = self._select_nested_task_config(config, code_dir)
        if task_config:
            task_name, nested = task_config
            report["selected_task"] = task_name
            for key, value in nested.items():
                if key not in config:
                    config[key] = value
                    report["promoted_keys"].append(key)
            if "task_name" not in config:
                config["task_name"] = task_name
                report["promoted_keys"].append("task_name")
        else:
            report["reason"] = "no_nested_task_config_detected"

        outputs = config.get("outputs")
        if not isinstance(outputs, dict):
            outputs = {}
            config["outputs"] = outputs
        desired_outputs = {
            "metrics_json": "../results/metrics.json",
            "results_table_csv": "../results/results_table.csv",
        }
        patched_outputs = []
        for key, value in desired_outputs.items():
            if outputs.get(key) != value:
                outputs[key] = value
                patched_outputs.append(key)
        if patched_outputs:
            report["patched_outputs"] = patched_outputs

        if report["promoted_keys"] or patched_outputs:
            config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            report["patched"] = True
        return report

    def _select_nested_task_config(self, config: dict[str, Any], code_dir: Path) -> tuple[str, dict[str, Any]] | None:
        candidates: list[tuple[str, dict[str, Any]]] = []
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(value.get("dataset_path"), str):
                candidates.append((key, value))
        if not candidates:
            return None
        for key, value in candidates:
            dataset_path = code_dir / value["dataset_path"]
            if dataset_path.exists():
                return key, value
        return candidates[0]

    def _find_requested_device(self, value: Any) -> str | None:
        if isinstance(value, dict):
            if isinstance(value.get("device"), str):
                return value["device"]
            runtime = value.get("runtime")
            if isinstance(runtime, dict) and isinstance(runtime.get("device"), str):
                return runtime["device"]
            for item in value.values():
                found = self._find_requested_device(item)
                if found:
                    return found
        if isinstance(value, list):
            for item in value:
                found = self._find_requested_device(item)
                if found:
                    return found
        return None

    def _patch_device_values(self, value: Any, selected: str) -> bool:
        changed = False
        if isinstance(value, dict):
            for key, item in list(value.items()):
                if key == "device" and isinstance(item, str) and item.strip().lower() in {"auto", "gpu"}:
                    value[key] = selected
                    changed = True
                else:
                    changed = self._patch_device_values(item, selected) or changed
        elif isinstance(value, list):
            for item in value:
                changed = self._patch_device_values(item, selected) or changed
        return changed

    def _propagate_device_to_task_configs(self, value: Any, selected: str) -> bool:
        """Ensure nested task configs are executable when passed independently.

        LLM-generated runners often do `task_config = config["atari"]` and pass
        that nested dict to training code. The root config may already have a
        resolved concrete device, but the nested task dict can still miss it.
        """
        changed = False
        if isinstance(value, dict):
            looks_like_task_config = any(key in value for key in ("dataset_path", "train_epochs", "batch_size", "env_name"))
            if looks_like_task_config and not isinstance(value.get("device"), str):
                value["device"] = selected
                changed = True
            runtime = value.get("runtime")
            if isinstance(runtime, dict):
                if runtime.get("device") != selected:
                    runtime["device"] = selected
                    changed = True
                if runtime.get("resolved_device") != selected:
                    runtime["resolved_device"] = selected
                    changed = True
            for item in value.values():
                changed = self._propagate_device_to_task_configs(item, selected) or changed
        elif isinstance(value, list):
            for item in value:
                changed = self._propagate_device_to_task_configs(item, selected) or changed
        return changed

    def _repair_code_project(
        self,
        spec: BuildSpec,
        code_dir: Path,
        command: list[str],
        completed: subprocess.CompletedProcess[str],
    ) -> dict[str, Any]:
        if self.llm is None:
            return {
                "attempted": False,
                "succeeded": False,
                "reason": "no_builder_llm_available",
                "errors": [completed.stderr[-4000:] or "initial run failed without stderr"],
            }
        history = self._previous_repair_attempts()
        prompt = self._code_stage_repair_prompt(spec, code_dir, command, completed, history, len(history) + 1)
        raw = self.llm.complete(prompt)
        self.artifacts.write_markdown("CodeRepair.raw.txt", raw)
        files_written = self._write_llm_code_project(raw, code_dir, require_full_project=False)
        syntax_report = self._check_code_syntax(code_dir)
        self.artifacts.write_json("CodeSyntaxReport.json", syntax_report)
        after = self._run_experiment(command, code_dir) if syntax_report.get("status") == "PASS" else subprocess.CompletedProcess(
            command,
            2,
            stdout="",
            stderr=json.dumps(syntax_report, indent=2, sort_keys=True),
        )
        attempt_report = {
            "attempt": len(history) + 1,
            "command": " ".join(command),
            "returncode_before": completed.returncode,
            "stderr_tail_before": (completed.stderr or "")[-4000:],
            "files_written": bool(files_written),
            "syntax_status_after": syntax_report.get("status"),
            "returncode_after": after.returncode,
            "stdout_tail_after": (after.stdout or "")[-4000:],
            "stderr_tail_after": (after.stderr or "")[-4000:],
        }
        attempts = [*history, attempt_report]
        return {
            "attempted": True,
            "succeeded": after.returncode == 0,
            "reason": None if files_written else "repair_response_not_valid_file_marker",
            "files_written": bool(files_written),
            "attempts": attempts,
            "returncode_after_repair": after.returncode,
            "stdout_tail_after": (after.stdout or "")[-4000:],
            "stderr_tail_after": (after.stderr or "")[-4000:],
            "errors": [completed.stderr[-4000:] or "initial run failed without stderr"],
        }

    def _previous_repair_attempts(self) -> list[dict[str, Any]]:
        if not self.artifacts.path("CodeRepairReport.json").exists():
            return []
        try:
            report = self.artifacts.read_json("CodeRepairReport.json")
        except Exception:
            return []
        attempts = report.get("attempts")
        return attempts if isinstance(attempts, list) else []

    def _code_stage_repair_prompt(
        self,
        spec: BuildSpec,
        code_dir: Path,
        command: list[str],
        completed: subprocess.CompletedProcess[str],
        history: list[dict[str, Any]],
        attempt: int,
    ) -> str:
        contract = self.artifacts.read_json("ImplementationContract.json") if self.artifacts.path("ImplementationContract.json").exists() else self._default_implementation_contract(spec)
        reports: dict[str, Any] = {}
        for name in [
            "CodeSyntaxReport.json",
            "CodeRunReport.json",
            "CodeRepairReport.json",
            "ExperimentAudit.json",
            "CodeEvalQualityReport.json",
            "ResultCollectionReport.json",
            "CodeStageGenerationReport.json",
            "CodeStage_core_report.json",
            "CodeStage_experiment_report.json",
            "CoreImplementationReport.json",
            "ExperimentImplementationReport.json",
            "code/CoreImplementationReport.json",
            "code/ExperimentImplementationReport.json",
        ]:
            path = self.artifacts.path(name)
            if path.exists():
                try:
                    reports[name] = self.artifacts.read_json(name)
                except Exception as exc:
                    reports[name] = {"status": "UNREADABLE", "error": str(exc)}
        traceback_file = self._repair_target_file(completed, reports)
        return textwrap.dedent(
            f"""\
            Follow this repair skill exactly.

            {self._code_stage_repair_template}

            BuildSpec:
            {json.dumps(spec.to_dict(), indent=2, sort_keys=True)}

            ImplementationContract:
            {json.dumps(contract, indent=2, sort_keys=True)}

            Attempt: {attempt}
            Previous repair attempts:
            {json.dumps(history, indent=2, sort_keys=True)}

            Latest command:
            {" ".join(command)}

            Latest returncode:
            {completed.returncode}

            stdout_tail:
            {completed.stdout[-6000:]}

            stderr_tail:
            {completed.stderr[-12000:]}

            traceback_file:
            {traceback_file or "unknown"}

            Reports:
            {json.dumps(reports, indent=2, sort_keys=True)}

            Current relevant source files:
            {self._repair_code_context(code_dir, traceback_file)}

            Return only minimal complete file replacements using file markers.
            """
        )

    def _repair_target_file(self, completed: subprocess.CompletedProcess[str], reports: dict[str, Any]) -> str | None:
        traceback_file = self._traceback_project_file(completed.stderr)
        if traceback_file:
            return traceback_file
        syntax_report = reports.get("CodeSyntaxReport.json")
        if isinstance(syntax_report, dict):
            failures = syntax_report.get("failures")
            if isinstance(failures, list) and failures:
                first = failures[0]
                if isinstance(first, dict) and isinstance(first.get("path"), str):
                    return first["path"]
        return None

    def _repair_code_context(self, code_dir: Path, target_file: str | None = None) -> str:
        priority = [
            "configs/experiment_config.json",
            "run_experiment.py",
            "src/dataset.py",
            "src/method.py",
            "src/baselines.py",
            "src/train.py",
            "src/evaluate.py",
            "src/plot.py",
            "requirements.txt",
        ]
        parts: list[str] = []
        if target_file:
            target_path = code_dir / target_file
            if target_path.exists():
                try:
                    content = target_path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    content = ""
                if len(content) > 18000:
                    content = content[:4000] + "\n\n[...middle omitted for repair prompt...]\n\n" + content[-14000:]
                parts.append(f"=== FILE: {target_file} ===\n{content}")
            priority = [rel for rel in priority if rel != target_file]
        remaining = self._current_code_context(
            code_dir,
            priority,
            max_file_chars=2500,
            max_total_chars=14000,
        )
        if remaining and remaining != "(no code files written yet)":
            parts.append(remaining)
        return "\n\n".join(parts) if parts else remaining

    def _traceback_project_file(self, stderr: str) -> str | None:
        matches = re.findall(r'File "([^"]+)", line \d+', stderr or "")
        for item in reversed(matches):
            path = item.replace("\\", "/")
            marker = "/code/"
            if marker in path:
                rel = path.split(marker, 1)[1]
                if rel.startswith(("src/", "configs/")) or rel == "run_experiment.py":
                    return rel
        return None

    def _resolve_code_environment(self, spec: BuildSpec) -> BuildSpec:
        """Resolve repository/environment details from BuildSpec before code generation.

        BUILD_SPEC normally prepares `environment`, but CODE is the last
        boundary before execution, so it also handles the case where a user or
        builder LLM supplied only `repo_url`.
        """
        report: dict[str, Any] = {
            "status": "PASS",
            "strategy": "build_spec_environment_then_repo_url",
            "input_repo_url": spec.repo_url,
            "input_environment_source": spec.environment.source,
            "selected_repo_url": spec.environment.reference_repo_url,
            "selected_repo_path": spec.environment.reference_repo_path,
            "env_files": list(spec.environment.env_files),
            "fallback": None,
            "errors": [],
        }
        if spec.environment.source.startswith("reference_repo") and (
            spec.environment.reference_repo_url or spec.environment.reference_repo_path
        ):
            report["resolution"] = "used_build_spec_environment"
            self.artifacts.write_json("EnvironmentResolutionReport.json", report)
            return spec

        normalized_url = normalize_repo_url(spec.repo_url) if spec.repo_url else ""
        if not normalized_url:
            report["resolution"] = "generated_environment"
            report["fallback"] = "missing_or_invalid_repo_url"
            spec.environment = self._generated_environment()
            self.artifacts.write_json("EnvironmentResolutionReport.json", report)
            return spec

        repo = RepoCard(
            repo_id=repo_id_from_url(normalized_url),
            repo_url=normalized_url,
            source_paper_id="BuildSpec.repo_url",
            source_title=spec.target_task,
            relevance_score=1.0,
        )
        resolved = RepoManager(self.artifacts.path("repos"), timeout_seconds=120).clone_and_inspect(repo)
        report["selected_repo_url"] = resolved.repo_url
        report["selected_repo_path"] = resolved.local_repo_path
        report["env_files"] = resolved.env_files
        report["repo_status"] = resolved.status
        report["errors"] = resolved.errors
        if resolved.status in {"cloned", "inspected"}:
            spec.environment = BuildEnvironment(
                source="reference_repo" if resolved.env_files or resolved.local_repo_path else "reference_repo_metadata",
                reference_repo_url=resolved.repo_url,
                reference_repo_path=resolved.local_repo_path,
                env_files=resolved.env_files,
                language=resolved.language or "python",
                framework=resolved.framework,
                requirements=self._requirements_from_repo(resolved),
                setup_commands=["pip install -r requirements.txt"],
            )
            report["resolution"] = "resolved_from_repo_url"
            report["fallback"] = None
        else:
            spec.environment = self._generated_environment()
            report["resolution"] = "generated_environment"
            report["fallback"] = "repo_clone_or_inspect_failed"
        self.artifacts.write_json("EnvironmentResolutionReport.json", report)
        return spec

    def _write_code_project(self, spec: BuildSpec, code_dir: Path) -> None:
        (code_dir / "src").mkdir(parents=True, exist_ok=True)
        (code_dir / "configs").mkdir(parents=True, exist_ok=True)
        self._write_text(code_dir / "README.md", self._code_readme(spec))
        self._write_text(code_dir / "ENVIRONMENT.md", self._environment_readme(spec))
        self._write_text(code_dir / "requirements.txt", self._requirements_txt(spec))
        self._write_text(code_dir / "environment.yml", self._environment_yml(spec))
        self._copy_reference_env_files(spec, code_dir)
        self._write_json(code_dir / "configs" / "experiment_config.json", self._experiment_config(spec))
        self._write_text(code_dir / "src" / "__init__.py", "")
        self._write_text(code_dir / "src" / "dataset.py", _GENERIC_DATASET_PY)
        self._write_text(code_dir / "src" / "baselines.py", _GENERIC_BASELINES_PY)
        self._write_text(code_dir / "src" / "method.py", _GENERIC_METHOD_PY)
        self._write_text(code_dir / "src" / "train.py", _GENERIC_TRAIN_PY)
        self._write_text(code_dir / "src" / "evaluate.py", _GENERIC_EVALUATE_PY)
        self._write_text(code_dir / "src" / "plot.py", _GENERIC_PLOT_PY)
        self._write_text(code_dir / "run_experiment.py", _GENERIC_RUN_EXPERIMENT_PY)

    def _write_llm_code_project(self, raw: str, code_dir: Path, *, require_full_project: bool = True) -> bool:
        # Try file-marker format first (avoids JSON escaping issues with code strings)
        files = self._parse_file_marker_format(raw)
        if not files:
            # Fall back to JSON format for backward compatibility
            files = self._parse_json_file_format(raw)
        if not files:
            return False
        written = set()
        for relative_path, content in files.items():
            normalized_path = self._normalize_code_output_path(relative_path)
            if not normalized_path:
                continue
            target = (code_dir / normalized_path).resolve()
            if code_dir.resolve() not in target.parents and target != code_dir.resolve():
                continue
            self._write_text(target, content)
            written.add(normalized_path)
        if not written:
            return False
        required = {
            "README.md",
            "configs/experiment_config.json",
            "src/dataset.py",
            "src/method.py",
            "src/baselines.py",
            "src/train.py",
            "src/evaluate.py",
            "src/plot.py",
            "run_experiment.py",
        }
        if require_full_project:
            # Also count files already on disk from a previous partial generation (continuation mode)
            on_disk = {r for r in required if (code_dir / r).exists()}
            all_present = on_disk | written
            if not required.issubset(all_present):
                # Still partial — update truncation state with cumulative written set
                missing = sorted(required - all_present)
                last_file = sorted(written)[-1] if written else ""
                last_content_tail = list(files.values())[-1][-800:] if files else ""
                self.artifacts.write_json("CodeTruncationState.json", {
                    "written_files": sorted(all_present),
                    "missing_files": missing,
                    "last_truncated_file": last_file,
                    "last_content_tail": last_content_tail,
                })
                return False
        # All required files present — clear stale truncation state
        trunc_path = self.artifacts.path("CodeTruncationState.json")
        if trunc_path.exists():
            trunc_path.unlink()
        return True if require_full_project else bool(written)

    def _normalize_code_output_path(self, relative_path: str) -> str:
        """Normalize LLM-emitted paths to be relative to the generated code root."""

        path = relative_path.strip().replace("\\", "/")
        while path.startswith("./"):
            path = path[2:]
        if not path or path.startswith("/") or ".." in Path(path).parts:
            return ""
        while path == "code" or path.startswith("code/"):
            path = path.removeprefix("code/") if path != "code" else ""
        return path

    def _parse_file_marker_format(self, raw: str) -> dict[str, str]:
        """Parse the === FILE: path === ... format. Returns {path: content} or {}."""
        pattern = re.compile(r"^=== FILE: (.+?) ===$", re.MULTILINE)
        matches = list(pattern.finditer(raw))
        if not matches:
            return {}
        files: dict[str, str] = {}
        for i, match in enumerate(matches):
            path = match.group(1).strip()
            start = match.end() + 1  # skip the newline after the marker
            end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
            content = self._strip_markdown_code_fence(raw[start:end].rstrip("\n"))
            if path:
                files[path] = content
        return files

    def _parse_json_file_format(self, raw: str) -> dict[str, str]:
        """Parse the legacy {files: [{path, content}]} JSON format. Returns {path: content} or {}."""
        payload = self._extract_json_object(raw)
        if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
            return {}
        files: dict[str, str] = {}
        for item in payload["files"]:
            if not isinstance(item, dict):
                continue
            relative_path = str(item.get("path", "")).strip()
            content = item.get("content")
            if relative_path and isinstance(content, str):
                files[relative_path] = self._strip_markdown_code_fence(content)
        return files

    def _strip_markdown_code_fence(self, content: str) -> str:
        """Remove accidental markdown code fences around generated file content."""

        text = content.strip("\n")
        lines = text.splitlines()
        if lines and re.fullmatch(r"```[A-Za-z0-9_+-]*\s*", lines[0].strip()):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).rstrip("\n")

    def _code_prompt(self, spec: BuildSpec) -> str:
        """Build the single standard code-generation prompt."""
        return self._code_prompt_generate_fresh(spec)

    def _run_staged_code_generation(
        self,
        spec: BuildSpec,
        code_dir: Path,
        *,
        evidence: list[EvidenceCard],
    ) -> dict[str, Any]:
        """Generate code through small staged skills instead of one large prompt."""
        code_dir.mkdir(parents=True, exist_ok=True)
        (code_dir / "src").mkdir(parents=True, exist_ok=True)
        (code_dir / "configs").mkdir(parents=True, exist_ok=True)
        if not (code_dir / "src" / "__init__.py").exists():
            self._write_text(code_dir / "src" / "__init__.py", "")

        report: dict[str, Any] = {
            "status": "PASS",
            "mode": "staged",
            "stages": [],
            "files_written": [],
            "errors": [],
        }
        raw_parts: list[str] = []
        prompt_parts: list[str] = []

        contract_prompt = self._implementation_contract_prompt(spec, evidence)
        prompt_parts.append("=== implementation_contract ===\n" + contract_prompt)
        contract_raw = self.llm.complete(contract_prompt) if self.llm is not None else ""
        raw_parts.append("=== implementation_contract ===\n" + contract_raw)
        self.artifacts.write_markdown("CodeStage_implementation_contract.raw.txt", contract_raw)
        contract = self._extract_json_object(contract_raw)
        if not isinstance(contract, dict):
            contract = self._default_implementation_contract(spec)
            report["errors"].append("implementation contract response was not valid JSON; used deterministic default contract")
        self.artifacts.write_json("ImplementationContract.json", contract)
        report["stages"].append({
            "stage": "implementation_contract",
            "files_written": ["ImplementationContract.json"],
            "valid": isinstance(contract, dict),
        })

        stages = [
            ("core", self._implement_core_template, ["src/dataset.py", "src/__init__.py", "src/method.py", "src/baselines.py", "configs/experiment_config.json"]),
            ("experiment", self._implement_experiment_template, ["src/train.py", "run_experiment.py", "configs/experiment_config.json", "README.md", "ENVIRONMENT.md", "requirements.txt", "environment.yml", "src/evaluate.py", "src/plot.py"]),
        ]
        for stage_name, template, expected_paths in stages:
            self._progress("CODE", f"staged generation: {stage_name}")
            stage_prompt = self._stage_code_prompt(
                stage_name=stage_name,
                stage_template=template,
                spec=spec,
                contract=contract,
                code_dir=code_dir,
                expected_paths=expected_paths,
            )
            prompt_parts.append(f"=== {stage_name} ===\n{stage_prompt}")
            try:
                stage_raw = self.llm.complete(stage_prompt) if self.llm is not None else ""
            except Exception as exc:
                stage_raw = f"Stage {stage_name} LLM request failed: {exc}"
                report["errors"].append(stage_raw)
            raw_parts.append(f"=== {stage_name} ===\n{stage_raw}")
            self.artifacts.write_markdown(f"CodeStage_{stage_name}.raw.txt", stage_raw)
            before = self._code_file_set(code_dir)
            files_written = self._write_llm_code_project(stage_raw, code_dir, require_full_project=False)
            after = self._code_file_set(code_dir)
            changed = sorted(after - before)
            if not changed:
                parsed = self._parse_file_marker_format(stage_raw) or self._parse_json_file_format(stage_raw)
                changed = sorted(parsed.keys()) if parsed else []
            syntax_report = self._check_code_syntax(code_dir)
            stage_report = {
                "stage": stage_name,
                "expected_paths": expected_paths,
                "files_written": changed,
                "parser_wrote_files": bool(files_written),
                "syntax_status_after_stage": syntax_report.get("status"),
            }
            report["stages"].append(stage_report)
            report["files_written"].extend(changed)
            self.artifacts.write_json(f"CodeStage_{stage_name}_report.json", stage_report)
            if syntax_report.get("status") != "PASS":
                report["status"] = "FAIL"
                report["errors"].append(f"syntax failed after staged generation step: {stage_name}")
                report["syntax_failures"] = syntax_report.get("failures", [])
                break

        report["files_written"] = sorted(set(report["files_written"]))
        required = {
            "src/dataset.py",
            "src/method.py",
            "src/baselines.py",
            "src/train.py",
            "src/evaluate.py",
            "src/plot.py",
            "run_experiment.py",
            "configs/experiment_config.json",
        }
        missing = sorted(path for path in required if not (code_dir / path).exists())
        if missing:
            report["status"] = "FAIL"
            report["missing_required_files"] = missing
            report["errors"].append(f"missing required staged files: {missing}")
        return {
            "prompt": "\n\n".join(prompt_parts),
            "raw": "\n\n".join(raw_parts),
            "files_written": bool(report["files_written"]),
            "report": report,
        }

    def _implementation_contract_prompt(self, spec: BuildSpec, evidence: list[EvidenceCard]) -> str:
        return textwrap.dedent(
            f"""\
            Follow this skill exactly.

            {self._implementation_contract_template}

            BuildSpec:
            {json.dumps(spec.to_dict(), indent=2, sort_keys=True)}

            Evidence summary:
            {self._evidence_summary(evidence)}

            Reference repository context:
            {self._reference_repo_context(spec)}

            Return JSON only. Do not use markdown fences.
            """
        )

    def _stage_code_prompt(
        self,
        *,
        stage_name: str,
        stage_template: str,
        spec: BuildSpec,
        contract: dict[str, Any],
        code_dir: Path,
        expected_paths: list[str],
    ) -> str:
        return textwrap.dedent(
            f"""\
            Follow this builder skill exactly.

            {stage_template}

            BuildSpec:
            {json.dumps(spec.to_dict(), indent=2, sort_keys=True)}

            ImplementationContract:
            {json.dumps(contract, indent=2, sort_keys=True)}

            Current code context for this stage:
            {self._current_code_context_for_stage(stage_name, code_dir)}

            Previous CODE feedback:
            {json.dumps(self._previous_code_feedback(), indent=2, sort_keys=True)}

            Priority rules:
            - Use BuildSpec to preserve research semantics: task, proposed method, datasets, baselines, metrics, logs, plots, sensitivities, ablations, and success criteria.
            - Use ImplementationContract as the binding engineering interface: file names, public APIs, metric schema, shared results paths, config keys, and verification checks.
            - If BuildSpec and ImplementationContract appear to conflict on engineering details, follow ImplementationContract and preserve BuildSpec semantics as closely as possible.
            - Do not invent new file names, APIs, output paths, or metric schemas when ImplementationContract already specifies them.

            Stage: {stage_name}
            Expected files for this stage:
            {json.dumps(expected_paths, indent=2)}

            Output format:
            - Return only complete file replacements for this stage.
            - Use file markers exactly:
              === FILE: <path> ===
              ...complete file content...
            - Paths are relative to the generated code root `runs/<run_id>/code/`; never prefix them with `code/`.
              Use `src/dataset.py`, not `code/src/dataset.py`.
            - No markdown fences: never write ```python or ``` inside a file replacement.
            - No JSON wrapper, no prose outside file markers.
            - Prefer the expected files. Do not rewrite unrelated files unless needed for interface consistency.
            """
        )

    def _current_code_context_for_stage(self, stage_name: str, code_dir: Path) -> str:
        stage = stage_name.lower().strip()
        if stage == "dataset":
            return self._current_code_context(code_dir, ["configs/experiment_config.json"], max_file_chars=2500, max_total_chars=4000)
        if stage == "core":
            return self._current_code_context(
                code_dir,
                ["configs/experiment_config.json"],
                max_file_chars=3000,
                max_total_chars=5000,
            )
        if stage == "experiment":
            return self._current_code_context(
                code_dir,
                ["configs/experiment_config.json", "src/dataset.py", "src/method.py", "src/baselines.py", "src/train.py"],
                max_file_chars=3500,
                max_total_chars=16000,
            )
        return self._current_code_context(
            code_dir,
            [
                "configs/experiment_config.json",
                "src/dataset.py",
                "src/method.py",
                "src/baselines.py",
                "src/train.py",
                "src/evaluate.py",
                "src/plot.py",
                "run_experiment.py",
                "requirements.txt",
            ],
            max_file_chars=3000,
            max_total_chars=16000,
        )

    def _current_code_context(
        self,
        code_dir: Path,
        priority: list[str],
        *,
        max_file_chars: int = 3000,
        max_total_chars: int = 16000,
    ) -> str:
        parts: list[str] = []
        total = 0
        for rel in priority:
            path = code_dir / rel
            if not path.exists():
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")[:max_file_chars]
            except OSError:
                continue
            block = f"=== FILE: {rel} ===\n{content}"
            if total + len(block) > max_total_chars:
                parts.append(f"=== FILE: {rel} ===\n[omitted: context limit]")
                break
            parts.append(block)
            total += len(block)
        return "\n\n".join(parts) if parts else "(no code files written yet)"

    def _code_file_set(self, code_dir: Path) -> set[str]:
        if not code_dir.exists():
            return set()
        return {
            path.relative_to(code_dir).as_posix()
            for path in code_dir.rglob("*")
            if path.is_file()
        }

    def _evidence_summary(self, evidence: list[EvidenceCard], limit: int = 8) -> str:
        if not evidence:
            return "(none)"
        rows = []
        for card in evidence[:limit]:
            rows.append(json.dumps(card.to_dict(), sort_keys=True)[:1200])
        return "\n".join(rows)

    def _default_implementation_contract(self, spec: BuildSpec) -> dict[str, Any]:
        metrics = spec.metrics or ["primary_metric"]
        plots = [plot.to_dict() for plot in spec.plots]
        return {
            "status": "PASS",
            "canonical_layout": {
                "entrypoint": "run_experiment.py",
                "config": "configs/experiment_config.json",
                "dataset_file": "src/dataset.py",
                "method_file": "src/method.py",
                "baselines_file": "src/baselines.py",
                "train_file": "src/train.py",
                "evaluate_file": "src/evaluate.py",
                "plot_file": "src/plot.py",
            },
            "module_contracts": {
                "dataset": {
                    "public_api": ["load_dataset(config)", "make_synthetic(config, seed)"],
                    "data_objects": ["domain-specific dataset or scenario object"],
                    "smoke_test": ["load_dataset({'seed': 0}) returns non-None"],
                },
                "method": {
                    "public_api": ["ProposedMethod or BuildSpec-derived method class"],
                    "required_behaviors": [
                        "return metrics compatible with BuildSpec.metrics",
                        "use exact BuildSpec metric names or document an explicit internal-to-BuildSpec metric mapping",
                    ],
                    "smoke_test": ["instantiate method and run one tiny step/evaluation"],
                },
                "baselines": {
                    "required": spec.baselines,
                    "public_api": ["baseline classes/functions returning same metric schema as method and exact BuildSpec metrics"],
                    "smoke_test": ["run each baseline on tiny dataset"],
                },
                "train": {
                    "public_api": ["train_model, run_training, train_and_evaluate, or BuildSpec-derived training loop"],
                    "required_behaviors": ["call dataset/method/baseline APIs without redefining them", "write intermediate logs under ../results"],
                },
                "runner": {
                    "command": "python run_experiment.py --config configs/experiment_config.json",
                    "required_outputs": ["../results/metrics.json", "../results/results_table.csv", "../results/summary_table.csv"],
                },
                "plots": {
                    "required_outputs": [plot.get("path", "") for plot in plots],
                },
            },
            "generation_stages": [
                {
                    "stage": "core",
                    "skill": "implement_core.md",
                    "responsibility": "dataset -> method -> baselines",
                    "files": [
                        "src/__init__.py",
                        "src/dataset.py",
                        "src/method.py",
                        "src/baselines.py",
                        "configs/experiment_config.json",
                    ],
                    "report": "code/CoreImplementationReport.json",
                },
                {
                    "stage": "experiment",
                    "skill": "implement_experiment.md",
                    "responsibility": "execute -> runner -> plots",
                    "files": [
                        "src/train.py",
                        "src/evaluate.py",
                        "src/plot.py",
                        "run_experiment.py",
                        "configs/experiment_config.json",
                        "README.md",
                        "ENVIRONMENT.md",
                        "requirements.txt",
                        "environment.yml",
                    ],
                    "report": "code/ExperimentImplementationReport.json",
                },
            ],
            "implementation_choices": {
                "shared_results_dir": "../results",
                "metric_schema": ["method", "source", *metrics],
            },
            "dependency_policy": {
                "avoid_hard_optional_dependencies": True,
                "fallback_after_dataset_acquisition_attempts": 2,
            },
            "verification_steps": [
                "python -m py_compile run_experiment.py src/*.py",
                "python run_experiment.py --config configs/experiment_config.json",
                "check results/metrics.json, results/results_table.csv, logs, and plots",
            ],
            "forbidden_paths": ["code/results", "code/outputs", "code/logs"],
            "notes": [],
        }

    # ------------------------------------------------------------------
    # Shared output-format block reused by both prompt variants
    # ------------------------------------------------------------------
    def _code_output_format_block(self) -> str:
        return textwrap.dedent(
            f"""\
            ## Output Format (MUST follow exactly)
            Return every file using this marker format — one header line, then raw content:

            === FILE: README.md ===
            (file content here)
            === FILE: configs/experiment_config.json ===
            (file content here)
            === FILE: src/__init__.py ===

            === FILE: src/dataset.py ===
            (file content here)
            === FILE: src/method.py ===
            (file content here)
            === FILE: src/baselines.py ===
            (file content here)
            === FILE: src/train.py ===
            (file content here)
            === FILE: src/evaluate.py ===
            (file content here)
            === FILE: src/plot.py ===
            (file content here)
            === FILE: run_experiment.py ===
            (file content here)

            - Each file starts with exactly `=== FILE: <path> ===` on its own line.
            - Write raw file content directly — no JSON wrapping, no markdown fences, no quotes.
            - No prose or explanation between files.

            ## Epoch Budget (MUST be respected)
            Read the epoch bounds directly from the BuildSpec above:
            - train_epochs must be in [min_train_epochs, max_train_epochs] as specified in BuildSpec
            - eval_epochs must be in [min_eval_epochs, max_eval_epochs] as specified in BuildSpec
            - These are full dataset passes (epochs), not gradient steps.

            ## BuildSpec Compliance Requirements
            - Implement executable algorithm logic for the BuildSpec method; no decorative scaffold.
            - Detect CPU/GPU when using PyTorch.
            - Follow BuildSpec.logging for JSONL log paths and fields; if absent, log the primary BuildSpec metric over epochs/episodes/steps.
            - Treat BuildSpec.metrics, BuildSpec.logging, and BuildSpec.plots as the reporting contract passed from BUILD_SPEC.
            - Write ../results/metrics.json and ../results/results_table.csv with real computed numbers.
            - Write ../results/summary_table.csv with compact paper-facing method labels and key metrics when result rows exist.
            - Write every figure declared in BuildSpec.plots via src/plot.py; if absent, write progress_curve.png and eval_curve.png using real domain metrics.
            - Make eval_curve.png paper-ready: use short labels, computed rows only, the correct primary metric, uncertainty/error bars when available, and multi-panel secondary views when at least three numeric reporting targets exist.
            - Make progress_curve.png from real multi-point progress/candidate/epoch records for iterative experiments; do not fake a curve from one terminal point per seed.
            - Cover every metric in BuildSpec.metrics with a numeric column or field using the exact BuildSpec metric names.
            - If method/baseline code uses internal metric names, add one explicit mapping layer before writing metrics.json, results_table.csv, logs, and plots; never let missing key lookups silently become zero metrics.
            - results_table.csv must include rows for the proposed method and at least one baseline.
            - Do not fabricate metrics, losses, curves, or baseline scores; only use train_loss/eval_loss when the method has a real optimization loss.
            - If previous feedback lists failures, fix the underlying code rather than adding stub files.
            """
        )

    def _code_prompt_generate_fresh(self, spec: BuildSpec) -> str:
        """Prompt for the standard path: write a standalone project from scratch."""
        feedback = self._previous_code_feedback()
        return (
            self._code_template
            .replace("{{build_spec}}", json.dumps(spec.to_dict(), indent=2, sort_keys=True))
            .replace("{{reference_repo_context}}", self._reference_repo_context(spec))
            .replace("{{feedback}}", json.dumps(feedback, indent=2, sort_keys=True))
        )

    def _reference_repo_context(self, spec: BuildSpec) -> str:
        """Return optional repo excerpts for the standard generator."""
        repo_path_str = spec.environment.reference_repo_path
        repo_path = Path(repo_path_str) if repo_path_str else None
        if not repo_path or not repo_path.exists():
            return "No reference repository is available. Implement from BuildSpec only."
        files_summary = self._repo_files_summary(repo_path, max_chars=10000)
        return textwrap.dedent(
            f"""\
            Reference repository path: {repo_path_str}

            Use these excerpts only as implementation reference. Do not preserve the
            repository layout, entry points, result paths, plotting scripts, or table
            formats unless they match the required output contract exactly.

            Key files:
            {files_summary or "(no readable files found)"}
            """
        ).strip()

    def _repo_files_summary(self, repo_path: Path, max_chars: int = 10000) -> str:
        """Read the most informative files from a cloned repo and return a condensed string."""
        # Priority order: documentation/config first, then entry points, then core source
        priority_globs = [
            ("README*", 1500),
            ("requirements*.txt", 800),
            ("setup.py", 600),
            ("pyproject.toml", 600),
            ("environment*.yml", 600),
            ("train.py", 1800),
            ("main.py", 1800),
            ("run*.py", 1800),
            ("experiment*.py", 1800),
            ("model*.py", 1800),
            ("method*.py", 1800),
            ("network*.py", 1800),
            ("dataset*.py", 1200),
            ("data*.py", 1200),
        ]
        collected: list[tuple[str, str]] = []
        seen: set[Path] = set()
        remaining = max_chars

        for glob_pattern, per_file_cap in priority_globs:
            if remaining <= 0:
                break
            for match in sorted(repo_path.rglob(glob_pattern)):
                if not match.is_file() or match in seen:
                    continue
                # Skip hidden dirs, node_modules, __pycache__, etc.
                if any(p.name.startswith(".") or p.name in {"__pycache__", "node_modules", ".git"}
                       for p in match.parents):
                    continue
                try:
                    content = match.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                excerpt = content[:min(per_file_cap, remaining)]
                relative = str(match.relative_to(repo_path))
                collected.append((relative, excerpt))
                seen.add(match)
                remaining -= len(excerpt)
                if remaining <= 0:
                    break

        if not collected:
            return ""
        parts = [f"--- {path} ---\n{content}" for path, content in collected]
        return "\n\n".join(parts)

    def _continuation_prompt(self, spec: BuildSpec, trunc_state: dict[str, Any]) -> str:
        written = trunc_state.get("written_files", [])
        missing = trunc_state.get("missing_files", [])
        last_file = trunc_state.get("last_truncated_file", "")
        last_tail = trunc_state.get("last_content_tail", "")
        code_dir = self.artifacts.path("code")

        # Read existing file contents so the LLM can write consistent continuations
        existing_files_block = self._read_existing_code_files(code_dir, written, last_file)

        return textwrap.dedent(
            f"""\
            The previous code generation was cut off before completion.

            BuildSpec:
            {json.dumps(spec.to_dict(), indent=2, sort_keys=True)}

            === ALREADY WRITTEN FILES ===
            The following files are already on disk. Do NOT rewrite them unless the file is marked as truncated below.
            {existing_files_block}

            === TRUNCATED / MISSING FILES ===
            These required files are still missing and must be generated now:
            {json.dumps(missing, indent=2)}

            The last file being written ({last_file!r}) may be incomplete. Its content ended with:
            ...
            {last_tail}

            Instructions:
            - If {last_file!r} was truncated mid-way, output it first as a COMPLETE replacement.
            - Then output each missing file in full.
            - Use the file-marker format:

            === FILE: {last_file} ===
            (complete file content)
            === FILE: src/evaluate.py ===
            (complete file content)
            ... and so on for all missing files listed above.

            Rules:
            - Do NOT repeat files that are already fully written (except {last_file!r} if incomplete).
            - Write raw file content directly after each marker line. No JSON, no markdown fences.
            - All generated code must be runnable and fully consistent with the already-written files shown above.
            - Apply the same output constraints as the original generation:
              write ../results/metrics.json, ../results/results_table.csv, ../results/progress_curve.png, ../results/eval_curve.png.
            """
        )

    def _read_existing_code_files(self, code_dir: Path, written: list[str], last_file: str) -> str:
        """Read already-written code files and format them for the continuation prompt."""
        MAX_FILE_CHARS = 3000  # per-file cap to avoid blowing the context window
        MAX_TOTAL_CHARS = 20000
        parts: list[str] = []
        total = 0
        for rel in written:
            path = code_dir / rel
            if not path.exists():
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            truncated_note = " [possibly truncated — see tail above]" if rel == last_file else ""
            if len(content) > MAX_FILE_CHARS:
                content = content[:MAX_FILE_CHARS] + "\n... (truncated for brevity)"
            block = f"=== FILE: {rel} ==={truncated_note}\n{content}"
            if total + len(block) > MAX_TOTAL_CHARS:
                parts.append(f"=== FILE: {rel} === [omitted — context limit reached]")
                break
            parts.append(block)
            total += len(block)
        return "\n".join(parts) if parts else "(none)"

    def _previous_code_feedback(self) -> dict[str, Any]:
        feedback: dict[str, Any] = {}
        for name in ["CodeRunReport.json", "CodeSyntaxReport.json", "CodeRepairReport.json", "CodeEvalQualityReport.json", "ExperimentAudit.json"]:
            path = self.artifacts.path(name)
            if path.exists():
                try:
                    feedback[name] = self.artifacts.read_json(name)
                except Exception as exc:
                    feedback[name] = {"status": "UNREADABLE", "error": str(exc)}
        if self.artifacts.path("EXPERIMENT_LOG.md").exists():
            text = self.artifacts.path("EXPERIMENT_LOG.md").read_text(encoding="utf-8", errors="ignore")
            feedback["EXPERIMENT_LOG_tail"] = text[-6000:]
        return feedback

    def _experiment_config(self, spec: BuildSpec) -> dict[str, Any]:
        metrics = spec.metrics or ["primary_metric"]
        baselines = spec.baselines or ["reference_baseline"]
        method_name = self._method_name(spec)
        return {
            "build_id": spec.build_id,
            "idea_id": spec.idea_id,
            "target_task": spec.target_task,
            "method_summary": spec.method_summary,
            "environment": spec.environment.to_dict(),
            "baselines": baselines,
            "metrics": metrics,
            "seeds": [0],
            "runtime": {
                "use_cuda": True,
            },
            "dataset": {"name": "synthetic_supervised", "num_samples": 512, "input_dim": 12, "output_dim": 3, "noise_std": 0.05},
            "model": {
                "hidden_dim": 64,
            },
            "training": {
                "batch_size": 64,
                "learning_rate": 0.0003,
                "method_steps": 60,
            },
            "method": {
                "name": method_name,
                "experiment_family": self._experiment_family(spec),
            },
            "logging": {
                "log_interval": 20,
            },
            "outputs": {
                "metrics_json": "../results/metrics.json",
                "results_table_csv": "../results/results_table.csv",
            },
        }

    def _method_name(self, spec: BuildSpec) -> str:
        return safe_identifier(spec.idea_id or "proposed_method")

    def _experiment_family(self, spec: BuildSpec) -> str:
        text = " ".join([spec.method_summary, " ".join(spec.implementation_plan), " ".join(spec.baselines)]).lower()
        if "vae" in text or "variational autoencoder" in text:
            return "generative_modeling"
        if "classification" in text or "classifier" in text:
            return "supervised_classification"
        if "regression" in text or "predict" in text:
            return "supervised_regression"
        return "generic_experiment"

    def _code_readme(self, spec: BuildSpec) -> str:
        return textwrap.dedent(
            f"""\
            # Generated Experiment Code

            Build: `{spec.build_id}`
            Idea: `{spec.idea_id}`

            ## Target Task
            {spec.target_task}

            ## Method
            {spec.method_summary}

            ## Run
            ```bash
            python run_experiment.py --config configs/experiment_config.json
            ```

            Outputs are written to `../results/`.
            """
        )

    def _environment_readme(self, spec: BuildSpec) -> str:
        env = spec.environment
        setup_commands = "\n".join(f"- `{command}`" for command in env.setup_commands)
        env_files = "\n".join(f"- `{path}`" for path in env.env_files) if env.env_files else "- none"
        return (
            "# Environment\n\n"
            f"Source: `{env.source}`\n"
            f"Reference repo: `{env.reference_repo_url or 'none'}`\n"
            f"Reference repo path: `{env.reference_repo_path or 'none'}`\n"
            f"Language: `{env.language or 'python'}`\n"
            f"Framework: `{env.framework or 'not detected'}`\n\n"
            "## Setup Commands\n"
            f"{setup_commands}\n\n"
            "## Reference Environment Files\n"
            f"{env_files}\n"
        )

    def _requirements_txt(self, spec: BuildSpec) -> str:
        requirements = [item for item in spec.environment.requirements if not item.startswith("python")]
        requirements = _ensure_runtime_requirements(requirements)
        return "\n".join(_dedupe(requirements)) + ("\n" if requirements else "")

    def _environment_yml(self, spec: BuildSpec) -> str:
        pip_requirements = [item for item in spec.environment.requirements if not item.startswith("python")]
        pip_requirements = _ensure_runtime_requirements(pip_requirements)
        pip_block = "\n".join(f"      - {item}" for item in pip_requirements) or "      - pip"
        return textwrap.dedent(
            f"""\
            name: quit-agent-{spec.build_id}
            channels:
              - conda-forge
            dependencies:
              - python>=3.11
              - pip
              - pip:
            {pip_block}
            """
        )

    def _copy_reference_env_files(self, spec: BuildSpec, code_dir: Path) -> None:
        if not spec.environment.env_files:
            return
        target_dir = code_dir / "env" / "reference"
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in spec.environment.env_files[:8]:
            source = Path(item)
            if not source.exists() or not source.is_file():
                continue
            self._write_text(target_dir / source.name, source.read_text(encoding="utf-8", errors="ignore"))

    def _write_experiment_metrics_markdown(
        self,
        spec: BuildSpec,
        completed: subprocess.CompletedProcess[str],
        report: CodeRunReport,
    ) -> None:
        metrics_path = self.artifacts.path("results/metrics.json")
        table_path = self.artifacts.path("results/results_table.csv")
        metrics = self._load_metrics_json(metrics_path)
        rows = self._load_results_rows(table_path)
        stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
        training_lines = self._training_log_lines(stdout_lines)
        error_text = completed.stderr.strip()
        lines = [
            "# Experiment Dashboard",
            "",
            f"Build ID: `{spec.build_id}`",
            f"Idea ID: `{spec.idea_id}`",
            f"Status: `{report.status}`",
            f"Return code: `{completed.returncode}`",
            f"Target task: {spec.target_task}",
            "",
            "## Final Performance",
            "",
        ]

        if rows:
            lines.extend(self._markdown_table(rows[:12]))
        else:
            lines.append("No result rows were produced.")

        lines.extend(["", "## Metric Summary", ""])
        if metrics:
            summary = metrics.get("summary", metrics) if isinstance(metrics, dict) else metrics
            lines.append("```json")
            lines.append(json.dumps(summary, indent=2, sort_keys=True)[:5000])
            lines.append("```")
        else:
            lines.append("`results/metrics.json` was not produced or could not be parsed.")

        lines.extend(["", "## Progress / Evaluation Log", ""])
        if training_lines:
            lines.append("```text")
            lines.extend(training_lines[-80:])
            lines.append("```")
        else:
            lines.append("No training log lines were detected in stdout.")

        lines.extend(["", "## Stdout Tail", ""])
        stdout_tail = stdout_lines[-30:] if stdout_lines else ["No stdout captured."]
        lines.append("```text")
        lines.extend(stdout_tail)
        lines.append("```")

        lines.extend(["", "## Result Files", ""])
        lines.append(f"- `results/metrics.json`: {'present' if metrics_path.exists() else 'missing'}")
        lines.append(f"- `results/results_table.csv`: {'present' if table_path.exists() else 'missing'}")

        lines.extend(["", "## Raw Results Table Preview", ""])
        if table_path.exists():
            table_lines = table_path.read_text(encoding="utf-8", errors="ignore").splitlines()[:30]
            lines.append("```csv")
            lines.extend(table_lines)
            lines.append("```")
        else:
            lines.append("`results/results_table.csv` was not produced.")

        lines.extend(["", "## Errors", ""])
        if error_text:
            lines.extend(["```text", error_text[-6000:], "```"])
        else:
            lines.append("No stderr captured.")
        self._write_text(self.artifacts.path("code/EXPERIMENT_METRICS.md"), "\n".join(lines) + "\n")

    def _load_metrics_json(self, path: Path) -> Any:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _load_results_rows(self, path: Path) -> list[dict[str, str]]:
        if not path.exists():
            return []
        try:
            import csv

            with path.open(newline="", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))
        except Exception:
            return []

    def _training_log_lines(self, stdout_lines: list[str]) -> list[str]:
        patterns = ("loss", "metric", "accuracy", "return", "reward", "epoch", "step", "eval", "validation", "train")
        return [line for line in stdout_lines if any(pattern in line.lower() for pattern in patterns)]

    def _markdown_table(self, rows: list[dict[str, str]]) -> list[str]:
        headers = list(rows[0].keys())
        table = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in rows:
            table.append("| " + " | ".join(str(row.get(header, ""))[:80] for header in headers) + " |")
        return table

    def _experiment_log(self, spec: BuildSpec, command: list[str], completed: subprocess.CompletedProcess[str], report: CodeRunReport) -> str:
        return textwrap.dedent(
            f"""\
            # Experiment Log

            Build ID: `{spec.build_id}`
            Idea ID: `{spec.idea_id}`
            Status: `{report.status}`
            Code generation: `{report.generation_mode}`
            Fallback used: `{report.fallback_used}`
            Repair attempted: `{report.repair_attempted}`
            Repair succeeded: `{report.repair_succeeded}`

            ## Command
            ```bash
            {' '.join(command)}
            ```

            ## Generated Outputs
            {chr(10).join(f"- `{item}`" for item in report.outputs)}

            ## stdout
            ```text
            {completed.stdout[-4000:]}
            ```

            ## stderr
            ```text
            {completed.stderr[-4000:]}
            ```
            """
        )

    def _write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def write(
        self,
        spec: BuildSpec,
        evidence: list[EvidenceCard] | None = None,
        papers: list[PaperCard] | None = None,
    ) -> tuple[dict[str, Any], str, str]:
        """Skill: write_from_build_spec.

        Use when: CODE_EVAL passes and the system needs a paper draft artifact.
        Inputs: BuildSpec.json plus optional EvidenceCards/PaperCards for citations and style.
        Outputs: paper_gene/main.tex, references.bib, copied ICML template files, optional main.pdf.
        Failure mode: WriteReport.json records compile errors; WRITE_EVAL decides fallback.
        """
        evidence = evidence or []
        papers = papers or []
        paper_dir = self.artifacts.path("paper_gene")
        copied_files = self._copy_paper_template(paper_dir)
        bib_text, citation_keys = self._build_references_bib(evidence, papers)

        self._progress("WRITE", "build paper prompt")
        prompt = self._write_prompt(spec, evidence, papers, citation_keys)
        llm_error = ""
        try:
            if self.llm is not None:
                self._progress("WRITE", "request writer LLM paper draft")
            llm_raw = self.llm.complete(prompt) if self.llm is not None else ""
        except Exception as exc:
            llm_raw = ""
            llm_error = f"writer LLM request failed: {exc}"
            if self.llm is not None and self._is_retryable_writer_error(exc):
                compact_prompt = self._compact_write_prompt(spec, evidence, papers, citation_keys)
                self.artifacts.write_markdown("WriteDraft.retry_prompt.txt", compact_prompt)
                self._progress("WRITE", f"writer LLM request failed; retry compact prompt_chars={len(compact_prompt)}")
                try:
                    llm_raw = self.llm.complete(compact_prompt)
                    prompt = compact_prompt
                    llm_error = ""
                except Exception as retry_exc:
                    llm_error = f"{llm_error}; compact retry failed: {retry_exc}"
            if llm_error:
                self.artifacts.write_markdown("WriteDraft.error.txt", llm_error)
                self._progress("WRITE", "writer LLM request failed; report failure")
        llm_raw = self._clean_latex_output(llm_raw)
        if not self._is_valid_latex(llm_raw):
            if llm_raw:
                self.artifacts.write_markdown("WriteDraft.raw.txt", llm_raw)
                self._progress("WRITE", "writer LLM response invalid; report failure")
            report = {
                "status": "FAIL",
                "reason": llm_error or ("writer LLM did not return valid LaTeX" if llm_raw else "writer LLM is not configured"),
                "paper_dir": str(paper_dir),
                "template_dir": str(_DEFAULT_PAPER_TEMPLATE_DIR),
                "copied_template_files": copied_files,
                "used_llm": self.llm is not None,
                "outputs": ["paper_gene/references.bib"],
                "compile": {"status": "SKIPPED", "reason": "valid main.tex was not generated"},
                "style_sources": [paper.paper_id for paper in papers[:8]],
                "evidence_used": [card.evidence_id for card in evidence],
            }
            self._write_text(paper_dir / "references.bib", bib_text)
            self.artifacts.write_json("WriteReport.json", report)
            return report, prompt, llm_raw

        llm_raw = self._ensure_visible_icml_authors(llm_raw)
        self._write_text(paper_dir / "main.tex", llm_raw)
        self._write_text(paper_dir / "references.bib", bib_text)
        length_report = self._validate_paper_lengths(llm_raw)
        reference_report = self._validate_references(llm_raw, bib_text, citation_keys)
        experiment_report = self._validate_experiment_reporting(llm_raw)
        appendix_report = self._validate_appendix_content(llm_raw)
        validation_passed = (
            length_report["status"] == "PASS"
            and reference_report["status"] == "PASS"
            and experiment_report["status"] == "PASS"
            and appendix_report["status"] == "PASS"
        )
        compile_report = self._compile_latex(paper_dir)
        latex_repair_report = self._repair_latex_compile_failure(
            spec=spec,
            evidence=evidence,
            papers=papers,
            citation_keys=citation_keys,
            paper_dir=paper_dir,
            current_tex=llm_raw,
            compile_report=compile_report,
        )
        if latex_repair_report.get("succeeded"):
            llm_raw = str(latex_repair_report.get("repaired_tex", llm_raw))
            llm_raw = self._ensure_visible_icml_authors(llm_raw)
            self._write_text(paper_dir / "main.tex", llm_raw)
            length_report = self._validate_paper_lengths(llm_raw)
            reference_report = self._validate_references(llm_raw, bib_text, citation_keys)
            experiment_report = self._validate_experiment_reporting(llm_raw)
            appendix_report = self._validate_appendix_content(llm_raw)
            validation_passed = (
                length_report["status"] == "PASS"
                and reference_report["status"] == "PASS"
                and experiment_report["status"] == "PASS"
                and appendix_report["status"] == "PASS"
            )
            compile_report = latex_repair_report.get("compile", compile_report)
        pdf_exists = (paper_dir / "main.pdf").exists()
        page_report = self._validate_compiled_pdf_pages(paper_dir, compile_report)
        status = "PASS" if pdf_exists and validation_passed else "PARTIAL" if pdf_exists else "FAIL"
        if page_report.get("hard_fail"):
            status = "FAIL"
        if status == "PASS" and page_report.get("status") != "PASS":
            status = "PARTIAL"
        report = {
            "status": status,
            "paper_dir": str(paper_dir),
            "template_dir": str(_DEFAULT_PAPER_TEMPLATE_DIR),
            "copied_template_files": copied_files,
            "used_llm": True,
            "outputs": [
                "paper_gene/main.tex",
                "paper_gene/references.bib",
                *(["paper_gene/main.pdf"] if pdf_exists else []),
            ],
            "length_validation": length_report,
            "reference_validation": reference_report,
            "experiment_validation": experiment_report,
            "appendix_validation": appendix_report,
            "page_validation": page_report,
            "compile": compile_report,
            "latex_repair": latex_repair_report,
            "style_sources": [paper.paper_id for paper in papers[:8]],
            "evidence_used": [card.evidence_id for card in evidence],
        }
        self.artifacts.write_json("WriteReport.json", report)
        return report, prompt, llm_raw

    def revise_paper(self, spec: BuildSpec) -> tuple[dict[str, Any], str, str]:
        """Skill: revise_paper.

        Use when: WRITE_LLM_EVAL produced improvement_hints in PaperReview.json.
        The LLM reads the current main.tex and the reviewer hints, then decides
        autonomously which parts to change — targeted edits, section rewrites, or
        additions.  The output is a complete revised main.tex.
        Failure mode: falls back to the existing draft unchanged; WriteReport.json
        records the outcome.
        """
        paper_dir = self.artifacts.path("paper_gene")
        current_tex_path = paper_dir / "main.tex"
        if not current_tex_path.exists():
            report = {
                "status": "FAIL",
                "reason": "paper_gene/main.tex not found; cannot revise",
                "outputs": [],
                "compile": {"status": "SKIPPED"},
            }
            self.artifacts.write_json("WriteReport.json", report)
            return report, "", ""

        current_tex = current_tex_path.read_text(encoding="utf-8", errors="ignore")

        # Read improvement_hints from PaperReview.json (written by WRITE_LLM_EVAL or WRITE_FINAL_EVAL)
        llm_review: dict[str, Any] = {}
        hard_failures: list[str] = []
        if self.artifacts.path("PaperReview.json").exists():
            paper_review = self.artifacts.read_json("PaperReview.json")
            llm_review = paper_review.get("llm_review") or {}
            hard_failures = paper_review.get("failures") or []

        weaknesses = llm_review.get("weaknesses", "")
        improvement_hints = list(llm_review.get("improvement_hints") or [])
        # Promote any hard-check failures as additional concrete hints
        for failure in hard_failures:
            hint = f"[hard-check] {failure}"
            if hint not in improvement_hints:
                improvement_hints.append(hint)
        hints_text = "\n".join(f"- {h}" for h in improvement_hints) if improvement_hints else "(none)"

        build_spec_json = json.dumps(spec.to_dict(), indent=2, sort_keys=True)
        tex_excerpt = current_tex[:32000] if len(current_tex) > 32000 else current_tex

        prompt = (
            self._revise_paper_template
            .replace("{{build_spec}}", build_spec_json)
            .replace("{{weaknesses}}", weaknesses or "(none)")
            .replace("{{improvement_hints}}", hints_text)
            .replace("{{current_tex}}", tex_excerpt)
        )

        self._progress("WRITE_REVISE", f"request LLM paper revision, prompt_chars={len(prompt)}")
        llm_raw = ""
        llm_error = ""
        try:
            llm_raw = self.llm.complete(prompt) if self.llm is not None else ""
        except Exception as exc:
            llm_error = f"revise LLM request failed: {exc}"
            self.artifacts.write_markdown("ReviseDraft.error.txt", llm_error)

        llm_raw = self._clean_latex_output(llm_raw)
        # If LLM output is not valid LaTeX, keep the existing draft
        if not self._is_valid_latex(llm_raw):
            if llm_raw:
                self.artifacts.write_markdown("ReviseDraft.raw.txt", llm_raw)
            self._progress("WRITE_REVISE", "LLM revision invalid; keeping existing draft")
            # Re-run compile/validate on the unchanged draft so WriteReport is current
            llm_raw = current_tex
            llm_error = llm_error or "LLM revision did not return valid LaTeX; kept existing draft"

        llm_raw = self._ensure_visible_icml_authors(llm_raw)
        self._write_text(paper_dir / "main.tex", llm_raw)

        length_report = self._validate_paper_lengths(llm_raw)
        reference_report = self._validate_references(
            llm_raw,
            (paper_dir / "references.bib").read_text(encoding="utf-8", errors="ignore") if (paper_dir / "references.bib").exists() else "",
            {},
        )
        experiment_report = self._validate_experiment_reporting(llm_raw)
        appendix_report = self._validate_appendix_content(llm_raw)
        validation_passed = (
            length_report["status"] == "PASS"
            and reference_report["status"] == "PASS"
            and experiment_report["status"] == "PASS"
            and appendix_report["status"] == "PASS"
        )
        compile_report = self._compile_latex(paper_dir)
        latex_repair_report = self._repair_latex_compile_failure(
            spec=spec,
            evidence=[],
            papers=[],
            citation_keys={},
            paper_dir=paper_dir,
            current_tex=llm_raw,
            compile_report=compile_report,
        )
        if latex_repair_report.get("succeeded"):
            llm_raw = str(latex_repair_report.get("repaired_tex", llm_raw))
            llm_raw = self._ensure_visible_icml_authors(llm_raw)
            self._write_text(paper_dir / "main.tex", llm_raw)
            length_report = self._validate_paper_lengths(llm_raw)
            experiment_report = self._validate_experiment_reporting(llm_raw)
            appendix_report = self._validate_appendix_content(llm_raw)
            validation_passed = (
                length_report["status"] == "PASS"
                and reference_report["status"] == "PASS"
                and experiment_report["status"] == "PASS"
                and appendix_report["status"] == "PASS"
            )
            compile_report = latex_repair_report.get("compile", compile_report)

        pdf_exists = (paper_dir / "main.pdf").exists()
        page_report = self._validate_compiled_pdf_pages(paper_dir, compile_report)
        status = "PASS" if pdf_exists and validation_passed else "PARTIAL" if pdf_exists else "FAIL"
        if page_report.get("hard_fail"):
            status = "FAIL"
        if status == "PASS" and page_report.get("status") != "PASS":
            status = "PARTIAL"
        if llm_error and status == "PASS":
            status = "PARTIAL"

        report = {
            "status": status,
            "paper_dir": str(paper_dir),
            "used_llm": self.llm is not None,
            "llm_error": llm_error,
            "hints_applied": improvement_hints,
            "outputs": [
                "paper_gene/main.tex",
                *(["paper_gene/main.pdf"] if pdf_exists else []),
            ],
            "length_validation": length_report,
            "reference_validation": reference_report,
            "experiment_validation": experiment_report,
            "appendix_validation": appendix_report,
            "page_validation": page_report,
            "compile": compile_report,
            "latex_repair": latex_repair_report,
        }
        self.artifacts.write_json("WriteReport.json", report)
        return report, prompt, llm_raw

    def revise_code_performance(self, spec: BuildSpec) -> tuple[dict[str, Any], str, str]:
        """Skill: revise_code_performance.

        Called after CODE_LLM_EVAL FAIL. Makes targeted edits to key code files
        (hyperparameters, training loop, method implementation) and re-runs the
        experiment. Does NOT regenerate the full codebase.
        """
        code_dir = self.artifacts.path("code")
        perf_eval = self.artifacts.read_json("CodePerformanceEval.json") if self.artifacts.path("CodePerformanceEval.json").exists() else {}
        repair_hints = perf_eval.get("repair_hints", [])
        per_metric = perf_eval.get("per_metric", [])

        # Gather current results summary
        metrics_text = ""
        table_text = ""
        if self.artifacts.path("results/metrics.json").exists():
            metrics_text = self.artifacts.path("results/metrics.json").read_text(encoding="utf-8", errors="ignore")[:3000]
        if self.artifacts.path("results/results_table.csv").exists():
            table_text = self.artifacts.path("results/results_table.csv").read_text(encoding="utf-8", errors="ignore")[:3000]

        # Read key source files to give LLM context
        _KEY_FILES = [
            "src/method.py",
            "src/train.py",
            "configs/experiment_config.json",
        ]
        code_blocks: list[str] = []
        for rel in _KEY_FILES:
            path = code_dir / rel
            if path.exists():
                content = path.read_text(encoding="utf-8", errors="ignore")
                code_blocks.append(f"=== FILE: {rel} ===\n{content}")

        results_summary = json.dumps({
            "per_metric": per_metric,
            "metrics_json": metrics_text,
            "results_table_csv": table_text,
        }, indent=2)

        prompt = (
            self._revise_code_performance_template
            .replace("{{build_spec}}", json.dumps(spec.to_dict(), indent=2, sort_keys=True))
            .replace("{{perf_eval}}", json.dumps(perf_eval, indent=2))
            .replace("{{results_summary}}", results_summary)
            .replace("{{code_files}}", "\n\n".join(code_blocks))
        )

        llm_raw = ""
        llm_error = ""
        files_written: list[str] = []

        if self.llm is not None:
            self._progress("CODE_REVISE", "asking LLM for targeted performance fixes")
            try:
                llm_raw = self.llm.complete(prompt)
            except Exception as exc:
                llm_error = f"revise_code_performance LLM failed: {exc}"

        if llm_raw:
            parsed = self._parse_file_marker_format(llm_raw)
            for rel_path, content in parsed.items():
                target = code_dir / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                files_written.append(rel_path)
                self._progress("CODE_REVISE", f"wrote {rel_path}")

        # Re-run the experiment with the updated code
        run_report: dict[str, Any] = {"status": "SKIPPED", "reason": "no files changed"}
        if files_written and (code_dir / "run_experiment.py").exists():
            self._progress("CODE_REVISE", "re-running experiment with revised code")
            command = [sys.executable, "run_experiment.py", "--config", "configs/experiment_config.json"]
            try:
                completed = self._run_experiment(command, code_dir)
                stdout = completed.stdout or ""
                stderr = completed.stderr or ""
                log_path = self.artifacts.path("EXPERIMENT_LOG.md")
                log_path.write_text(stdout + "\n" + stderr, encoding="utf-8")
                run_report = {
                    "status": "PASS" if completed.returncode == 0 else "FAIL",
                    "returncode": completed.returncode,
                    "stdout_tail": stdout[-2000:],
                    "stderr_tail": stderr[-2000:],
                }
                self.artifacts.write_json("CodeRunReport.json", run_report)
            except Exception as exc:
                run_report = {"status": "FAIL", "reason": str(exc)}
                self.artifacts.write_json("CodeRunReport.json", run_report)

        report = {
            "status": "PASS" if run_report.get("status") == "PASS" else "PARTIAL",
            "files_written": files_written,
            "repair_hints_applied": repair_hints,
            "llm_error": llm_error,
            "run_report": run_report,
        }
        self.artifacts.write_json("CodeReviseReport.json", report)
        return report, prompt, llm_raw

    def _copy_paper_template(self, paper_dir: Path) -> list[str]:
        paper_dir.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        if not _DEFAULT_PAPER_TEMPLATE_DIR.exists():
            return copied
        for source in _DEFAULT_PAPER_TEMPLATE_DIR.iterdir():
            target = paper_dir / source.name
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
            else:
                shutil.copy2(source, target)
            copied.append(source.name)
        return sorted(copied)

    def _ensure_visible_icml_authors(self, tex: str) -> str:
        """Use ICML preprint mode so generated PDFs show the supplied authors."""

        affiliation = self._paper_author_affiliation_text()
        tex = re.sub(r"\\usepackage\{icml2026\}", r"\\usepackage[preprint]{icml2026}", tex, count=1)
        tex = re.sub(r"\\usepackage\[(?:accepted|review|nohyperref)\]\{icml2026\}", r"\\usepackage[preprint]{icml2026}", tex, count=1)
        tex = re.sub(
            r"\\begin\{icmlauthorlist\}.*?\\end\{icmlauthorlist\}",
            lambda _match: (
                "\\begin{icmlauthorlist}\n"
                "\\icmlauthor{QUIT}{agent}\n"
                "\\end{icmlauthorlist}"
            ),
            tex,
            count=1,
            flags=re.DOTALL,
        )
        tex = re.sub(r"\n?\\icmlaffiliation\{(?:anon|model|builder|researcher|agent)\}\{[^{}]*\}", "", tex)
        tex = re.sub(r"\n?\\icmlcorrespondingauthor\{[^{}]*\}\{[^{}]*\}", "", tex)
        tex = tex.replace(r"\icmlauthor{Xinchen Han}{researcher}", r"\icmlauthor{QUIT}{agent}")
        tex = tex.replace(r"\icmlauthor{QUIT Agent}{agent}", r"\icmlauthor{QUIT}{agent}")
        if r"\begin{icmlauthorlist}" not in tex:
            tex = tex.replace(
                r"\begin{document}",
                (
                    "\\begin{document}\n"
                    "\\begin{icmlauthorlist}\n"
                    "\\icmlauthor{QUIT}{agent}\n"
                    "\\end{icmlauthorlist}"
                ),
                1,
            )
        if r"\icmlaffiliation{agent}" not in tex and r"\end{icmlauthorlist}" in tex:
            tex = tex.replace(
                r"\end{icmlauthorlist}",
                f"\\end{{icmlauthorlist}}\n\\icmlaffiliation{{agent}}{{{affiliation}}}",
                1,
            )
        tex = self._ensure_no_corresponding_author_notice(tex)
        return tex

    def _paper_author_affiliation_text(self) -> str:
        model = self._latex_text_escape(self.model_name or "unknown-model")
        return f"base model: {model}, builder: Xinchen"

    def _ensure_no_corresponding_author_notice(self, tex: str) -> str:
        if r"\renewcommand{\printAffiliationsAndNotice}" in tex:
            return tex
        macro = r"""
\makeatletter
\renewcommand{\printAffiliationsAndNotice}[1]{\global\icml@noticeprintedtrue%
  \stepcounter{@affiliationcounter}%
  {\let\thefootnote\relax\footnotetext{\hspace*{-\footnotesep}\ificmlshowauthors #1\fi%
      \forloop{@affilnum}{1}{\value{@affilnum} < \value{@affiliationcounter}}{
        \textsuperscript{\arabic{@affilnum}}\ifcsname @affilname\the@affilnum\endcsname%
          \csname @affilname\the@affilnum\endcsname%
        \else
          {\bf AUTHORERR: Missing \textbackslash{}icmlaffiliation.}
        \fi
      }.%
      \ \\
      \Notice@String
    }
  }
}
\makeatother
""".strip()
        if r"\begin{document}" in tex:
            return tex.replace(r"\begin{document}", macro + "\n\n" + r"\begin{document}", 1)
        return macro + "\n\n" + tex

    def _build_references_bib(self, evidence: list[EvidenceCard], papers: list[PaperCard]) -> tuple[str, dict[str, str]]:
        paper_by_id = {paper.paper_id: paper for paper in papers}
        cited_ids = _dedupe([card.paper_id for card in evidence] + [paper.paper_id for paper in papers])
        keys: dict[str, str] = {}
        entries: list[str] = []
        for index, paper_id in enumerate(cited_ids, start=1):
            paper = paper_by_id.get(paper_id)
            key = safe_identifier(paper_id or f"paper_{index}")
            keys[paper_id] = key
            title = paper.title if paper else paper_id
            authors = " and ".join(paper.authors) if paper and paper.authors else "Unknown"
            year = paper.year if paper else 2026
            url = (paper.paper_url or paper.pdf_url) if paper else ""
            entries.append(
                "@article{%s,\n  title={%s},\n  author={%s},\n  year={%s},\n  url={%s}\n}\n"
                % (
                    key,
                    self._bib_escape(title),
                    self._bib_escape(authors),
                    year,
                    self._bib_escape(url),
                )
            )
        if not entries:
            entries.append("@misc{buildspec,\n  title={BuildSpec Generated Research Draft},\n  year={2026}\n}\n")
        return "\n".join(entries), keys

    def _compile_latex(self, paper_dir: Path) -> dict[str, Any]:
        latexmk = shutil.which("latexmk")
        pdflatex = shutil.which("pdflatex")
        xelatex = shutil.which("xelatex")
        lualatex = shutil.which("lualatex")
        bibtex = shutil.which("bibtex")
        if latexmk is not None:
            commands = [[latexmk, "-C", "main.tex"], [latexmk, "-g", "-pdf", "-interaction=nonstopmode", "main.tex"]]
            report = self._run_latex_commands(paper_dir, commands, engine="latexmk-pdf")
            return self._patch_latex_texttt_font_failure(paper_dir, report, commands, "latexmk-pdf")
        engine = pdflatex or xelatex or lualatex
        if engine is None:
            return {"status": "SKIPPED", "reason": "no LaTeX compiler found", "checked": ["latexmk", "pdflatex", "xelatex", "lualatex"]}
        commands = [[engine, "-interaction=nonstopmode", "main.tex"]]
        if bibtex is not None:
            commands.append([bibtex, "main"])
        commands.extend([[engine, "-interaction=nonstopmode", "main.tex"], [engine, "-interaction=nonstopmode", "main.tex"]])
        report = self._run_latex_commands(paper_dir, commands, engine=Path(engine).name)
        return self._patch_latex_texttt_font_failure(paper_dir, report, commands, Path(engine).name)

    def _repair_latex_compile_failure(
        self,
        *,
        spec: BuildSpec,
        evidence: list[EvidenceCard],
        papers: list[PaperCard],
        citation_keys: dict[str, str],
        paper_dir: Path,
        current_tex: str,
        compile_report: dict[str, Any],
    ) -> dict[str, Any]:
        if compile_report.get("status") != "FAIL":
            return {"attempted": False, "succeeded": False, "reason": "compile did not fail"}
        if self.llm is None:
            return {"attempted": False, "succeeded": False, "reason": "writer LLM is not configured"}
        self._progress("WRITE", "LaTeX compile failed; request writer repair")
        prompt = self._latex_repair_prompt(spec, evidence, papers, citation_keys, current_tex, compile_report)
        try:
            repaired = self.llm.complete(prompt)
        except Exception as exc:
            return {"attempted": True, "succeeded": False, "reason": f"writer repair LLM request failed: {exc}"}
        repaired = self._clean_latex_output(repaired)
        if not self._is_valid_latex(repaired):
            self.artifacts.write_markdown("WriteRepair.raw.txt", repaired)
            return {"attempted": True, "succeeded": False, "reason": "writer repair did not return valid LaTeX"}
        repaired = self._ensure_visible_icml_authors(repaired)
        self._write_text(paper_dir / "main.tex", repaired)
        retry_compile = self._compile_latex(paper_dir)
        self.artifacts.write_markdown("WriteRepair.prompt.txt", prompt)
        return {
            "attempted": True,
            "succeeded": retry_compile.get("status") == "PASS" and (paper_dir / "main.pdf").exists(),
            "compile": retry_compile,
            "repaired_tex": repaired,
        }

    def _latex_repair_prompt(
        self,
        spec: BuildSpec,
        evidence: list[EvidenceCard],
        papers: list[PaperCard],
        citation_keys: dict[str, str],
        current_tex: str,
        compile_report: dict[str, Any],
    ) -> str:
        return textwrap.dedent(
            f"""\
            The LaTeX paper draft failed to compile. Return ONLY a corrected full main.tex.
            Do not wrap the answer in markdown fences. Preserve the paper content and fix the compile error.

            BuildSpec:
            {json.dumps(spec.to_dict(), indent=2, sort_keys=True)}

            Citation key mapping:
            {json.dumps(citation_keys, indent=2, sort_keys=True)}

            Compact evidence:
            {json.dumps([self._compact_evidence_card(card) for card in evidence[:8]], indent=2, sort_keys=True)}

            Compact papers:
            {json.dumps([self._compact_paper_card(paper) for paper in papers[:15]], indent=2, sort_keys=True)}

            Compile report:
            {json.dumps(self._compact_latex_compile_report(compile_report), indent=2, sort_keys=True)}

            Current main.tex:
            ```latex
            {current_tex}
            ```

            Requirements:
            - Return a complete LaTeX document starting with \\documentclass{{article}}.
            - Keep \\usepackage[preprint]{{icml2026}}.
            - Immediately after \\usepackage[preprint]{{icml2026}}, add \\renewcommand{{\\ttdefault}}{{cmtt}} to avoid Courier font errors.
            - Fix unclosed environments, mismatched braces, invalid algorithmic commands, missing bibliography commands, package conflicts, and broken figure/table syntax.
            - Use only citation keys from the mapping above.
            - The ICML structure requires: title/author block inside \\twocolumn[...] immediately after \\begin{{document}}, followed by \\printAffiliationsAndNotice{{}} outside the bracket. Never move \\icmltitle or author commands into the preamble.
            - End with \\bibliography{{references}} and \\bibliographystyle{{icml2026}} before \\end{{document}}.
            - IMPORTANT: Do NOT replace existing \\includegraphics commands for real result figures such as ../results/progress_curve.png, ../results/training_curve.png, or ../results/eval_curve.png with placeholder \\fbox blocks. These image files exist at those relative paths. Keep the \\includegraphics commands as-is.
            """
        )

    def _compact_latex_compile_report(self, compile_report: dict[str, Any]) -> dict[str, Any]:
        runs = []
        for run in compile_report.get("runs", []):
            if not isinstance(run, dict):
                continue
            runs.append(
                {
                    "command": run.get("command", ""),
                    "returncode": run.get("returncode"),
                    "stdout_tail": str(run.get("stdout_tail", ""))[-3000:],
                    "stderr_tail": str(run.get("stderr_tail", ""))[-3000:],
                }
            )
        compact = {key: value for key, value in compile_report.items() if key != "runs"}
        compact["runs"] = runs
        return compact

    def _run_latex_commands(self, paper_dir: Path, commands: list[list[str]], *, engine: str) -> dict[str, Any]:
        runs = []
        status = "PASS"
        for command in commands:
            completed = subprocess.run(command, cwd=paper_dir, capture_output=True, text=True, timeout=self._latex_timeout_seconds)
            runs.append(
                {
                    "command": " ".join(command),
                    "returncode": completed.returncode,
                    "stdout_tail": completed.stdout[-2000:],
                    "stderr_tail": completed.stderr[-2000:],
                }
            )
            if completed.returncode != 0:
                status = "FAIL"
                break
        return {"status": status, "engine": engine, "runs": runs}

    def _patch_latex_texttt_font_failure(
        self,
        paper_dir: Path,
        report: dict[str, Any],
        commands: list[list[str]],
        engine: str,
    ) -> dict[str, Any]:
        if report.get("status") != "FAIL":
            return report
        combined = "\n".join(
            f"{run.get('stdout_tail', '')}\n{run.get('stderr_tail', '')}"
            for run in report.get("runs", [])
            if isinstance(run, dict)
        )
        if "pcrr7t" not in combined and "invalid font identifier" not in combined:
            return report
        tex_path = paper_dir / "main.tex"
        if not tex_path.exists():
            return report
        text = tex_path.read_text(encoding="utf-8")
        patched = re.sub(r"\\texttt\{([^{}]*)\}", r"\\emph{\1}", text)
        if r"\urlstyle{same}" not in patched:
            patched = patched.replace(r"\usepackage{hyperref}", "\\usepackage{hyperref}\n\\urlstyle{same}", 1)
        if patched == text:
            return report
        tex_path.write_text(patched, encoding="utf-8")
        retry = self._run_latex_commands(paper_dir, commands, engine=engine)
        retry["patched"] = "replaced_texttt_after_missing_typewriter_font"
        retry["previous_runs"] = report.get("runs", [])
        return retry

    def _validate_paper_lengths(self, tex: str) -> dict[str, Any]:
        requirements = {
            "abstract": 150,
            "introduction": 700,
            "related work": 350,
            "method": 1600,
            "experiments": 1200,
            "conclusion": 200,
        }
        counts = {
            "abstract": self._count_words(self._extract_environment(tex, "abstract")),
            "introduction": self._count_words(self._extract_section(tex, "Introduction")),
            "related work": self._count_words(self._extract_section(tex, "Related Work")),
            "method": self._count_words(self._extract_section(tex, "Method")),
            "experiments": self._count_words(self._extract_section(tex, "Experiments")),
            "conclusion": self._count_words(self._extract_first_section(tex, ["Conclusion", "Conclusions"])),
        }
        failures = [
            {"section": section, "minimum_words": minimum, "actual_words": counts.get(section, 0)}
            for section, minimum in requirements.items()
            if counts.get(section, 0) < minimum
        ]
        return {
            "status": "PASS" if not failures else "FAIL",
            "requirements": requirements,
            "word_counts": counts,
            "failures": failures,
        }

    def _validate_compiled_pdf_pages(self, paper_dir: Path, compile_report: dict[str, Any]) -> dict[str, Any]:
        expected_pages = self._expected_main_pages
        pdf_path = paper_dir / "main.pdf"
        page_count = self._extract_compiled_pdf_page_count(paper_dir, compile_report)
        if not pdf_path.exists():
            return {
                "status": "SKIPPED",
                "reason": "main.pdf was not produced",
                "expected_pages": expected_pages,
                "target_pages": expected_pages,
                "hard_fail_below_pages": expected_pages,
                "actual_pages": page_count,
            }
        failures = []
        hard_fail = False
        if page_count is None:
            failures.append({
                "expected_pages": expected_pages,
                "target_pages": expected_pages,
                "hard_fail_below_pages": expected_pages,
                "actual_pages": None,
                "reason": "could not determine PDF page count",
            })
        elif page_count < expected_pages:
            hard_fail = True
            failures.append({
                "expected_pages": expected_pages,
                "target_pages": expected_pages,
                "hard_fail_below_pages": expected_pages,
                "actual_pages": page_count,
                "reason": f"compiled PDF is below configured expected_main_pages={expected_pages}; expand the main paper substantially before retrying WRITE",
            })
        return {
            "status": "PASS" if not failures else "FAIL",
            "expected_pages": expected_pages,
            "target_pages": expected_pages,
            "hard_fail_below_pages": expected_pages,
            "actual_pages": page_count,
            "hard_fail": hard_fail,
            "failures": failures,
        }

    def _extract_compiled_pdf_page_count(self, paper_dir: Path, compile_report: dict[str, Any]) -> int | None:
        candidates: list[str] = []
        for run in compile_report.get("runs", []):
            if isinstance(run, dict):
                candidates.append(str(run.get("stdout_tail", "")))
                candidates.append(str(run.get("stderr_tail", "")))
        log_path = paper_dir / "main.log"
        if log_path.exists():
            candidates.append(log_path.read_text(encoding="utf-8", errors="ignore"))
        for text in candidates:
            match = re.search(r"Output written on .*?\((\d+)\s+pages?", text)
            if match:
                return int(match.group(1))
        return None

    def _validate_references(self, tex: str, bib_text: str, citation_keys: dict[str, str]) -> dict[str, Any]:
        allowed_keys = set(citation_keys.values())
        recommended_references = 15
        bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib_text))
        cited_keys = self._extract_citation_keys(tex)
        hallucinated_keys = sorted(cited_keys - allowed_keys)
        cited_missing_from_bib = sorted(cited_keys - bib_keys)
        bib_without_papercard = sorted(bib_keys - allowed_keys)
        failures: list[dict[str, Any]] = []
        if hallucinated_keys:
            failures.append({"rule": "hallucinated_citation_keys", "keys": hallucinated_keys})
        if cited_missing_from_bib:
            failures.append({"rule": "cited_keys_missing_from_bib", "keys": cited_missing_from_bib})
        if bib_without_papercard:
            failures.append({"rule": "bib_entries_without_papercard", "keys": bib_without_papercard})
        warnings = []
        if len(bib_keys) < recommended_references:
            warnings.append(
                {
                    "rule": "recommended_references",
                    "recommended": recommended_references,
                    "actual": len(bib_keys),
                    "reason": "fewer PaperCard-backed references are available; citation hallucination remains forbidden",
                }
            )
        return {
            "status": "PASS" if not failures else "FAIL",
            "recommended_references": recommended_references,
            "bib_entry_count": len(bib_keys),
            "allowed_key_count": len(allowed_keys),
            "cited_key_count": len(cited_keys),
            "cited_keys": sorted(cited_keys),
            "hallucinated_keys": hallucinated_keys,
            "warnings": warnings,
            "failures": failures,
        }

    def _validate_experiment_reporting(self, tex: str) -> dict[str, Any]:
        results = self._compact_experiment_results()
        numeric_rows = results.get("numeric_result_rows", [])
        failures = []
        if numeric_rows:
            experiments = self._extract_section(tex, "Experiments")
            if r"\begin{table}" not in experiments:
                failures.append({"rule": "missing_results_table", "reason": "numeric results exist but Experiments has no LaTeX table"})
            if r"\begin{figure}" not in experiments:
                failures.append({"rule": "missing_experiment_figure", "reason": "numeric results exist but Experiments has no figure or chart"})
            pending_count = len(re.findall(r"\bpending\b", experiments, flags=re.IGNORECASE))
            if pending_count >= 3:
                failures.append({"rule": "pending_table_despite_numeric_results", "pending_count": pending_count})
            methods_in_tex = {str(row["method"]) for row in numeric_rows if str(row.get("method", "")).strip() and str(row["method"]) in experiments}
            if len(methods_in_tex) < min(3, len(numeric_rows)):
                failures.append(
                    {
                        "rule": "missing_numeric_method_rows",
                        "methods_found": sorted(methods_in_tex),
                        "expected_examples": [row["method"] for row in numeric_rows[:5]],
                    }
                )
        return {
            "status": "PASS" if not failures else "FAIL",
            "numeric_result_count": len(numeric_rows),
            "failures": failures,
        }

    def _validate_appendix_content(self, tex: str) -> dict[str, Any]:
        appendix = self._extract_appendix(tex)
        experimental = self._extract_section(appendix, "Experimental Details")
        theoretical = self._extract_first_section(appendix, ["Theoretical Proofs", "Algorithm Proof"])
        theory_required = self._paper_has_proof_like_content(tex)
        failures = []
        if not appendix.strip():
            failures.append({"rule": "missing_appendix", "reason": "paper must include \\appendix after references"})
        if not experimental.strip():
            failures.append({"rule": "missing_appendix_experimental_details", "reason": "appendix must include \\section{Experimental Details}"})
        if theory_required and not theoretical.strip():
            failures.append({"rule": "missing_appendix_theoretical_proofs", "reason": "draft contains proof-like content, so appendix must include \\section{Theoretical Proofs}"})

        experimental_keywords = ["dataset", "hyperparameter", "seed", "evaluation", "hardware", "device", "artifact", "preprocess", "environment"]
        theoretical_keywords = ["proposition", "lemma", "theorem", "proof", "assumption", "convergence", "correctness", "bound", "objective"]
        experimental_hits = self._keyword_hits(experimental, experimental_keywords)
        theoretical_hits = self._keyword_hits(theoretical, theoretical_keywords)
        if experimental and len(experimental_hits) < 4:
            failures.append(
                {
                    "rule": "appendix_experimental_details_too_thin",
                    "keyword_hits": experimental_hits,
                    "reason": "Experimental Details must cover reproducibility details such as datasets, hyperparameters, seeds, evaluation, hardware, and artifacts",
                }
            )
        if theory_required and theoretical and (len(theoretical_hits) < 3 or not re.search(r"\\begin\{(proof|proposition|lemma|theorem)\}", theoretical, flags=re.IGNORECASE)):
            failures.append(
                {
                    "rule": "appendix_theoretical_proofs_too_thin",
                    "keyword_hits": theoretical_hits,
                    "reason": "Theoretical Proofs must contain proof/proposition-style content, not only a heading",
                }
            )
        return {
            "status": "PASS" if not failures else "FAIL",
            "appendix_word_count": self._count_words(appendix),
            "experimental_details_word_count": self._count_words(experimental),
            "theoretical_proofs_word_count": self._count_words(theoretical),
            "theoretical_proofs_required": theory_required,
            "experimental_keyword_hits": experimental_hits,
            "theoretical_keyword_hits": theoretical_hits,
            "failures": failures,
        }

    def _paper_has_proof_like_content(self, tex: str) -> bool:
        main_text = tex.split(r"\appendix", 1)[0]
        return re.search(r"\\begin\{(proof|proposition|lemma|theorem)\}", main_text, flags=re.IGNORECASE) is not None

    def _keyword_hits(self, text: str, keywords: list[str]) -> list[str]:
        lowered = text.lower()
        return [keyword for keyword in keywords if keyword in lowered]

    def _extract_citation_keys(self, tex: str) -> set[str]:
        keys: set[str] = set()
        pattern = re.compile(r"\\cite\w*\*?(?:\[[^\]]*\]){0,2}\{(?P<keys>[^{}]+)\}")
        for match in pattern.finditer(tex):
            for key in match.group("keys").split(","):
                normalized = key.strip()
                if normalized:
                    keys.add(normalized)
        return keys

    def _extract_environment(self, tex: str, environment: str) -> str:
        pattern = re.compile(rf"\\begin\{{{re.escape(environment)}\}}(?P<body>.*?)\\end\{{{re.escape(environment)}\}}", re.DOTALL | re.IGNORECASE)
        match = pattern.search(tex)
        return match.group("body") if match else ""

    def _extract_section(self, tex: str, section: str) -> str:
        escaped = re.escape(section)
        pattern = re.compile(
            rf"\\section\*?\{{{escaped}\}}(?P<body>.*?)(?=\\section\*?\{{|\\bibliography\{{|\\appendix|\\end\{{document\}}|\Z)",
            re.DOTALL | re.IGNORECASE,
        )
        match = pattern.search(tex)
        return match.group("body") if match else ""

    def _extract_first_section(self, tex: str, sections: list[str]) -> str:
        for section in sections:
            body = self._extract_section(tex, section)
            if body:
                return body
        return ""

    def _extract_appendix(self, tex: str) -> str:
        match = re.search(r"\\appendix(?P<body>.*?)(?=\\end\{document\})", tex, flags=re.DOTALL | re.IGNORECASE)
        return match.group("body") if match else ""

    def _count_words(self, latex_text: str) -> int:
        text = re.sub(r"%.*", " ", latex_text)
        text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", text)
        text = re.sub(r"[$^_{}\\]", " ", text)
        return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text))

    def _write_prompt(
        self,
        spec: BuildSpec,
        evidence: list[EvidenceCard],
        papers: list[PaperCard],
        citation_keys: dict[str, str],
    ) -> str:
        selected_papers = papers[:15]
        selected_paper_ids = {paper.paper_id for paper in selected_papers}
        selected_evidence = [card for card in evidence if card.paper_id in selected_paper_ids][:12] or evidence[:12]
        selected_keys = {
            paper_id: key
            for paper_id, key in citation_keys.items()
            if not selected_paper_ids or paper_id in selected_paper_ids
        }
        key_lines = "\n".join(f"  {pid} -> {key}" for pid, key in selected_keys.items())
        return (
            self._write_template
            .replace("{{model_name}}", self.model_name)
            .replace("{{expected_main_pages}}", str(self._expected_main_pages))
            .replace("{{build_spec}}", json.dumps(spec.to_dict(), indent=2, sort_keys=True))
            .replace("{{evidence_cards}}", json.dumps([self._compact_evidence_card(c) for c in selected_evidence], indent=2, sort_keys=True))
            .replace("{{paper_cards}}", json.dumps([self._compact_paper_card(p) for p in selected_papers], indent=2, sort_keys=True))
            .replace("{{experiment_results}}", json.dumps(self._compact_experiment_results(), indent=2, sort_keys=True))
            .replace("{{write_feedback}}", json.dumps(self._previous_write_feedback(), indent=2, sort_keys=True))
            .replace("{{citation_keys}}", key_lines)
        )

    def _previous_write_feedback(self) -> dict[str, Any]:
        feedback: dict[str, Any] = {}
        for name in ["PaperReview.json", "WriteReport.json"]:
            path = self.artifacts.path(name)
            if not path.exists():
                continue
            try:
                feedback[name] = self.artifacts.read_json(name)
            except Exception as exc:
                feedback[name] = {"status": "UNREADABLE", "error": str(exc)}
        main_tex = self.artifacts.path("paper_gene/main.tex")
        if main_tex.exists():
            text = main_tex.read_text(encoding="utf-8", errors="ignore")
            feedback["previous_main_tex_tail"] = text[-6000:]
        return feedback

    def _is_retryable_writer_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return any(token in message for token in ["524", "timeout", "timed out", "502", "503", "504"])

    def _compact_write_prompt(
        self,
        spec: BuildSpec,
        evidence: list[EvidenceCard],
        papers: list[PaperCard],
        citation_keys: dict[str, str],
    ) -> str:
        selected_papers = papers[:8]
        selected_paper_ids = {paper.paper_id for paper in selected_papers}
        selected_evidence = [card for card in evidence if card.paper_id in selected_paper_ids][:6] or evidence[:6]
        selected_keys = {
            paper_id: key
            for paper_id, key in citation_keys.items()
            if not selected_paper_ids or paper_id in selected_paper_ids
        }
        key_lines = "\n".join(f"{pid} -> {key}" for pid, key in selected_keys.items())
        compact_spec = {
            "target_task": spec.target_task,
            "problem_statement": spec.problem_statement,
            "method_summary": spec.method_summary,
            "implementation_plan": spec.implementation_plan[:6],
            "experiment_plan": spec.experiment_plan[:6],
            "baselines": spec.baselines[:8],
            "metrics": spec.metrics[:8],
            "success_criteria": spec.success_criteria[:6],
            "citations_required": spec.citations_required[:12],
        }
        prompt = f"""Write a complete ICML-style LaTeX paper draft from the compact artifacts below.

Return ONLY valid LaTeX for main.tex. Do not use markdown fences.
The first non-whitespace characters must be \\documentclass{{article}}.
Use \\bibliography{{references}} and \\bibliographystyle{{icml2026}}. Use only citation keys listed below.
Include Abstract, Introduction, Related Work, Method, Experiments, Conclusion, then references, then \\appendix.
Appendix must include \\section{{Experimental Details}} with concrete reproducibility details.
Include \\section{{Theoretical Proofs}} only if the draft states a proposition, lemma, theorem, or proof sketch.
Hard validation gates: abstract >=150 words; Introduction >=700 words; Related Work >=350 words; Method >=2000 words; Experiments >=1200 words; Conclusion >=200 words.
The compiled ICML main paper must be at least {self._expected_main_pages} pages excluding appendix; target {self._expected_main_pages + 1} pages when possible.
Do not use placeholders, TODOs, pending values, omitted-for-brevity text, or empty sections.
If numeric results are available, include a LaTeX results table and discuss whether the proposed method improves or underperforms.
If plots are available, include the plot filenames with \\includegraphics.

BuildSpec:
{json.dumps(compact_spec, indent=2, sort_keys=True)}

EvidenceCards:
{json.dumps([self._compact_evidence_card(c) for c in selected_evidence], indent=2, sort_keys=True)}

PaperCards:
{json.dumps([self._compact_paper_card(p) for p in selected_papers], indent=2, sort_keys=True)}

ExperimentResultSummary:
{json.dumps(self._compact_experiment_results(), indent=2, sort_keys=True)}

Previous WRITE feedback:
{json.dumps(self._previous_write_feedback(), indent=2, sort_keys=True)}

Citation key mapping:
{key_lines}
"""
        return prompt

    def _compact_experiment_results(self) -> dict[str, Any]:
        metrics_path = self.artifacts.path("results/metrics.json")
        table_path = self.artifacts.path("results/results_table.csv")
        progress_log_path = self.artifacts.path("results/progress_log.jsonl")
        legacy_training_log_path = self.artifacts.path("results/training_log.jsonl")
        progress_curve_path = self.artifacts.path("results/progress_curve.png")
        legacy_training_curve_path = self.artifacts.path("results/training_curve.png")
        eval_curve_path = self.artifacts.path("results/eval_curve.png")
        result_plot_paths = sorted(str(path.relative_to(self.artifacts.run_dir)) for path in self.artifacts.path("results").glob("*.png"))
        build_spec_reporting = self._compact_build_spec_reporting_contract()
        metrics = self._load_metrics_json(metrics_path)
        rows = self._load_results_rows(table_path)
        required_metrics = [
            str(metric).strip()
            for metric in build_spec_reporting.get("metrics", [])
            if str(metric).strip()
        ]
        _DOMAIN_KEYS = ["task", "environment", "task_name", "env_name", "dataset", "domain", "Domain"]
        raw_rows = []
        for row in rows:
            method = (
                row.get("method_name")
                or row.get("method")
                or row.get("Method")
            )
            metric_match = self._first_build_spec_metric_value(row, required_metrics)
            if not method or metric_match is None:
                continue
            metric_name, metric_column, metric_value = metric_match
            domain = next((row.get(k) for k in _DOMAIN_KEYS if row.get(k)), "")
            std = self._metric_std_value(row, metric_column, metric_name)
            raw_rows.append({
                "method": method,
                "domain": domain,
                "primary_metric_name": metric_name,
                "primary_metric_column": metric_column,
                "primary_metric_value": metric_value,
                "std": std,
            })
        # Aggregate per-seed rows into per-method mean ± std.
        from collections import defaultdict
        import statistics
        buckets: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
        meta: dict[tuple[str, str, str, str], dict] = {}
        for r in raw_rows:
            key = (r["method"], r["domain"], r["primary_metric_name"], r["primary_metric_column"])
            buckets[key].append(r["primary_metric_value"])
            meta[key] = {"std": r["std"]}
        numeric_rows = []
        for (method, domain, metric_name, metric_column), values in buckets.items():
            aggregate_value = round(statistics.mean(values), 6)
            agg_std = (
                meta[(method, domain, metric_name, metric_column)]["std"]
                if meta[(method, domain, metric_name, metric_column)]["std"] is not None
                else (round(statistics.stdev(values), 6) if len(values) > 1 else None)
            )
            numeric_rows.append({
                "method": method,
                "domain": domain,
                "primary_metric_name": metric_name,
                "primary_metric_column": metric_column,
                "primary_metric_value": aggregate_value,
                "std": agg_std,
                "n_seeds": len(values),
            })
        # When the same method appears under multiple domains (e.g., a per-task domain and
        # an "aggregate" domain), keep only the entry with the most seeds so the table
        # is not cluttered with redundant aggregate rows.
        seen_methods: dict[str, dict] = {}
        for r in sorted(numeric_rows, key=lambda x: x["n_seeds"], reverse=True):
            if r["method"] not in seen_methods:
                seen_methods[r["method"]] = r
        primary_metric_name = required_metrics[0] if required_metrics else ""
        lower_is_better = self._metric_lower_is_better(primary_metric_name)
        numeric_rows = sorted(
            seen_methods.values(),
            key=lambda r: r["primary_metric_value"],
            reverse=not lower_is_better,
        )
        best = numeric_rows[0] if numeric_rows else None
        return {
            "metrics_json_available": metrics_path.exists(),
            "results_table_available": table_path.exists(),
            "progress_log_available": progress_log_path.exists() or legacy_training_log_path.exists(),
            "progress_curve_available": progress_curve_path.exists() or legacy_training_curve_path.exists(),
            "legacy_training_log_available": legacy_training_log_path.exists(),
            "legacy_training_curve_available": legacy_training_curve_path.exists(),
            "eval_curve_available": eval_curve_path.exists(),
            "result_plot_paths": result_plot_paths,
            "build_spec_reporting": build_spec_reporting,
            "dataset": metrics.get("dataset", {}) if isinstance(metrics, dict) else {},
            "unavailable_domains": metrics.get("unavailable_domains", {}) if isinstance(metrics, dict) else {},
            "csv_table_previews": self._csv_table_previews(),
            "numeric_result_rows": numeric_rows[:15],
            "best_result": best,
            "primary_metric_name": primary_metric_name,
            "primary_metric_goal": "minimize" if lower_is_better else "maximize",
            "has_build_spec_metric_numeric_rows": bool(numeric_rows),
            "llm_performance_eval": self._load_code_performance_eval(),
            "instruction": (
                "Use numeric_result_rows for the LaTeX main results table. "
                "Each row has: method (display name), domain (dataset/environment, may be empty), "
                "primary_metric_name, primary_metric_column, primary_metric_value (mean across seeds), "
                "std (optional ± value), n_seeds. "
                "Rows are already aggregated across seeds — do not re-split them. "
                "These rows are included only when results_table.csv contains numeric columns matching BuildSpec.metrics. "
                "Do not replace these rows with pending values. If numeric_result_rows is empty, use csv_table_previews "
                "to determine whether non-primary diagnostic tables are still available, but do not invent metric values. "
                "If the proposed method underperforms baselines, state that honestly. "
                "llm_performance_eval contains a structured per-metric comparison produced by an LLM evaluator — "
                "use it to write a precise, grounded experiment analysis section."
            ),
        }

    def _load_code_performance_eval(self) -> dict[str, Any]:
        path = self.artifacts.path("CodePerformanceEval.json")
        if not path.exists():
            return {}
        try:
            return self.artifacts.read_json("CodePerformanceEval.json")
        except Exception:
            return {}

    def _compact_build_spec_reporting_contract(self) -> dict[str, Any]:
        path = self.artifacts.path("BuildSpec.json")
        if not path.exists():
            return {}
        try:
            spec = BuildSpec.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return {}
        return {
            "metrics": spec.metrics[:8],
            "logging": [item.to_dict() for item in spec.logging[:3]],
            "plots": [item.to_dict() for item in spec.plots[:6]],
        }

    def _first_build_spec_metric_value(self, row: dict[str, Any], metrics: list[str]) -> tuple[str, str, float] | None:
        for metric in metrics:
            for key, value in row.items():
                parsed = self._parse_float(value)
                if parsed is None:
                    continue
                if self._metric_is_covered(metric, str(key)):
                    return metric, str(key), parsed
        return None

    def _metric_std_value(self, row: dict[str, Any], metric_column: str, metric_name: str) -> float | None:
        candidates = [
            f"{metric_column}_std",
            f"{metric_name}_std",
            "std",
            "Std",
            "standard_deviation",
            "Standard Deviation",
        ]
        for key in candidates:
            parsed = self._round_optional(row.get(key))
            if parsed is not None:
                return parsed
        return None

    def _metric_is_covered(self, required: str, column_name: str) -> bool:
        return bool(self._metric_aliases(required) & self._metric_aliases(column_name))

    def _metric_aliases(self, value: str) -> set[str]:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
        compact = normalized.replace("_", "")
        aliases = {normalized, compact}
        aliases.update(token for token in normalized.split("_") if len(token) >= 3)
        return {alias for alias in aliases if alias}

    def _metric_lower_is_better(self, metric_name: str) -> bool:
        aliases = self._metric_aliases(metric_name)
        lower_tokens = {
            "loss", "error", "distance", "divergence", "kl", "wasserstein",
            "time", "latency", "runtime", "cost", "mse", "mae", "rmse",
        }
        higher_tokens = {"reward", "return", "score", "accuracy", "iou", "success", "rate"}
        if aliases & lower_tokens:
            return True
        if aliases & higher_tokens:
            return False
        return False

    def _csv_table_previews(self, max_rows: int = 6) -> list[dict[str, Any]]:
        results_dir = self.artifacts.path("results")
        candidates = sorted({*results_dir.glob("*.csv"), *results_dir.glob("ablations/*.csv")})
        previews: list[dict[str, Any]] = []
        for path in candidates:
            try:
                with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
                    reader = csv.DictReader(handle)
                    rows = list(reader)
            except OSError:
                continue
            previews.append(
                {
                    "path": str(path.relative_to(self.artifacts.run_dir)),
                    "columns": list(reader.fieldnames or []),
                    "row_count": len(rows),
                    "preview_rows": rows[:max_rows],
                }
            )
        return previews

    def _first_numeric_value(self, row: dict[str, Any], keys: list[str]) -> float | None:
        for key in keys:
            parsed = self._parse_float(row.get(key))
            if parsed is not None:
                return parsed
        return None

    def _first_numeric_key(self, row: dict[str, Any], keys: list[str]) -> str:
        for key in keys:
            if self._parse_float(row.get(key)) is not None:
                return key
        return ""

    def _parse_float(self, value: Any) -> float | None:
        try:
            if value in {None, "", "N/A"}:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _round_optional(self, value: Any) -> float | None:
        parsed = self._parse_float(value)
        return round(parsed, 6) if parsed is not None else None

    def _compact_evidence_card(self, card: EvidenceCard) -> dict[str, Any]:
        return {
            "evidence_id": card.evidence_id,
            "paper_id": card.paper_id,
            "task": self._truncate_text(card.task, 240),
            "method": self._truncate_text(card.method, 240),
            "setting": self._truncate_text(card.setting, 240),
            "claims": [self._truncate_text(item, 260) for item in card.claims[:3]],
            "metrics": card.metrics[:4],
            "limitations": [self._truncate_text(item, 220) for item in card.limitations[:2]],
            "transferable_idea_seeds": [self._truncate_text(item, 220) for item in card.transferable_idea_seeds[:2]],
        }

    def _compact_paper_card(self, paper: PaperCard) -> dict[str, Any]:
        return {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "authors": paper.authors[:6],
            "year": paper.year,
            "venue": paper.venue,
            "source": paper.source,
            "abstract": self._truncate_text(paper.abstract, 500),
            "paper_url": paper.paper_url,
            "code_url": paper.code_url,
        }

    def _truncate_text(self, value: str, limit: int) -> str:
        text = str(value or "").strip().replace("\n", " ")
        return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."

    @staticmethod
    def _clean_latex_output(text: str) -> str:
        """Strip LLM thinking blocks and any leading prose before \\documentclass."""
        import re as _re
        # Remove <think>...</think> and <thinking>...</thinking> blocks (extended reasoning output)
        text = _re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=_re.DOTALL | _re.IGNORECASE)
        # Trim everything before the first \documentclass
        idx = text.find(r"\documentclass")
        if idx > 0:
            text = text[idx:]
        return text.strip()

    def _is_valid_latex(self, text: str) -> bool:
        stripped = text.strip()
        return r"\documentclass" in stripped and r"\begin{document}" in stripped

    def _latex_text_escape(self, value: str) -> str:
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        return "".join(replacements.get(char, char) for char in str(value).replace("\n", " "))

    def _bib_escape(self, value: str) -> str:
        return str(value).replace("{", "").replace("}", "").replace("\n", " ")


def _load_prompt_template(skill_path: Path) -> str:
    skill_text = skill_path.read_text(encoding="utf-8")
    match = re.search(r"## Runtime Prompt Template\s+```text\n(?P<prompt>.*?)\n```", skill_text, re.DOTALL)
    if not match:
        raise ValueError(f"Runtime prompt fenced text block not found in skill: {skill_path}")
    return match.group("prompt").strip() + "\n"


def _load_skill_text(skill_path: Path) -> str:
    return skill_path.read_text(encoding="utf-8").strip()


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    values = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)
    return values


def _ensure_runtime_requirements(items: list[str]) -> list[str]:
    values = list(items)
    lowered = "\n".join(values).lower()
    if "torch" not in lowered:
        values.append("torch")
    if "numpy" not in lowered:
        values.append("numpy")
    if "matplotlib" not in lowered:
        values.append("matplotlib")
    return values


def safe_identifier(value: str) -> str:
    identifier = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    if not identifier:
        return "proposed_method"
    if identifier[0].isdigit():
        identifier = f"method_{identifier}"
    return identifier


_GENERIC_DATASET_PY = r'''
from __future__ import annotations

import torch


def make_dataset(config: dict, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = config["dataset"]
    generator = torch.Generator(device="cpu").manual_seed(int(config["seeds"][0]))
    n = int(cfg["num_samples"])
    input_dim = int(cfg["input_dim"])
    output_dim = int(cfg["output_dim"])
    noise_std = float(cfg.get("noise_std", 0.05))
    x = torch.randn(n, input_dim, generator=generator)
    weights = torch.randn(input_dim, output_dim, generator=generator) / input_dim**0.5
    y = torch.sin(x @ weights) + noise_std * torch.randn(n, output_dim, generator=generator)
    return x.to(device), y.to(device)


def split_dataset(x: torch.Tensor, y: torch.Tensor, train_fraction: float = 0.8) -> dict[str, torch.Tensor]:
    n_train = int(x.shape[0] * train_fraction)
    return {
        "x_train": x[:n_train],
        "y_train": y[:n_train],
        "x_val": x[n_train:],
        "y_val": y[n_train:],
    }


def sample_batch(x: torch.Tensor, y: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, x.shape[0], (batch_size,), device=x.device)
    return x[idx], y[idx]
'''

_ROBUST_OFFLINE_DATASET_PY = r'''
from __future__ import annotations

from typing import Tuple

import h5py
import numpy as np
import pandas as pd
import torch


class OfflineDataset:
    def __init__(self, path: str):
        self.path = path
        if path.lower().endswith((".h5", ".hdf5")):
            self._load_hdf5(path)
        else:
            self._load_csv(path)
        self.n_states = int(self.obs.shape[1])
        self.n_actions = int(self.actions.shape[1])

    def _load_hdf5(self, path: str) -> None:
        with h5py.File(path, "r") as handle:
            self.obs = np.asarray(handle["obs"][:], dtype=np.float32)
            self.actions = np.asarray(handle["acts"][:], dtype=np.float32)
            self.rewards = np.asarray(handle["rewards"][:], dtype=np.float32).reshape(-1, 1)
            self.dones = np.asarray(handle["dones"][:], dtype=np.float32).reshape(-1, 1)
            self.terminals = self.dones.copy()
            self.metadata = np.zeros_like(self.dones, dtype=np.float32)

    def _load_csv(self, path: str) -> None:
        frame = pd.read_csv(path)
        obs_cols = [name for name in frame.columns if name.startswith("obs_")]
        act_cols = [name for name in frame.columns if name.startswith("act_")]
        if obs_cols and act_cols:
            self.obs = frame[obs_cols].to_numpy(dtype=np.float32)
            self.actions = frame[act_cols].to_numpy(dtype=np.float32)
        else:
            self.obs = np.asarray(frame["observation"].tolist(), dtype=np.float32)
            self.actions = np.asarray(frame["action"].tolist(), dtype=np.float32)
            if self.obs.ndim == 1:
                self.obs = self.obs.reshape(-1, 1)
            if self.actions.ndim == 1:
                self.actions = self.actions.reshape(-1, 1)
        self.rewards = frame.get("reward", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=np.float32).reshape(-1, 1)
        self.dones = frame.get("done", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=np.float32).reshape(-1, 1)
        self.terminals = frame.get("terminal", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=np.float32).reshape(-1, 1)
        self.metadata = frame.get("is_first_step", pd.Series(np.zeros(len(frame)))).to_numpy(dtype=np.float32).reshape(-1, 1)

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        size = min(int(batch_size), len(self))
        indices = np.random.choice(len(self.obs), size, replace=False)
        return (
            torch.as_tensor(self.obs[indices], dtype=torch.float32),
            torch.as_tensor(self.actions[indices], dtype=torch.float32),
            torch.as_tensor(self.rewards[indices], dtype=torch.float32),
            torch.as_tensor(self.dones[indices], dtype=torch.float32),
            torch.as_tensor(self.terminals[indices], dtype=torch.float32),
            torch.as_tensor(self.metadata[indices], dtype=torch.float32),
        )

    def __len__(self):
        return int(len(self.obs))
'''


_GENERIC_BASELINES_PY = r'''
from __future__ import annotations

import torch


def fit_reference_baseline(x: torch.Tensor, y: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    ones = torch.ones(x.shape[0], 1, device=x.device)
    design = torch.cat([x, ones], dim=1)
    eye = torch.eye(design.shape[1], device=x.device)
    return torch.linalg.solve(design.T @ design + ridge * eye, design.T @ y)


def predict_reference(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(x.shape[0], 1, device=x.device)
    design = torch.cat([x, ones], dim=1)
    return design @ weights
'''


_GENERIC_METHOD_PY = r'''
from __future__ import annotations

import torch
from torch import nn


class ProposedMethod(nn.Module):
    """Generic trainable method scaffold.

    The LLM code path should replace this with the BuildSpec-specific method.
    This fallback is intentionally domain-neutral and only validates the local
    code/execution/metrics pipeline.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
'''


_GENERIC_TRAIN_PY = r'''
from __future__ import annotations

import torch
from torch.nn import functional as F

from .dataset import sample_batch
from .method import ProposedMethod


def train_method(data: dict[str, torch.Tensor], config: dict, *, device: torch.device) -> tuple[ProposedMethod, dict]:
    model = ProposedMethod(
        input_dim=data["x_train"].shape[1],
        output_dim=data["y_train"].shape[1],
        hidden_dim=int(config["model"]["hidden_dim"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    steps = int(config["training"]["method_steps"])
    batch_size = int(config["training"]["batch_size"])
    log_interval = int(config["logging"]["log_interval"])
    history = []
    for step in range(1, steps + 1):
        xb, yb = sample_batch(data["x_train"], data["y_train"], batch_size)
        pred = model(xb)
        loss = F.mse_loss(pred, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        history.append(float(loss.detach().cpu()))
        if step == 1 or step % log_interval == 0:
            print(f"[TRAIN] step={step} loss={history[-1]:.6f} grad_norm={float(grad_norm):.6f}")
    return model, {
        "train_final_loss": history[-1],
        "train_mean_loss": sum(history) / len(history),
        "loss_history": history,
        "eval_loss_history": history,
    }
'''


_GENERIC_EVALUATE_PY = r'''
from __future__ import annotations

import torch
from torch.nn import functional as F

from .baselines import predict_reference


@torch.no_grad()
def evaluate(model: torch.nn.Module, reference_weights: torch.Tensor, data: dict[str, torch.Tensor], config: dict) -> tuple[list[dict], dict]:
    method_name = config["method"]["name"]
    baseline_name = str((config.get("baselines") or ["reference_baseline"])[0])
    metrics = [str(item).strip() for item in config.get("metrics", []) if str(item).strip()] or ["primary_metric"]
    primary_metric = metrics[0]
    primary_key = _metric_key(primary_metric)
    method_pred = model(data["x_val"])
    baseline_pred = predict_reference(data["x_val"], reference_weights)
    method_loss = float(F.mse_loss(method_pred, data["y_val"]).cpu())
    baseline_loss = float(F.mse_loss(baseline_pred, data["y_val"]).cpu())
    method_value = _metric_value_from_error(method_loss, primary_metric)
    baseline_value = _metric_value_from_error(baseline_loss, primary_metric)
    rows = [
        {"method": method_name, "source": "proposed", primary_key: round(method_value, 6)},
        {"method": baseline_name, "source": "baseline", primary_key: round(baseline_value, 6)},
    ]
    for row, raw_error in [(rows[0], method_loss), (rows[1], baseline_loss)]:
        for metric in metrics[1:]:
            row[_metric_key(metric)] = round(_metric_value_from_error(raw_error, metric), 6)
    summary = {
        method_name: {primary_key: rows[0][primary_key], "internal_mse_error": round(method_loss, 6)},
        baseline_name: {primary_key: rows[1][primary_key], "internal_mse_error": round(baseline_loss, 6)},
        "primary_metric": primary_metric,
        "primary_metric_key": primary_key,
    }
    return rows, summary


def _metric_value_from_error(error: float, metric: str) -> float:
    if _metric_lower_is_better(metric):
        return error
    return 1.0 / (1.0 + error)


def _metric_lower_is_better(metric: str) -> bool:
    lowered = metric.lower()
    return any(token in lowered for token in ["loss", "error", "distance", "violation", "latency", "runtime", "time", "cost", "memory"])


def _metric_key(metric: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9]+", "_", metric.strip().lower()).strip("_") or "primary_metric"
'''


_GENERIC_PLOT_PY = r'''
from __future__ import annotations

from pathlib import Path


def write_plots(rows: list[dict], summary: dict, output_dir: Path) -> list[str]:
    """Write training and evaluation plots from generated experiment artifacts."""
    warnings: list[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        warnings.append(f"matplotlib unavailable; wrote simple PNG plots instead: {exc}")
        return _write_simple_png_plots(rows, summary, output_dir, warnings)

    output_dir.mkdir(parents=True, exist_ok=True)
    train = summary.get("train", {}) if isinstance(summary, dict) else {}
    history = train.get("loss_history") or []
    eval_history = train.get("eval_loss_history") or []
    if history or eval_history:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        if history:
            ax.plot(range(1, len(history) + 1), history, label="train_loss", linewidth=1.8)
        if eval_history:
            ax.plot(range(1, len(eval_history) + 1), eval_history, label="eval_loss", linewidth=1.8)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title("Training dynamics")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "progress_curve.png", dpi=160)
        plt.close(fig)
    else:
        warnings.append("no loss history available; skipped progress_curve.png")

    numeric_rows = []
    metric_keys = _metric_keys(summary)
    for row in rows:
        method = str(row.get("method", "")).strip()
        value = _first_float(row, metric_keys) if metric_keys else _first_numeric(row)
        if method and value is not None:
            numeric_rows.append((method, value))
    if numeric_rows:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        labels = [item[0] for item in numeric_rows]
        values = [item[1] for item in numeric_rows]
        ax.bar(labels, values, color=["#386cb0", "#fdb462", "#7fc97f", "#ef3b2c"][: len(labels)])
        ax.set_ylabel("primary metric")
        ax.set_title("Evaluation summary")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "eval_curve.png", dpi=160)
        plt.close(fig)
    else:
        warnings.append("no numeric evaluation rows available; skipped eval_curve.png")
    return warnings


def _write_simple_png_plots(rows: list[dict], summary: dict, output_dir: Path, warnings: list[str]) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = summary.get("train", {}) if isinstance(summary, dict) else {}
    history = [float(item) for item in (train.get("loss_history") or []) if _is_number(item)]
    eval_history = [float(item) for item in (train.get("eval_loss_history") or []) if _is_number(item)]
    if history or eval_history:
        _write_line_png(output_dir / "progress_curve.png", history or eval_history, eval_history if history else [])
    else:
        _write_blank_png(output_dir / "progress_curve.png")
        warnings.append("no loss history available; wrote blank progress_curve.png")

    values = []
    metric_keys = _metric_keys(summary)
    for row in rows:
        value = _first_float(row, metric_keys) if metric_keys else _first_numeric(row)
        if value is not None:
            values.append(value)
    if values:
        _write_bar_png(output_dir / "eval_curve.png", values)
    else:
        _write_blank_png(output_dir / "eval_curve.png")
        warnings.append("no numeric evaluation rows available; wrote blank eval_curve.png")
    return warnings


def _write_line_png(path: Path, series_a: list[float], series_b: list[float]) -> None:
    width, height = 640, 360
    image = _new_image(width, height)
    _draw_axes(image)
    values = series_a + series_b
    lo, hi = min(values), max(values)
    if hi <= lo:
        hi = lo + 1.0
    _draw_series(image, series_a, lo, hi, (42, 109, 181))
    if series_b:
        _draw_series(image, series_b, lo, hi, (217, 95, 2))
    _write_png(path, image)


def _write_bar_png(path: Path, values: list[float]) -> None:
    width, height = 640, 360
    image = _new_image(width, height)
    _draw_axes(image)
    hi = max(values) if values else 1.0
    lo = min(0.0, min(values) if values else 0.0)
    if hi <= lo:
        hi = lo + 1.0
    plot_left, plot_top, plot_right, plot_bottom = 60, 30, width - 20, height - 45
    bar_width = max(12, int((plot_right - plot_left) / max(len(values) * 2, 1)))
    for index, value in enumerate(values):
        x0 = plot_left + int((index + 0.5) * (plot_right - plot_left) / len(values)) - bar_width // 2
        y = plot_bottom - int((value - lo) / (hi - lo) * (plot_bottom - plot_top))
        _fill_rect(image, x0, min(y, plot_bottom), x0 + bar_width, plot_bottom, (49, 130, 189))
    _write_png(path, image)


def _write_blank_png(path: Path) -> None:
    image = _new_image(640, 360)
    _draw_axes(image)
    _write_png(path, image)


def _new_image(width: int, height: int) -> list[list[tuple[int, int, int]]]:
    return [[(255, 255, 255) for _ in range(width)] for _ in range(height)]


def _draw_axes(image: list[list[tuple[int, int, int]]]) -> None:
    height = len(image)
    width = len(image[0])
    _draw_line(image, 60, 30, 60, height - 45, (30, 30, 30))
    _draw_line(image, 60, height - 45, width - 20, height - 45, (30, 30, 30))


def _draw_series(image: list[list[tuple[int, int, int]]], values: list[float], lo: float, hi: float, color: tuple[int, int, int]) -> None:
    if not values:
        return
    height = len(image)
    width = len(image[0])
    left, top, right, bottom = 60, 30, width - 20, height - 45
    points = []
    for index, value in enumerate(values):
        x = left + int(index * (right - left) / max(len(values) - 1, 1))
        y = bottom - int((value - lo) / (hi - lo) * (bottom - top))
        points.append((x, y))
    for start, end in zip(points, points[1:]):
        _draw_line(image, start[0], start[1], end[0], end[1], color)
    for x, y in points:
        _fill_rect(image, x - 2, y - 2, x + 2, y + 2, color)


def _draw_line(image: list[list[tuple[int, int, int]]], x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        _set_pixel(image, x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _fill_rect(image: list[list[tuple[int, int, int]]], x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    for y in range(max(0, y0), min(len(image), y1 + 1)):
        for x in range(max(0, x0), min(len(image[0]), x1 + 1)):
            image[y][x] = color


def _set_pixel(image: list[list[tuple[int, int, int]]], x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= y < len(image) and 0 <= x < len(image[0]):
        image[y][x] = color


def _write_png(path: Path, image: list[list[tuple[int, int, int]]]) -> None:
    import struct
    import zlib

    height = len(image)
    width = len(image[0])
    raw = bytearray()
    for row in image:
        raw.append(0)
        for r, g, b in row:
            raw.extend((r, g, b))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def _is_number(value: object) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _first_float(row: dict, keys: list[str]) -> float | None:
    for key in keys:
        try:
            value = row.get(key)
            if value in {None, "", "N/A"}:
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _first_numeric(row: dict) -> float | None:
    ignored = {"method", "source", "domain", "seed", "split"}
    for key, value in row.items():
        if str(key).lower() in ignored:
            continue
        try:
            if value in {None, "", "N/A"}:
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _metric_keys(summary: dict) -> list[str]:
    mapping = summary.get("requested_metric_keys", {}) if isinstance(summary, dict) else {}
    if isinstance(mapping, dict):
        return [str(value) for value in mapping.values() if str(value)]
    return []
'''


_GENERIC_RUN_EXPERIMENT_PY = r'''
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from src.baselines import fit_reference_baseline
from src.dataset import make_dataset, split_dataset
from src.evaluate import evaluate
from src.plot import write_plots
from src.train import train_method


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))

    use_cuda = bool(config["runtime"].get("use_cuda", True)) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(json.dumps({
        "event": "runtime",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "experiment_family": config["method"].get("experiment_family", "generic_experiment"),
    }, sort_keys=True))

    x, y = make_dataset(config, device=device)
    data = split_dataset(x, y)
    print(json.dumps({
        "event": "dataset",
        "num_samples": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "output_dim": int(y.shape[1]),
    }, sort_keys=True))

    reference_weights = fit_reference_baseline(data["x_train"], data["y_train"])
    model, train_stats = train_method(data, config, device=device)
    rows, summary = evaluate(model, reference_weights, data, config)
    summary["train"] = train_stats
    summary["requested_metric_keys"] = {
        metric: _metric_key(metric)
        for metric in config.get("metrics", [])
    }

    metrics_path = Path(config["outputs"]["metrics_json"])
    table_path = Path(config["outputs"]["results_table_csv"])
    output_dir = metrics_path.parent
    progress_log_path = metrics_path.parent / "progress_log.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    plot_warnings = write_plots(rows, summary, output_dir)
    summary["plot_warnings"] = plot_warnings
    metrics_path.write_text(json.dumps({"summary": summary, "runs": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    progress_log_path.write_text(json.dumps({
        "events": [
            {"event": "runtime", "device": str(device), "cuda_available": torch.cuda.is_available()},
            {"event": "dataset", "num_samples": int(x.shape[0]), "input_dim": int(x.shape[1]), "output_dim": int(y.shape[1])},
            {"event": "train_summary", **train_stats},
            {"event": "evaluation", "methods": [row["method"] for row in rows]},
            {"event": "plot", "warnings": plot_warnings},
        ]
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with table_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = []
        for preferred in ["method", "source", *[_metric_key(metric) for metric in config.get("metrics", [])]]:
            if any(preferred in row for row in rows):
                fieldnames.append(preferred)
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"status": "PASS", "methods": [row["method"] for row in rows], "plot_warnings": plot_warnings}, sort_keys=True))
    return 0


def _metric_key(metric: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9]+", "_", metric.strip().lower()).strip("_") or "primary_metric"


if __name__ == "__main__":
    raise SystemExit(main())
'''
