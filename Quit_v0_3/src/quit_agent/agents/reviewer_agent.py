from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quit_agent.artifacts.manager import ArtifactManager
from quit_agent.schemas.build_spec import BuildSpec

_MIN_REPORTING_TARGETS = 5
from quit_agent.schemas.enums import FallbackTarget, IdeaDecisionType
from quit_agent.schemas.idea_card import IdeaCard
from quit_agent.schemas.research_brief import ResearchBrief
from quit_agent.schemas.review_artifacts import ExperimentAudit, IdeaDecision, LLMPaperReview, PaperReview
from quit_agent.tools.code_quality import evaluate_code_quality
from quit_agent.tools.llm_interface import LLMClient


_EVALUATE_IDEA_SKILL_PATH = Path(__file__).resolve().parents[1] / "skills" / "reviewer" / "evaluate_idea.md"
_EVALUATE_WRITE_LLM_SKILL_PATH = Path(__file__).resolve().parents[1] / "skills" / "reviewer" / "evaluate_write_llm.md"
_EXTRACT_SUMMARY_SKILL_PATH = Path(__file__).resolve().parents[1] / "skills" / "reviewer" / "extract_summary.md"
_EVALUATE_CODE_PERFORMANCE_SKILL_PATH = Path(__file__).resolve().parents[1] / "skills" / "reviewer" / "evaluate_code_performance.md"

_MAX_EXTRACTS = 5


class ReviewerAgent:
    def __init__(self, artifacts: ArtifactManager, llm: LLMClient | None = None) -> None:
        self.artifacts = artifacts
        self.llm = llm
        self._idea_review_template = _load_prompt_template(_EVALUATE_IDEA_SKILL_PATH)
        self._write_llm_review_template = _load_prompt_template(_EVALUATE_WRITE_LLM_SKILL_PATH)
        self._extract_summary_template = _load_prompt_template(_EXTRACT_SUMMARY_SKILL_PATH)
        self._evaluate_code_performance_template = _load_prompt_template(_EVALUATE_CODE_PERFORMANCE_SKILL_PATH)

    def evaluate_idea(self, brief: ResearchBrief, ideas: list[IdeaCard]) -> tuple[IdeaDecision, str, str]:
        """Skill: isolated_idea_review.

        Use when: candidate ideas need independent acceptance.
        Inputs: ResearchBrief.json, IdeaLibrary.jsonl, red lines, acceptance criteria.
        Output: IdeaDecision.json.
        Failure mode: routes to IDEATE, READ, or RETRIEVE based only on explicit decision fields.
        """
        prompt = self._idea_review_prompt(brief, ideas)
        if self.llm is not None:
            raw = self.llm.complete(prompt)
            decision = self._parse_decision(raw)
            if decision is not None:
                self.artifacts.write_json("IdeaDecision.json", decision)
                return decision, prompt, raw
            self.artifacts.write_markdown("IdeaDecision.raw.txt", raw)

        decision = self._fallback_idea_decision(brief, ideas)
        self.artifacts.write_json("IdeaDecision.json", decision)
        return decision, prompt, decision.to_json_text()

    def _fallback_idea_decision(self, brief: ResearchBrief, ideas: list[IdeaCard]) -> IdeaDecision:
        if not ideas:
            return IdeaDecision(
                idea_id="",
                decision=IdeaDecisionType.REJECT,
                reason="no ideas available",
                fallback_target=FallbackTarget.READ,
                missing_evidence=["no evidence-backed ideas"],
            )
        idea = ideas[0]
        violations = [
            red_line for red_line in brief.red_lines if red_line.lower() in idea.novelty_claim.lower()
        ]
        if violations:
            return IdeaDecision(
                idea_id=idea.idea_id,
                decision=IdeaDecisionType.REVISE,
                reason="idea violates red lines",
                fallback_target=FallbackTarget.IDEATE,
                violations=violations,
                required_changes=["remove red-line violation and regenerate idea"],
            )
        if not idea.supporting_evidence_ids:
            return IdeaDecision(
                idea_id=idea.idea_id,
                decision=IdeaDecisionType.REJECT,
                reason="missing supporting evidence",
                fallback_target=FallbackTarget.READ,
                missing_evidence=["supporting_evidence_ids"],
            )
        return IdeaDecision(
            idea_id=idea.idea_id,
            decision=IdeaDecisionType.PASS,
            reason="idea is evidence-backed and does not violate red lines",
            fallback_target=FallbackTarget.BUILD_SPEC,
        )

    def _idea_review_prompt(self, brief: ResearchBrief, ideas: list[IdeaCard]) -> str:
        return (
            self._idea_review_template.replace("{{research_brief}}", json.dumps(brief.to_dict(), indent=2, sort_keys=True))
            .replace("{{idea_library}}", json.dumps([idea.to_dict() for idea in ideas], indent=2, sort_keys=True))
        )

    def _parse_decision(self, raw: str) -> IdeaDecision | None:
        value = self._extract_json_object(raw)
        if not isinstance(value, dict):
            return None
        try:
            return IdeaDecision.model_validate(value)
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

    def evaluate_code(self) -> ExperimentAudit:
        if not self.artifacts.path("BuildSpec.json").exists():
            audit = ExperimentAudit(status="FAIL", failures=["BuildSpec.json is missing"], fallback_target=FallbackTarget.BUILD_SPEC)
            self.artifacts.write_json("ExperimentAudit.json", audit)
            return audit

        spec = self.artifacts.load_model("BuildSpec.json", BuildSpec)
        code_report = self.artifacts.read_json("CodeRunReport.json") if self.artifacts.path("CodeRunReport.json").exists() else {}
        stdout = ""
        if self.artifacts.path("EXPERIMENT_LOG.md").exists():
            stdout = self.artifacts.path("EXPERIMENT_LOG.md").read_text(encoding="utf-8", errors="ignore")
        quality = evaluate_code_quality(
            run_dir=self.artifacts.run_dir,
            spec=spec,
            returncode=code_report.get("returncode"),
            stdout=stdout,
        )
        self.artifacts.write_json("CodeEvalQualityReport.json", quality)
        failures = [
            f"{failure.get('rule', 'code_quality_failed')}: {failure.get('reason', '')}"
            for failure in quality.get("failures", [])
        ]
        if code_report.get("status") == "FAIL":
            failures.extend(str(error) for error in code_report.get("errors", []))
        audit = ExperimentAudit(
            status="PASS" if not failures else "FAIL",
            failures=failures,
            fallback_target=FallbackTarget.CODE,
        )
        self.artifacts.write_json("ExperimentAudit.json", audit)
        return audit

    def evaluate_code_performance(self) -> dict[str, Any] | None:
        """LLM-based judgment: does the proposed method outperform baselines overall?

        Returns the parsed verdict dict, or None if skipped (no LLM / no results).
        verdict == "FAIL" means CODE should be retried with the repair_hints.
        """
        if self.llm is None:
            return None

        metrics_path = self.artifacts.path("results/metrics.json")
        table_path = self.artifacts.path("results/results_table.csv")
        if not metrics_path.exists() or not table_path.exists():
            return None

        spec = self.artifacts.load_model("BuildSpec.json", BuildSpec) if self.artifacts.path("BuildSpec.json").exists() else None
        build_spec_json = json.dumps(spec.to_dict(), indent=2, sort_keys=True) if spec else "{}"
        metrics_json = metrics_path.read_text(encoding="utf-8", errors="ignore")
        results_table = table_path.read_text(encoding="utf-8", errors="ignore")

        prompt = (
            self._evaluate_code_performance_template
            .replace("{{build_spec}}", build_spec_json)
            .replace("{{metrics_json}}", metrics_json[:4000])
            .replace("{{results_table}}", results_table[:6000])
        )

        try:
            raw = self.llm.complete(prompt)
            value = self._extract_json_object(raw)
            if isinstance(value, dict) and value.get("verdict") in {"PASS", "FAIL"}:
                self.artifacts.write_json("CodePerformanceEval.json", value)
                return value
        except Exception:
            pass
        return None

    def evaluate_write(self) -> PaperReview:
        report = self.artifacts.read_json("WriteReport.json") if self.artifacts.path("WriteReport.json").exists() else {}
        failures: list[str] = []
        spec = self.artifacts.load_model("BuildSpec.json", BuildSpec) if self.artifacts.path("BuildSpec.json").exists() else None
        if not self.artifacts.path("paper_gene/main.tex").exists():
            failures.append("paper_gene/main.tex is missing")
            tex = ""
        else:
            tex = self.artifacts.path("paper_gene/main.tex").read_text(encoding="utf-8", errors="ignore")
        if report.get("status") != "PASS":
            reason = str(report.get("reason") or "WriteReport status is not PASS")
            failures.append(reason)
            compile_report = report.get("compile", {})
            if isinstance(compile_report, dict) and compile_report.get("status") not in {None, "PASS"}:
                failures.append(f"compile {compile_report.get('status')}: {compile_report.get('reason') or compile_report.get('engine') or ''}".strip())
            for key in ["length_validation", "reference_validation", "experiment_validation", "appendix_validation", "page_validation"]:
                validation = report.get(key, {})
                if isinstance(validation, dict) and validation.get("status") == "FAIL":
                    failures.append(f"{key} failed: {validation.get('failures', [])}")
        page_validation = report.get("page_validation", {})
        if isinstance(page_validation, dict) and page_validation.get("status") == "FAIL":
            actual_pages = page_validation.get("actual_pages")
            expected_pages = page_validation.get(
                "expected_pages",
                page_validation.get("target_pages", page_validation.get("minimum_pages", 8)),
            )
            if page_validation.get("hard_fail"):
                failures.append(
                    f"compiled paper is direct FAIL: {actual_pages} pages is below configured expected page threshold {expected_pages}; "
                    f"expand the paper and retry WRITE"
                )
            else:
                failures.append(f"compiled paper is too short: {actual_pages} pages, expected at least {expected_pages}")
        if spec is None:
            failures.append("BuildSpec.json is missing")
        elif tex:
            failures.extend(self._evaluate_paper_against_build_spec(spec, tex))
            failures.extend(self._evaluate_paper_result_artifacts(tex))
        review = PaperReview(
            status="PASS" if not failures else "FAIL",
            failures=failures,
            fallback_target=FallbackTarget.WRITE,
        )
        self.artifacts.write_json("PaperReview.json", review)
        return review

    def evaluate_write_llm(self) -> LLMPaperReview | None:
        """Soft LLM review called only after hard WRITE_EVAL passes.

        Patches the existing PaperReview.json with the llm_review block so that
        the next WRITE_REVISE step automatically receives improvement_hints via
        _previous_write_feedback().  Always returns (never blocks progression).
        """
        tex = ""
        if self.artifacts.path("paper_gene/main.tex").exists():
            tex = self.artifacts.path("paper_gene/main.tex").read_text(encoding="utf-8", errors="ignore")
        spec = self.artifacts.load_model("BuildSpec.json", BuildSpec) if self.artifacts.path("BuildSpec.json").exists() else None
        llm_result = self._run_write_llm_review(tex, spec)

        # Patch PaperReview.json in place so _previous_write_feedback picks it up
        if llm_result is not None and self.artifacts.path("PaperReview.json").exists():
            existing = self.artifacts.read_json("PaperReview.json")
            existing["llm_review"] = llm_result.to_dict()
            self.artifacts.write_json("PaperReview.json", existing)

        return llm_result

    def _run_write_llm_review(self, tex: str, spec: BuildSpec | None) -> LLMPaperReview | None:
        if self.llm is None or not tex:
            return None
        build_spec_json = json.dumps(spec.to_dict(), indent=2, sort_keys=True) if spec else "{}"
        write_report_json = (
            json.dumps(self.artifacts.read_json("WriteReport.json"), indent=2, sort_keys=True)
            if self.artifacts.path("WriteReport.json").exists()
            else "{}"
        )
        tex_excerpt = tex[:24000] if len(tex) > 24000 else tex
        prompt = (
            self._write_llm_review_template
            .replace("{{build_spec}}", build_spec_json)
            .replace("{{paper_tex}}", tex_excerpt)
            .replace("{{write_report}}", write_report_json)
        )
        try:
            raw = self.llm.complete(prompt)
            value = self._extract_json_object(raw)
            if isinstance(value, dict):
                return LLMPaperReview.model_validate(value)
        except Exception:
            pass
        return None

    def extract_summary(self) -> dict[str, Any] | None:
        """Call LLM to extract a structured research summary and append it to the per-topic queue.

        Returns the extracted entry dict, or None if extraction could not be performed.
        """
        if self.llm is None:
            return None

        tex = ""
        if self.artifacts.path("paper_gene/main.tex").exists():
            tex = self.artifacts.path("paper_gene/main.tex").read_text(encoding="utf-8", errors="ignore")
        if not tex:
            return None

        spec = self.artifacts.load_model("BuildSpec.json", BuildSpec) if self.artifacts.path("BuildSpec.json").exists() else None
        llm_review_dict: dict[str, Any] = {}
        if self.artifacts.path("PaperReview.json").exists():
            existing = self.artifacts.read_json("PaperReview.json")
            llm_review_dict = existing.get("llm_review") or {}

        build_spec_json = json.dumps(spec.to_dict(), indent=2, sort_keys=True) if spec else "{}"
        llm_review_json = json.dumps(llm_review_dict, indent=2, sort_keys=True)
        tex_excerpt = tex[:24000] if len(tex) > 24000 else tex

        prompt = (
            self._extract_summary_template
            .replace("{{build_spec}}", build_spec_json)
            .replace("{{llm_review}}", llm_review_json)
            .replace("{{paper_tex}}", tex_excerpt)
        )

        extracted: dict[str, Any] | None = None
        try:
            raw = self.llm.complete(prompt)
            value = self._extract_json_object(raw)
            if isinstance(value, dict):
                extracted = value
        except Exception:
            pass

        if extracted is None:
            return None

        topic = spec.target_task if spec else ""
        project_key = _topic_to_filename(topic) or "general"
        extracts_dir = self.artifacts.paths.extracts_dir
        extracts_dir.mkdir(parents=True, exist_ok=True)
        extracts_file = extracts_dir / f"{project_key}.json"

        queue: list[dict[str, Any]] = []
        if extracts_file.exists():
            try:
                queue = json.loads(extracts_file.read_text(encoding="utf-8"))
                if not isinstance(queue, list):
                    queue = []
            except Exception:
                queue = []

        entry: dict[str, Any] = {
            "run_id": self.artifacts.paths.run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            **{k: extracted.get(k, "") for k in ("contribution", "method_novelty", "key_results", "limitations", "future_directions")},
        }
        queue.append(entry)
        if len(queue) > _MAX_EXTRACTS:
            queue = queue[-_MAX_EXTRACTS:]

        extracts_file.write_text(json.dumps(queue, indent=2, ensure_ascii=False), encoding="utf-8")
        return entry

    def _evaluate_paper_against_build_spec(self, spec: BuildSpec, tex: str) -> list[str]:
        failures: list[str] = []
        normalized_tex = _normalize_text(tex)
        missing_metrics = [
            metric for metric in spec.metrics
            if metric and not _text_mentions_metric(normalized_tex, metric)
        ]
        if missing_metrics:
            failures.append(f"paper_missing_build_spec_metrics: {missing_metrics}")

        expected_plot_paths = [
            str(plot.path).strip()
            for plot in spec.plots
            if str(plot.path).strip()
        ]
        missing_plots: list[str] = []
        for plot_path in expected_plot_paths:
            artifact_path = _run_relative_path(self.artifacts.run_dir, plot_path)
            if not artifact_path.exists():
                continue
            candidates = {
                plot_path,
                "../" + plot_path if not plot_path.startswith("../") else plot_path,
                Path(plot_path).name,
            }
            if not any(candidate in tex for candidate in candidates):
                missing_plots.append(plot_path)
        if missing_plots:
            failures.append(f"paper_missing_build_spec_plots: {missing_plots}")

        reporting_targets = len([m for m in spec.metrics if m]) + len(expected_plot_paths)
        if reporting_targets >= _MIN_REPORTING_TARGETS:
            mentioned_metrics = sum(1 for metric in spec.metrics if _text_mentions_metric(normalized_tex, metric))
            mentioned_plots = len(expected_plot_paths) - len(missing_plots)
            if mentioned_metrics + mentioned_plots < min(_MIN_REPORTING_TARGETS, reporting_targets):
                failures.append(
                    "paper_under_discusses_build_spec_reporting: "
                    f"mentioned {mentioned_metrics + mentioned_plots} of {reporting_targets} metrics/plots"
                )
        return failures

    def _evaluate_paper_result_artifacts(self, tex: str) -> list[str]:
        failures: list[str] = []
        png_paths = sorted(self.artifacts.path("results").glob("*.png"))
        missing_pngs = []
        for path in png_paths:
            rel = f"../results/{path.name}"
            if rel not in tex:
                missing_pngs.append(f"results/{path.name}")
        if missing_pngs:
            failures.append(f"paper_missing_result_pngs: {missing_pngs}")

        csv_paths = sorted(self.artifacts.path("results").glob("*.csv"))
        if csv_paths:
            if "\\begin{table" not in tex or "\\begin{tabular" not in tex:
                failures.append(f"paper_missing_csv_latex_tables: {[f'results/{path.name}' for path in csv_paths]}")
            else:
                missing_csv_content = []
                for path in csv_paths:
                    try:
                        header = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
                    except (OSError, IndexError):
                        continue
                    columns = [column.strip() for column in header.split(",") if column.strip()]
                    if columns and not any(_latex_text_contains(tex, column) for column in columns):
                        missing_csv_content.append(f"results/{path.name}")
                if missing_csv_content:
                    failures.append(f"paper_table_missing_csv_columns: {missing_csv_content}")
        return failures


def _run_relative_path(run_dir: Path, raw_path: str) -> Path:
    path = Path(str(raw_path).strip())
    if not str(path):
        return run_dir / "results" / "artifact"
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "..":
        path = Path(*parts[1:])
        parts = path.parts
    if parts and parts[0] == "code":
        path = Path(*parts[1:])
    return run_dir / path


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower())


def _text_mentions_metric(normalized_tex: str, metric: str) -> bool:
    aliases = _metric_text_aliases(metric)
    return any(alias and alias in normalized_tex for alias in aliases)


def _metric_text_aliases(metric: str) -> set[str]:
    raw = str(metric).strip().lower()
    if not raw:
        return set()
    without_parenthetical = re.sub(r"\([^)]*\)", "", raw).strip()
    aliases = {_normalize_text(raw).strip(), _normalize_text(without_parenthetical).strip()}
    snake = re.sub(r"[^a-z0-9]+", "_", without_parenthetical).strip("_")
    aliases.add(snake.replace("_", " "))
    aliases.update(token for token in _normalize_text(without_parenthetical).split() if len(token) >= 4)
    return {alias for alias in aliases if alias}


def _latex_text_contains(tex: str, text: str) -> bool:
    normalized_tex = _normalize_text(tex)
    normalized_text = _normalize_text(text).strip()
    return bool(normalized_text and normalized_text in normalized_tex)


def _topic_to_filename(topic: str) -> str:
    normalized = re.sub(r"[^\w\s-]", "", topic.lower()).strip()
    normalized = re.sub(r"[\s_-]+", "_", normalized)
    return normalized[:80]


def _load_prompt_template(skill_path: Path) -> str:
    skill_text = skill_path.read_text(encoding="utf-8")
    match = re.search(r"## Runtime Prompt Template\s+```text\n(?P<prompt>.*?)\n```", skill_text, re.DOTALL)
    if not match:
        raise ValueError(f"Runtime prompt fenced text block not found in skill: {skill_path}")
    return match.group("prompt").strip() + "\n"
