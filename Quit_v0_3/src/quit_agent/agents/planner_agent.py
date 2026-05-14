from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from quit_agent.artifacts.manager import ArtifactManager
from quit_agent.config import BuildBudgetConfig, SearchBudgetConfig
from quit_agent.tools.llm_interface import LLMClient


_REPLAN_SKILL_PATH = Path(__file__).resolve().parents[1] / "skills" / "planner" / "replan_research_brief.md"
_PLAN_SKILL_PATH = Path(__file__).resolve().parents[1] / "skills" / "planner" / "plan_research_brief.md"


class PlannerAgent:
    def __init__(
        self,
        llm: LLMClient,
        artifacts: ArtifactManager,
        search_budget: SearchBudgetConfig | None = None,
        build_budget: BuildBudgetConfig | None = None,
    ) -> None:
        self.llm = llm
        self.artifacts = artifacts
        self._search_budget = search_budget or SearchBudgetConfig()
        self._build_budget = build_budget or BuildBudgetConfig()
        self._plan_prompt_template = _load_prompt_template(_PLAN_SKILL_PATH)
        self._replan_prompt_template = _load_prompt_template(_REPLAN_SKILL_PATH)

    def _fill_budget_placeholders(self, prompt: str) -> str:
        sb = self._search_budget
        bb = self._build_budget
        return (
            prompt
            .replace("{{max_queries}}", str(sb.max_queries))
            .replace("{{max_papers_screened}}", str(sb.max_papers_screened))
            .replace("{{max_papers_selected}}", str(sb.max_papers_selected))
            .replace("{{max_repo_checked}}", str(sb.max_repo_checked))
            .replace("{{stop_if_no_new_signal_rounds}}", str(sb.stop_if_no_new_signal_rounds))
            .replace("{{max_code_iterations}}", str(bb.max_code_iterations))
            .replace("{{max_experiments}}", str(bb.max_experiments))
            .replace("{{max_review_revisions}}", str(bb.max_review_revisions))
        )

    def generate_brief(self, user_request: str) -> tuple[str, str, str]:
        """First-pass planning prompt.

        Use when: starting a run from a user request.
        Inputs: user prompt only.
        Output: raw ResearchBrief JSON candidate written to ResearchBrief.raw.json.
        Failure mode: malformed JSON is handled by VALIDATE_BRIEF repair/fallback.
        """
        prompt = self._fill_budget_placeholders(
            self._plan_prompt_template.replace("{{user_request}}", user_request)
        )
        response = self.llm.complete(prompt)
        self.artifacts.write_markdown("ResearchBrief.raw.json", response)
        return prompt, response, "ResearchBrief.raw.json"

    def regenerate_brief_after_failure(
        self,
        *,
        user_request: str,
        previous_raw_response: str,
        validation_report: dict[str, Any],
    ) -> tuple[str, str, str]:
        """Skill: replan_research_brief.

        Use when: the previous ResearchBrief failed parse/validation/repair.
        Inputs: original user request, previous raw output, validation report artifact.
        Output: replacement ResearchBrief.raw.json.
        Failure mode: repeated malformed JSON is bounded by the orchestrator.
        """
        prompt = self._fill_budget_placeholders(
            self._replan_prompt_template
            .replace("{{user_request}}", user_request)
            .replace("{{previous_raw_response}}", previous_raw_response)
            .replace("{{validation_report}}", json.dumps(validation_report, indent=2, sort_keys=True))
        )
        response = self.llm.complete(prompt)
        self.artifacts.write_markdown("ResearchBrief.raw.json", response)
        return prompt, response, "ResearchBrief.raw.json"


def _load_prompt_template(skill_path: Path) -> str:
    """Extract the text fenced block from a skill's runtime prompt section."""
    skill_text = skill_path.read_text(encoding="utf-8")
    match = re.search(r"## (?:Runtime )?Prompt Template\s+.*?```text\n(?P<prompt>.*?)\n```", skill_text, re.DOTALL)
    if not match:
        raise ValueError(f"Runtime prompt fenced text block not found in skill: {skill_path}")
    return match.group("prompt").strip() + "\n"
