from __future__ import annotations

import json
from json import JSONDecoder
from typing import Any

from pydantic import ValidationError

from quit_agent.artifacts.manager import ArtifactManager
from quit_agent.schemas.enums import ValidationStatus
from quit_agent.schemas.research_brief import (
    BuildBudget,
    FallbackPolicy,
    ResearchBrief,
    ResearchBriefValidationReport,
    SearchBudget,
)


class ResearchBriefValidator:
    """Parse, validate, repair, and report on ResearchBrief artifacts."""

    def __init__(self, artifacts: ArtifactManager) -> None:
        self.artifacts = artifacts

    def validate_or_repair(self, raw: str | dict[str, Any]) -> tuple[ResearchBrief | None, ResearchBriefValidationReport]:
        parsed, parse_errors = self._parse(raw)
        if parsed is None:
            report = ResearchBriefValidationReport(
                status=ValidationStatus.FAIL,
                errors=parse_errors,
                repair_attempted=True,
                repaired=False,
                fallback_target="PLAN",
            )
            self.artifacts.write_json("ResearchBriefValidationReport.json", report)
            return None, report
        try:
            brief = ResearchBrief.model_validate(parsed)
            report = ResearchBriefValidationReport(status=ValidationStatus.PASS, fallback_target="RETRIEVE")
            self.artifacts.write_json("ResearchBrief.json", brief)
            self.artifacts.write_json("ResearchBriefValidationReport.json", report)
            return brief, report
        except ValidationError as first_error:
            repaired_payload = self._repair_payload(parsed)
            try:
                brief = ResearchBrief.model_validate(repaired_payload)
                report = ResearchBriefValidationReport(
                    status=ValidationStatus.REPAIR,
                    errors=self._format_errors(first_error),
                    repair_attempted=True,
                    repaired=True,
                    fallback_target="RETRIEVE",
                )
                self.artifacts.write_json("ResearchBrief.json", brief)
                self.artifacts.write_json("ResearchBriefValidationReport.json", report)
                return brief, report
            except ValidationError as second_error:
                report = ResearchBriefValidationReport(
                    status=ValidationStatus.FAIL,
                    errors=self._format_errors(first_error) + self._format_errors(second_error),
                    repair_attempted=True,
                    repaired=False,
                    fallback_target="PLAN",
                )
                self.artifacts.write_json("ResearchBriefValidationReport.json", report)
                return None, report

    def _parse(self, raw: str | dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
        if isinstance(raw, dict):
            return raw, []
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            value = self._extract_json_object(raw)
            if value is None:
                return None, [f"invalid JSON: {exc}"]
        if not isinstance(value, dict):
            return None, ["ResearchBrief payload must be a JSON object"]
        return value, []

    def _extract_json_object(self, raw: str) -> dict[str, Any] | None:
        """Recover a JSON object from raw model output with extra text."""
        decoder = JSONDecoder()
        for index, char in enumerate(raw):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(raw[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return None

    def _repair_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        repaired = dict(payload)
        for key in ["domain", "constraints", "deliverable", "red_lines", "acceptance_criteria"]:
            repaired[key] = self._as_list(repaired.get(key, []))
        repaired.setdefault("topic", "untitled research request")
        if "objective" in repaired:
            repaired["objective"] = str(repaired["objective"])
        repaired["search_budget"] = self._merge_budget(SearchBudget().to_dict(), repaired.get("search_budget"))
        repaired["build_budget"] = self._merge_budget(BuildBudget().to_dict(), repaired.get("build_budget"))
        repaired["fallback_policy"] = self._merge_budget(FallbackPolicy().to_dict(), repaired.get("fallback_policy"))
        return repaired

    def _as_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)]

    def _merge_budget(self, defaults: dict[str, Any], value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return defaults
        merged = dict(defaults)
        for key, item in value.items():
            if key in merged and isinstance(merged[key], int):
                try:
                    merged[key] = int(item)
                except (TypeError, ValueError):
                    continue
            elif key in merged:
                merged[key] = str(item)
        return merged

    def _format_errors(self, error: ValidationError) -> list[str]:
        return [f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}" for item in error.errors()]
