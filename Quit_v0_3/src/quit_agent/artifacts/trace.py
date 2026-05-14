from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .io import read_json, write_json
from .paths import RunPaths


class TraceManager:
    """Writes per-step traces plus uniform prompt/response files."""

    def __init__(self, paths: RunPaths) -> None:
        self.paths = paths
        self.paths.trace_dir.mkdir(parents=True, exist_ok=True)
        self.paths.llm_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.paths.run_dir / "run_trace.json"
        if not self.summary_path.exists():
            write_json(self.summary_path, {"run_id": paths.run_id, "steps": []})

    def record_step(
        self,
        *,
        step_id: str,
        state: str,
        agent: str,
        skill: str,
        input_artifacts: list[str],
        output_artifacts: list[str],
        prompt_text: str,
        response_text: str,
        parsed_output_preview: Any,
        validation_result: Any,
        next_state: str | None,
        fallback_decision: str | None,
        started_at: str | None = None,
    ) -> Path:
        started = started_at or datetime.now(UTC).isoformat()
        finished = datetime.now(UTC).isoformat()
        base = f"{step_id}_{state.lower()}"
        prompt_path = self.paths.llm_dir / f"{base}_prompt.txt"
        response_path = self.paths.llm_dir / f"{base}_response.txt"
        prompt_path.write_text(prompt_text, encoding="utf-8")
        response_path.write_text(response_text, encoding="utf-8")
        trace_path = self.paths.trace_dir / f"{base}.json"
        trace = {
            "step_id": step_id,
            "state": state,
            "agent": agent,
            "skill": skill,
            "input_artifacts": input_artifacts,
            "output_artifacts": output_artifacts,
            "prompt_file": str(prompt_path.relative_to(self.paths.run_dir)),
            "response_file": str(response_path.relative_to(self.paths.run_dir)),
            "parsed_output_preview": parsed_output_preview,
            "validation_result": validation_result,
            "next_state": next_state,
            "fallback_decision": fallback_decision,
            "started_at": started,
            "finished_at": finished,
        }
        write_json(trace_path, trace)
        summary = read_json(self.summary_path)
        summary.setdefault("steps", []).append(
            {
                "step_id": step_id,
                "state": state,
                "agent": agent,
                "skill": skill,
                "trace_file": str(trace_path.relative_to(self.paths.run_dir)),
                "next_state": next_state,
                "fallback_decision": fallback_decision,
                "started_at": started,
                "finished_at": finished,
            }
        )
        write_json(self.summary_path, summary)
        return trace_path
