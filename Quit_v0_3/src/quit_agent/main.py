from __future__ import annotations

import argparse
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from quit_agent.agents.builder_agent import BuilderAgent
from quit_agent.agents.planner_agent import PlannerAgent
from quit_agent.agents.research_agent import ResearchAgent
from quit_agent.agents.reviewer_agent import ReviewerAgent
from quit_agent.artifacts.manager import ArtifactManager
from quit_agent.artifacts.trace import TraceManager
from quit_agent.config import QuitAgentConfig, llm_config_for_agent
from quit_agent.orchestrator.state_machine import StateMachine
from quit_agent.schemas.enums import WorkflowState
from quit_agent.providers.config import LLMProviderConfig
from quit_agent.providers.factory import create_llm_client
from quit_agent.providers.local_vllm import ManagedVLLMServer
from quit_agent.tools.repo_tools import RepoManager
from quit_agent.tools.retrievers import PdfDownloader, create_retriever


def _progress(message: str) -> None:
    print(message, flush=True)


class _MissingLLMClient:
    def complete(self, prompt: str) -> str:
        raise RuntimeError("No LLM provider configured. Set llm.provider in config.json.")

    def complete_with_files(self, prompt: str, file_paths: list[str]) -> str:
        raise RuntimeError("No LLM provider configured. Set llm.provider in config.json.")


def build_machine(run_id: str, config: QuitAgentConfig, llm_config: LLMProviderConfig | None) -> StateMachine:
    artifacts = ArtifactManager(config.runtime.data_dir, run_id)
    trace = TraceManager(artifacts.paths)
    planner_llm = (
        create_llm_client(llm_config_for_agent(llm_config, config.agents.planner))
        if llm_config is not None
        else _MissingLLMClient()
    )
    planner = PlannerAgent(planner_llm, artifacts, search_budget=config.search_budget, build_budget=config.build_budget)
    retriever = create_retriever(config.retrieval)
    pdf_dir = config.retrieval.pdf_dir if config.retrieval.pdf_dir.is_absolute() else artifacts.path(str(config.retrieval.pdf_dir))
    downloader = (
        PdfDownloader(pdf_dir, timeout_seconds=config.retrieval.timeout_seconds)
        if config.retrieval.download_pdfs
        else None
    )
    research_llm = (
        create_llm_client(llm_config_for_agent(llm_config, config.agents.research))
        if llm_config is not None and config.retrieval.use_llm_query_planning
        else None
    )
    read_llm = (
        create_llm_client(llm_config_for_agent(llm_config, config.agents.research))
        if llm_config is not None
        else None
    )
    researcher = ResearchAgent(
        retriever,
        artifacts,
        downloader=downloader,
        query_llm=research_llm,
        read_llm=read_llm,
        read_mode=config.read.mode,
        max_downloads=config.retrieval.max_downloads,
        min_downloads=config.retrieval.min_downloads,
        per_source_results=config.retrieval.per_source_results,
        relevance_selection_ratio=config.retrieval.relevance_selection_ratio,
        repo_manager=RepoManager(artifacts.path("repos"), timeout_seconds=config.retrieval.timeout_seconds),
        max_direct_pdf_mb=config.read.max_direct_pdf_mb,
        read_workers=config.read.read_workers,
    )
    reviewer_llm = (
        create_llm_client(llm_config_for_agent(llm_config, config.agents.reviewer))
        if llm_config is not None
        else None
    )
    reviewer = ReviewerAgent(artifacts, llm=reviewer_llm)
    builder_llm = (
        create_llm_client(llm_config_for_agent(llm_config, config.agents.builder))
        if llm_config is not None
        else None
    )
    builder_model_name = llm_config.model if llm_config is not None else "unknown-model"
    builder = BuilderAgent(
        artifacts,
        llm=builder_llm,
        model_name=builder_model_name,
        min_train_epochs=config.run_budget.min_train_epochs,
        max_train_epochs=config.run_budget.max_train_epochs,
        min_eval_epochs=config.run_budget.min_eval_epochs,
        max_eval_epochs=config.run_budget.max_eval_epochs,
        max_download_attempts=config.dataset.max_download_attempts,
        expected_main_pages=config.write.expected_main_pages,
        experiment_timeout_seconds=config.run_budget.experiment_timeout_seconds,
        latex_timeout_seconds=config.write.latex_timeout_seconds,
        reporter=_progress,
    )
    return StateMachine(
        artifacts=artifacts,
        trace=trace,
        planner=planner,
        researcher=researcher,
        reviewer=reviewer,
        builder=builder,
        reporter=_progress,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QUIT Agent.")
    parser.add_argument("request", nargs="*", help="Optional user request. Defaults to config.project.")
    parser.add_argument("--config", default="config.json", help="Path to config JSON.")
    parser.add_argument("--run-id", default=None, help="Optional run id. Defaults to '<project_id> - <UTC timestamp>'.")
    parser.add_argument(
        "--stop-after",
        choices=[state.value for state in WorkflowState],
        default=None,
        help="Execute through this state, then stop. Example: VALIDATE_BRIEF.",
    )
    parser.add_argument(
        "--start-at",
        choices=[state.value for state in WorkflowState],
        default="PLAN",
        help="Start execution from this state using artifacts already present in --run-id.",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum state executions before stopping.")
    return parser.parse_args(argv)


def format_run_id(project_id: str, timestamp: str) -> str:
    safe_project_id = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", project_id).strip()
    safe_project_id = re.sub(r"\s+", " ", safe_project_id).strip(" .")
    return f"{safe_project_id} - {timestamp}" if safe_project_id else timestamp


def _run_sort_key(path: Path) -> tuple[str, str]:
    match = re.search(r"(\d{14})$", path.name)
    timestamp = match.group(1) if match else path.name
    return (timestamp, path.name)


def resolve_run_id(args_run_id: str | None, data_dir: Path, start_at: WorkflowState, project_id: str = "") -> str:
    if args_run_id:
        return args_run_id
    if start_at == WorkflowState.PLAN:
        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        return format_run_id(project_id, timestamp)
    candidates = sorted((path for path in data_dir.iterdir() if path.is_dir()), key=_run_sort_key)
    if not candidates:
        raise ValueError(f"--start-at {start_at.value} needs an existing run under {data_dir} or an explicit --run-id.")
    return candidates[-1].name


_WORKFLOW_ORDER = [
    WorkflowState.PLAN,
    WorkflowState.VALIDATE_BRIEF,
    WorkflowState.RETRIEVE,
    WorkflowState.READ,
    WorkflowState.IDEATE,
    WorkflowState.IDEA_EVAL,
    WorkflowState.BUILD_SPEC,
    WorkflowState.CODE,
    WorkflowState.CODE_EVAL,
    WorkflowState.CODE_LLM_EVAL,
    WorkflowState.CODE_REVISE,
    WorkflowState.WRITE,
    WorkflowState.WRITE_EVAL,
    WorkflowState.WRITE_LLM_EVAL,
    WorkflowState.WRITE_REVISE,
    WorkflowState.WRITE_FINAL_EVAL,
    WorkflowState.EXTRACT,
]


def llm_needed_for_run(start_at: WorkflowState, stop_after: WorkflowState, config: QuitAgentConfig) -> bool:
    """Return whether any state in this requested execution window needs an LLM."""

    states_requiring_llm = {
        WorkflowState.PLAN,
        WorkflowState.VALIDATE_BRIEF,
        WorkflowState.READ,
        WorkflowState.IDEATE,
        WorkflowState.IDEA_EVAL,
        WorkflowState.BUILD_SPEC,
        WorkflowState.CODE,
        WorkflowState.CODE_EVAL,
        WorkflowState.CODE_LLM_EVAL,
        WorkflowState.CODE_REVISE,
        WorkflowState.WRITE,
        WorkflowState.WRITE_LLM_EVAL,
        WorkflowState.WRITE_REVISE,
    }
    if config.retrieval.use_llm_query_planning:
        states_requiring_llm.add(WorkflowState.RETRIEVE)

    try:
        start_index = _WORKFLOW_ORDER.index(start_at)
        stop_index = _WORKFLOW_ORDER.index(stop_after)
    except ValueError:
        return start_at in states_requiring_llm

    if stop_index < start_index:
        window = [start_at]
    else:
        window = _WORKFLOW_ORDER[start_index : stop_index + 1]
    return any(state in states_requiring_llm for state in window)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    config = QuitAgentConfig.load(args.config)

    # CLI args override config.project if provided
    user_request = " ".join(args.request) if args.request else config.project.as_user_request()
    stop_after = WorkflowState(args.stop_after or config.runtime.stop_after)
    start_at = WorkflowState(args.start_at)
    run_id = resolve_run_id(args.run_id, config.runtime.data_dir, start_at, config.project.project_id)
    max_steps = args.max_steps or config.runtime.max_steps

    provider = config.llm.provider.lower().strip()
    llm_needed = llm_needed_for_run(start_at, stop_after, config)
    if llm_needed and not provider:
        raise ValueError("No LLM provider configured. Set llm.provider in config.json.")

    if provider == "local-vllm" and llm_needed:
        if config.local_vllm is None:
            raise ValueError("provider='local-vllm' requires a [local_vllm] section in config.json.")
        with ManagedVLLMServer(config.local_vllm) as runtime:
            print(f"vLLM ready at {runtime.base_url} (model={runtime.served_model_name})")
            extra_body = config.local_vllm.extra_body
            if extra_body is None:
                extra_body = config.llm.extra_body
            stream = config.local_vllm.stream
            if stream is None:
                stream = config.llm.stream
            llm_config = LLMProviderConfig(
                provider="vllm",
                model=runtime.served_model_name,
                base_url=runtime.base_url,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
                timeout_seconds=config.llm.timeout_seconds,
                stream=stream,
                extra_body=extra_body,
            )
            result = build_machine(run_id, config, llm_config).run(
                user_request,
                start_at=start_at,
                stop_after=stop_after,
                max_steps=max_steps,
            )
    else:
        llm_config = config.llm if llm_needed else None
        result = build_machine(run_id, config, llm_config).run(
            user_request,
            start_at=start_at,
            stop_after=stop_after,
            max_steps=max_steps,
        )

    print(f"run_id={result.run_id}")
    print(f"run_dir={result.run_dir}")
    print(f"final_state={result.final_state.value}")
    print(f"next_state={result.next_state.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
