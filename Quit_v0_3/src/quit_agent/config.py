from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quit_agent.providers.config import LLMProviderConfig
from quit_agent.providers.env_config import load_env_file
from quit_agent.providers.local_vllm import LocalVLLMServerConfig


DEFAULT_CONFIG_PATH = Path("config.json")


@dataclass(frozen=True)
class ProjectConfig:
    topic: str = ""
    objective: str = ""
    domain: str = ""
    constraints: list[str] = field(default_factory=list)
    project_id: str = ""

    def as_user_request(self) -> str:
        """Build a single prompt string from project fields."""
        parts = []
        if self.topic:
            parts.append(f"Topic: {self.topic}")
        if self.objective:
            parts.append(f"Objective: {self.objective}")
        if self.domain:
            parts.append(f"Domain: {self.domain}")
        if self.constraints:
            parts.append(f"Constraints: {', '.join(self.constraints)}")
        return "\n".join(parts) if parts else "Build an artifact-driven local-first research agent system."


@dataclass(frozen=True)
class RuntimeConfig:
    data_dir: Path = Path("runs")
    env_file: str = ".env"
    stop_after: str = "BUILD_SPEC"
    max_steps: int = 50


@dataclass(frozen=True)
class AgentConfig:
    max_tokens: int
    timeout_seconds: int
    temperature: float | None = None


@dataclass(frozen=True)
class AgentsConfig:
    planner: AgentConfig = field(default_factory=lambda: AgentConfig(max_tokens=1024, timeout_seconds=180))
    research: AgentConfig = field(default_factory=lambda: AgentConfig(max_tokens=2048, timeout_seconds=180))
    reviewer: AgentConfig = field(default_factory=lambda: AgentConfig(max_tokens=1024, timeout_seconds=180))
    builder: AgentConfig = field(default_factory=lambda: AgentConfig(max_tokens=4096, timeout_seconds=300))


@dataclass(frozen=True)
class RetrievalConfig:
    sources: list[str] = field(default_factory=lambda: ["local", "mock"])
    local_database_path: Path = Path("paper_database/local_papers")
    per_source_results: int = 8
    timeout_seconds: int = 20
    use_llm_query_planning: bool = False
    download_pdfs: bool = False
    max_downloads: int = 20
    min_downloads: int = 0
    relevance_selection_ratio: float = 0.7
    pdf_dir: Path = Path("paper_retrieve")


@dataclass(frozen=True)
class RunBudgetConfig:
    """Epoch bounds applied to ALL runs.

    The LLM chooses experiment-appropriate values within [min, max].
    min_* is a floor; max_* is a ceiling.
    train_epochs / eval_epochs are full dataset passes, not gradient steps.
    """
    min_train_epochs: int = 100
    max_train_epochs: int = 10000
    min_eval_epochs: int = 20
    max_eval_epochs: int = 200
    experiment_timeout_seconds: int = 120


@dataclass(frozen=True)
class DatasetConfig:
    """Controls dataset acquisition behaviour.

    The system tries to download a real dataset up to max_download_attempts times.
    If all attempts fail it generates synthetic fallback data and marks the run as a smoke run.
    """
    max_download_attempts: int = 3


@dataclass(frozen=True)
class WriteConfig:
    """Controls paper generation validation."""

    expected_main_pages: int = 7
    latex_timeout_seconds: int = 120


@dataclass(frozen=True)
class SearchBudgetConfig:
    max_queries: int = 4
    max_papers_screened: int = 50
    max_papers_selected: int = 20
    max_repo_checked: int = 5
    stop_if_no_new_signal_rounds: int = 2


@dataclass(frozen=True)
class BuildBudgetConfig:
    max_code_iterations: int = 3
    max_experiments: int = 5
    max_review_revisions: int = 2


@dataclass(frozen=True)
class ReadConfig:
    mode: str = "local_text"
    fallback_to_local_text: bool = False
    max_direct_pdf_mb: float = 10.0
    read_workers: int = 4


@dataclass(frozen=True)
class QuitAgentConfig:
    """Top-level configuration loaded from config.json."""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    llm: LLMProviderConfig = field(default_factory=LLMProviderConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    search_budget: SearchBudgetConfig = field(default_factory=SearchBudgetConfig)
    build_budget: BuildBudgetConfig = field(default_factory=BuildBudgetConfig)
    read: ReadConfig = field(default_factory=ReadConfig)
    run_budget: RunBudgetConfig = field(default_factory=RunBudgetConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    write: WriteConfig = field(default_factory=WriteConfig)
    # Kept for compatibility with older config.json files.
    agent_max_tokens: int = 8192
    builder_max_tokens: int = 16384
    local_vllm: LocalVLLMServerConfig | None = None

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> QuitAgentConfig:
        """Load config from a JSON file.

        Missing file → all defaults.
        Missing keys → individual field defaults.
        .env file is loaded first so api_key_env lookups work.
        """
        raw: dict[str, Any] = {}
        config_path = Path(path)
        if config_path.exists():
            raw = json.loads(config_path.read_text(encoding="utf-8"))

        # Load .env before resolving api keys
        runtime = _load_runtime_config(raw.get("runtime", {}))
        if runtime.env_file:
            load_env_file(runtime.env_file, override=False)

        project = _load_project_config(raw.get("project", {}))
        llm_raw = raw.get("llm", {})
        agent_max_tokens = int(llm_raw.get("agent_max_tokens", 8192))
        builder_max_tokens = int(llm_raw.get("builder_max_tokens", 16384))
        llm = _load_llm_config(llm_raw)
        agents = _load_agents_config(raw.get("agents", {}), llm)
        retrieval = _load_retrieval_config(raw.get("retrieval", {}))
        search_budget = _load_search_budget_config(raw.get("search_budget", {}))
        build_budget = _load_build_budget_config(raw.get("build_budget", {}))
        read = _load_read_config(raw.get("read", {}))

        vllm_raw = raw.get("local_vllm")
        local_vllm = _load_local_vllm_config(vllm_raw) if vllm_raw else None

        run_budget = _load_run_budget_config(raw.get("run_budget", raw.get("smoke", {})))
        dataset_cfg = _load_dataset_config(raw.get("dataset", {}))
        write_cfg = _load_write_config(raw.get("write", {}))

        return cls(
            project=project,
            runtime=runtime,
            llm=llm,
            agents=agents,
            retrieval=retrieval,
            search_budget=search_budget,
            build_budget=build_budget,
            read=read,
            run_budget=run_budget,
            dataset=dataset_cfg,
            write=write_cfg,
            agent_max_tokens=agent_max_tokens,
            builder_max_tokens=builder_max_tokens,
            local_vllm=local_vllm,
        )


# ── section loaders ───────────────────────────────────────────────────────────

def _load_project_config(raw: dict[str, Any]) -> ProjectConfig:
    constraints = raw.get("constraints", [])
    if isinstance(constraints, str):
        constraints = [c.strip() for c in constraints.split(",") if c.strip()]
    return ProjectConfig(
        topic=raw.get("topic", ""),
        objective=raw.get("objective", ""),
        domain=raw.get("domain", ""),
        constraints=list(constraints),
        project_id=raw.get("project_id", ""),
    )


def _load_runtime_config(raw: dict[str, Any]) -> RuntimeConfig:
    return RuntimeConfig(
        data_dir=Path(raw.get("data_dir", "runs")),
        env_file=raw.get("env_file", ".env"),
        stop_after=raw.get("stop_after", "BUILD_SPEC"),
        max_steps=int(raw.get("max_steps", 50)),
    )


def _load_llm_config(raw: dict[str, Any]) -> LLMProviderConfig:
    api_key_env = raw.get("api_key_env")
    api_key = raw.get("api_key") or (os.environ.get(api_key_env) if api_key_env else None)
    # agent_max_tokens is the default cap for all agents; builder gets its own
    max_tokens = int(raw.get("agent_max_tokens", raw.get("max_tokens", 8192)))
    provider_raw = raw.get("provider", "").lower().strip().replace("_", "-")
    return LLMProviderConfig(
        provider=provider_raw,
        model=raw.get("model", "claude-sonnet-4-6"),
        base_url=raw.get("base_url"),
        api_key=api_key,
        api_key_env=api_key_env,
        temperature=float(raw.get("temperature", 0.0)),
        max_tokens=max_tokens,
        timeout_seconds=int(raw.get("timeout_seconds", 60)),
        stream=_optional_bool(raw.get("stream")),
        extra_body=_load_extra_body(raw),
    )


def _load_agents_config(raw: dict[str, Any], llm: LLMProviderConfig) -> AgentsConfig:
    def load_agent(name: str, default_tokens: int, default_timeout: int) -> AgentConfig:
        item = raw.get(name, {})
        return AgentConfig(
            max_tokens=int(item.get("max_tokens", default_tokens)),
            timeout_seconds=int(item.get("timeout_seconds", default_timeout)),
            temperature=float(item["temperature"]) if "temperature" in item else None,
        )

    return AgentsConfig(
        planner=load_agent("planner", llm.max_tokens, llm.timeout_seconds),
        research=load_agent("research", llm.max_tokens, llm.timeout_seconds),
        reviewer=load_agent("reviewer", llm.max_tokens, llm.timeout_seconds),
        builder=load_agent("builder", max(llm.max_tokens, 4096), max(llm.timeout_seconds, 180)),
    )


def _load_retrieval_config(raw: dict[str, Any]) -> RetrievalConfig:
    sources = raw.get("sources", ["local", "mock"])
    if isinstance(sources, str):
        sources = [item.strip() for item in sources.split(",") if item.strip()]
    return RetrievalConfig(
        sources=list(sources),
        local_database_path=Path(raw.get("local_database_path", "paper_database/local_papers")),
        per_source_results=int(raw.get("per_source_results", 8)),
        timeout_seconds=int(raw.get("timeout_seconds", 20)),
        use_llm_query_planning=bool(raw.get("use_llm_query_planning", False)),
        download_pdfs=bool(raw.get("download_pdfs", False)),
        max_downloads=int(raw.get("max_downloads", 20)),
        min_downloads=int(raw.get("min_downloads", 0)),
        relevance_selection_ratio=float(raw.get("relevance_selection_ratio", 0.7)),
        pdf_dir=Path(raw.get("pdf_dir", "paper_retrieve")),
    )


def _load_search_budget_config(raw: dict[str, Any]) -> SearchBudgetConfig:
    return SearchBudgetConfig(
        max_queries=int(raw.get("max_queries", 4)),
        max_papers_screened=int(raw.get("max_papers_screened", 50)),
        max_papers_selected=int(raw.get("max_papers_selected", 20)),
        max_repo_checked=int(raw.get("max_repo_checked", 5)),
        stop_if_no_new_signal_rounds=int(raw.get("stop_if_no_new_signal_rounds", 2)),
    )


def _load_build_budget_config(raw: dict[str, Any]) -> BuildBudgetConfig:
    return BuildBudgetConfig(
        max_code_iterations=int(raw.get("max_code_iterations", 3)),
        max_experiments=int(raw.get("max_experiments", 5)),
        max_review_revisions=int(raw.get("max_review_revisions", 2)),
    )


def _load_read_config(raw: dict[str, Any]) -> ReadConfig:
    mode = str(raw.get("mode", "local_text")).lower().strip()
    if mode not in {"local_text", "direct_pdf"}:
        mode = "local_text"
    return ReadConfig(
        mode=mode,
        fallback_to_local_text=bool(raw.get("fallback_to_local_text", False)),
        max_direct_pdf_mb=float(raw.get("max_direct_pdf_mb", 10.0)),
        read_workers=int(raw.get("read_workers", 4)),
    )


def _load_local_vllm_config(raw: dict[str, Any]) -> LocalVLLMServerConfig:
    model_path = raw.get("model_path") or ""
    return LocalVLLMServerConfig(
        model_path=model_path,
        served_model_name=raw.get("served_model_name") or Path(model_path).name or "model",
        host=raw.get("host", "127.0.0.1"),
        port=int(_value_or_default(raw.get("port"), 8000)),
        gpu_memory_utilization=raw.get("gpu_memory_utilization"),
        gpu_memory_safety_margin=float(_value_or_default(raw.get("gpu_memory_safety_margin"), 0.10)),
        tensor_parallel_size=_value_or_default(raw.get("tensor_parallel_size"), 1),
        startup_timeout_seconds=int(
            _value_or_default(raw.get("startup_timeout_seconds"), _value_or_default(raw.get("vllm_startup_timeout"), 300))
        ),
        keep_alive=bool(raw.get("keep_alive", False)),
        stream=_optional_bool(raw.get("stream")),
        extra_body=_load_extra_body(raw),
    )


def _value_or_default(value: Any, default: Any) -> Any:
    return default if value is None else value


def _optional_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    raise ValueError("extra_body must be an object when provided.")


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ValueError("stream must be a boolean when provided.")


def _load_extra_body(raw: dict[str, Any]) -> dict[str, Any] | None:
    extra_body = _optional_dict(raw.get("extra_body")) or {}
    # Backward-compatible alias for older configs.
    chat_template_kwargs = _optional_dict(raw.get("chat_template_kwargs"))
    if chat_template_kwargs is not None:
        extra_body["chat_template_kwargs"] = chat_template_kwargs
    return extra_body or None


def _load_run_budget_config(raw: dict[str, Any]) -> RunBudgetConfig:
    return RunBudgetConfig(
        min_train_epochs=int(raw.get("min_train_epochs", 10)),
        max_train_epochs=int(raw.get("max_train_epochs", 50)),
        min_eval_epochs=int(raw.get("min_eval_epochs", 5)),
        max_eval_epochs=int(raw.get("max_eval_epochs", 20)),
        experiment_timeout_seconds=max(1, int(raw.get("experiment_timeout_seconds", 120))),
    )


def _load_dataset_config(raw: dict[str, Any]) -> DatasetConfig:
    return DatasetConfig(
        max_download_attempts=int(raw.get("max_download_attempts", 3)),
    )


def _load_write_config(raw: dict[str, Any]) -> WriteConfig:
    expected_main_pages = max(1, int(raw.get("expected_main_pages", 7)))
    latex_timeout_seconds = max(1, int(raw.get("latex_timeout_seconds", 120)))
    return WriteConfig(expected_main_pages=expected_main_pages, latex_timeout_seconds=latex_timeout_seconds)


def llm_config_for_agent(base: LLMProviderConfig, agent: AgentConfig, *, provider: str | None = None, model: str | None = None, base_url: str | None = None) -> LLMProviderConfig:
    return LLMProviderConfig(
        provider=provider or base.provider,
        model=model or base.model,
        base_url=base_url if base_url is not None else base.base_url,
        api_key=base.api_key,
        api_key_env=base.api_key_env,
        temperature=base.temperature if agent.temperature is None else agent.temperature,
        max_tokens=agent.max_tokens,
        timeout_seconds=agent.timeout_seconds,
        stream=base.stream,
        extra_body=base.extra_body,
    )
