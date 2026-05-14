from __future__ import annotations

import json

from quit_agent.config import QuitAgentConfig, llm_config_for_agent
from quit_agent.main import format_run_id, llm_needed_for_run, resolve_run_id
from quit_agent.schemas.enums import WorkflowState


def test_loads_runtime_and_per_agent_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "runtime": {"stop_after": "VALIDATE_BRIEF", "max_steps": 7},
                "llm": {
                    "provider": "anthropic",
                    "agent_max_tokens": 999,
                    "timeout_seconds": 88,
                    "stream": True,
                    "extra_body": {"top_p": 0.9},
                },
                "retrieval": {
                    "sources": ["local", "arxiv"],
                    "local_database_path": "papers.jsonl",
                    "use_llm_query_planning": True,
                },
                "agents": {
                    "planner": {"max_tokens": 123, "timeout_seconds": 45},
                    "builder": {"max_tokens": 456, "timeout_seconds": 90},
                },
                "run_budget": {"experiment_timeout_seconds": 321},
                "write": {"expected_main_pages": 6, "latex_timeout_seconds": 222},
            }
        ),
        encoding="utf-8",
    )

    config = QuitAgentConfig.load(path)
    planner_llm = llm_config_for_agent(config.llm, config.agents.planner)

    assert config.runtime.stop_after == "VALIDATE_BRIEF"
    assert config.runtime.max_steps == 7
    assert planner_llm.max_tokens == 123
    assert planner_llm.timeout_seconds == 45
    assert config.agents.builder.max_tokens == 456
    assert config.llm.stream is True
    assert config.llm.extra_body == {"top_p": 0.9}
    assert config.retrieval.sources == ["local", "arxiv"]
    assert config.retrieval.use_llm_query_planning is True
    assert config.run_budget.experiment_timeout_seconds == 321
    assert config.write.expected_main_pages == 6
    assert config.write.latex_timeout_seconds == 222


def test_retrieval_defaults_use_pdf_directories(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")

    config = QuitAgentConfig.load(path)

    assert str(config.retrieval.local_database_path) == "paper_database/local_papers"
    assert str(config.retrieval.pdf_dir) == "paper_retrieve"
    assert config.run_budget.experiment_timeout_seconds == 120
    assert config.write.expected_main_pages == 7
    assert config.write.latex_timeout_seconds == 120


def test_local_vllm_null_values_use_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "llm": {"provider": "local-vllm"},
                "local_vllm": {
                    "model_path": "../models/example",
                    "gpu_memory_utilization": None,
                    "gpu_memory_safety_margin": None,
                    "tensor_parallel_size": None,
                    "startup_timeout_seconds": None,
                    "stream": False,
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                },
            }
        ),
        encoding="utf-8",
    )

    config = QuitAgentConfig.load(path)

    assert config.local_vllm is not None
    assert config.local_vllm.gpu_memory_utilization is None
    assert config.local_vllm.gpu_memory_safety_margin == 0.10
    assert config.local_vllm.tensor_parallel_size == 1
    assert config.local_vllm.startup_timeout_seconds == 300
    assert config.local_vllm.stream is False
    assert config.local_vllm.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_chat_template_kwargs_is_backward_compatible_extra_body_alias(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "llm": {
                    "provider": "local-vllm",
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            }
        ),
        encoding="utf-8",
    )

    config = QuitAgentConfig.load(path)

    assert config.llm.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_resume_without_run_id_uses_latest_run_directory(tmp_path):
    (tmp_path / "20260428120000").mkdir()
    (tmp_path / "Offline RL - 20260428174457").mkdir()

    assert resolve_run_id(None, tmp_path, WorkflowState.CODE) == "Offline RL - 20260428174457"


def test_formats_new_run_id_with_project_prefix():
    assert format_run_id("Offline RL", "20260428174457") == "Offline RL - 20260428174457"


def test_run_id_project_prefix_is_path_safe():
    assert format_run_id("Offline/RL:Flow?", "20260428174457") == "Offline_RL_Flow_ - 20260428174457"


def test_llm_needed_uses_full_resume_window():
    config = QuitAgentConfig()

    assert llm_needed_for_run(WorkflowState.RETRIEVE, WorkflowState.WRITE, config) is True
    assert llm_needed_for_run(WorkflowState.CODE_EVAL, WorkflowState.WRITE, config) is True
    assert llm_needed_for_run(WorkflowState.RETRIEVE, WorkflowState.RETRIEVE, config) is False
