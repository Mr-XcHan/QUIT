from __future__ import annotations

from pydantic import Field

from .research_brief import JsonModel


class ArtifactsRequired(JsonModel):
    coder: list[str] = Field(default_factory=list)
    writer: list[str] = Field(default_factory=list)


class BuildEnvironment(JsonModel):
    """Environment contract used by CODE before generating experiment files."""

    source: str = "generated"
    reference_repo_url: str = ""
    reference_repo_path: str = ""
    env_files: list[str] = Field(default_factory=list)
    language: str = "python"
    framework: str = ""
    requirements: list[str] = Field(default_factory=lambda: ["python>=3.11"])
    setup_commands: list[str] = Field(default_factory=lambda: ["python -m venv .venv", "pip install -r requirements.txt"])
    # CODE always generates the standard standalone project; reference repos are optional context.
    code_strategy: str = "generate_fresh"


class ExperimentLogSpec(JsonModel):
    """Run-level logging contract consumed by CODE and WRITE."""

    path: str = "results/progress_log.jsonl"
    record_type: str = "execution"
    fields: list[str] = Field(default_factory=lambda: ["epoch", "primary_metric", "timestamp"])
    x_axis: str = "epoch"
    interval: str = "log_interval"
    description: str = "Domain-appropriate progress log; ML/DL/RL may use train/eval losses, but non-trainable methods should use real domain metrics."


class ExperimentPlotSpec(JsonModel):
    """Plot contract produced by CODE from logs, tables, or metrics."""

    path: str
    title: str
    source: str = "log"
    x: str = "epoch"
    y: str = "primary_metric"
    kind: str = "line"
    series: str = "method"
    description: str = ""


class BuildSpec(JsonModel):
    build_id: str
    idea_id: str
    target_task: str
    problem_statement: str
    method_summary: str
    repo_url: str = ""
    implementation_plan: list[str] = Field(default_factory=list)
    experiment_plan: list[str] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    logging: list[ExperimentLogSpec] = Field(default_factory=list)
    plots: list[ExperimentPlotSpec] = Field(
        default_factory=lambda: [
            ExperimentPlotSpec(
                path="results/progress_curve.png",
                title="Progress Over Experiment Steps",
                source="log",
                x="epoch",
                y="primary_metric",
                kind="line",
                description="Plot the BuildSpec-selected progress metric; ML/DL/RL may plot train/eval loss, but non-trainable methods must use real domain metrics.",
            ),
            ExperimentPlotSpec(
                path="results/eval_curve.png",
                title="Evaluation Metric Comparison",
                source="results_table",
                x="method",
                y="primary_metric",
                kind="bar",
                description="Compare proposed and baseline methods using the primary BuildSpec metric.",
            ),
        ]
    )
    success_criteria: list[str] = Field(default_factory=list)
    artifacts_required: ArtifactsRequired = Field(default_factory=ArtifactsRequired)
    environment: BuildEnvironment = Field(default_factory=BuildEnvironment)
    paper_outline: list[str] = Field(default_factory=list)
    citations_required: list[str] = Field(default_factory=list)
    # Epoch bounds injected from run_budget config; code generation must respect these.
    min_train_epochs: int = 100
    max_train_epochs: int = 10000
    min_eval_epochs: int = 20
    max_eval_epochs: int = 200
