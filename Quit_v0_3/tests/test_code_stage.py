from __future__ import annotations

import json
import shutil
import subprocess

from quit_agent.agents.builder_agent import BuilderAgent
from quit_agent.artifacts.manager import ArtifactManager
from quit_agent.schemas.build_spec import ArtifactsRequired, BuildEnvironment, BuildSpec
from quit_agent.schemas.evidence_card import EvidenceCard
from quit_agent.tools.code_quality import evaluate_code_quality


def test_code_stage_generates_standalone_project_and_results(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="sample-efficient offline RL",
        problem_statement="Improve offline RL under shift.",
        method_summary="Use pessimistic flow matching.",
        implementation_plan=["implement method"],
        experiment_plan=["run smoke experiment"],
        baselines=["BCQ", "CQL"],
        metrics=["normalized return", "robustness under shift"],
        success_criteria=["beat baseline"],
        artifacts_required=ArtifactsRequired(coder=["working code module"], writer=[]),
        citations_required=["p1"],
    )

    report, prompt, log = builder.code(spec)

    assert "BuildSpec" in prompt
    assert report.status == "PASS"
    assert report.executed is True
    assert report.generation_mode == "generic_fallback_no_llm"
    assert report.fallback_used is True
    assert "results/metrics.json" in report.outputs
    assert "results/results_table.csv" in report.outputs
    assert "code/requirements.txt" in report.outputs
    assert "code/environment.yml" in report.outputs
    assert "code/ENVIRONMENT.md" in report.outputs
    assert "code/EXPERIMENT_METRICS.md" in report.outputs
    assert "EnvironmentResolutionReport.json" in report.outputs
    assert "DeviceReport.json" in report.outputs
    assert artifacts.path("code/run_experiment.py").exists()
    assert artifacts.path("code/ENVIRONMENT.md").exists()
    assert artifacts.path("code/EXPERIMENT_METRICS.md").exists()
    assert artifacts.path("code/src/method.py").exists()
    assert artifacts.path("code/src/plot.py").exists()
    assert artifacts.path("results/metrics.json").exists()
    assert artifacts.path("results/results_table.csv").exists()
    assert artifacts.path("results/progress_curve.png").exists()
    assert artifacts.path("results/eval_curve.png").exists()
    assert artifacts.path("EXPERIMENT_LOG.md").exists()
    assert artifacts.path("CodeRunReport.json").exists()
    results_table = artifacts.path("results/results_table.csv").read_text(encoding="utf-8")
    assert "normalized_return" in results_table
    assert "robustness_under_shift" in results_table
    assert "validation_loss" not in results_table
    assert "baseline_validation_loss" not in results_table
    assert "linear_baseline" not in results_table
    assert artifacts.read_json("EnvironmentResolutionReport.json")["resolution"] == "generated_environment"
    device_report = artifacts.read_json("DeviceReport.json")
    selected_device = device_report["selection"]["selected"]
    assert selected_device == "cpu" or selected_device.startswith("cuda")
    experiment_config = artifacts.read_json("code/configs/experiment_config.json")
    assert experiment_config["device"] == selected_device
    assert experiment_config["runtime"]["resolved_device"] == selected_device
    metrics_markdown = artifacts.path("code/EXPERIMENT_METRICS.md").read_text(encoding="utf-8")
    assert "Experiment Dashboard" in metrics_markdown
    assert "Final Performance" in metrics_markdown
    assert "Progress / Evaluation Log" in metrics_markdown
    assert "Errors" in metrics_markdown
    assert "Experiment Log" in log


def test_code_prompt_uses_reference_repo_as_context_only(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    repo_path = tmp_path / "reference_repo"
    repo_path.mkdir()
    (repo_path / "README.md").write_text("Reference repo for flow matching experiments.\n", encoding="utf-8")
    (repo_path / "train.py").write_text("def train_reference_model():\n    return 'reference'\n", encoding="utf-8")
    builder = BuilderAgent(artifacts)
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="sample-efficient offline RL",
        problem_statement="Improve offline RL under shift.",
        method_summary="Use pessimistic flow matching.",
        environment=BuildEnvironment(
            source="reference_repo",
            reference_repo_path=str(repo_path),
            code_strategy="generate_fresh",
        ),
    )

    prompt = builder._code_prompt(spec)

    assert "Optional Reference Repository Context" in prompt
    assert "Reference repo for flow matching experiments" in prompt
    assert "Use these excerpts only as implementation reference" in prompt
    assert "up to 3 times" in prompt
    assert "select_torch_device(requested or \"auto\")" in prompt
    assert "runtime.resolved_device" in prompt
    assert "never pass `\"auto\"` or `\"gpu\"` directly to `torch.device`" in prompt
    assert "=== FILE: run_experiment.py ===" in prompt
    assert "code_from_build_spec_adapt_repo" not in prompt


def test_llm_code_writer_strips_redundant_code_prefix(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    code_dir = artifacts.path("code")
    raw = """=== FILE: code/src/dataset.py ===
DATASET = True
=== FILE: code/configs/experiment_config.json ===
{"ok": true}
=== FILE: code/CoreImplementationReport.json ===
{"status": "PASS"}
"""

    assert builder._write_llm_code_project(raw, code_dir, require_full_project=False) is True

    assert artifacts.path("code/src/dataset.py").read_text(encoding="utf-8") == "DATASET = True"
    assert artifacts.path("code/configs/experiment_config.json").exists()
    assert artifacts.path("code/CoreImplementationReport.json").exists()
    assert not artifacts.path("code/code").exists()


def test_stage_prompt_tells_model_not_to_prefix_code_dir(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="offline RL",
        problem_statement="problem",
        method_summary="method",
    )

    prompt = builder._stage_code_prompt(
        stage_name="dataset",
        stage_template="Implement dataset.",
        spec=spec,
        contract={},
        code_dir=artifacts.path("code"),
        expected_paths=["src/dataset.py"],
    )

    assert "never prefix them with `code/`" in prompt
    assert "Use `src/dataset.py`, not `code/src/dataset.py`" in prompt
    assert "never write ```python or ``` inside a file replacement" in prompt


def test_llm_code_writer_strips_markdown_fences(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    code_dir = artifacts.path("code")
    raw = """=== FILE: src/plot.py ===
```python
VALUE = 1
```
"""

    assert builder._write_llm_code_project(raw, code_dir, require_full_project=False) is True

    assert artifacts.path("code/src/plot.py").read_text(encoding="utf-8") == "VALUE = 1"
    compile(artifacts.path("code/src/plot.py").read_text(encoding="utf-8"), "plot.py", "exec")


def test_stage_code_context_is_scoped_by_stage(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    code_dir = artifacts.path("code")
    artifacts.path("code/configs/experiment_config.json").parent.mkdir(parents=True)
    artifacts.path("code/src").mkdir(parents=True)
    artifacts.path("code/configs/experiment_config.json").write_text('{"dataset": "cfg"}\n', encoding="utf-8")
    artifacts.path("code/src/dataset.py").write_text("DATASET_API = 'dataset'\n", encoding="utf-8")
    artifacts.path("code/src/method.py").write_text("METHOD_API = 'method'\n" + "x = 1\n" * 1000, encoding="utf-8")
    artifacts.path("code/src/baselines.py").write_text("BASELINE_API = 'baseline'\n", encoding="utf-8")
    artifacts.path("code/src/evaluate.py").write_text("EVAL_API = 'eval'\n", encoding="utf-8")
    artifacts.path("code/run_experiment.py").write_text("RUNNER_API = 'runner'\n", encoding="utf-8")

    core_context = builder._current_code_context_for_stage("core", code_dir)
    assert '{"dataset": "cfg"}' in core_context
    assert "DATASET_API" not in core_context
    assert "METHOD_API" not in core_context
    assert "BASELINE_API" not in core_context

    experiment_context = builder._current_code_context_for_stage("experiment", code_dir)
    assert "DATASET_API" in experiment_context
    assert "METHOD_API" in experiment_context
    assert "BASELINE_API" in experiment_context
    assert "RUNNER_API" not in experiment_context
    assert "EVAL_API" not in experiment_context


def test_code_eval_quality_tool_rejects_empty_successful_results(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="controllable synthesis",
        problem_statement="problem",
        method_summary="method",
        baselines=["baseline"],
    )
    artifacts.path("results").mkdir(parents=True)
    artifacts.path("results/metrics.json").write_text("{}\n", encoding="utf-8")
    artifacts.path("results/results_table.csv").write_text("method,score\n", encoding="utf-8")
    completed = subprocess.CompletedProcess(["python", "run_experiment.py"], 0, stdout="", stderr="")

    report = evaluate_code_quality(
        run_dir=artifacts.run_dir,
        spec=spec,
        returncode=completed.returncode,
        stdout=completed.stdout,
    )

    assert report["status"] == "FAIL"
    rules = {failure["rule"] for failure in report["failures"]}
    assert "empty_results_table" in rules
    assert "insufficient_numeric_results" in rules
    assert "missing_experiment_log" in rules


def test_code_eval_quality_tool_requires_build_spec_metric_coverage(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="3D pattern synthesis",
        problem_statement="problem",
        method_summary="method",
        baselines=["baseline"],
        metrics=["micro reconstruction IoU", "Chamfer distance", "full-scene generation time"],
    )
    artifacts.path("results").mkdir(parents=True)
    artifacts.path("results/metrics.json").write_text(
        json.dumps({"summary": {"validation_loss": 0.1}}) + "\n",
        encoding="utf-8",
    )
    artifacts.path("results/results_table.csv").write_text(
        "method,validation_loss\nproposed,0.1\nbaseline,0.2\n",
        encoding="utf-8",
    )
    artifacts.path("results/progress_log.jsonl").write_text(
        json.dumps({"epoch": 1, "train_loss": 0.2, "eval_loss": 0.1}) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.CompletedProcess(["python", "run_experiment.py"], 0, stdout="", stderr="")

    report = evaluate_code_quality(
        run_dir=artifacts.run_dir,
        spec=spec,
        returncode=completed.returncode,
        stdout=completed.stdout,
    )

    assert report["status"] == "FAIL"
    metric_failure = next(failure for failure in report["failures"] if failure["rule"] == "missing_build_spec_metrics")
    assert "micro reconstruction IoU" in metric_failure["missing_metrics"]
    assert "Chamfer distance" in metric_failure["missing_metrics"]
    assert "full-scene generation time" in metric_failure["missing_metrics"]


def test_code_eval_quality_tool_rejects_poor_paper_artifacts(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="offline RL optimization",
        problem_statement="problem",
        method_summary="iterative training with candidate checkpoint selection",
        baselines=["baseline"],
        metrics=["D4RL normalized score", "95% bootstrapped confidence interval", "action generation running time"],
    )
    artifacts.path("results").mkdir(parents=True)
    artifacts.path("results/metrics.json").write_text(
        json.dumps({"D4RL normalized score": 48.0, "95% bootstrapped confidence interval": 2.0, "action generation running time": 0.01}) + "\n",
        encoding="utf-8",
    )
    artifacts.path("results/results_table.csv").write_text(
        "method,source,D4RL normalized score,95% bootstrapped confidence interval,action generation running time,evaluation_source\n"
        "proposed,computed,48,2,0.01,unit\n"
        "baseline,computed,45,3,0.02,unit\n",
        encoding="utf-8",
    )
    artifacts.path("results/progress_log.jsonl").write_text(
        json.dumps({"epoch": 1, "D4RL normalized score": 48.0, "timestamp": 1.0}) + "\n",
        encoding="utf-8",
    )
    _write_test_png(artifacts.path("results/progress_curve.png"), width=900, height=520)
    _write_test_png(artifacts.path("results/eval_curve.png"), width=800, height=480)

    report = evaluate_code_quality(
        run_dir=artifacts.run_dir,
        spec=spec,
        returncode=0,
        stdout="",
    )

    assert report["status"] == "FAIL"
    rules = {failure["rule"] for failure in report["failures"]}
    assert "missing_summary_table" in rules
    assert "single_point_progress_log" in rules
    assert "eval_plot_not_summary_figure" in rules


def _write_test_png(path, width=1200, height=800):
    import struct
    import zlib

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            row.extend(((x * 3 + y) % 256, (x + y * 5) % 256, (x * 7 + y * 11) % 256))
        rows.append(b"\x00" + bytes(row))
    raw = b"".join(rows)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    payload = b"\x89PNG\r\n\x1a\n"
    payload += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += chunk(b"IDAT", zlib.compress(raw, 1))
    payload += chunk(b"IEND", b"")
    path.write_bytes(payload)


class BrokenCodeLLM:
    def complete(self, prompt: str) -> str:
        return """{
          "files": [
            {"path": "README.md", "content": "broken"},
            {"path": "configs/experiment_config.json", "content": "{\\"device\\": \\"cpu\\"}"},
            {"path": "src/__init__.py", "content": ""},
            {"path": "src/dataset.py", "content": ""},
            {"path": "src/method.py", "content": ""},
            {"path": "src/baselines.py", "content": ""},
            {"path": "src/train.py", "content": ""},
            {"path": "src/evaluate.py", "content": ""},
            {"path": "src/plot.py", "content": ""},
            {"path": "run_experiment.py", "content": "raise RuntimeError('specific code failed')\\n"}
          ]
        }"""


class FailingCodeLLM:
    def complete(self, prompt: str) -> str:
        raise RuntimeError("HTTP provider request failed: 524 upstream timeout")


def test_llm_generated_code_failure_does_not_fallback_to_generic_pass(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts, llm=BrokenCodeLLM())
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="specific task",
        problem_statement="problem",
        method_summary="specific method",
    )

    report, _, _ = builder.code(spec)

    assert report.status == "FAIL"
    assert report.generation_mode == "llm_generated"
    assert report.fallback_used is False
    assert "specific code failed" in "".join(report.errors)


def test_code_stage_falls_back_when_llm_request_fails(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts, llm=FailingCodeLLM())
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="specific task",
        problem_statement="problem",
        method_summary="specific method",
    )

    report, _, raw = builder.code(spec)

    assert report.status == "FAIL"
    assert report.generation_mode == "generic_fallback_after_llm_error"
    assert report.fallback_used is True
    assert "524 upstream timeout" in raw
    assert "generic fallback only validates pipeline execution" in "".join(report.errors)
    assert artifacts.path("CodeGeneration.error.txt").exists()
    experiment_config = artifacts.read_json("code/configs/experiment_config.json")
    assert experiment_config["metrics"] == ["primary_metric"]
    assert experiment_config["baselines"] == ["reference_baseline"]


def test_code_preflight_patches_missing_numpy_import(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    code_dir = artifacts.path("code")
    code_dir.mkdir(parents=True)
    script = code_dir / "run_experiment.py"
    script.write_text("x = np.random.randn(2, 2)\nprint(x.shape)\n", encoding="utf-8")
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="generic task",
        problem_statement="problem",
        method_summary="method",
    )

    builder._finalize_code_project_scaffold(spec, code_dir)

    text = script.read_text(encoding="utf-8")
    assert "import numpy as np" in text
    assert artifacts.path("code/requirements.txt").exists()


def test_code_preflight_rejects_mock_hdf5_dataset_fallback(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    code_dir = artifacts.path("code")
    code_dir.mkdir(parents=True)
    script = code_dir / "run_experiment.py"
    script.write_text(
        """
import os

def main():
    atari_config = {"dataset_path": "runs/datasets/atari_breakout.hdf5"}
    if not os.path.exists(atari_config["dataset_path"]):
        print(f"Dataset not found at {atari_config['dataset_path']}, creating mock...")
        os.makedirs(os.path.dirname(atari_config["dataset_path"]), exist_ok=True)
        atari_config["dataset_path"] = "runs/datasets/atari_breakout_mock.hdf5"
    print(atari_config["dataset_path"])
""".lstrip(),
        encoding="utf-8",
    )
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="generic task",
        problem_statement="problem",
        method_summary="method",
    )

    builder._finalize_code_project_scaffold(spec, code_dir)

    text = script.read_text(encoding="utf-8")
    assert "creating mock" not in text
    assert "_mock.hdf5" not in text
    assert "raise FileNotFoundError" in text
    assert "mock hdf5 fallback is forbidden" in text


def test_code_uses_same_candidate_download_policy_for_offline_rl(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    code_dir = artifacts.path("code")
    (code_dir / "configs").mkdir(parents=True)
    (code_dir / "configs" / "experiment_config.json").write_text(
        json.dumps(
                {
                    "dataset_path": "runs/datasets/atari_breakout.hdf5",
                    "train_epochs": 1000,
                    "eval_epochs": 500,
                    "atari": {"dataset_path": "runs/datasets/atari_breakout.hdf5", "train_epochs": 1000},
                    "mujoco": {"dataset_path": "runs/datasets/mujoco_halfcheetah.hdf5", "eval_epochs": 500},
                }
            ),
        encoding="utf-8",
    )

    def fake_download(_url, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"raw d4rl artifact")

    builder._download_file = fake_download
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="offline RL MuJoCo validation",
        problem_statement="Evaluate offline reinforcement learning on one dataset.",
        method_summary="Use D4RL data for verification.",
    )

    report = builder._acquire_verification_dataset(spec, code_dir)

    assert report["status"] == "PASS"
    assert report["dataset_id"] == "hopper-medium-v2"
    assert report["reason"] == "downloaded_evidence_informed_dataset_and_generated_smoke_view"
    assert report["download_attempts"][0]["status"] == "PASS"
    config = artifacts.read_json("code/configs/experiment_config.json")
    assert config["dataset_path"] == "datasets/synthetic_offline_rl_smoke.jsonl"
    assert config["train_epochs"] == 50
    assert config["eval_epochs"] == 20
    assert config["atari"]["dataset_path"] == "datasets/synthetic_offline_rl_smoke.jsonl"
    assert config["atari"]["train_epochs"] == 50
    assert config["mujoco"]["dataset_path"] == "datasets/synthetic_offline_rl_smoke.jsonl"
    assert config["mujoco"]["eval_epochs"] == 20
    first_record = json.loads((code_dir / config["dataset_path"]).read_text(encoding="utf-8").splitlines()[0])
    assert {"obs", "action", "reward", "next_obs", "done"} <= set(first_record)
    assert first_record["source_dataset"]["dataset_id"] == "hopper-medium-v2"


def test_code_generates_domain_smoke_dataset_for_non_rl_dataset_path(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    code_dir = artifacts.path("code")
    (code_dir / "configs").mkdir(parents=True)
    (code_dir / "configs" / "experiment_config.json").write_text(
        json.dumps(
            {
                "dataset_path": "datasets/3d_pattern_dataset.jsonl",
                "train_epochs": 1000,
                "eval_epochs": 500,
            }
        ),
        encoding="utf-8",
    )
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="Interactive 3D Pattern Synthesis",
        problem_statement="Evaluate structure-appearance decoupling in 3D Gaussian patterns.",
        method_summary="Use synthetic 3D Gaussian pattern scenes for smoke validation.",
    )

    def fake_download(url, destination):
        raise RuntimeError("network unavailable")

    builder._download_file = fake_download

    report = builder._acquire_verification_dataset(spec, code_dir)

    assert report["status"] == "PASS"
    assert report["dataset_id"] == "synthetic-smoke"
    assert report["reason"] == "generated_domain_agnostic_smoke_dataset_after_download_failures"
    config = artifacts.read_json("code/configs/experiment_config.json")
    assert config["dataset_path"] == "datasets/synthetic_3d_patterns.jsonl"
    assert config["train_epochs"] == 50
    assert config["eval_epochs"] == 20
    dataset_path = code_dir / config["dataset_path"]
    assert dataset_path.exists()
    first_record = json.loads(dataset_path.read_text(encoding="utf-8").splitlines()[0])
    assert "gaussians" in first_record
    assert "structure_latent" in first_record
    assert "appearance_latent" in first_record


def test_code_tries_three_evidence_informed_dataset_downloads_before_smoke_view(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    code_dir = artifacts.path("code")
    (code_dir / "configs").mkdir(parents=True)
    (code_dir / "configs" / "experiment_config.json").write_text(
        json.dumps({"dataset_path": "datasets/shape_dataset.jsonl", "train_epochs": 1000}),
        encoding="utf-8",
    )
    attempts = []

    def fake_download(url, destination):
        attempts.append(url)
        if len(attempts) < 3:
            raise RuntimeError(f"temporary download failure {len(attempts)}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"raw dataset")

    builder._download_file = fake_download
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="3D pattern synthesis",
        problem_statement="Evaluate 3D geometry editing.",
        method_summary="Use ShapeNet or ModelNet style 3D object data.",
    )
    evidence = [
        EvidenceCard(
            evidence_id="ev-1",
            paper_id="paper-1",
            task="3D shape reconstruction on ShapeNet",
            method="3D Gaussian grouping",
            setting="ShapeNet and ModelNet object benchmarks",
        )
    ]

    report = builder._acquire_verification_dataset(spec, code_dir, evidence=evidence)

    assert report["status"] == "PASS"
    assert len(attempts) == 3
    assert [item["status"] for item in report["download_attempts"]] == ["FAIL", "FAIL", "PASS"]
    assert report["reason"] == "downloaded_evidence_informed_dataset_and_generated_smoke_view"
    assert report["dataset_id"] == "tiny-nerf"
    config = artifacts.read_json("code/configs/experiment_config.json")
    assert config["dataset_path"] == "datasets/synthetic_3d_patterns.jsonl"
    first_record = json.loads((code_dir / config["dataset_path"]).read_text(encoding="utf-8").splitlines()[0])
    assert first_record["source_dataset"]["dataset_id"] == "tiny-nerf"


def test_code_falls_back_to_synthetic_dataset_after_three_download_failures(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    code_dir = artifacts.path("code")
    (code_dir / "configs").mkdir(parents=True)
    (code_dir / "configs" / "experiment_config.json").write_text(
        json.dumps({"dataset_path": "datasets/shape_dataset.jsonl"}),
        encoding="utf-8",
    )

    def fake_download(url, destination):
        raise RuntimeError("network unavailable")

    builder._download_file = fake_download
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="3D pattern synthesis",
        problem_statement="Evaluate 3D geometry editing.",
        method_summary="Use common 3D object datasets.",
    )

    report = builder._acquire_verification_dataset(spec, code_dir)

    assert report["status"] == "PASS"
    assert len(report["download_attempts"]) == 3
    assert all(item["status"] == "FAIL" for item in report["download_attempts"])
    assert report["reason"] == "generated_domain_agnostic_smoke_dataset_after_download_failures"
    assert report["dataset_id"] == "synthetic-smoke"
    config = artifacts.read_json("code/configs/experiment_config.json")
    assert config["dataset_path"] == "datasets/synthetic_3d_patterns.jsonl"


def test_code_preflight_patches_missing_h5py_import(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    code_dir = artifacts.path("code")
    code_dir.mkdir(parents=True)
    script = code_dir / "run_experiment.py"
    script.write_text("with h5py.File('data.h5', 'w') as handle:\n    pass\n", encoding="utf-8")
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="generic task",
        problem_statement="problem",
        method_summary="method",
    )

    builder._finalize_code_project_scaffold(spec, code_dir)

    text = script.read_text(encoding="utf-8")
    assert "import h5py" in text


def test_code_preflight_patches_missing_csv_import(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    code_dir = artifacts.path("code")
    code_dir.mkdir(parents=True)
    script = code_dir / "evaluate.py"
    script.write_text("writer = csv.writer(open('x.csv', 'w'))\n", encoding="utf-8")
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="generic task",
        problem_statement="problem",
        method_summary="method",
    )

    builder._finalize_code_project_scaffold(spec, code_dir)

    text = script.read_text(encoding="utf-8")
    assert "import csv" in text


def test_code_preflight_patches_missing_offline_dataset_import(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    code_dir = artifacts.path("code")
    (code_dir / "src").mkdir(parents=True)
    (code_dir / "src" / "dataset.py").write_text(
        "from src.dataset import OfflineDataset\n\nclass OfflineDataset:\n    pass\n",
        encoding="utf-8",
    )
    script = code_dir / "run_experiment.py"
    script.write_text("dataset = OfflineDataset('data.h5', 32, 'cpu')\n", encoding="utf-8")
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="generic task",
        problem_statement="problem",
        method_summary="method",
    )

    builder._finalize_code_project_scaffold(spec, code_dir)

    text = script.read_text(encoding="utf-8")
    assert "from src.dataset import OfflineDataset" in text
    dataset_text = (code_dir / "src" / "dataset.py").read_text(encoding="utf-8")
    assert "from src.dataset import OfflineDataset" not in dataset_text


def test_code_preflight_patches_common_generated_python_errors(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    code_dir = artifacts.path("code")
    (code_dir / "src").mkdir(parents=True)
    (code_dir / "src" / "method.py").write_text(
        "class FlowMatchingEnsemble(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__\n\n"
        "class FlowMatchingTrainer:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (code_dir / "src" / "evaluate.py").write_text(
        "from typing import Dict\n"
        "from src.method import FlowMatchingEnsemble\n"
        "def evaluate(trainer: FlowMatchingTrainer) -> Dict:\n"
        "    return {}\n",
        encoding="utf-8",
    )
    (code_dir / "run_experiment.py").write_text(
        "baselines = BaselineModels(dataset.n_states, dataset.n_actions)\n",
        encoding="utf-8",
    )
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="generic task",
        problem_statement="problem",
        method_summary="method",
    )

    builder._finalize_code_project_scaffold(spec, code_dir)

    assert "super().__init__()" in (code_dir / "src" / "method.py").read_text(encoding="utf-8")
    assert "FlowMatchingEnsemble, FlowMatchingTrainer" in (code_dir / "src" / "evaluate.py").read_text(encoding="utf-8")
    assert "trainer.dataset.n_states" in (code_dir / "run_experiment.py").read_text(encoding="utf-8")


def test_code_preflight_replaces_fragile_pandas_offline_dataset_loader(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    dataset_path = artifacts.path("code/src/dataset.py")
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text(
        "import pandas as pd\n\n"
        "class OfflineDataset:\n"
        "    def __init__(self, path):\n"
        "        self.data = pd.read_csv(path)\n",
        encoding="utf-8",
    )

    report = builder._patch_generated_offline_dataset_loader(dataset_path)

    assert report["patched"] is True
    text = dataset_path.read_text(encoding="utf-8")
    assert "def _load_hdf5" in text
    assert "torch.as_tensor" in text


def test_runtime_patch_reuses_dataset_for_none_evaluation(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    run_path = artifacts.path("code/run_experiment.py")
    run_path.parent.mkdir(parents=True)
    run_path.write_text(
        "evaluations['mujoco'] = evaluator.run_evaluation(None, 'mujoco') # Passing None for dummy\n",
        encoding="utf-8",
    )

    patch = builder._patch_none_dataset_evaluation(run_path)

    assert patch["patched"] is True
    text = run_path.read_text(encoding="utf-8")
    assert "run_evaluation(atari_dataset, 'mujoco')" in text
    assert "run_evaluation(None" not in text


def test_runtime_patch_handles_rem_q_values_shape_error(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    method_path = artifacts.path("code/src/method.py")
    method_path.parent.mkdir(parents=True)
    method_path.write_text(
        """
def get_ensemble_values(self, obs, acts):
    batch_size = obs.shape[0]
    device = obs.device
    weights = torch.rand(self.ensemble_size, batch_size, 1, device=device)
    weights = weights / weights.sum(dim=0, keepdim=True)
    q_values = torch.stack(q_values_list, dim=1)
    combined_q = torch.sum(weights * q_values, dim=1)
    return combined_q
""".lstrip(),
        encoding="utf-8",
    )
    completed = subprocess.CompletedProcess(
        args=["python", "run_experiment.py"],
        returncode=1,
        stdout="",
        stderr="RuntimeError: The size of tensor a (64) must match the size of tensor b (5) at non-singleton dimension 1\n    combined_q = torch.sum(weights * q_values, dim=1)",
    )

    report = builder._patch_runtime_error_from_failure(artifacts.path("code"), completed)

    assert report["patched"] is True
    text = method_path.read_text(encoding="utf-8")
    assert "torch.rand(batch_size, self.ensemble_size, 1, device=device)" in text
    assert "weights.sum(dim=1, keepdim=True)" in text


def test_code_collects_results_only_from_run_results_dir(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    root_results = artifacts.path("results")
    code_results = artifacts.path("code/results")
    root_results.mkdir(parents=True)
    code_results.mkdir(parents=True)
    (root_results / "metrics.json").write_text('{"status":"ok"}\n', encoding="utf-8")
    (root_results / "results_table.csv").write_text("method,score\nx,1\n", encoding="utf-8")
    (code_results / "metrics.json").write_text('{"status":"wrong"}\n', encoding="utf-8")
    (code_results / "results_table.csv").write_text("method,score\nwrong,0\n", encoding="utf-8")

    report = builder._collect_generated_result_files(artifacts.path("code"))

    assert len(report["present"]) == 2
    assert len(report["removed_misplaced"]) == 2
    assert artifacts.path("results/metrics.json").exists()
    assert artifacts.path("results/results_table.csv").exists()
    assert not artifacts.path("code/results/metrics.json").exists()
    assert not artifacts.path("code/results/results_table.csv").exists()
    assert artifacts.path("ResultCollectionReport.json").exists()


def test_config_normalization_forces_outputs_to_run_results_dir(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    config_path = artifacts.path("code/configs/experiment_config.json")
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "outputs": {
                    "metrics_json": "results/metrics.json",
                    "results_table_csv": "results/results_table.csv",
                }
            }
        ),
        encoding="utf-8",
    )

    report = builder._normalize_generated_experiment_config(artifacts.path("code"))
    config = artifacts.read_json("code/configs/experiment_config.json")

    assert report["patched"] is True
    assert config["outputs"]["metrics_json"] == "../results/metrics.json"
    assert config["outputs"]["results_table_csv"] == "../results/results_table.csv"


def test_run_experiment_timeout_returns_failed_completed_process(tmp_path, monkeypatch):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts, experiment_timeout_seconds=77)
    seen = {}

    def timeout_run(*_args, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        raise subprocess.TimeoutExpired(cmd=kwargs.get("args", ["python"]), timeout=kwargs.get("timeout"))

    monkeypatch.setattr("quit_agent.agents.builder_agent.subprocess.run", timeout_run)

    completed = builder._run_experiment(["python", "run_experiment.py"], artifacts.path("code"))

    assert completed.returncode == 124
    assert seen["timeout"] == 77
    assert "77" in completed.stderr


def test_code_repair_prompt_uses_code_stage_eval_repair_skill(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    code_dir = artifacts.path("code")
    (code_dir / "src").mkdir(parents=True)
    (code_dir / "src" / "method.py").write_text("def broken():\n    pass\n", encoding="utf-8")
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="generic task",
        problem_statement="problem",
        method_summary="method",
    )
    completed = subprocess.CompletedProcess(
        args=["python", "run_experiment.py"],
        returncode=1,
        stdout="training started",
        stderr='File "/tmp/run/code/src/method.py", line 1\nRuntimeError: shape mismatch',
    )

    prompt = builder._code_stage_repair_prompt(spec, code_dir, ["python", "run_experiment.py"], completed, [], 1)

    assert "Follow this repair skill exactly" in prompt
    assert "Code Stage Eval Repair" in prompt
    assert "No markdown fences" in prompt
    assert "traceback_file:" in prompt
    assert "src/method.py" in prompt
    assert "Tensor shape errors" not in prompt


def test_code_eval_failure_repairs_existing_project_without_regenerating(tmp_path):
    class EvalRepairLLM:
        def __init__(self):
            self.prompt = ""

        def complete(self, prompt: str) -> str:
            self.prompt = prompt
            return json.dumps(
                {
                    "files": [
                        {
                            "path": "src/plot.py",
                            "content": "REPAIRED_FROM_CODE_EVAL = True\n",
                        }
                    ]
                }
            )

    artifacts = ArtifactManager(tmp_path, "run")
    llm = EvalRepairLLM()
    builder = BuilderAgent(artifacts, llm=llm)
    code_dir = artifacts.path("code")
    (code_dir / "src").mkdir(parents=True)
    (code_dir / "configs").mkdir(parents=True)
    for rel in ["src/__init__.py", "src/dataset.py", "src/method.py", "src/baselines.py", "src/train.py", "src/evaluate.py"]:
        (code_dir / rel).write_text("\n", encoding="utf-8")
    (code_dir / "src/plot.py").write_text("REPAIRED_FROM_CODE_EVAL = False\n", encoding="utf-8")
    (code_dir / "configs/experiment_config.json").write_text('{"outputs": {}}\n', encoding="utf-8")
    (code_dir / "run_experiment.py").write_text(
        """
from pathlib import Path

print("existing project still running")
results = Path("../results")
results.mkdir(parents=True, exist_ok=True)
(results / "metrics.json").write_text('{"summary": {"score": 1.0}}\\n')
(results / "results_table.csv").write_text("method,score\\nproposed,1.0\\nbaseline,0.5\\n")
(results / "progress_log.jsonl").write_text('{"step": 1, "score": 1.0}\\n')
""",
        encoding="utf-8",
    )
    artifacts.write_json(
        "BuildSpec.json",
        BuildSpec(
            build_id="build-idea-1",
            idea_id="idea-1",
            target_task="generic task",
            problem_statement="problem",
            method_summary="method",
        ),
    )
    artifacts.write_json(
        "CodeRunReport.json",
        {
            "status": "PASS",
            "code_dir": "code",
            "generation_mode": "llm_generated",
            "fallback_used": False,
            "outputs": ["code/run_experiment.py"],
            "executed": True,
            "returncode": 0,
            "errors": [],
        },
    )
    artifacts.write_json(
        "ExperimentAudit.json",
        {
            "status": "FAIL",
            "failures": ["missing_experiment_figure: numeric results exist but no figure was written"],
            "fallback_target": "CODE",
        },
    )
    artifacts.write_json(
        "CodeEvalQualityReport.json",
        {
            "status": "FAIL",
            "failures": [{"rule": "missing_experiment_figure", "reason": "numeric results exist but no figure was written"}],
        },
    )
    spec = artifacts.load_model("BuildSpec.json", BuildSpec)

    report, prompt, _ = builder.code(spec)

    assert report.status == "PASS"
    assert report.generation_mode == "code_eval_repair"
    assert report.repair_attempted is True
    assert "CODE_EVAL failed" in prompt
    assert "missing_experiment_figure" in llm.prompt
    assert "existing project still running" in artifacts.path("EXPERIMENT_LOG.md").read_text(encoding="utf-8")
    assert "REPAIRED_FROM_CODE_EVAL = True" in (code_dir / "src/plot.py").read_text(encoding="utf-8")
    assert "existing project still running" in (code_dir / "run_experiment.py").read_text(encoding="utf-8")


def test_code_failure_repairs_existing_project_without_full_regeneration(tmp_path):
    class CodeFailureRepairLLM:
        def __init__(self):
            self.prompts: list[str] = []

        def complete(self, prompt: str) -> str:
            self.prompts.append(prompt)
            assert "Repair the existing generated project only" in prompt
            return """=== FILE: run_experiment.py ===
from pathlib import Path

print("repaired existing code")
results = Path("../results")
results.mkdir(parents=True, exist_ok=True)
(results / "metrics.json").write_text('{"score": 1.0}\\n')
(results / "results_table.csv").write_text("method,score\\nproposed,1.0\\nbaseline,0.5\\n")
(results / "progress_log.jsonl").write_text('{"step": 1, "score": 1.0}\\n')
"""

    artifacts = ArtifactManager(tmp_path, "run")
    llm = CodeFailureRepairLLM()
    builder = BuilderAgent(artifacts, llm=llm)
    code_dir = artifacts.path("code")
    (code_dir / "src").mkdir(parents=True)
    (code_dir / "configs").mkdir(parents=True)
    for rel in ["src/__init__.py", "src/dataset.py", "src/method.py", "src/baselines.py", "src/train.py", "src/evaluate.py", "src/plot.py"]:
        (code_dir / rel).write_text("\n", encoding="utf-8")
    (code_dir / "configs/experiment_config.json").write_text('{"outputs": {}}\n', encoding="utf-8")
    (code_dir / "run_experiment.py").write_text("raise RuntimeError('old failure')\n", encoding="utf-8")
    artifacts.write_json(
        "CodeRunReport.json",
        {
            "status": "FAIL",
            "code_dir": "code",
            "generation_mode": "llm_generated",
            "fallback_used": False,
            "outputs": ["code/run_experiment.py"],
            "executed": True,
            "returncode": 1,
            "errors": ["old failure"],
        },
    )
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="generic task",
        problem_statement="problem",
        method_summary="method",
    )

    report, prompt, _ = builder.code(spec)

    assert report.status == "PASS"
    assert report.generation_mode == "code_failure_repair"
    assert report.repair_attempted is True
    assert len(llm.prompts) == 1
    assert "Follow this repair skill exactly" in prompt
    assert "Follow this repair skill exactly" in llm.prompts[0]
    assert not artifacts.path("CodeStageGenerationReport.json").exists()
    assert "repaired existing code" in (code_dir / "run_experiment.py").read_text(encoding="utf-8")


def test_staged_generation_stops_after_first_syntax_failure(tmp_path):
    class SyntaxFailStageLLM:
        def __init__(self):
            self.calls = 0

        def complete(self, _prompt: str) -> str:
            self.calls += 1
            if self.calls == 1:
                return '{"entrypoint": "run_experiment.py"}'
            if self.calls == 2:
                return """=== FILE: src/dataset.py ===
def load_dataset(config=None):
    return {}
=== FILE: src/__init__.py ===

=== FILE: configs/experiment_config.json ===
{}
=== FILE: src/baselines.py ===
def run_baseline(name, dataset, config=None):
    return {}
=== FILE: src/method.py ===
def broken(
"""
            raise AssertionError("generation should stop after core syntax failure")

    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts, llm=SyntaxFailStageLLM())
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="generic task",
        problem_statement="problem",
        method_summary="method",
    )

    result = builder._run_staged_code_generation(spec, artifacts.path("code"), evidence=[])

    assert result["report"]["status"] == "FAIL"
    assert result["report"]["stages"][-1]["stage"] == "core"
    assert "syntax failed after staged generation step: core" in result["report"]["errors"]


def test_device_resolution_propagates_device_to_nested_task_configs(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    config_path = artifacts.path("code/configs/experiment_config.json")
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        '{"device":"cpu","atari":{"dataset_path":"data.h5","batch_size":64},"mujoco":{"train_epochs":10}}\n',
        encoding="utf-8",
    )

    report = builder._resolve_generated_code_device(artifacts.path("code"))
    config = artifacts.read_json("code/configs/experiment_config.json")

    assert report["status"] == "PASS"
    assert config["device"] in {"cpu"} or config["device"].startswith("cuda")
    assert config["atari"]["device"] == config["device"]
    assert config["mujoco"]["device"] == config["device"]


def test_runtime_patch_moves_batch_tensors_to_model_device(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    method_path = artifacts.path("code/src/method.py")
    method_path.parent.mkdir(parents=True)
    method_path.write_text(
        "def train_step(self, batch):\n"
        "        self.optimizer.zero_grad()\n"
        "        obs = batch['obs'].float()\n"
        "        acts = batch['act'].float()\n"
        "        rewards = batch['reward'].float()\n"
        "        dones = batch['done'].float()\n"
        "        next_obs = batch['next_obs'].float()\n"
        "        return obs\n",
        encoding="utf-8",
    )

    patch = builder._patch_batch_tensors_to_model_device(method_path)

    assert patch["patched"] is True
    text = method_path.read_text(encoding="utf-8")
    assert "next(self.model.parameters()).device" in text
    assert "batch['obs'].float().to(device)" in text
    assert "batch['reward'].float().view(-1, 1).to(device)" in text


def test_runtime_patch_reports_missing_hdf5_without_creating_mock(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    stderr = "FileNotFoundError: unable to open file: name = 'runs/datasets/mujoco_halfcheetah_mock.hdf5'"

    patch = builder._report_missing_hdf5_dataset(stderr)

    assert patch["patched"] is False
    assert patch["file"] == "runs/datasets/mujoco_halfcheetah_mock.hdf5"
    assert "will not create a mock dataset" in patch["reason"]


def test_code_augments_results_with_paper_reported_baselines(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    artifacts.path("results").mkdir(parents=True)
    artifacts.path("results/results_table.csv").write_text(
        "Method,Score,Std,Baseline_Score,Improvement\nREM-Flow Matching,57.0,2.5,N/A,N/A\n",
        encoding="utf-8",
    )
    artifacts.write_jsonl(
        "EvidenceCards.jsonl",
        [
            {
                "evidence_id": "ev-rem",
                "paper_id": "1907.04543v4",
                "task": "Offline RL on Atari",
                "method": "Random Ensemble Mixture and Offline QR-DQN baseline",
                "setting": "Baselines include Offline QR-DQN, Offline BCQ, and fully-trained online DQN.",
                "claims": ["REM surpasses strong baselines like QR-DQN and the best online DQN policy."],
                "metrics": ["Median normalized scores"],
                "limitations": [],
                "transferable_idea_seeds": [],
            }
        ],
    )
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="offline RL",
        problem_statement="problem",
        method_summary="method",
        baselines=["Offline QR-DQN", "Offline BCQ"],
    )

    report = builder._augment_results_with_paper_baselines(spec)

    assert report["status"] == "PASS"
    assert report["appended_count"] == 2
    table = artifacts.path("results/results_table.csv").read_text(encoding="utf-8")
    assert "Offline QR-DQN" in table
    assert "paper_reported" in table
    assert "1907.04543v4" in table
    baseline_report = artifacts.read_json("PaperBaselineResults.json")
    assert baseline_report["baselines"][0]["evidence_ids"] == ["ev-rem"]


def test_code_augments_baselines_from_all_retrieved_papers_not_only_citations(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    artifacts.path("results").mkdir(parents=True)
    artifacts.path("results/results_table.csv").write_text(
        "Method,Score,Std,Baseline_Score,Improvement\nREM-Flow Matching,57.0,2.5,N/A,N/A\n",
        encoding="utf-8",
    )
    artifacts.write_jsonl("EvidenceCards.jsonl", [])
    artifacts.write_jsonl(
        "PaperCards.jsonl",
        [
            {
                "paper_id": "1907.04543v4",
                "title": "An Optimistic Perspective on Offline Reinforcement Learning",
                "abstract": (
                    "Offline REM trained on the DQN replay dataset surpasses strong RL baselines, "
                    "including Offline QR-DQN, Offline DQN, TD3, DDPG, and BCQ."
                ),
                "query_source": "offline rl",
            }
        ],
    )
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="offline RL",
        problem_statement="problem",
        method_summary="method",
        baselines=["Offline QR-DQN", "Offline BCQ", "Unmentioned Baseline"],
        citations_required=["ev-unrelated"],
    )

    report = builder._augment_results_with_paper_baselines(spec)

    assert report["status"] == "PASS"
    table = artifacts.path("results/results_table.csv").read_text(encoding="utf-8")
    assert "Offline QR-DQN" in table
    assert "Offline BCQ" in table
    assert "paper-1907.04543v4" in table
    assert "Unmentioned Baseline,N/A,N/A,N/A,N/A,build_spec_only" in table
    baseline_report = artifacts.read_json("PaperBaselineResults.json")
    unmentioned = next(item for item in baseline_report["baselines"] if item["baseline"] == "Unmentioned Baseline")
    assert unmentioned["confirmed_no_paper_baseline_result"] is True
    assert len(unmentioned["confirmation_attempts"]) == 3
    assert {attempt["status"] for attempt in unmentioned["confirmation_attempts"]} == {"NOT_FOUND"}


def test_baseline_confirmation_records_three_checks_before_declaring_absent(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    artifacts.path("results").mkdir(parents=True)
    artifacts.path("results/results_table.csv").write_text(
        "Method,Score\nProposed,1.0\n",
        encoding="utf-8",
    )
    artifacts.write_jsonl("EvidenceCards.jsonl", [])
    artifacts.write_jsonl(
        "PaperCards.jsonl",
        [
            {
                "paper_id": "paper-unrelated",
                "title": "Unrelated Planning Method",
                "abstract": "This paper studies model predictive control.",
                "query_source": "planning",
            }
        ],
    )
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="offline RL",
        problem_statement="problem",
        method_summary="method",
        baselines=["Offline QR-DQN"],
    )

    report = builder._augment_results_with_paper_baselines(spec)

    assert report["status"] == "PASS"
    assert report["baselines"][0]["source"] == "build_spec_only"
    assert report["baselines"][0]["confirmed_no_paper_baseline_result"] is True
    assert [attempt["check"] for attempt in report["baselines"][0]["confirmation_attempts"]] == [
        "evidence_cards",
        "paper_cards",
        "combined_relaxed",
    ]
    assert [attempt["status"] for attempt in report["baselines"][0]["confirmation_attempts"]] == [
        "NOT_FOUND",
        "NOT_FOUND",
        "NOT_FOUND",
    ]


def test_baseline_matching_does_not_match_short_alias_inside_other_words(tmp_path):
    builder = BuilderAgent(ArtifactManager(tmp_path, "run"))

    matches = builder._evidence_for_baseline(
        "Offline DQN",
        [
            {
                "evidence_id": "ev-modqn",
                "paper_id": "p1",
                "method": "MODQN for fraud evaluation",
                "setting": "e-commerce",
                "claims": ["MODQN improves net revenue."],
                "metrics": [],
            }
        ],
    )

    assert matches == []
