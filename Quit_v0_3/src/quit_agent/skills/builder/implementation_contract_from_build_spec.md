# Skill: Implementation Contract From BuildSpec

Create `ImplementationContract.json` from `BuildSpec.json`.

This skill removes ambiguity before code generation. It must not restate the research narrative. It translates BuildSpec into concrete module interfaces, file names, output paths, dependency decisions, and verification steps.

## Inputs

Required:

```text
BuildSpec.json
```

Optional:

```text
EvidenceCards.jsonl
RepoCards.jsonl
EnvironmentResolutionReport.json
Previous CODE feedback reports
```

## Output Artifact

Write:

```text
ImplementationContract.json
```

## Output Contract

Return exactly one JSON object:

```json
{
  "status": "PASS",
  "canonical_layout": {
    "entrypoint": "run_experiment.py",
    "config": "configs/experiment_config.json",
    "dataset_file": "src/dataset.py",
    "method_file": "src/method.py",
    "baselines_file": "src/baselines.py",
    "train_file": "src/train.py",
    "evaluate_file": "src/evaluate.py",
    "plot_file": "src/plot.py"
  },
  "module_contracts": {
    "dataset": {
      "public_api": [],
      "data_objects": [],
      "smoke_test": []
    },
    "method": {
      "public_api": [],
      "required_behaviors": [],
      "smoke_test": []
    },
    "baselines": {
      "required": [],
      "public_api": [],
      "smoke_test": []
    },
    "runner": {
      "command": "python run_experiment.py --config configs/experiment_config.json",
      "required_outputs": []
    },
    "plots": {
      "required_outputs": []
    }
  },
  "generation_stages": [
    {
      "stage": "core",
      "skill": "implement_core.md",
      "responsibility": "dataset -> method -> baselines",
      "files": ["src/__init__.py", "src/dataset.py", "src/method.py", "src/baselines.py", "configs/experiment_config.json"],
      "report": "code/CoreImplementationReport.json"
    },
    {
      "stage": "experiment",
      "skill": "implement_experiment.md",
      "responsibility": "execute -> runner -> plots",
      "files": ["src/train.py", "src/evaluate.py", "src/plot.py", "run_experiment.py", "configs/experiment_config.json", "README.md", "ENVIRONMENT.md", "requirements.txt", "environment.yml"],
      "report": "code/ExperimentImplementationReport.json"
    }
  ],
  "implementation_choices": {},
  "dependency_policy": {},
  "verification_steps": [],
  "forbidden_paths": [],
  "notes": []
}
```

## Rules

- Do not repeat `problem_statement`, full `method_summary`, paper outline, citations, or success criteria prose.
- Freeze canonical filenames. Prefer `src/method.py`; if legacy code uses `src/methods.py`, require a compatibility shim or migration.
- Convert every `results/...` BuildSpec path to `../results/...` when code writes from inside `code/`.
- Include every BuildSpec metric, log, plot, baseline, sensitivity, and ablation as an engineering output/check.
- Define the two CODE generation stages exactly as `core` and `experiment`; do not split them back into per-file stages.
- The `core` stage owns dataset, method, and baseline interfaces. The `experiment` stage owns train, runner, evaluation, plotting, final metrics, and result tables.
- Specify optional dependency fallback policy, especially for solvers, simulators, and heavy ML libraries.
- Include smoke tests for import, dataset creation/loading, method one-step execution, runner execution, and output existence.
- Include forbidden output paths such as `code/results`, `code/outputs`, and `code/logs`.
