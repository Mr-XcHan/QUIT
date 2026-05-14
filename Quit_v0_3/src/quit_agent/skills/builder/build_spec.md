# Skill: Build Spec

## Description
Convert an approved research idea into a compact, executable `BuildSpec` artifact.

This skill is used by the `BUILD_SPEC` workflow state. `BuildSpec.json` becomes the research/product source of truth for downstream CODE and WRITE. Later stages should not depend on long chat history or the full set of prior artifacts.

CODE now runs as staged implementation. `BuildSpec.json` should define what must be implemented and evaluated; `ImplementationContract.json` will later define exact code interfaces, file names, module APIs, and verification commands. Do not overload BuildSpec with low-level code interface decisions unless they are scientifically necessary.

## When To Use
Use this skill when:

- `IdeaDecision.json` exists and has `decision = PASS`.
- `IdeaLibrary.jsonl` contains the approved idea.
- Supporting `EvidenceCards.jsonl` are available for citations and experimental grounding.
- Optional `RepoCards.jsonl` exists from RETRIEVE and contains linked code repositories.

## Input Artifacts
Required:

```text
ResearchBrief.json
IdeaDecision.json
IdeaLibrary.jsonl
EvidenceCards.jsonl
```

Use only the approved idea and its supporting evidence. Do not include unrelated evidence or free-form historical context.
If repository cards are available, try to clone the repositories attached to the approved idea's supporting evidence in relevance order. Use the first cloneable or inspectable repo as the reference environment. If none is available, summarize a generated minimal Python environment into `environment`.

## Output Artifact
Write:

```text
BuildSpec.json
```

Downstream CODE will derive:

```text
ImplementationContract.json
code/src/dataset.py
code/src/method.py
code/src/baselines.py
code/src/train.py
code/run_experiment.py
code/src/plot.py
results/
```

Optional malformed model output:

```text
BuildSpec.raw.txt
```

## Output Schema
Return exactly one JSON object:

```json
{
  "build_id": "",
  "idea_id": "",
  "target_task": "",
  "problem_statement": "",
  "method_summary": "",
  "repo_url": "",
  "implementation_plan": [],
  "experiment_plan": [],
  "baselines": [],
  "metrics": [],
  "dataset": {
    "name": "",
    "public_sources": [
      {
        "name": "",
        "url": "",
        "reason": "why this source is relevant"
      }
    ],
    "fallback_policy": "try public sources first; use domain-appropriate synthetic smoke data only after acquisition/load failures"
  },
  "logging": [
    {
      "path": "results/progress_log.jsonl",
      "record_type": "training|evaluation|execution|simulation",
      "fields": ["epoch", "primary_metric", "timestamp"],
      "x_axis": "epoch",
      "interval": "log_interval",
      "description": "what each record means"
    }
  ],
  "plots": [
    {
      "path": "results/progress_curve.png",
      "title": "domain-specific figure title",
      "source": "log|metrics|results_table",
      "x": "epoch|step|seed|method|parameter_name",
      "y": "metric name from metrics/logging",
      "kind": "line|bar|scatter|box",
      "series": "method|seed|scenario",
      "description": "what this figure should show"
    }
  ],
  "success_criteria": [],
  "artifacts_required": {
    "coder": [
      "ImplementationContract.json",
      "CoreImplementationReport.json",
      "ExperimentImplementationReport.json",
      "working code module",
      "experiment logs",
      "result table/figures"
    ],
    "writer": [
      "latex section draft"
    ]
  },
  "environment": {
    "source": "generated|reference_repo|reference_repo_metadata",
    "reference_repo_url": "",
    "reference_repo_path": "",
    "env_files": [],
    "language": "python",
    "framework": "",
    "requirements": ["python>=3.11"],
    "setup_commands": ["pip install -r requirements.txt"]
  },
  "paper_outline": [
    "Abstract",
    "Introduction",
    "Related Work",
    "Preliminaries",
    "Methods",
    "Experiments",
    "Conclusion",
    "Appendix"
  ],
  "citations_required": []
}
```

## Requirements
The spec must:

- be compact enough for coder/writer stages
- include concrete implementation steps
- include a runnable experiment plan
- name baselines and at least one domain-appropriate metric
- describe dataset expectations when relevant: real public dataset candidates, local path expectation, or why no public dataset is appropriate
- define `logging` records and `plots` from the experiment design; losses are required only when the method is genuinely trainable and has a real optimization loss
- describe at least five reporting targets in total across `metrics` and `plots`; for example, three metrics plus two plots is acceptable when that is the right reporting contract
- each plot must have a concrete `path`, `title`, `source`, `x`, `y`, `kind`, `series`, and `description` so CODE can implement it and WRITE can discuss it without guessing
- define the final method-comparison plot, normally `results/eval_curve.png`, as a paper-facing evaluation summary. When the metric set contains primary, uncertainty, domain, runtime, robustness, ablation, or sensitivity information, its description should request those secondary views instead of only a single generic bar chart.
- define the progress plot, normally `results/progress_curve.png`, around a metric that can actually be logged across epochs, steps, episodes, simulation iterations, or candidate evaluations. Do not request a curve if the experiment is inherently one-shot; request a final diagnostic plot instead.
- for simulation, optimization, scheduling, control, queueing, search, or analytical experiments, choose domain metrics such as profit, satisfaction rate, regret, queue backlog, runtime, objective value, constraint violation, success rate, or sensitivity values instead of train/eval loss
- for ML, deep learning, reinforcement learning, representation learning, supervised learning, or other genuinely trainable experiments, it is appropriate to include training and evaluation losses plus training diagnostics such as train loss, eval/validation loss, accuracy, F1, AUROC, reward/return, success rate, gradient norm, learning rate, convergence steps, sample efficiency, calibration error, or wall-clock training time
- for trainable experiments, plots may include loss curves (`train_loss` and `eval_loss` over epoch/step), validation metric curves, reward/return curves, accuracy/F1 curves, convergence plots, ablations, and final method-vs-baseline comparison bars; use these only when the code can compute them from real training/evaluation records
- use artifact paths under `results/`; CODE will write them from `code/` using `../results/...`
- keep method and baseline metrics schema-compatible: the proposed method and every baseline must be reportable in one `results_table.csv`
- include all required sensitivity studies and ablations in `experiment_plan` when they are part of the claim; CODE will write additional shared `results/` artifacts such as `sensitivity_results.csv`
- choose baselines from the approved idea's domain and supporting evidence; do not use unrelated domain defaults such as offline RL baselines unless the ResearchBrief explicitly concerns offline RL
- include success criteria tied to `expected_gain`
- include required artifacts for coder and writer
- include citation identifiers from supporting evidence paper IDs
- include `environment`; prefer a relevant reference repo environment when available, otherwise generate a small standalone Python environment

## Boundary With ImplementationContract

BuildSpec should specify:

- scientific target task and problem
- proposed method behavior at a high level
- dataset/public-data expectations
- required baselines
- metrics, logging, plots
- experiment plan, sensitivities, ablations
- success criteria and citations

BuildSpec should not specify unless necessary:

- exact Python class names
- whether the method file is `src/method.py` or `src/methods.py`
- internal function signatures
- precise smoke-test commands
- exact plotting helper names

Those engineering details belong in `ImplementationContract.json`, generated as the first staged CODE artifact.

## Runtime Prompt Template
```text
You are converting an approved research idea into a compact implementation source of truth.

The BuildSpec must be executable by downstream CODE and WRITE stages without reading the full prior history.

ResearchBrief:
{{research_brief}}

Approved IdeaCard:
{{idea_card}}

IdeaDecision:
{{idea_decision}}

Supporting EvidenceCards:
{{supporting_evidence}}

Optional RepoCards:
{{repo_cards}}

Return exactly one valid JSON object and nothing else.
Do not include markdown or explanations.

Required schema:
{
  "build_id": "build-<idea_id>",
  "idea_id": "<approved idea id>",
  "target_task": "<specific task>",
  "problem_statement": "<concise problem statement>",
  "method_summary": "<proposed method summary>",
  "repo_url": "",
  "implementation_plan": ["step 1", "step 2"],
  "experiment_plan": ["experiment 1", "experiment 2"],
  "baselines": ["baseline"],
  "metrics": ["primary metric"],
  "dataset": {
    "name": "dataset or scenario family",
    "public_sources": [
      {
        "name": "small credible public source if known",
        "url": "https://...",
        "reason": "why it supports the experiment"
      }
    ],
    "fallback_policy": "try public sources first; use domain-appropriate synthetic smoke data after acquisition/load failures"
  },
  "logging": [
    {
      "path": "results/progress_log.jsonl",
      "record_type": "execution",
      "fields": ["epoch", "metric", "timestamp"],
      "x_axis": "epoch",
      "interval": "log_interval",
      "description": "Record the domain progress metric over experiment epochs/episodes."
    }
  ],
  "plots": [
    {
      "path": "results/progress_curve.png",
      "title": "Progress of the primary metric",
      "source": "log",
      "x": "epoch",
      "y": "metric",
      "kind": "line",
      "series": "method",
      "description": "Show how the primary metric changes during the run."
    },
    {
      "path": "results/eval_curve.png",
      "title": "Final method comparison",
      "source": "results_table",
      "x": "method",
      "y": "metric",
      "kind": "bar",
      "series": "method",
      "description": "Compare proposed and baseline methods on the primary metric."
    },
    {
      "path": "results/secondary_metric_curve.png",
      "title": "Secondary metric analysis",
      "source": "results_table",
      "x": "method",
      "y": "secondary metric",
      "kind": "bar",
      "series": "method",
      "description": "Compare methods on an additional BuildSpec metric or diagnostic."
    }
  ],
  "success_criteria": ["criterion"],
  "artifacts_required": {
    "coder": ["ImplementationContract.json", "CoreImplementationReport.json", "ExperimentImplementationReport.json", "working code module", "experiment logs", "result table/figures"],
    "writer": ["latex section draft"]
  },
  "environment": {
    "source": "generated|reference_repo|reference_repo_metadata",
    "reference_repo_url": "",
    "reference_repo_path": "",
    "env_files": [],
    "language": "python",
    "framework": "",
    "requirements": ["python>=3.11"],
    "setup_commands": ["pip install -r requirements.txt"]
  },
  "paper_outline": ["Abstract", "Introduction", "Related Work", "Preliminaries", "Methods", "Experiments", "Conclusion", "Appendix"],
  "citations_required": ["paper_id_or_evidence_source"]
}
```
