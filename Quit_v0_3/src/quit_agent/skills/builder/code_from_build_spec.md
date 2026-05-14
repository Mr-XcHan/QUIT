# Skill: Code From BuildSpec

Status: compatibility orchestrator.

The preferred CODE workflow is no longer a single large code-generation request. Use these stage skills instead:

```text
BuildSpec.json
  -> implementation_contract_from_build_spec.md
  -> implement_core.md          # dataset -> method -> baselines
  -> implement_experiment.md    # execute -> runner -> plots
  -> code_stage_eval_repair.md
```

`BuildSpec.json` is the research/product specification. `ImplementationContract.json` is the engineering interface contract. Downstream code skills must follow `ImplementationContract.json` exactly for file names, module APIs, output paths, dependencies, and verification checks.

Keep this file only as the old single-shot compatibility prompt until `BuilderAgent.code()` is updated to call the staged skills explicitly.

---

## Legacy Single-Shot Prompt

Generate a standalone Python experiment project from `BuildSpec.json`.

Reference repositories, when available, are read-only implementation context. The generated project must always use the standard `code/` layout and the standard `../results/` artifacts.

## Source Of Truth

Use `BuildSpec.json` as the required input. Use reference repository excerpts only for reusable algorithms, model definitions, data parsing details, hyperparameters, or evaluation formulas.

Do not depend on chat history. Do not preserve repository-specific paths, entry points, plots, tables, logs, or directory structure unless they exactly match this skill's contract.

## Output Artifact

The LLM must return complete file contents using the file-marker format in the runtime prompt. The caller writes those files under:

```text
runs/<run_id>/code/
```

Generated code must write experiment results under:

```text
runs/<run_id>/results/
```

## Runtime Prompt Template
```text
Generate a standalone Python experiment project that implements the BuildSpec below.

The project must be self-contained and runnable from the `code/` directory. If reference repository context is provided, use it only as read-only implementation guidance.

## BuildSpec
{{build_spec}}

## Optional Reference Repository Context
{{reference_repo_context}}

Rules for reference context:
- You may reuse algorithmic ideas, model definitions, data parsing details, hyperparameter conventions, and evaluation formulas.
- Do not copy the repository's directory layout, entry point names, output paths, plot file names, table schema, or logging format unless they exactly match the required output contract below.
- If reference context conflicts with BuildSpec or the output contract, follow BuildSpec and the output contract.

## Previous CODE Feedback
{{feedback}}

## Output Format (MUST follow exactly)
Return every file using this marker format — one header line, then raw content:

=== FILE: README.md ===
(file content here)
=== FILE: requirements.txt ===
(file content here)
=== FILE: environment.yml ===
(file content here)
=== FILE: ENVIRONMENT.md ===
(file content here)
=== FILE: configs/experiment_config.json ===
(file content here)
=== FILE: src/__init__.py ===

=== FILE: src/dataset.py ===
(file content here)
=== FILE: src/method.py ===
(file content here)
=== FILE: src/baselines.py ===
(file content here)
=== FILE: src/train.py ===
(file content here)
=== FILE: src/evaluate.py ===
(file content here)
=== FILE: src/plot.py ===
(file content here)
=== FILE: run_experiment.py ===
(file content here)

- Each file starts with exactly `=== FILE: <path> ===` on its own line.
- Write raw file content directly. No JSON wrapper, no markdown fences, no prose.
- All files must be complete replacements, not diffs.

## Required Runtime Behavior
- `python run_experiment.py --config configs/experiment_config.json` must exit with code 0.
- Implement executable algorithm logic from BuildSpec. Do not output a decorative scaffold.
- Include training or execution loops when the method is trainable.
- Use `train_epochs` and `eval_epochs` from BuildSpec as full dataset passes, not gradient steps.
- Keep `train_epochs` within `[min_train_epochs, max_train_epochs]` and `eval_epochs` within `[min_eval_epochs, max_eval_epochs]`.
- Include `"log_interval": 10` in `configs/experiment_config.json`.
- Follow `BuildSpec.logging` exactly for log file paths and fields. Convert each `results/...` path to `../results/...` when writing from `code/`.
- Treat `BuildSpec.metrics`, `BuildSpec.logging`, and `BuildSpec.plots` as the reporting contract passed from BUILD_SPEC. Do not replace it with a generic train/eval-loss contract.
- Implement every BuildSpec metric and all BuildSpec-declared plot artifacts when enough computed data exists.
- If `BuildSpec.logging` is missing, append JSONL records to `../results/progress_log.jsonl` at each log interval with `epoch`, the primary BuildSpec metric, and `timestamp`.
- Use `train_loss`/`eval_loss` only when the method is genuinely trainable and those losses are real optimization objectives. For simulation, optimization, scheduling, control, queueing, search, or analytical experiments, log domain metrics from BuildSpec.metrics such as profit, satisfaction rate, queue backlog, objective value, regret, runtime, or constraint violation.
- Compute `eval_loss` on a fixed held-out validation batch using evaluation mode and no gradients only when using PyTorch and a real loss objective exists.
- Do not fabricate metrics, losses, result rows, or baseline scores.
- If previous feedback lists failures, fix the underlying code rather than adding stub files.

## Results Contract
Generated code must write:

- `../results/metrics.json`
- `../results/results_table.csv`
- `../results/summary_table.csv` when result rows are available, with compact paper-facing method labels and the main reporting metrics
- every log artifact listed in `BuildSpec.logging` (or `../results/progress_log.jsonl` if missing)
- every figure artifact listed in `BuildSpec.plots` (or `../results/progress_curve.png` and `../results/eval_curve.png` if missing)

`metrics.json`:
- Must be a flat JSON object.
- Must contain every metric listed in the current `BuildSpec.metrics` as an exact key.
- If internal short metric keys are used, derive an explicit mapping from the current `BuildSpec.metrics` and the implemented metric semantics, then map internal keys back to the original BuildSpec metric names before writing outputs. Do not hard-code mappings for one task; mappings must be specific to the current BuildSpec.
- Must preserve enough fields for WRITE to explain the BuildSpec reporting targets across metrics and figures.
- Values must be numeric.

`results_table.csv`:
- Must include `method` and `source` columns.
- Must include one numeric column per metric listed in BuildSpec.metrics, using the exact BuildSpec metric text as the CSV header.
- If method/baseline code uses internal metric keys, define one explicit per-run mapping layer in evaluation/output code and apply it to proposed and baseline rows before writing `metrics.json`, `results_table.csv`, logs, or plots.
- Must contain at least two computed rows: one proposed method row and at least one baseline row.
- Every BuildSpec baseline must have a computed baseline row; `source="paper_reported"` rows are useful context but do not count as generated baseline evaluation.
- Baseline rows must not use all-zero metric placeholders. If a baseline cannot be trained/evaluated, fail clearly instead of writing successful-looking zeros.
- Each computed baseline row must include a provenance column such as `evaluation_source`, `evaluation_protocol`, `train_status`, `fit_status`, `n_eval_samples`, `num_eval_episodes`, or `dataset_size` showing the row came from real training/evaluation.
- Use `source="paper_reported"` only for external numbers that were not computed by this run.
- Do not leave computed metric cells blank or set them to `N/A`.
- Keep `results_table.csv` provenance-rich but still parseable. Put compact presentation labels and selected columns in `summary_table.csv`; do not rely on a later writer stage to clean up unreadable tables.

`src/plot.py`:
- Must be the single plotting module.
- Must write the plot files declared in `BuildSpec.plots` after training/evaluation/execution.
- Each plot must use its declared `source`, `x`, `y`, `kind`, and `series` where possible.
- Generate all declared plots. If a requested y metric is not available under the exact name, use a clear alias from BuildSpec.metrics and document the alias in `metrics.json`.
- If a declared source is unavailable, use the closest real computed artifact and clearly label the plotted metric; do not draw a fake loss curve.
- Use clean publication-style matplotlib defaults: readable labels, legend when useful, grid, and tight layout.
- The final comparison plot, normally `eval_curve.png`, must be paper-ready. If at least three numeric reporting targets exist, it should be a multi-panel summary figure rather than a single bare bar chart: include the primary metric comparison plus at least two secondary views such as uncertainty/error bars, domain breakdown, runtime/cost, robustness/sensitivity, ablation, reward/objective, or constraint violation.
- Method labels in plots must be short and readable. Do not put full paragraph-length baseline descriptions on x-axis ticks.
- Filter `paper_reported`, blank, `N/A`, and non-computed context rows out of computed method-comparison plots unless the plot explicitly compares to literature-reported values.
- Progress plots must be based on real progress records. For iterative training, simulation, search, or candidate-selection workflows, `progress_log.jsonl` must contain multiple x-values; a single terminal point per seed is not a valid curve. For one-shot deterministic workflows, label the plot as a final diagnostic instead of a curve.

## Dataset Contract
- Always prefer real public data when BuildSpec or evidence implies a public dataset.
- In `configs/experiment_config.json`, declare `dataset_path` and, when known, `dataset_source_url`. The outer pipeline will attempt to download candidate public datasets up to 3 times before running the experiment.
- Do not set `dataset.source` to `"synthetic"` in config to bypass acquisition.
- Generated code must support both cases: load the downloaded real dataset when `dataset_path` exists; otherwise run a small domain-appropriate smoke experiment.
- In `src/dataset.py`, implement this policy:

    def load_dataset(config):
        path = config.get("dataset_path")
        if path and pathlib.Path(path).exists():
            return _load_real(path, config)
        return _make_synthetic(config)

- The synthetic fallback is only for failed/missing real-data acquisition and must be domain-appropriate and small enough for local smoke execution.
- Never create fake `.h5` or `.hdf5` files to make missing real data appear present.

## Device And Dependency Contract
- Do not implement custom GPU probing in generated code. Device selection is handled by `tools/device.py` before execution.
- The outer pipeline calls `select_torch_device(requested or "auto")`, where `requested` is discovered from `configs/experiment_config.json`.
- `select_torch_device` resolves `"auto"` or `"gpu"` to `cuda:<index>` with the most free memory when CUDA is available, otherwise `cpu`. Explicit unavailable CUDA requests are downgraded to `cpu`.
- Before `run_experiment.py` is executed, the pipeline patches `configs/experiment_config.json` with concrete fields such as `device`, `runtime.device`, `runtime.resolved_device`, and `runtime.cuda_available`, and propagates the concrete device into nested task configs.
- Generated code should read the concrete device from config, preferably `runtime.resolved_device`, then `runtime.device`, then `device`, defaulting to `cpu` only if all are missing.
- Generated code must use that concrete value consistently for tensors/models and must never pass `"auto"` or `"gpu"` directly to `torch.device`.
- Keep dependencies minimal in `requirements.txt` and `environment.yml`.
- Generated code must still run when reference-repo-only dependencies are unavailable.

## Forbidden Shortcuts
- No decorative scaffolds.
- No invented benchmark scores.
- No skipped baseline rows when BuildSpec lists baselines.
- No all-zero baseline placeholder rows.
- No baseline rows without a real training/evaluation provenance field.
- No missing plots.
- No repo-specific output paths such as `results/...` from inside `code/`; use `../results/...`.
- No broad exception swallowing that hides a failed experiment while writing successful-looking artifacts.
```
