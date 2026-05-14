# Skill: Implement Experiment

Implement the experiment execution layer in one generation pass:

```text
execute -> runner -> plots
```

This stage consumes the already-written core modules and writes the orchestration, evaluation, reporting, and plotting files together so interfaces stay consistent without additional LLM calls.

## Inputs

Required:

```text
BuildSpec.json
ImplementationContract.json
code/src/dataset.py
code/src/method.py
code/src/baselines.py
code/configs/experiment_config.json
```

Optional:

```text
CoreImplementationReport.json
DatasetAcquisitionReport.json
Previous CODE feedback
```

## Output Artifacts

Write complete replacements for:

```text
code/src/train.py
code/src/evaluate.py
code/src/plot.py
code/run_experiment.py
code/configs/experiment_config.json
code/README.md
code/ENVIRONMENT.md
code/requirements.txt
code/environment.yml
ExperimentImplementationReport.json
```

## Required Runtime Behavior

This command must exit with code 0:

```text
python run_experiment.py --config configs/experiment_config.json
```

The runner must write:

```text
../results/metrics.json
../results/results_table.csv
```

It must also write every log artifact from BuildSpec logging, or `../results/progress_log.jsonl` when logging is absent, and every BuildSpec plot artifact, or reasonable default progress/eval figures when plots are absent.

Every `results/...` path declared by BuildSpec must be written from inside `code/` as `../results/...`. Never write final artifacts under `code/results/`, `code/outputs/`, or `code/logs/`.

The experiment stage must make result artifacts paper-ready by default. Do not rely on a later manual inspection step to repair plots or tables.

## Required Behavior

- **Device usage**: read the device from config (`runtime.resolved_device` → `runtime.device` → `device`). Pass this resolved string to `torch.device(...)`. Do not hard-code `"cpu"` or `"cuda"`. If the config says `"gpu"`, the runtime patches it to `"cuda:0"` (or the best available GPU) before execution, so the resolved value will already be a valid `torch.device` string.
- `src/train.py` owns proposed-method training or deterministic execution loops. Use training terminology only when the method is genuinely trained; otherwise treat this as the execution/progress loop.
- `run_experiment.py` owns orchestration, config loading, baseline execution, aggregation, result tables, and final status.
- `src/evaluate.py` owns reusable evaluation helpers when useful.
- `src/plot.py` owns figure generation from real result/log data.
- Do not redefine dataset, method, or baseline implementations in this stage.
- Call the public APIs from `src/dataset.py`, `src/method.py`, and `src/baselines.py`.
- Use one shared results directory: `../results/`.
- Preserve the exact artifact names declared by `BuildSpec.logging` and `BuildSpec.plots`; do not rename, move, or silently replace them with generic files.
- Do not let proposed-method, baseline, sensitivity, or ablation runs overwrite each other silently.
- Aggregate final result rows in memory, then write `metrics.json` and `results_table.csv` once.
- Also write `../results/summary_table.csv` when possible: a compact paper-facing table with short method labels and the primary metric, uncertainty/error metric when available, key secondary metrics, runtime/cost when available, and seeds. Keep the full provenance-rich table in `results_table.csv`.
- Include a proposed method row and every required baseline row.
- Cover every metric in `BuildSpec.metrics` with numeric values or documented aliases.
- For each baseline row, record real evaluation provenance with a column such as `evaluation_source`, `evaluation_protocol`, `train_status`, `fit_status`, `n_eval_samples`, `num_eval_episodes`, or `dataset_size`.
- Do not write all-zero baseline metric rows as placeholders. Train/evaluate the baseline on the shared protocol or fail clearly.
- Follow `BuildSpec.logging` exactly for JSONL fields and x-axis semantics when present. If logging is absent, write `../results/progress_log.jsonl` with `epoch` or `step`, the primary BuildSpec metric, and `timestamp`.
- Progress logs must contain a real progression when a method has iterations, epochs, candidate evaluations, episodes, simulation steps, or hyperparameter search. A single terminal record per seed is only acceptable for genuinely one-shot deterministic experiments; otherwise log intermediate domain metrics or validation/candidate scores.
- Implement all BuildSpec sensitivity studies and ablations when listed. Write separate shared result tables such as `../results/sensitivity_results.csv` when needed, preserving the same metric columns.
- Do not fabricate metrics, losses, curves, or baseline scores.
- Do not write blank placeholder figures when numeric data for the requested plot exists.
- If a declared plot source is unavailable, use the closest real computed artifact and record a clear warning in `metrics.json` or `ExperimentImplementationReport.json`; do not draw a fake loss curve.
- **The proposed method must outperform baselines overall.** It is acceptable to be weaker on a minority of metrics, but the proposed method must win on the majority of BuildSpec metrics. If early training results suggest the method is underperforming, increase training budget, tune key hyperparameters (learning rate, batch size, network capacity, regularization strength), or adjust the method's core implementation before writing final results. Never stop training early if the method has not yet surpassed the baseline.
- Keep default config small enough for local execution while respecting BuildSpec run-count bounds where feasible.

## Results Contract

`metrics.json`:

- Flat JSON object.
- Numeric values only.
- Contains each BuildSpec metric as an exact key.
- Maps any internal method/baseline metric keys to the original current `BuildSpec.metrics` names before writing. Build this mapping dynamically from the current BuildSpec and the implemented metric semantics; do not hard-code mappings for a particular task.

`results_table.csv`:

- Includes `method` and `source` columns.
- Includes one numeric column per BuildSpec metric, using the exact BuildSpec metric text as the header.
- Applies the same explicit per-run metric mapping to proposed and baseline rows; do not let missing key lookups silently become `0.0`.
- Contains proposed and baseline rows.
- Contains one computed row for every BuildSpec baseline; paper-reported rows do not count.
- Baseline rows include real training/evaluation provenance and are not all-zero placeholders.
- Does not use blank, `N/A`, or stringified exception values for computed metrics.
- Main comparison rows must stay in `results_table.csv`; sensitivity and ablation rows may go to separate CSVs but must keep metric columns compatible.
- Method labels in paper-facing artifacts must be readable. Keep long exact baseline descriptions in provenance fields if useful, but provide short labels in `summary_table.csv` and plots.

`src/plot.py`:

- Must write every figure declared in `BuildSpec.plots`.
- Must use the declared `source`, `x`, `y`, `kind`, and `series` where possible.
- Must use real metrics/logs/results-table values.
- Must generate readable publication-style matplotlib plots when matplotlib is available, with labels, legend when useful, grid, and tight layout.
- If matplotlib is unavailable, use a documented lightweight PNG fallback that still represents real numeric values.
- `eval_curve.png` or any final method-comparison plot must be paper-ready, not a bare single-axis bar chart with long labels. When at least three numeric reporting targets exist, render a multi-panel summary figure that includes the primary method comparison and at least two useful secondary views such as uncertainty/error bars, domain breakdowns, runtime/cost, robustness, sensitivity, ablation, or final reward/objective. Use short labels and include error bars when a CI/std/standard-error metric is available.
- `progress_curve.png` or any progress plot must use the real progress log and should contain more than one x-value for iterative/candidate/epoch-based experiments. If only a terminal value exists, plot a clearly labeled final-value diagnostic instead of pretending it is a curve, and record the limitation in `ExperimentImplementationReport.json`.
- Plot code must filter out `paper_reported`, blank, `N/A`, or non-computed context rows from computed method-comparison figures unless the figure is explicitly about literature context.
- Plot code must not silently fall back to the wrong y-axis. If a requested y metric is unavailable, choose a BuildSpec metric alias deliberately, label the plotted metric, and record the alias/warning.

`run_experiment.py`:

- Must exit nonzero only for real unrecovered failures.
- Must leave enough stdout/stderr detail for CODE repair when it fails.
- Must not swallow exceptions and emit successful fake artifacts.

## Verification

Minimum checks:

```text
python -m py_compile run_experiment.py src/train.py src/evaluate.py src/plot.py
python run_experiment.py --config configs/experiment_config.json
test -s ../results/metrics.json
test -s ../results/results_table.csv
```

Return only file-marker output. No markdown fences, JSON wrapper, prose, or diffs.
