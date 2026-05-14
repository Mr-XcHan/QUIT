# Skill: Code Stage Eval Repair

Audit and repair generated code against `BuildSpec.json` and `ImplementationContract.json`.

Use this after any code stage fails or after the full runner succeeds but output artifacts do not satisfy the contract.

This skill is an iterative LLM repair workflow. It must feed the model the real error information from the latest execution/audit and let the model infer the repair. Do not rely only on hard-coded Python-specific patches or fixed pattern substitutions.

## Inputs

Required:

```text
BuildSpec.json
ImplementationContract.json
code/
latest command stdout/stderr or CodeRunReport.json
CodeSyntaxReport.json
```

Optional:

```text
CoreImplementationReport.json
ExperimentImplementationReport.json
code/CoreImplementationReport.json
code/ExperimentImplementationReport.json
ExperimentAudit.json
results/
```

## Required Error Context

Every repair attempt must include the latest concrete failure context. At minimum, pass:

```text
latest_command
latest_returncode
stdout_tail
stderr_tail
traceback_file if any
CodeSyntaxReport.json
CodeRunReport.json if present
CodeRepairReport.json if present
ExperimentAudit.json or CodeEvalQualityReport.json if present
ResultCollectionReport.json if present
current relevant source files
```

The repair model must see the actual error text, not only a summarized rule such as "fix import bug" or "missing plot".

For runtime failures, include the traceback target file plus directly related imports/callees. For contract failures, include the artifact/report that states the missing requirement plus the files responsible for producing it.

## Output Contract

Return minimal complete file replacements using file-marker format:

```text
=== FILE: src/method.py ===
...complete file content...
```

Rules:

- No markdown fences.
- No prose outside file markers.
- Prefer repairing one stage/file.
- Return multiple files only when the interface between files must change.
- Do not rewrite the entire project.

## Repair Retry Policy

One repair pass may not be enough. Do not hard-code a separate retry budget inside this skill. The outer workflow already controls total retries with `runtime.max_steps` / state-machine steps.

Each CODE step should normally perform one model repair attempt:

```text
1. Run the narrowest relevant check first.
2. If it fails, collect stdout/stderr/report tails.
3. Build a repair prompt with BuildSpec, ImplementationContract, latest errors, relevant files, and previous repair history.
4. Ask the model for minimal complete file replacements.
5. Apply returned files.
6. Re-run syntax check and the failed command/audit.
7. Write the repair attempt to CodeRepairReport.json.
8. If it still fails, return FAIL so the state machine can enter CODE again, as allowed by max_steps.
```

Each attempt must write or update a repair report with:

```json
{
  "attempt": 1,
  "command": "...",
  "returncode_before": 1,
  "stderr_tail_before": "...",
  "files_written": [],
  "returncode_after": null,
  "remaining_errors": []
}
```

If a later attempt sees a new error caused by a previous repair, include both the latest error and a short history of previous attempts.

## Repair Policy

Fix the actual failure directly, based on the latest error context:

- Missing canonical file: create the canonical file or a compatibility shim when legacy code is otherwise valid.
- Import mismatch: repair the importing file and/or canonical module.
- Interface mismatch: make the callee match `ImplementationContract.json`.
- Missing dependency: remove unnecessary hard dependency or add a documented optional fallback.
- Missing outputs: repair runner/plot generation instead of fabricating artifacts.
- Missing baseline quality: train/evaluate each BuildSpec baseline on the shared protocol, write non-placeholder numeric metrics, and add a provenance column such as `evaluation_source`, `evaluation_protocol`, `train_status`, `fit_status`, `n_eval_samples`, `num_eval_episodes`, or `dataset_size`.
- Metric key mismatch: if proposed or baseline code returns internal metric keys, repair the evaluation/output layer to derive an explicit mapping from the current `BuildSpec.metrics` and the implemented metric semantics, then map to the exact BuildSpec metric names before writing `metrics.json`, `results_table.csv`, logs, and plots. Never hard-code mappings for one task, and never allow missing metric lookups to silently become all-zero rows.
- Method underperforms baseline overall (`method_underperforms_baseline_overall`): the proposed method must win on the majority of BuildSpec metrics. Repair by: (1) increasing training epochs/steps within the configured budget; (2) tuning key hyperparameters in `configs/experiment_config.json` (learning rate, batch size, network capacity, regularisation, exploration schedule); (3) improving the algorithmic implementation in `src/method.py` if the baseline advantage is large. A few metrics may remain lower, but the method must win on more metrics than it loses. Never fabricate results.
- Poor result artifacts: repair `src/plot.py`, `run_experiment.py`, and logging/table generation so the outputs are paper-ready. In particular, final comparison plots such as `eval_curve.png` must use computed rows, short method labels, the correct primary metric, error bars when uncertainty exists, and multi-panel secondary views when enough metrics exist. Progress plots must use real multi-point progress/candidate/epoch records when the experiment is iterative.
- Unreadable or incomplete tables: add a compact `results/summary_table.csv` with short method labels and the main reporting metrics while preserving the full `results_table.csv`.
- BuildSpec coverage gap: add the missing baseline, metric, sensitivity, ablation, log, or plot.

These bullets are examples, not a deterministic patch list. The model should inspect the provided code and reports, infer the cause, and return a minimal coherent repair.

The repair must not:

- Fabricate metrics.
- Skip baseline execution silently.
- Write all-zero baseline placeholder rows.
- Write baseline result rows without real training/evaluation provenance.
- Produce a bare, unreadable `eval_curve.png` that plots the wrong metric, uses long paragraph labels, omits available uncertainty/domain/runtime views, or mixes computed rows with non-computed paper-reported context rows.
- Produce a fake progress curve from one terminal point per seed when intermediate iterations/candidates/epochs exist.
- Replace a domain method with a generic unrelated fallback.
- Hide exceptions with broad `try/except`.
- Create fake datasets.

## Required Checks After Repair

Run the smallest relevant set first, then broaden:

```text
python -m py_compile run_experiment.py src/*.py
python run_experiment.py --config configs/experiment_config.json
```

Then check:

```text
results/metrics.json
results/results_table.csv
results/summary_table.csv when result rows are available
all BuildSpec logging paths
all BuildSpec plot paths
```
