# Skill: Revise Code for Better Performance (CODE_REVISE)

## Description
Make targeted, minimal edits to improve the proposed method's performance over baselines.
Called after CODE_LLM_EVAL returns a FAIL verdict with concrete repair_hints.
Do NOT regenerate the full codebase — only modify the specific files responsible for
the performance gap.

## When To Use
- `CODE_EVAL` hard check passed.
- `CODE_LLM_EVAL` produced a FAIL verdict with `repair_hints` in `CodePerformanceEval.json`.
- Experiment results exist in `results/`.

## Input Artifacts
```
CodePerformanceEval.json   — verdict, per_metric comparison, repair_hints
BuildSpec.json             — target task, metrics, baselines, success criteria
results/metrics.json       — current numeric results
results/results_table.csv  — current per-method result rows
code/src/method.py         — proposed method implementation
code/src/train.py          — training / execution loop
code/configs/experiment_config.json  — hyperparameters and runtime config
```

## Output Artifacts
Return minimal file replacements in file-marker format, then re-run the experiment.
Typical files to change (only what is needed):
```
code/src/method.py                   — algorithmic improvements
code/src/train.py                    — training loop fixes
code/configs/experiment_config.json  — hyperparameter tuning
```
Also write:
```
CodeReviseReport.json   — what was changed and why
```

## Revision Policy
You are an expert ML engineer fixing a performance gap between the proposed method and baselines.

1. **Read `repair_hints` carefully.** Each hint is a concrete, actionable suggestion.
2. **Identify the root cause.** Look at `per_metric` to see where the gap is largest.
3. **Make targeted edits only.** Change the minimum needed:
   - Hyperparameter tuning (learning rate, batch size, network size, regularisation,
     number of training epochs/steps) → edit `configs/experiment_config.json`
   - Training loop bug or suboptimal schedule → edit `src/train.py`
   - Algorithmic weakness in the core method → edit `src/method.py`
4. **Do not touch** dataset, baseline, plotting, or reporting code unless a hint
   explicitly targets them.
5. **Do not fabricate results.** The experiment will be re-run after edits.
6. The method must outperform baselines on the majority of BuildSpec metrics after revision.
   A minority of weaker metrics is acceptable.

## Output Format
Return file replacements using file-marker format. No markdown fences. No prose outside markers.

```
=== FILE: src/method.py ===
...complete file content...

=== FILE: configs/experiment_config.json ===
...complete file content...
```

## Runtime Prompt Template
```text
You are an ML engineer making targeted fixes to improve a proposed method's performance.

BuildSpec:
{{build_spec}}

LLM Performance Evaluation (CodePerformanceEval.json):
{{perf_eval}}

Current results summary:
{{results_summary}}

Current code files:
{{code_files}}

Based on the repair_hints and per_metric gap, make the minimum targeted edits needed
to help the proposed method outperform the baselines on the majority of metrics.
Tune hyperparameters or fix algorithmic issues as indicated.
Do not regenerate the full codebase. Only return files that need to change.

Return file replacements in file-marker format only. No markdown fences. No prose.
```
