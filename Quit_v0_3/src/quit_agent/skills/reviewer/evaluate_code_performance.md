# Skill: Evaluate Code Performance (CODE_PERF_EVAL)

## Description
LLM-based judgment of whether the proposed method outperforms baselines overall.
Called after the hard CODE_EVAL checks pass. Reads experiment results and decides
whether the method is genuinely better. If not, it emits concrete repair hints
(hyperparameter tuning or algorithm improvement) that are fed back into the next
CODE repair iteration.

## Input Artifacts
```
results/metrics.json       — final metric values
results/results_table.csv  — per-method rows with metric columns
results/summary_table.csv  — compact paper-facing table (optional)
BuildSpec.json             — target task, metrics, baselines
```

## Output Schema
Return exactly one JSON object:

```json
{
  "verdict": "PASS" | "FAIL",
  "reason": "<one sentence overall judgment>",
  "per_metric": [
    {"metric": "<name>", "proposed": <value>, "best_baseline": <value>, "winner": "proposed" | "baseline" | "tie"}
  ],
  "repair_hints": ["<concrete hint 1>", "<concrete hint 2>"]
}
```

- `verdict`: **PASS** if the proposed method wins on the majority of metrics overall.
  A minority of weaker metrics is acceptable as long as the method leads in most.
  **FAIL** if the baseline is better on more metrics than the proposed method.
- `reason`: one clear sentence summarising your overall judgment.
- `per_metric`: list one entry per BuildSpec metric that has numeric data for both
  proposed and at least one baseline.
- `repair_hints`: only when verdict is FAIL — list 2–4 concrete, actionable suggestions
  such as increasing training epochs, adjusting learning rate, batch size, network
  capacity, regularisation strength, or fixing a specific algorithmic weakness you
  observe in the results.

## Runtime Prompt Template
```text
You are evaluating whether a proposed research method outperforms its baselines.

BuildSpec:
{{build_spec}}

results/metrics.json:
{{metrics_json}}

results/results_table.csv:
{{results_table}}

Judge whether the proposed method outperforms the baselines overall.
The method is allowed to be weaker on a minority of metrics, but it must win
on the majority of BuildSpec metrics to receive a PASS verdict.

Return exactly one JSON object with no surrounding text:

{
  "verdict": "PASS or FAIL",
  "reason": "<one sentence>",
  "per_metric": [
    {"metric": "<name>", "proposed": <number>, "best_baseline": <number>, "winner": "proposed | baseline | tie"}
  ],
  "repair_hints": ["<hint — only when FAIL>"]
}
```
