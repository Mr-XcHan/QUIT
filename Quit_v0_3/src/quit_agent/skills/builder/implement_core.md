# Skill: Implement Core

Implement the core experiment modules in one generation pass:

```text
dataset -> method -> baselines
```

This grouped stage is intentionally larger than the old per-file stages to reduce repeated LLM calls and duplicated prompt context.

## Inputs

Required:

```text
BuildSpec.json
ImplementationContract.json
```

Optional:

```text
EvidenceCards.jsonl
Reference repository excerpts
Previous CODE feedback
```

## Output Artifacts

Write complete replacements for:

```text
code/src/__init__.py
code/src/dataset.py
code/src/method.py
code/src/baselines.py
code/configs/experiment_config.json
CoreImplementationReport.json
```

## Required Behavior

- Implement dataset loading, synthetic fallback, preprocessing, splits, and a small smoke dataset path.
- Implement executable proposed-method logic from BuildSpec, not a decorative scaffold.
- Implement every required baseline from BuildSpec unless clearly unavailable with documented reason.
- Keep module APIs consistent with `ImplementationContract.json`.
- Treat `BuildSpec.metrics`, `BuildSpec.baselines`, sensitivity studies, ablations, and success criteria as binding requirements, not suggestions.
- Use one shared metric schema across proposed method and baselines.
- Prefer returning metric dictionaries keyed by the exact `BuildSpec.metrics` names. If internal short keys are useful inside the method, document them and ensure the experiment stage can map them unambiguously to the exact BuildSpec metric names.
- Include every metric listed in `BuildSpec.metrics` in returned metric dictionaries when enough data exists; do not rely on downstream code to convert missing internal keys into zeros.
- Do not replace domain metrics with generic `train_loss` or `eval_loss` unless the method genuinely optimizes those losses.
- Keep defaults small enough for local smoke execution while preserving BuildSpec semantics.
- Avoid optional hard dependencies unless required by the contract; provide documented fallbacks when public datasets or solvers are unavailable.
- Do not create fake `.h5` / `.hdf5` datasets or mock public datasets just to make execution pass.
- If a required dataset artifact is unavailable, expose a clear fallback path to the verified smoke dataset and record the limitation in `CoreImplementationReport.json`.
- **Device selection in `configs/experiment_config.json`**: set `"device"` to `"gpu"` if the method uses neural networks, deep learning, or any GPU-acceleratable computation; set `"device"` to `"cpu"` only if the method is genuinely CPU-only (e.g. pure tabular, tree-based, or symbolic algorithms). The runtime will auto-detect the best available GPU when `"gpu"` is requested and fall back to CPU if none is found. Never hard-code `"cpu"` for a GPU-appropriate task.
- Use resolved device fields only: `runtime.resolved_device`, then `runtime.device`, then `device`, default `"cpu"`.
- Do not pass `"auto"` or `"gpu"` directly to `torch.device`; use the resolved string from config (which will be e.g. `"cuda:0"` or `"cpu"` after patching).
- Do not write final `metrics.json`, `results_table.csv`, or plots; those are owned by the experiment stage.
- Do not create output roots under `code/results`, `code/outputs`, or `code/logs`.
- Do not hide failures with broad `try/except`; optional fallback paths must be explicit and logged.
- Keep generated code importable with only the dependencies declared in `requirements.txt`/`environment.yml` from the experiment stage.

## Shared Interface Contract

The generated core modules must be usable by the next grouped stage:

```text
src.dataset      loads data/scenarios and exposes the contract dataset API
src.method       defines the proposed method/model/policy/optimizer
src.baselines    defines baseline runners/classes using the same data and metrics
```

Prefer simple stable APIs when the contract leaves names open:

```text
load_dataset(config)
ProposedMethod(config)
run_baseline(name, dataset, config)
available_baselines(config)
```

All metric rows returned by core modules must be compatible with:

```text
method,source,<BuildSpec metric 1>,<BuildSpec metric 2>,...
```

Use `source="computed"` for values computed in this run. Use `source="paper_reported"` only for external paper numbers with provenance recorded in `CoreImplementationReport.json`.

## Verification

Minimum checks:

```text
python -m py_compile src/dataset.py src/method.py src/baselines.py
python - <<'PY'
from src.dataset import load_dataset
import src.method as method
import src.baselines as baselines
data = load_dataset({"seed": 0})
assert data is not None and method is not None and baselines is not None
PY
```

Return only file-marker output. No markdown fences, JSON wrapper, prose, or diffs.
