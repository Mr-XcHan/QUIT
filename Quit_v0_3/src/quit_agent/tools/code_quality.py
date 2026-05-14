from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from quit_agent.schemas.build_spec import BuildSpec


def evaluate_code_quality(
    *,
    run_dir: Path,
    spec: BuildSpec,
    returncode: int | None = 0,
    stdout: str = "",
) -> dict[str, Any]:
    """Evaluate whether generated CODE artifacts satisfy the BuildSpec contract."""
    metrics_path = run_dir / "results" / "metrics.json"
    table_path = run_dir / "results" / "results_table.csv"
    summary_table_path = run_dir / "results" / "summary_table.csv"
    log_paths = _expected_log_paths(run_dir, spec)
    plot_paths = _expected_plot_paths(run_dir, spec)
    training_paths = log_paths + plot_paths
    metrics = _load_metrics_json(metrics_path)
    rows = _load_results_rows(table_path)
    required_metrics = [str(item).strip() for item in spec.metrics if str(item).strip()]
    numeric_rows = [row for row in rows if _row_has_numeric_result(row, required_metrics)]
    generated_numeric_rows = [
        row for row in numeric_rows
        if str(row.get("Source") or row.get("source") or "").strip().lower() != "paper_reported"
    ]
    generated_baseline_rows = [
        row for row in generated_numeric_rows
        if _is_generated_baseline_row(row)
    ]
    method_names = {
        str(row.get("method") or row.get("Method") or "").strip().lower()
        for row in generated_numeric_rows
        if str(row.get("method") or row.get("Method") or "").strip()
    }
    baseline_names = [item for item in spec.baselines if str(item).strip()]
    covered_metrics = _covered_metric_names(metrics, rows)
    missing_metrics = [
        metric for metric in required_metrics
        if not _metric_is_covered(metric, covered_metrics)
    ]
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if returncode not in {0, None}:
        failures.append({"rule": "experiment_returncode", "reason": "experiment command failed", "returncode": returncode})
    if not metrics_path.exists():
        failures.append({"rule": "missing_metrics_json", "reason": "results/metrics.json was not produced"})
    elif not isinstance(metrics, dict):
        failures.append({"rule": "invalid_metrics_json", "reason": "results/metrics.json is not a JSON object"})
    if not table_path.exists():
        failures.append({"rule": "missing_results_table", "reason": "results/results_table.csv was not produced"})
    elif not rows:
        failures.append({"rule": "empty_results_table", "reason": "results/results_table.csv has no data rows"})
    if table_path.exists() and len(generated_numeric_rows) < 2:
        failures.append(
            {
                "rule": "insufficient_numeric_results",
                "reason": "results_table.csv must contain at least two generated rows with numeric metrics",
                "generated_numeric_row_count": len(generated_numeric_rows),
            }
        )
    if generated_numeric_rows and not summary_table_path.exists():
        failures.append(
            {
                "rule": "missing_summary_table",
                "reason": "CODE must write results/summary_table.csv with compact paper-facing labels and key metrics when numeric result rows exist",
            }
        )
    if baseline_names and len(method_names) < 2:
        failures.append(
            {
                "rule": "missing_baseline_or_ablation_rows",
                "reason": "BuildSpec lists baselines, but CODE did not produce comparable generated numeric rows",
                "baseline_count": len(baseline_names),
                "methods_seen": sorted(method_names),
            }
        )
    if baseline_names and len(generated_baseline_rows) < len(baseline_names):
        failures.append(
            {
                "rule": "missing_generated_baseline_rows",
                "reason": "Each BuildSpec baseline must have a generated numeric result row; paper-reported rows do not count",
                "baseline_count": len(baseline_names),
                "generated_baseline_row_count": len(generated_baseline_rows),
                "baseline_methods_seen": [
                    str(row.get("method") or row.get("Method") or "").strip()
                    for row in generated_baseline_rows
                ],
            }
        )
    zero_baseline_rows = [
        str(row.get("method") or row.get("Method") or "").strip() or "<unknown>"
        for row in generated_baseline_rows
        if _all_required_metric_values_zero(row, required_metrics)
    ]
    if zero_baseline_rows:
        failures.append(
            {
                "rule": "zero_placeholder_baseline_results",
                "reason": "Generated baseline rows cannot report all-zero BuildSpec metrics; train/evaluate the baseline or fail clearly",
                "baseline_methods": zero_baseline_rows,
            }
        )
    baselines_without_provenance = [
        str(row.get("method") or row.get("Method") or "").strip() or "<unknown>"
        for row in generated_baseline_rows
        if not _has_baseline_evaluation_provenance(row)
    ]
    if baselines_without_provenance:
        failures.append(
            {
                "rule": "missing_baseline_evaluation_source",
                "reason": "Each generated baseline row must record a real training/evaluation provenance column, not only a generic baseline label",
                "baseline_methods": baselines_without_provenance,
                "accepted_columns": sorted(_BASELINE_PROVENANCE_COLUMNS),
            }
        )
    if missing_metrics:
        failures.append(
            {
                "rule": "missing_build_spec_metrics",
                "reason": "CODE did not report all metrics requested by BuildSpec.metrics",
                "missing_metrics": missing_metrics,
                "covered_metrics": sorted(covered_metrics),
            }
        )
    existing_logs = [p for p in log_paths if p.exists()]
    if log_paths and not existing_logs:
        failures.append(
            {
                "rule": "missing_experiment_log",
                "reason": "CODE did not produce any BuildSpec-declared experiment log",
                "expected_logs": [str(p.relative_to(run_dir)) for p in log_paths],
            }
        )
    for log_spec in getattr(spec, "logging", []):
        if not getattr(log_spec, "path", ""):
            continue
        log_path = _run_relative_path(run_dir, log_spec.path)
        if not log_path.exists():
            continue
        missing_fields = _missing_jsonl_fields(log_path, [str(field).strip() for field in log_spec.fields if str(field).strip()])
        if missing_fields:
            failures.append(
                {
                    "rule": "missing_declared_log_fields",
                    "reason": "experiment log does not contain fields declared in BuildSpec.logging",
                    "log_path": str(log_path.relative_to(run_dir)),
                    "missing_fields": missing_fields,
                }
            )
        progress_issue = _progress_log_quality_issue(log_path, log_spec, spec)
        if progress_issue:
            failures.append(progress_issue)
    if not getattr(spec, "logging", []):
        for log_path in existing_logs:
            progress_issue = _progress_log_quality_issue(log_path, None, spec)
            if progress_issue:
                failures.append(progress_issue)
    existing_images = [p for p in plot_paths if p.exists()]
    missing_images = [p for p in plot_paths if not p.exists()]
    if missing_images:
        failures.append(
            {
                "rule": "missing_result_plots",
                "reason": "CODE must produce the figure artifacts declared in BuildSpec.plots",
                "missing_images": [str(p.relative_to(run_dir)) for p in missing_images],
                "existing_images": [str(p.relative_to(run_dir)) for p in existing_images],
            }
        )
    for image_path in existing_images:
        image_issue = _plot_quality_issue(image_path, run_dir, spec)
        if image_issue:
            failures.append(image_issue)

    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "warnings": warnings,
        "metrics_json_available": metrics_path.exists(),
        "results_table_available": table_path.exists(),
        "summary_table_available": summary_table_path.exists(),
        "progress_artifacts_available": [str(path.relative_to(run_dir)) for path in training_paths if path.exists()],
        "training_artifacts_available": [str(path.relative_to(run_dir)) for path in training_paths if path.exists()],
        "result_row_count": len(rows),
        "generated_numeric_row_count": len(generated_numeric_rows),
        "generated_baseline_row_count": len(generated_baseline_rows),
        "method_names": sorted(method_names),
        "required_metrics": required_metrics,
        "covered_metrics": sorted(covered_metrics),
        "missing_metrics": missing_metrics,
        "expected_logs": [str(path.relative_to(run_dir)) for path in log_paths],
        "expected_plots": [str(path.relative_to(run_dir)) for path in plot_paths],
    }


def _progress_log_quality_issue(log_path: Path, log_spec: Any, spec: BuildSpec) -> dict[str, Any] | None:
    if not log_path.exists() or log_path.suffix.lower() not in {".jsonl", ".ndjson", ".json"}:
        return None
    rows = _load_json_records(log_path)
    if not rows:
        return {
            "rule": "empty_progress_log",
            "reason": "progress log exists but contains no parseable records",
            "log_path": str(log_path),
        }
    x_key = str(getattr(log_spec, "x_axis", "") or "epoch")
    x_values = {
        str(row.get(x_key, row.get("epoch", row.get("step", ""))))
        for row in rows
        if isinstance(row, dict)
    }
    x_values.discard("")
    iterative = _spec_implies_iterative_progress(spec)
    if iterative and len(x_values) < 2:
        return {
            "rule": "single_point_progress_log",
            "reason": "iterative experiments must log multiple progress/candidate/epoch x-values; a single terminal point is not a valid progress curve",
            "log_path": str(log_path),
            "x_axis": x_key,
            "unique_x_values": sorted(x_values),
            "record_count": len(rows),
        }
    return None


def _plot_quality_issue(image_path: Path, run_dir: Path, spec: BuildSpec) -> dict[str, Any] | None:
    if not image_path.exists():
        return None
    rel = str(image_path.relative_to(run_dir))
    size = image_path.stat().st_size
    dims = _png_dimensions(image_path)
    if size < 8_000:
        return {
            "rule": "undersized_result_plot",
            "reason": "result plot is too small to be a useful paper artifact",
            "image": rel,
            "bytes": size,
        }
    if dims is not None:
        width, height = dims
        if width < 700 or height < 400:
            return {
                "rule": "low_resolution_result_plot",
                "reason": "result plot resolution is too small for paper use",
                "image": rel,
                "width": width,
                "height": height,
            }
        if image_path.name == "eval_curve.png" and len(spec.metrics) >= 3 and not (width >= 1200 and height >= 700):
            return {
                "rule": "eval_plot_not_summary_figure",
                "reason": "eval_curve.png should be a paper-ready multi-panel summary when at least three numeric reporting targets exist",
                "image": rel,
                "width": width,
                "height": height,
                "metric_count": len(spec.metrics),
            }
    return None


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            for key in ("records", "progress", "history", "logs"):
                items = value.get(key)
                if isinstance(items, list):
                    return [item for item in items if isinstance(item, dict)]
            return [value]
        return []
    records = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _spec_implies_iterative_progress(spec: BuildSpec) -> bool:
    if getattr(spec, "max_train_epochs", 1) > 1 or getattr(spec, "max_eval_epochs", 1) > 1:
        return True
    text = " ".join(
        [
            spec.target_task,
            spec.problem_statement,
            spec.method_summary,
            " ".join(spec.implementation_plan),
            " ".join(spec.experiment_plan),
        ]
    ).lower()
    iterative_terms = [
        "epoch",
        "iteration",
        "iterative",
        "candidate",
        "checkpoint",
        "episode",
        "step",
        "simulation",
        "training",
        "optimization",
        "search",
        "sensitivity",
    ]
    return any(term in text for term in iterative_terms)


def _png_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
    except OSError:
        return None
    if len(header) < 24 or not header.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


def _missing_jsonl_fields(path: Path, required_fields: list[str]) -> list[str]:
    if not required_fields:
        return []
    try:
        first_line = next((line for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()), "")
    except OSError:
        return required_fields
    if not first_line:
        return required_fields
    try:
        record = json.loads(first_line)
    except json.JSONDecodeError:
        return required_fields
    if not isinstance(record, dict):
        return required_fields
    present_aliases: set[str] = set()
    for key in record:
        present_aliases.update(_metric_aliases(str(key)))
    missing = [
        field for field in required_fields
        if not (_metric_aliases(field) & present_aliases)
    ]
    return missing


def _run_relative_path(run_dir: Path, raw_path: str) -> Path:
    path = Path(str(raw_path).strip())
    if not str(path):
        return run_dir / "results" / "artifact"
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "..":
        path = Path(*parts[1:])
        parts = path.parts
    if parts and parts[0] == "code":
        path = Path(*parts[1:])
    return run_dir / path


def _expected_log_paths(run_dir: Path, spec: BuildSpec) -> list[Path]:
    declared = [
        _run_relative_path(run_dir, item.path)
        for item in getattr(spec, "logging", [])
        if getattr(item, "path", "")
    ]
    if declared:
        return declared
    return [
        run_dir / "results" / "progress_log.json",
        run_dir / "results" / "progress_log.jsonl",
        run_dir / "results" / "training_log.json",
        run_dir / "results" / "training_log.jsonl",
        run_dir / "results" / "execution_log.jsonl",
    ]


def _expected_plot_paths(run_dir: Path, spec: BuildSpec) -> list[Path]:
    declared = [
        _run_relative_path(run_dir, item.path)
        for item in getattr(spec, "plots", [])
        if getattr(item, "path", "")
    ]
    if declared:
        return declared
    return [
        run_dir / "results" / "progress_curve.png",
        run_dir / "results" / "eval_curve.png",
    ]


def _load_metrics_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_results_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row_has_numeric_result(row: dict[str, Any], required_metrics: list[str]) -> bool:
    return any(
        _parse_float(value) is not None
        and any(_metric_is_covered(metric, {str(key)}) for metric in required_metrics)
        for key, value in row.items()
    )


def _is_generated_baseline_row(row: dict[str, Any]) -> bool:
    source = str(row.get("Source") or row.get("source") or "").strip().lower()
    if source == "paper_reported":
        return False
    method = str(row.get("method") or row.get("Method") or "").strip().lower()
    if source == "baseline":
        return True
    if source in {"proposed", "method", "ours", "ablation", "sensitivity"}:
        return False
    if method in {"", "proposed", "ours"} or method.startswith("proposed"):
        return False
    return True


def _all_required_metric_values_zero(row: dict[str, Any], required_metrics: list[str]) -> bool:
    values = [
        parsed
        for key, value in row.items()
        for parsed in [_parse_float(value)]
        if parsed is not None and any(_metric_is_covered(metric, {str(key)}) for metric in required_metrics)
    ]
    return bool(values) and all(abs(value) <= 1e-12 for value in values)


_BASELINE_PROVENANCE_COLUMNS = {
    "dataset_size",
    "eval_samples",
    "evaluation_protocol",
    "evaluation_source",
    "eval_protocol",
    "eval_source",
    "fit_status",
    "n_eval_samples",
    "num_eval_episodes",
    "offline_eval_samples",
    "train_status",
    "trained",
    "training_steps",
}


def _has_baseline_evaluation_provenance(row: dict[str, Any]) -> bool:
    for key, value in row.items():
        normalized_key = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
        if normalized_key not in _BASELINE_PROVENANCE_COLUMNS:
            continue
        text = str(value).strip().lower()
        if text and text not in {"0", "false", "none", "n/a", "na", "placeholder"}:
            return True
    return False


def _covered_metric_names(metrics: Any, rows: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if str(key).strip() and _parse_float(value) is not None:
                names.add(str(key))
        metric_name = row.get("metric") or row.get("Metric")
        value = row.get("value") or row.get("Value")
        if metric_name and _parse_float(value) is not None:
            names.add(str(metric_name))
    _collect_metric_names_from_json(metrics, names)
    return names


def _collect_metric_names_from_json(value: Any, names: set[str], prefix: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if _parse_float(item) is not None:
                names.add(key_text)
                names.add(path)
            else:
                _collect_metric_names_from_json(item, names, path)
    elif isinstance(value, list):
        for item in value:
            _collect_metric_names_from_json(item, names, prefix)


def _metric_is_covered(required: str, covered: set[str]) -> bool:
    required_aliases = _metric_aliases(required)
    return any(required_aliases & _metric_aliases(metric) for metric in covered)


def _metric_aliases(value: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    compact = normalized.replace("_", "")
    aliases = {normalized, compact}
    tokens = {token for token in normalized.split("_") if len(token) >= 3}
    aliases.update(tokens)
    synonyms = {
        "iou": {"iou", "intersectionoverunion", "intersection_union"},
        "chamfer": {"chamfer", "chamferdistance"},
        "sdf": {"sdf", "signeddistance", "signeddistanceerror"},
        "occupancy": {"occupancy", "occupancyerror"},
        "runtime": {"runtime", "time", "latency"},
        "latency": {"latency", "runtime", "time"},
        "memory": {"memory", "peakmemory", "peak_memory"},
        "connectivity": {"connectivity", "connectivityerror"},
        "boundary": {"boundary", "boundarymismatch"},
        "thickness": {"thickness", "minimumthickness", "minimum_thickness"},
        "watertightness": {"watertightness", "watertight"},
        "nonmanifold": {"nonmanifold", "non_manifold"},
        "diversity": {"diversity", "stylediversity"},
        "mask": {"mask", "maskadherence"},
        "region": {"region", "regioncompliance"},
        "chamferdistance": {"chamfer", "chamferdistance"},
    }
    for alias in list(aliases):
        aliases.update(synonyms.get(alias, set()))
    for token in list(tokens):
        aliases.update(synonyms.get(token, set()))
    return {alias for alias in aliases if alias}


def _training_log_lines(stdout_lines: list[str]) -> list[str]:
    return [line for line in stdout_lines if "loss" in line.lower() or "[train]" in line.lower()]


def _parse_float(value: Any) -> float | None:
    try:
        if value in {None, "", "N/A"}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
