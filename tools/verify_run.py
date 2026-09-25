"""Check that a recorded experiment directory is a complete, honest result.

The failure mode this exists to catch is the one that produced the worst bug in
this project: a run that finished, wrote its files, and reported success while
measuring nothing. An empty drift table, a NaN metric, an adaptation strategy
that silently failed, or a metadata block missing the version numbers all look
like success from the outside.

Exits non-zero and prints what is wrong, so CI fails on it.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REQUIRED_MODELS = [
    "majority",
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
]

REQUIRED_STRATEGIES = [
    "none",
    "threshold_recalibration",
    "recent_window_retrain",
    "rolling_window_retrain",
    "historical_plus_recent_retrain",
]

REQUIRED_METADATA = [
    "git_commit",
    "random_seed",
    "feature_schema_hash",
    "package_versions",
    "temporal_split",
    "dataset",
]

REQUIRED_PACKAGES = ["python", "numpy", "pandas", "scikit-learn"]


def _walk_floats(node, path=""):
    """Yield (path, value) for every float in a nested structure."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk_floats(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _walk_floats(value, f"{path}[{i}]")
    elif isinstance(node, float):
        yield path, node


def verify(run_dir: str, strict_models: bool = False) -> list:
    """Return a list of problems. An empty list means the run is complete."""
    problems = []
    run = Path(run_dir)

    if not run.is_dir():
        return [f"{run} is not a directory"]

    metrics_path = run / "metrics.json"
    if not metrics_path.is_file():
        return [f"{metrics_path} is missing, so this run never finished"]

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    meta_path = run / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    if meta.get("superseded"):
        return [f"{run.name} is marked superseded: {meta['superseded'].get('reason', '')[:120]}"]

    # ---- metadata completeness ----
    for field in REQUIRED_METADATA:
        if meta.get(field) in (None, "", {}):
            problems.append(f"metadata.{field} is missing or empty")

    versions = meta.get("package_versions") or {}
    for package in REQUIRED_PACKAGES:
        recorded = versions.get(package)
        if recorded in (None, "", "not installed"):
            problems.append(f"package_versions.{package} is {recorded!r}")

    split = meta.get("temporal_split") or {}
    for period in ("train_period", "validation_period", "backtest_period", "forward_period"):
        if not split.get(period):
            problems.append(f"temporal_split.{period} is missing")
    if not split.get("train_count"):
        problems.append("temporal_split.train_count is missing")

    # ---- models ----
    models = metrics.get("models") or []
    if not models:
        problems.append("metrics.models is empty")
    found = [m.get("result", {}).get("model") for m in models]
    if strict_models:
        for name in REQUIRED_MODELS:
            if name not in found:
                problems.append(f"required model missing: {name}")

    # ---- metrics and NaN ----
    for model in models:
        name = model.get("result", {}).get("model", "?")
        for period in ("backtest", "forward"):
            block = model.get("result", {}).get(period)
            if not isinstance(block, dict):
                problems.append(f"{name}.{period} is missing")
                continue
            for field in ("precision", "recall", "f1", "roc_auc", "pr_auc",
                          "false_positive_rate", "balanced_accuracy"):
                if field not in block:
                    problems.append(f"{name}.{period}.{field} is missing")
        for path, value in _walk_floats(model):
            if math.isnan(value) or math.isinf(value):
                problems.append(f"{name}: {path} is {value}")

        # Calibration is recorded per model in its own file, not inside
        # metrics.json, so it is checked where it is actually written.
        calibration_path = run / f"calibration_{name}.json"
        if not calibration_path.is_file():
            problems.append(f"{name}: calibration_{name}.json is missing")
        else:
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            for period in ("backtest", "forward"):
                block = calibration.get(period) or {}
                for field in ("brier", "ece"):
                    value = block.get(field)
                    if value is None:
                        problems.append(f"{name}.calibration.{period}.{field} is missing")
                    elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                        problems.append(f"{name}.calibration.{period}.{field} is {value}")

    # ---- adaptation ----
    if strict_models:
        for model in models:
            name = model.get("result", {}).get("model", "?")
            adaptation = model.get("adaptation") or {}
            # The strategies are a list of records, each carrying its own name.
            # Reading them as dict keys reports every real run as empty.
            strategies = adaptation.get("strategies") or []
            present = {s.get("strategy") for s in strategies if isinstance(s, dict)}
            missing = [s for s in REQUIRED_STRATEGIES if s not in present]
            if missing:
                problems.append(f"{name}: adaptation strategies missing: {missing}")
            for record in strategies:
                if not isinstance(record, dict):
                    continue
                label = record.get("strategy", "?")
                block = record.get("metrics") or {}
                for field in ("precision", "recall", "f1", "false_positive_rate"):
                    if field not in block:
                        problems.append(
                            f"{name}.{label}.{field} is missing")
                for path, value in _walk_floats(record):
                    if math.isnan(value) or math.isinf(value):
                        problems.append(f"{name}.{label}: {path} is {value}")

    # ---- drift ----
    if not metrics.get("drift_summary"):
        problems.append("drift_summary is empty, so the drift stage measured nothing")
    if not (run / "drift_events.csv").is_file():
        problems.append("drift_events.csv is missing")

    # ---- figures ----
    figures = run / "figures"
    if figures.is_dir():
        pngs = list(figures.glob("*.png"))
        if not pngs:
            problems.append("figures/ exists but contains no png")
        for png in pngs:
            if png.stat().st_size < 1000:
                problems.append(f"figure {png.name} is {png.stat().st_size} bytes, likely empty")

    return problems


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: verify_run.py <experiment_dir> [--strict]", file=sys.stderr)
        return 2
    strict = "--strict" in sys.argv
    target = [a for a in sys.argv[1:] if not a.startswith("--")][0]
    problems = verify(target, strict_models=strict)
    if problems:
        print(f"INCOMPLETE: {target}")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"complete: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
