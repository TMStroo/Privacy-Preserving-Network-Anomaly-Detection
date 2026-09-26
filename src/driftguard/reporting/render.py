"""Render README and report content from experiment output.

Every number this module emits is read from a recorded experiment directory. No
metric is written by hand, so the documentation cannot drift from the run that
produced it.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

BEGIN = "<!-- BEGIN RESULTS -->"
END = "<!-- END RESULTS -->"


def load_experiment(path: str) -> Dict:
    with open(Path(path) / "metrics.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_metadata(path: str) -> Dict:
    meta_path = Path(path) / "metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_experiment(experiments_dir: str, dataset: Optional[str] = None) -> Optional[str]:
    """The most recent finished run, preferring a real dataset.

    Sorting by directory name picks the synthetic fixture, because its
    directory is the newest. The fixture exists so the test suite can run the
    whole pipeline in seconds without a multi-gigabyte download, and it is
    never a research result. This function chose it, and the report was
    generated from 40,000 synthetic rows while the completed UNSW-NB15
    benchmark sat on disk unread: every table in the document described the
    fixture rather than the experiment.

    Where more than one real dataset has been run, ``dataset`` pins the choice.
    Without it the report followed alphabetical order into UGR'16 and then
    printed that run's numbers under a heading reading "UNSW-NB15", which is
    the same class of error one level up.
    """
    base = Path(experiments_dir)
    if not base.exists():
        return None
    candidates = [d for d in base.iterdir() if (d / "metrics.json").exists()]
    if not candidates:
        return None
    real = [d for d in candidates if "synthetic" not in d.name]
    pool = real or candidates
    if dataset:
        named = [d for d in pool if f"_{dataset}_" in d.name]
        if named:
            pool = named
    return str(sorted(pool, key=lambda p: p.name)[-1])


def all_experiments(experiments_dir: str) -> List[str]:
    base = Path(experiments_dir)
    if not base.exists():
        return []
    return [str(d) for d in sorted(base.iterdir()) if (d / "metrics.json").exists()]


def _fmt(value, digits: int = 4, dash: str = "n/a") -> str:
    if value is None:
        return dash
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        return f"{value:,}"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def main_comparison_table(metrics: Dict) -> str:
    """Backtest vs forward test per model. The two are never merged."""
    lines = [
        "| Model | Backtest F1 | Forward F1 | F1 degradation | Backtest recall | Forward recall | "
        "Backtest FPR | Forward FPR | FPR increase |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for entry in metrics["models"]:
        result = entry["result"]
        backtest, forward = result["backtest"], result.get("forward")
        lines.append(
            "| {model} | {bt_f1} | {fw_f1} | {deg} | {bt_recall} | {fw_recall} | {bt_fpr} | {fw_fpr} | {fpr_inc} |".format(
                model=result["model"],
                bt_f1=_fmt(backtest["f1"]),
                fw_f1=_fmt(forward["f1"]) if forward else "n/a",
                deg=_fmt(result.get("f1_degradation"), 4) if forward else "n/a",
                bt_recall=_fmt(backtest["recall"]),
                fw_recall=_fmt(forward["recall"]) if forward else "n/a",
                bt_fpr=_fmt(backtest["false_positive_rate"]),
                fw_fpr=_fmt(forward["false_positive_rate"]) if forward else "n/a",
                fpr_inc=_fmt(result.get("fpr_increase"), 4) if forward else "n/a",
            )
        )
    return "\n".join(lines)


def threshold_table(metrics: Dict) -> str:
    """Metrics at the fixed false-positive budget."""
    lines = [
        "| Model | Target FPR | Threshold | Validation recall | Validation precision | Validation F1 |",
        "|---|---|---|---|---|---|",
    ]
    for entry in metrics["models"]:
        result = entry["result"]
        budget = result["backtest"].get("target_fpr_metrics")
        if not budget:
            continue
        lines.append(
            f"| {result['model']} | {_fmt(budget.get('target_fpr'), 3)} | {_fmt(result['threshold'], 4)} | "
            f"{_fmt(budget['recall'])} | {_fmt(budget['precision'])} | {_fmt(budget['f1'])} |"
        )
    return "\n".join(lines)


def drift_table(drift_events_csv: str) -> str:
    if not os.path.exists(drift_events_csv):
        return "_No drift events recorded._"
    import pandas as pd

    table = pd.read_csv(drift_events_csv)
    if table.empty:
        return "_No drift events recorded._"
    grouped = table.groupby(["method", "feature"])["alert"].agg(["sum", "count"]).reset_index()
    grouped["rate"] = grouped["sum"] / grouped["count"]
    grouped = grouped.sort_values(["method", "rate"], ascending=[True, False]).head(20)
    lines = ["| Method | Feature | Alerting windows | Windows | Alert rate |", "|---|---|---|---|---|"]
    for _, row in grouped.iterrows():
        lines.append(
            f"| {row['method']} | {row['feature']} | {int(row['sum'])} | {int(row['count'])} | {_fmt(row['rate'], 3)} |"
        )
    return "\n".join(lines)


def adaptation_table(metrics: Dict) -> str:
    """Forward performance per adaptation strategy, per model."""
    lines = [
        "| Model | Strategy | Forward F1 | Forward recall | Forward FPR | F1 recovery % | Rows used | Refits |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for entry in metrics["models"]:
        for strategy in entry["adaptation"]["strategies"]:
            if "metrics" not in strategy:
                lines.append(f"| {entry['result']['model']} | {strategy['strategy']} | _error: {strategy.get('error', '?')}_ | | | | | |")
                continue
            recovery = strategy.get("recovery", {}).get("f1_recovery_pct")
            lines.append(
                f"| {entry['result']['model']} | {strategy['strategy']} | {_fmt(strategy['metrics']['f1'])} | "
                f"{_fmt(strategy['metrics']['recall'])} | {_fmt(strategy['metrics']['false_positive_rate'])} | "
                f"{_fmt(recovery, 1)} | {strategy.get('rows_used', 0):,} | {strategy.get('refits', 0)} |"
            )
    return "\n".join(lines)


def calibration_table(experiments_dir: str, experiment_path: str) -> str:
    """Brier and ECE on the backtest versus the forward test."""
    lines = ["| Model | Brier (backtest) | Brier (forward) | ECE (backtest) | ECE (forward) |", "|---|---|---|---|---|"]
    base = Path(experiment_path)
    for path in sorted(base.glob("calibration_*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        model = path.stem.replace("calibration_", "")
        backtest, forward = data.get("backtest", {}), data.get("forward", {})
        lines.append(
            f"| {model} | {_fmt(backtest.get('brier'))} | {_fmt(forward.get('brier'))} | "
            f"{_fmt(backtest.get('ece'))} | {_fmt(forward.get('ece'))} |"
        )
    return "\n".join(lines)


def failure_analysis_blocks(experiment_path: str) -> Dict[str, object]:
    """Pull the concrete failure findings a model produced."""
    base = Path(experiment_path)
    out: Dict[str, object] = {}
    for path in sorted(base.glob("failure_analysis_*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            out[path.stem.replace("failure_analysis_", "")] = json.load(handle)
    return out


def results_block(experiments_dir: str) -> str:
    """The full generated block written into README.md between markers."""
    experiment = latest_experiment(experiments_dir)
    if not experiment:
        return ("_No completed experiment found. Run "
                "`python -m driftguard benchmark --config configs/benchmark.yaml` first._")

    metrics = load_experiment(experiment)
    meta = load_metadata(experiment)
    split = metrics["split"]
    parts = [
        f"_Generated from experiment `{Path(experiment).name}` "
        f"(dataset `{metrics['dataset']}`, commit `{str(meta.get('git_commit', '?'))[:8]}`). "
        f"Regenerate with `python -m driftguard benchmark`._",
        "",
        "### Periods",
        "",
        f"- Train: {split['train_period'][0]} to {split['train_period'][1]} "
        f"({split['train_count']['rows']:,} rows, {split['train_count']['attack_rate']:.1%} attacks)",
        f"- Validation: {split['validation_period'][0]} to {split['validation_period'][1]} "
        f"({split['validation_count']['rows']:,} rows)",
        f"- Backtest: {split['backtest_period'][0]} to {split['backtest_period'][1]} "
        f"({split['backtest_count']['rows']:,} rows, {split['backtest_count']['attack_rate']:.1%} attacks)",
    ]
    if "forward_period" in split:
        parts.append(
            f"- Forward test: {split['forward_period'][0]} to {split['forward_period'][1]} "
            f"({split['forward_count']['rows']:,} rows, {split['forward_count']['attack_rate']:.1%} attacks)"
        )
    parts += [
        "",
        "### Backtest versus forward test",
        "",
        main_comparison_table(metrics),
        "",
        "### At the fixed false-positive budget",
        "",
        threshold_table(metrics),
        "",
        "### Adaptation",
        "",
        adaptation_table(metrics),
        "",
        "### Uncertainty and calibration",
        "",
        calibration_table(experiments_dir, experiment),
    ]
    return "\n".join(parts)


def render_readme(readme_path: str, experiments_dir: str) -> bool:
    """Replace the generated block in README.md. Returns True if it changed."""
    path = Path(readme_path)
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        return False
    head, _, rest = text.partition(BEGIN)
    _, _, tail = rest.partition(END)
    updated = f"{head}{BEGIN}\n{results_block(experiments_dir)}\n{END}{tail}"
    if updated != text:
        path.write_text(updated, encoding="utf-8")
        return True
    return False
