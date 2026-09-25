#!/usr/bin/env python
"""Render the README results block from recorded experiment output.

Run after the pipeline:

    python scripts/render_results.py

The README must not contain a hand-typed number, because a hand-typed number is
a number nobody checked. This script reads the experiment directories under
``results/experiments/`` and rewrites the block between the generated markers,
so the README can only show what an actual run produced.

It is deliberately quiet about missing experiments: a stage that has not been
run yet is reported as not run, rather than filled with a plausible number.

The V1 version of this script read ``results/metrics/`` and rendered a
FULL/RESTRICTED metadata table. That pipeline still exists under
``src/privacy_preserving_nad/`` and its results are still valid, but the
README now describes DriftGuard, so the renderer follows the experiments the
current pipeline writes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "results" / "experiments"
README = ROOT / "README.md"

BEGIN = "<!-- BEGIN GENERATED RESULTS -->"
END = "<!-- END GENERATED RESULTS -->"

MODEL_ORDER = [
    ("majority_baseline", "Majority baseline"),
    ("logistic_regression", "Logistic Regression"),
    ("random_forest", "Random Forest"),
    ("gradient_boosting", "Gradient Boosting"),
]

STRATEGY_ORDER = [
    ("none", "No adaptation"),
    ("threshold_recalibration", "Threshold recalibration"),
    ("recent_window_retrain", "Recent-window retraining"),
    ("rolling_window_retrain", "Rolling-window retraining"),
    ("historical_plus_recent_retrain", "Historical + recent retraining"),
]

# The pipeline records the unadapted result under its own key rather than under
# the strategy name, because it is the baseline every other strategy is compared
# to. This map is the single place that knows the two names differ.
STRATEGY_KEYS = {
    "none": "no_adaptation",
    "threshold_recalibration": "threshold_recalibration",
    "recent_window_retrain": "recent_window_retrain",
    "rolling_window_retrain": "rolling_window_retrain",
    "historical_plus_recent_retrain": "historical_plus_recent_retrain",
}

NUMERIC_COLUMNS = {
    "f1", "precision", "recall", "roc_auc", "pr_auc", "brier_score",
    "expected_calibration_error", "n_features", "target_fpr", "achieved_fpr",
    "train_rows", "eval_rows", "f1_degradation", "threshold", "mean_effect_size",
    "alert_rate", "f1_gain", "f1_recovery_pct", "history_rows", "alerts",
    "comparisons", "f1_degradation_backtest", "seconds",
}


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "-"
    return f"{number:.{digits}f}"


def pct(value, digits: int = 1) -> str:
    rendered = fmt(value, 6)
    if rendered == "-":
        return "-"
    return f"{float(value) * 100:.{digits}f}%"


def completed_runs(kind: str) -> List[Path]:
    """Every finished run of a given kind, oldest first.

    A directory counts as finished only if the artefact that kind is supposed to
    produce exists. An interrupted run leaves a directory behind with no metrics
    file, and reading one of those would report a partial experiment as a result.
    """
    required = {
        "temporal": "metrics.json",
        "shift": "controlled_shift_results.csv",
        "ablation": "ablation_results.csv",
    }.get(kind)
    if required is None or not EXPERIMENTS.exists():
        return []
    return sorted(
        d for d in EXPERIMENTS.iterdir()
        if d.is_dir() and f"_{kind}_" in d.name and (d / required).is_file()
    )


def latest(kind: str, dataset: Optional[str] = None) -> Optional[Path]:
    """The most recent finished run of a kind, on a real dataset.

    Sorting by name puts the synthetic fixture last, because its directory is
    the newest. The fixture exists so the test suite can run the whole pipeline
    without a multi-gigabyte download, and it is never a research result, so a
    real dataset always wins when one is available.
    """
    runs = completed_runs(kind)
    if not runs:
        return None
    real = [r for r in runs if "synthetic" not in r.name]
    candidates = real or runs
    if dataset:
        named = [r for r in candidates if f"_{dataset}_" in r.name]
        if named:
            candidates = named
    return candidates[-1]


def load_metrics(run: Path) -> Dict:
    return json.loads((run / "metrics.json").read_text(encoding="utf-8"))


def _cell(column: str, value) -> str:
    if column in NUMERIC_COLUMNS:
        return fmt(value)
    if value is None:
        return "-"
    try:
        if float(value) != float(value):
            return "-"
    except (TypeError, ValueError):
        pass
    return str(value)


def _table(header: List[str], rows: List[List[str]]) -> List[str]:
    lines = ["| " + " | ".join(str(h) for h in header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return lines


def temporal_block() -> List[str]:
    runs = completed_runs("temporal")
    if not runs:
        return ["_No temporal experiment has been completed yet._", ""]

    run = runs[-1]
    metrics = load_metrics(run)
    split = metrics.get("split", {})
    models = metrics.get("models", [])
    dataset = metrics.get("dataset", "?")
    train_rows = split.get("train_count", {}).get("rows", 0)
    forward_rows = split.get("forward_count", {}).get("rows", 0)
    forward_time = split.get("forward_time_range", {})

    span = ""
    if forward_time:
        span = (f" spanning {forward_time.get('start', '?')} to "
                f"{forward_time.get('end', '?')}")

    lines = [
        f"Latest run: `{run.name}` on **{dataset}**. Trained on the earliest {train_rows:,} flows, "
        f"scored unchanged on {forward_rows:,} later flows{span}.",
        "",
        "Each model is fitted on the historical period and evaluated twice: on a backtest held out inside the "
        "historical distribution, and on a forward period it never saw. Operating thresholds come from the "
        "validation split alone, under a fixed false-positive budget.",
        "",
    ]
    if not models:
        return lines + ["_No model results were recorded._", ""]

    header = ["Model", "Backtest F1", "Forward F1", "F1 change", "Forward recall", "Forward FPR",
              "Forward PR AUC", "Backtest recall", "Forward Brier", "Forward ECE"]
    rows = []
    for key, label in MODEL_ORDER:
        match = next((m["result"] for m in models if m["result"].get("model") == key), None)
        if match is None:
            continue
        forward = match.get("forward", {})
        backtest = match.get("backtest", {})
        rows.append([
            label, fmt(backtest.get("f1")), fmt(forward.get("f1")),
            fmt(match.get("f1_degradation")), fmt(forward.get("recall")),
            pct(forward.get("false_positive_rate")), fmt(forward.get("pr_auc")),
            fmt(backtest.get("recall")), fmt(forward.get("brier_score")),
            fmt(forward.get("expected_calibration_error")),
        ])
    if rows:
        lines += _table(header, rows) + [""]

    degraded = [m for m in models if (m["result"].get("f1_degradation") or 0) > 0]
    if degraded and len(degraded) == len(models):
        verdict = "Every model lost F1 on the forward period"
    elif degraded:
        verdict = (f"{len(degraded)} of {len(models)} models lost F1 on the forward period")
    else:
        verdict = "No model lost F1 on the forward period"
    worst = max(models, key=lambda m: m["result"].get("f1_degradation") or 0)
    best_forward = max(models, key=lambda m: (m["result"].get("forward", {}).get("f1") or 0))
    lines += [
        f"{verdict}. The largest loss was **{worst['result']['model']}** at "
        f"{fmt(worst['result'].get('f1_degradation'))} F1 "
        f"({fmt(worst['result'].get('backtest', {}).get('f1'))} to "
        f"{fmt(worst['result'].get('forward', {}).get('f1'))}). "
        f"The strongest forward performer was **{best_forward['result']['model']}** at F1 "
        f"{fmt(best_forward['result'].get('forward', {}).get('f1'))}.",
        "",
    ]
    return lines


def adaptation_block() -> List[str]:
    run = latest("temporal")
    if run is None:
        return ["_No temporal experiment has been completed yet._", ""]
    metrics = load_metrics(run)
    models = metrics.get("models", [])
    if not models:
        return ["_No model results were recorded._", ""]

    def strategy_entry(model, strategy):
        """The recorded row for one strategy.

        The pipeline stores strategies as a list of records under "strategies",
        each tagged with its own name, not as one dictionary key per strategy.
        Reading it as a keyed mapping finds nothing and prints a table of
        dashes for a run that in fact completed every strategy.
        """
        adaptation = model.get("adaptation", {}) or {}
        for record in adaptation.get("strategies", []) or []:
            if record.get("strategy") == strategy:
                return record
        return {}

    def cell(record, metric="f1"):
        metrics = record.get("metrics") or {}
        if metric not in metrics:
            return "-"
        return fmt(metrics.get(metric))

    lines = [
        "Forward-period F1 under each adaptation strategy, with the change against the unadapted model in "
        "brackets. Every strategy sets its threshold from permitted historical data, so no row is scored at a "
        "more forgiving operating point than the baseline.",
        "",
    ]
    header = ["Model"] + [label for _, label in STRATEGY_ORDER]
    rows = []
    for model in models:
        name = model["result"].get("model")
        label = next((lbl for key, lbl in MODEL_ORDER if key == name), name)
        cells = [label]
        for strategy, _ in STRATEGY_ORDER:
            record = strategy_entry(model, strategy)
            if not record:
                cells.append("-")
                continue
            shown = cell(record, "f1")
            gain = (record.get("recovery") or {}).get("f1_gain")
            if gain is not None:
                shown += f" ({float(gain):+.3f})"
            cells.append(shown)
        rows.append(cells)
    lines += _table(header, rows) + [""]

    lines += ["The same table for forward recall, because F1 alone hides the tradeoff:", ""]
    rows = []
    for model in models:
        name = model["result"].get("model")
        label = next((lbl for key, lbl in MODEL_ORDER if key == name), name)
        cells = [label]
        for strategy, _ in STRATEGY_ORDER:
            cells.append(cell(strategy_entry(model, strategy), "recall"))
        rows.append(cells)
    lines += _table(header, rows) + [""]
    return lines


def drift_block() -> List[str]:
    run = latest("temporal")
    if run is None:
        return ["_No temporal experiment has been completed yet._", ""]
    summary = load_metrics(run).get("drift_summary") or {}
    if not summary:
        return [
            "_The drift stage recorded no results. The pipeline raises rather than writing an empty summary, so "
            "this state means the stage never ran._",
            "",
        ]

    lines = [
        f"{summary['comparisons']} comparisons across {summary['windows']} windows and "
        f"{len(summary.get('by_method', {}))} detectors, of which {summary['alerts']} raised an alert. The first "
        f"alert fell on {summary.get('first_alert') or 'no window'}. An alert requires an effect size and a "
        "significance threshold together. A p-value alone is not treated as an alert: across hundreds of "
        "comparisons a nominal threshold fires almost everywhere, which measures the number of comparisons "
        "rather than the traffic.",
        "",
    ]

    def rates(section: str) -> List[List[str]]:
        out = []
        for name, stats in sorted(summary.get(section, {}).items()):
            comps = stats.get("comparisons", 0)
            hits = stats.get("alerts", 0)
            rate = (hits / comps) if comps else float("nan")
            out.append([name, str(comps), f"{hits}/{comps}", pct(rate)])
        return out

    lines += _table(["Detector", "Comparisons", "Alerts", "Alert rate"], rates("by_method"))
    lines += [""]
    lines += _table(["Feature", "Comparisons", "Alerts", "Alert rate"], rates("by_feature"))
    lines += [""]
    return lines


def shift_block() -> List[str]:
    run = latest("shift")
    if run is None:
        return ["_No controlled-shift experiment has been completed yet._", ""]
    table = pd.read_csv(run / "controlled_shift_results.csv")
    if table.empty:
        return ["_The controlled-shift experiment recorded no rows._", ""]

    lines = [
        f"Latest run: `{run.name}`. Every row records both the shift that was requested and the shift that was "
        "measured, so a perturbation that failed to move the distribution is visible instead of being assumed to "
        "have worked.",
        "",
    ]
    aggregate = (
        table.groupby("shift", as_index=False)
        .agg(
            shifts=("magnitude", "count"),
            verified=("realized_verified", "sum"),
            alerts=("drift_alerts", "sum"),
            mean_degradation=("f1_degradation", "mean"),
        )
        .sort_values("mean_degradation", ascending=False)
    )
    rows = [
        [str(r["shift"]), str(int(r["shifts"])),
         f"{int(r['verified'])}/{int(r['shifts'])}",
         f"{int(r['alerts'])}/{int(r['shifts'])}",
         fmt(r["mean_degradation"])]
        for _, r in aggregate.iterrows()
    ]
    return lines + _table(
        ["Shift family", "Shifts", "Realized as intended", "Drift alerts", "Mean F1 change"], rows
    ) + [""]


def ablation_block() -> List[str]:
    runs = completed_runs("ablation")
    if not runs:
        return ["_No ablation experiment has been completed yet._", ""]

    grouped: Dict[str, List[Dict]] = {}
    for run in runs:
        table = pd.read_csv(run / "ablation_results.csv")
        if table.empty or "ablation" not in table.columns:
            continue
        for name, group in table.groupby("ablation", sort=False):
            grouped.setdefault(str(name), []).extend(group.to_dict("records"))
    if not grouped:
        return ["_The ablation experiments recorded no rows._", ""]

    lines: List[str] = []
    notes: List[str] = []
    titles = {
        "split_mode": "**Temporal against random splitting.** Same rows and same model; only the split moves. "
                      "This is the difference between evaluating on contemporaneous traffic and on later traffic.",
        "target_fpr": "**The false-positive budget.** What each budget costs in recall, measured on identical scores.",
        "feature_policy": "**Feature policy.**",
    }
    keeps = {
        "split_mode": ["variant", "f1", "precision", "recall", "false_positive_rate", "pr_auc",
                       "train_rows", "eval_rows"],
        "target_fpr": ["variant", "target_fpr", "achieved_fpr", "recall", "precision", "f1"],
        "feature_policy": ["variant", "n_features", "f1", "roc_auc", "pr_auc"],
    }
    for name in ("split_mode", "target_fpr", "feature_policy"):
        rows = [r for r in grouped.get(name, []) if r.get("variant")]
        if not rows:
            continue
        keep = [c for c in keeps[name] if c in rows[0]]
        lines += [titles[name], ""]
        lines += _table(
            [c.replace("_", " ") for c in keep],
            [[_cell(c, r.get(c)) for c in keep] for r in rows],
        ) + [""]
        for r in rows:
            note = r.get("note")
            if note and not str(note).startswith(("trained", "same rows")):
                notes.append(str(note))
    if notes:
        lines += ["Notes:", ""] + [f"- {n}" for n in dict.fromkeys(notes)] + [""]
    return lines


def build_block() -> str:
    lines = [BEGIN, ""]
    for heading, builder in [
        ("### Temporal generalisation", temporal_block),
        ("### Drift detection", drift_block),
        ("### Adaptation", adaptation_block),
        ("### Controlled shifts", shift_block),
        ("### Ablations", ablation_block),
    ]:
        lines += [heading, ""]
        lines += builder()
    lines += [END]
    return "\n".join(lines)


def main() -> None:
    if not README.exists():
        raise SystemExit(f"{README} not found")
    text = README.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit(
            f"README.md is missing the {BEGIN} ... {END} markers, so the results block cannot be regenerated."
        )
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)
    README.write_text(pattern.sub(lambda _: build_block(), text), encoding="utf-8")
    print("README results block regenerated from results/experiments/")


if __name__ == "__main__":
    main()
