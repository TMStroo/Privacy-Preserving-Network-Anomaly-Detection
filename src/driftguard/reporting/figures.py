"""Figures for the README and the technical report.

Every figure is drawn from a recorded experiment's CSV or JSON, so a figure
cannot show a number the run did not produce.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE = "#2f6fb5"
ORANGE = "#d97b29"
GREY = "#8b93a1"
RED = "#c0392b"
GREEN = "#2e8b57"


def _style():
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 8.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    })


def backtest_vs_forward(metrics: Dict, out_path: str) -> Optional[str]:
    """Paired bars for each model's backtest and forward F1."""
    _style()
    models, backtest, forward = [], [], []
    for entry in metrics["models"]:
        result = entry["result"]
        if "forward" not in result:
            continue
        models.append(result["model"].replace("_", "\n"))
        backtest.append(result["backtest"]["f1"])
        forward.append(result["forward"]["f1"])

    if not models:
        return None
    x = np.arange(len(models))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.bar(x - width / 2, backtest, width, label="Backtest", color=BLUE)
    ax.bar(x + width / 2, forward, width, label="Forward test", color=ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=7.5)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.set_title("Does the detector keep working on later traffic?")
    ax.legend(frameon=False, fontsize=8)
    for bars in (ax.containers[0], ax.containers[1]):
        ax.bar_label(bars, fmt="%.3f", fontsize=6.5, padding=1)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def degradation_by_model(metrics: Dict, out_path: str) -> Optional[str]:
    """F1 and recall degradation per model, signed so improvements read as negative."""
    _style()
    models, f1_deg, recall_deg = [], [], []
    for entry in metrics["models"]:
        result = entry["result"]
        if "f1_degradation" not in result:
            continue
        models.append(result["model"].replace("_", "\n"))
        f1_deg.append(result["f1_degradation"])
        recall_deg.append(result["recall_degradation"])
    if not models:
        return None
    x = np.arange(len(models))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.bar(x - width / 2, f1_deg, width, label="F1 degradation", color=RED)
    ax.bar(x + width / 2, recall_deg, width, label="Recall degradation", color=ORANGE)
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=7.5)
    ax.set_ylabel("Forward minus backtest")
    ax.set_title("Performance change on later traffic (positive = worse)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def drift_timeline(drift_events_csv: str, out_path: str, top_features: int = 6) -> Optional[str]:
    """Alerts per time window, one row per feature."""
    if not Path(drift_events_csv).exists():
        return None
    table = pd.read_csv(drift_events_csv)
    if table.empty:
        return None
    table["window_start"] = pd.to_datetime(table["window_start"])
    counts = table.groupby(["feature", "window_start"])["alert"].sum().reset_index()
    features = counts.groupby("feature")["alert"].sum().nlargest(top_features).index.tolist()
    if not features:
        return None

    _style()
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    for feature in features:
        sub = counts[counts["feature"] == feature].sort_values("window_start")
        ax.plot(sub["window_start"], sub["alert"], marker="o", markersize=2.5, linewidth=1.1, label=feature)
    ax.set_ylabel("Alerts in window")
    ax.set_xlabel("Window start")
    ax.set_title("Drift alerts over the forward period")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def forward_windows(csv_path: str, out_path: str) -> Optional[str]:
    """Recall and false-positive rate per forward window."""
    if not Path(csv_path).exists():
        return None
    table = pd.read_csv(csv_path)
    if table.empty:
        return None
    table["bucket"] = pd.to_datetime(table["bucket"])
    _style()
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    ax.plot(table["bucket"], table["recall"], marker="o", markersize=3, color=BLUE, label="Recall")
    ax.plot(table["bucket"], table["fpr"], marker="s", markersize=3, color=ORANGE, label="False-positive rate")
    ax.axhline(0.5, color=GREY, linewidth=0.8, linestyle="--", label="0.5 reference")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Detection quality across the forward period")
    ax.legend(frameon=False, fontsize=8)
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def calibration_curves(experiment_path: str, out_path: str, model: str) -> Optional[str]:
    """Reliability diagram comparing the backtest and forward periods."""
    base = Path(experiment_path)
    path = base / f"calibration_{model}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    _style()
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.plot([0, 1], [0, 1], color=GREY, linewidth=0.9, linestyle="--", label="Perfect calibration")
    for key, color in [("backtest", BLUE), ("forward", ORANGE)]:
        curve = [row for row in data.get(key, {}).get("curve", []) if row.get("n", 0) > 0]
        if not curve:
            continue
        ax.plot([row["mean_pred"] for row in curve], [row["observed"] for row in curve],
                marker="o", markersize=3, color=color, label=key.capitalize())
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed attack rate")
    ax.set_title(f"Calibration: {model}")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def adaptation_bars(metrics: Dict, out_path: str, model: str) -> Optional[str]:
    """Forward F1 per adaptation strategy for one model."""
    _style()
    entry = next((e for e in metrics["models"] if e["result"]["model"] == model), None)
    if not entry:
        return None
    strategies = [s for s in entry["adaptation"]["strategies"] if "metrics" in s]
    if not strategies:
        return None
    names = [s["strategy"].replace("_", "\n") for s in strategies]
    values = [s["metrics"]["f1"] for s in strategies]
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    bars = ax.bar(names, values, color=[GREY if n == "none" else BLUE for n in [s["strategy"] for s in strategies]])
    ax.set_ylabel("Forward-test F1")
    ax.set_ylim(0, 1)
    ax.set_title(f"Adaptation strategies: {model}")
    ax.bar_label(bars, fmt="%.3f", fontsize=6.5, padding=1)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def class_timeline(experiment_path: str, model: str, out_path: str) -> Optional[str]:
    """Attack prevalence over the forward period from the prediction table."""
    path = Path(experiment_path) / f"predictions_forward_{model}.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path, parse_dates=["timestamp"])
    if table.empty:
        return None
    table["bucket"] = table["timestamp"].dt.floor("12h")
    grouped = table.groupby("bucket").agg(rate=("y_true", "mean"), n=("y_true", "size"))
    _style()
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    ax.plot(grouped.index, grouped["rate"], color=BLUE, linewidth=1.2)
    ax.fill_between(grouped.index, grouped["rate"], alpha=0.18, color=BLUE)
    ax.set_ylabel("Attack rate")
    ax.set_ylim(0, 1)
    ax.set_title(f"Attack prevalence across the forward period ({model})")
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def build_all_figures(experiment_path: str, metrics: Dict, figures_dir: str) -> Dict[str, str]:
    """Render every figure that the recorded data supports."""
    out = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(experiment_path)
    figures: Dict[str, str] = {}

    def keep(key, path):
        if path:
            figures[key] = path

    keep("backtest_vs_forward", backtest_vs_forward(metrics, str(out / "backtest_vs_forward.png")))
    keep("degradation", degradation_by_model(metrics, str(out / "degradation_by_model.png")))
    keep("drift_timeline", drift_timeline(str(base / "drift_events.csv"), str(out / "drift_timeline.png")))

    top_model = metrics["models"][-1]["result"]["model"] if metrics.get("models") else None
    if top_model:
        keep("forward_windows", forward_windows(str(base / f"forward_windows_{top_model}.csv"),
                                                 str(out / "forward_windows.png")))
        keep("calibration", calibration_curves(str(base), str(out / f"calibration_{top_model}.png"), top_model))
        keep("adaptation", adaptation_bars(metrics, str(out / "adaptation_strategies.png"), top_model))
        keep("class_timeline", class_timeline(str(base), top_model, str(out / "class_timeline.png")))
    return figures
