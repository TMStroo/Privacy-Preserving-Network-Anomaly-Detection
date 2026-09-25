"""Additional figures for the final report.

Kept separate from figures.py so the original set stays reviewable next to the
run that produced it. Same rule as everywhere else in this project: a figure is
drawn from a recorded CSV or JSON, so it cannot show a number the run did not
produce.
"""

import json
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from driftguard.reporting.figures import (  # noqa: E402
    BLUE, GREEN, ORANGE, RED, _style,
)

PERIOD_COLOURS = {
    "train": BLUE,
    "validation": GREEN,
    "backtest": ORANGE,
    "forward": RED,
}


def _read_csv(path) -> Optional[pd.DataFrame]:
    """Read a recorded table, or return None if it is absent or unreadable.

    Every figure here is optional. A run that did not run the shift stage should
    lose one figure, not fail the report.
    """
    if not path:
        return None
    target = Path(path)
    if not target.is_file():
        return None
    try:
        frame = pd.read_csv(target)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    return None if frame.empty else frame


def drift_detector_comparison(drift_events_csv: str, out_path: str) -> Optional[str]:
    """How the four detectors behaved on the same comparisons.

    Side by side deliberately: the interesting result in this project is that
    the methods disagree, and four bars show that faster than a table of prose
    does.
    """
    _style()
    events = _read_csv(drift_events_csv)
    if events is None or events.empty or "method" not in events.columns:
        return None

    present = set(events["method"])
    order = [m for m in ("ks", "wasserstein", "psi", "cusum") if m in present]
    order += sorted(present - set(order))
    if not order:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    x = np.arange(len(order))

    if "alert" in events.columns:
        rates = [float(events.loc[events["method"] == m, "alert"].mean()) for m in order]
        axes[0].bar(x, rates, color=ORANGE, width=0.6)
        axes[0].set_ylabel("Fraction of comparisons alerting")
        axes[0].set_title("Alert rate")
        axes[0].set_ylim(0, 1.08)
        for i, rate in enumerate(rates):
            axes[0].text(i, rate + 0.02, f"{rate:.0%}", ha="center", fontsize=7)
    else:
        axes[0].set_axis_off()

    if "effect_size" in events.columns:
        sizes = [float(events.loc[events["method"] == m, "effect_size"].abs().median()) for m in order]
        axes[1].bar(x, sizes, color=BLUE, width=0.6)
        axes[1].set_ylabel("Median |effect size|")
        axes[1].set_title("Effect size, alerted or not")
        for i, value in enumerate(sizes):
            axes[1].text(i, value, f"{value:.3g}", ha="center", va="bottom", fontsize=7)
    else:
        axes[1].set_axis_off()

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(order, fontsize=7.5, rotation=20)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order, fontsize=7.5, rotation=20)
    fig.suptitle("Drift detectors compared on the same windows", fontsize=10, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def recall_vs_fpr(metrics: dict, out_path: str) -> Optional[str]:
    """Recall against the false-positive rate each model actually achieved.

    A detector is only comparable under a stated operating point, so the x axis
    is the achieved rate rather than a fixed grid of target rates.
    """
    _style()
    if not metrics.get("models"):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, period, colour, title in [
        (axes[0], "backtest", BLUE, "Backtest (held out)"),
        (axes[1], "forward", ORANGE, "Forward (later traffic)"),
    ]:
        for entry in metrics["models"]:
            result = entry["result"]
            block = result.get(period) or {}
            if "recall" not in block or "false_positive_rate" not in block:
                continue
            fpr, recall = block["false_positive_rate"], block["recall"]
            ax.scatter(fpr, recall, s=48, color=colour, edgecolor="white", linewidth=0.6, zorder=3)
            ax.annotate(result["model"].replace("_", " "), (fpr, recall),
                        textcoords="offset points", xytext=(6, 3), fontsize=6.4)
        ax.set_xlabel("False-positive rate")
        ax.set_title(title)
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("Recall")
    fig.suptitle("Operating point per model, at the threshold chosen on validation", fontsize=10, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def shift_degradation(shift_csv: str, out_path: str) -> Optional[str]:
    """F1 change per controlled-shift family, labelled with whether it was real.

    The "verified as intended" fraction is printed beside every family, because
    a perturbation that did not move the distribution measures the perturbation
    rather than the detector, and reading it as detector damage would be wrong.
    """
    _style()
    table = _read_csv(shift_csv)
    if table is None or table.empty or "f1_degradation" not in table.columns:
        return None
    if "kind" not in table.columns:
        return None

    grouped = table.groupby("kind", as_index=False)["f1_degradation"].mean()
    grouped = grouped.sort_values("f1_degradation", ascending=True)

    verified = {}
    if "realized_verified" in table.columns:
        counts = table.groupby("kind")["realized_verified"].agg(["sum", "count"])
        verified = {k: (int(r["sum"]), int(r["count"])) for k, r in counts.iterrows()}

    labels, values, colours = [], [], []
    for _, row in grouped.iterrows():
        kind = str(row["kind"])
        ok, total = verified.get(kind, (1, 1))
        labels.append(f"{kind.replace('_', ' ')}\n{ok}/{total} realized")
        value = float(row["f1_degradation"])
        values.append(value)
        colours.append(RED if value > 0 else GREEN)

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.6, max(2.6, 0.46 * len(labels) + 1.5)))
    ax.barh(y, values, color=colours, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("F1 change (positive = worse after the shift)")
    ax.set_title("Controlled shifts: detector damage, and whether the shift happened at all")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def failure_summary(experiment_path: str, out_path: str) -> Optional[str]:
    """What each model actually got wrong on the forward period.

    Reads the per-model failure analysis, so this figure and the report's failure
    section cannot disagree with each other.
    """
    _style()
    base = Path(experiment_path)
    blocks = {}
    for path in sorted(base.glob("failure_analysis_*.json")):
        name = path.stem.replace("failure_analysis_", "")
        try:
            blocks[name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    if not blocks:
        return None

    models = list(blocks)
    fp_counts, fn_counts, fp_scores, fn_scores = [], [], [], []
    for name in models:
        block = blocks[name]
        fps = block.get("false_positives") or {}
        fns = block.get("false_negatives") or {}
        fp_counts.append(int(fps.get("count", 0) or 0))
        fn_counts.append(int(fns.get("count", 0) or 0))
        fp_scores.append(float(fps.get("mean_score", 0) or 0))
        fn_scores.append(float(fns.get("mean_score", 0) or 0))

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    x = np.arange(len(models))
    width = 0.36
    labels = [m.replace("_", "\n") for m in models]

    axes[0].bar(x - width / 2, fp_counts, width, label="False positives", color=ORANGE)
    axes[0].bar(x + width / 2, fn_counts, width, label="False negatives", color=RED)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].set_ylabel("Count on the forward period")
    axes[0].set_title("Error volume")
    axes[0].legend(frameon=False, fontsize=7.5)

    axes[1].bar(x - width / 2, fp_scores, width, label="Mean FP score", color=ORANGE)
    axes[1].bar(x + width / 2, fn_scores, width, label="Mean FN score", color=RED)
    axes[1].axhline(0.5, color="#333", linewidth=0.8, linestyle="--")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_ylabel("Mean predicted score")
    axes[1].set_title("How confident the errors were")
    axes[1].legend(frameon=False, fontsize=7.5)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def split_timeline(metrics: dict, out_path: str) -> Optional[str]:
    """The four periods on the real time axis, drawn to scale.

    Worth drawing because UNSW-NB15's forward period spans only a few hours, and
    every conclusion in this report is bounded by that fact.
    """
    _style()
    split = metrics.get("split") or {}
    if not split:
        return None

    spans = []
    for label in ("train", "validation", "backtest", "forward"):
        bounds = split.get(f"{label}_period")
        # The split records each period either as {"start": ..., "end": ...} or
        # as a two-element [start, end] list, depending on which writer produced
        # it. Accept both rather than assuming one.
        if isinstance(bounds, dict):
            start, end = bounds.get("start"), bounds.get("end")
        elif isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            start, end = bounds
        else:
            start = end = None
        count = int((split.get(f"{label}_count") or {}).get("rows", 0) or 0)
        if start and end:
            spans.append((label, pd.Timestamp(start), pd.Timestamp(end), count))
    if not spans:
        return None

    total_rows = max((s[3] for s in spans), default=0) or 1
    fig, ax = plt.subplots(figsize=(7.0, 1.9))
    for label, start, end, rows in spans:
        left = start.value / 1e9
        width = max((end - start).total_seconds(), 1.0)
        ax.barh(0, width, left=left, height=0.5, color=PERIOD_COLOURS[label], alpha=0.9)
        share = rows / total_rows * 100
        ax.text(left + width / 2, 0, f"{label}\n{rows:,} rows ({share:.0f}%)",
                ha="center", va="center", fontsize=7, color="white")

    ax.set_yticks([])
    ax.set_xlabel("Seconds since epoch, on the real capture axis")
    ax.set_title("Chronological split, drawn to scale")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def build_extra_figures(experiment_path: str, metrics: dict, out_dir: str,
                        shift_csv: Optional[str] = None) -> dict:
    """Render the extra figures the recorded data supports."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(experiment_path)
    figures = {}

    def keep(key, path):
        if path:
            figures[key] = path

    keep("split_timeline", split_timeline(metrics, str(out / "split_timeline.png")))
    keep("recall_vs_fpr", recall_vs_fpr(metrics, str(out / "recall_vs_fpr.png")))
    keep("drift_detectors", drift_detector_comparison(
        str(base / "drift_events.csv"), str(out / "drift_detector_comparison.png")))
    keep("failure_analysis", failure_summary(str(base), str(out / "failure_summary.png")))
    if shift_csv:
        keep("shift_degradation", shift_degradation(shift_csv, str(out / "shift_degradation.png")))
    return figures
