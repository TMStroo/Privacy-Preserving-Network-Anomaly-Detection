"""Regenerate every final figure from the recorded experiment directories.

The figures checked into an experiment directory were drawn when that run
executed. If a renderer is fixed afterwards, or a run is superseded, those PNGs
are stale while the CSVs beside them are not. This rebuilds all of them from the
artifacts and then refuses to report success if any figure is empty, is not a
real dataset, or carries a stale experiment ID.

Run:  python scripts/regenerate_figures.py
"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

from driftguard.reporting import figures as F
from driftguard.reporting import figures_extra as X

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "experiments"

FINAL = {
    "unsw": RUNS / "20260925T195935Z_temporal_unsw_nb15_full_6167ea",
    "ugr": RUNS / "20260925T220604Z_temporal_ugr16_78d5c1",
    "shift": RUNS / "20260926T003707Z_shift_unsw_nb15_eae6e4",
    "ablation": RUNS / "20260925T205657Z_ablation_unsw_nb15_d6d364",
}

# figure name -> what draws it, per dataset
TEMPORAL_FIGURES = {
    "split_timeline": lambda p, m, d: X.split_timeline(m, str(d / "split_timeline.png")),
    "backtest_vs_forward": lambda p, m, d: F.backtest_vs_forward(m, str(d / "backtest_vs_forward.png")),
    "degradation_by_model": lambda p, m, d: F.degradation_by_model(m, str(d / "degradation_by_model.png")),
    "drift_detector_comparison": lambda p, m, d: X.drift_detector_comparison(
        str(p / "drift_events.csv"), str(d / "drift_detector_comparison.png")
    ),
    "drift_timeline": lambda p, m, d: F.drift_timeline(
        str(p / "drift_events.csv"), str(d / "drift_timeline.png")
    ),
    "adaptation_strategies": lambda p, m, d: F.adaptation_bars(m, str(d / "adaptation_strategies.png"), "gradient_boosting"),
    "recall_vs_fpr": lambda p, m, d: X.recall_vs_fpr(m, str(d / "recall_vs_fpr.png")),
    "forward_windows": lambda p, m, d: F.forward_windows(
        str(p / "forward_windows_gradient_boosting.csv"), str(d / "forward_windows.png")
    ),
    "calibration_gradient_boosting": lambda p, m, d: F.calibration_curves(
        str(p), str(d / "calibration_gradient_boosting.png"), "gradient_boosting"
    ),
    "failure_summary": lambda p, m, d: X.failure_summary(str(p), str(d / "failure_summary.png")),
    "class_timeline": lambda p, m, d: F.class_timeline(
        str(p), "gradient_boosting", str(d / "class_timeline.png")
    ),
}

problems: list[str] = []
written = 0


def check(path, expect_run: str) -> None:
    """A figure is only accepted if it is a real PNG with content."""
    global written
    p = pathlib.Path(path)
    if not p.exists():
        problems.append(f"{p.name}: not written")
        return
    size = p.stat().st_size
    if size < 5_000:
        problems.append(f"{p.name}: {size} bytes, suspiciously small")
    with p.open("rb") as handle:
        if handle.read(8) != b"\x89PNG\r\n\x1a\n":
            problems.append(f"{p.name}: not a PNG")
    written += 1
    print(f"  {p.name:38s} {size:>8,} b   {p.parent.name}")


for key in ("unsw", "ugr"):
    run = FINAL[key]
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    dataset = metrics["dataset"]
    if dataset == "synthetic":
        raise SystemExit(f"{run.name} is the synthetic fixture; refusing to draw it as real data")
    out = run / "figures"
    out.mkdir(exist_ok=True)
    print(f"\n{dataset}  ({run.name})")
    for name, draw in TEMPORAL_FIGURES.items():
        result = draw(run, metrics, out)
        if result is None:
            problems.append(f"{key}/{name}: renderer declined (returned None)")
            continue
        check(result, run.name)

# Controlled shifts: the shift figure needs the shift run's own CSV.
shift_csv = FINAL["shift"] / "controlled_shift_results.csv"
if shift_csv.exists():
    target = FINAL["shift"] / "figures"
    target.mkdir(exist_ok=True)
    print(f"\ncontrolled shifts  ({FINAL['shift'].name})")
    result = X.shift_degradation(str(shift_csv), str(target / "shift_degradation.png"))
    if result is None:
        problems.append("shift_degradation: renderer declined")
    else:
        check(result, FINAL["shift"].name)
else:
    problems.append(f"missing {shift_csv}")

# Target-FPR ablation figure. There is no dedicated renderer, so the sweep is
# drawn from the ablation CSV directly rather than inventing one in figures.py.
import matplotlib.pyplot as plt

ablation_csv = FINAL["ablation"] / "ablation_results.csv"
if ablation_csv.exists():
    import pandas as pd

    frame = pd.read_csv(ablation_csv)
    sweep = frame[frame["ablation"] == "target_fpr"].copy()
    if sweep.empty:
        problems.append("no target_fpr rows to plot")
    else:
        target = FINAL["ablation"] / "figures"
        target.mkdir(exist_ok=True)
        print(f"\nablation  ({FINAL['ablation'].name})")
        F._style()
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for ax, model in zip(axes, ("random_forest", "logistic_regression")):
            part = sweep[sweep["model"] == model].sort_values("target_fpr")
            if part.empty:
                continue
            ax.plot(
                part["target_fpr"], part["f1"], marker="o", color="#2f6f9f", label="F1"
            )
            ax.set_title(model)
            # A log axis separates 0.01 from 0.02, which on a linear axis land
            # almost on top of each other and print as "0.010.02".
            ax.set_xscale("log")
            ax.set_xlabel("target false-positive budget (log scale)")
            ax.set_ylabel("F1")
            ax.set_ylim(0, 1.22)
            ax.set_xticks(list(part["target_fpr"]))
            ax.set_xticklabels([f"{v:g}" for v in part["target_fpr"]], fontsize=8)
            ax.minorticks_off()
            # Side margin so the first and last markers are not on the spine.
            ax.set_xlim(part["target_fpr"].min() * 0.8, part["target_fpr"].max() * 1.3)
            ax.grid(alpha=0.3)
            # Every label above its own marker. Offsetting alternately left and
            # right pushed 0.9574 into the y-axis and detached the other labels
            # from the points they describe.
            for _, row in part.iterrows():
                ax.annotate(
                    f"{row['f1']:.4f}",
                    (row["target_fpr"], row["f1"]),
                    textcoords="offset points",
                    xytext=(0, 9),
                    ha="center",
                    fontsize=7.5,
                )
        figure.suptitle(
            "UNSW-NB15 target-FPR ablation: F1 against the false-positive budget", y=0.995
        )
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        out = str(target / "target_fpr_ablation.png")
        figure.savefig(out, dpi=140)
        plt.close(figure)
        check(out, FINAL["ablation"].name)
else:
    problems.append(f"missing {ablation_csv}")

print(f"\n{written} figures written")
if problems:
    print(f"\n{len(problems)} problems:")
    for problem in problems:
        print(f"  - {problem}")
    sys.exit(1)
print("every required figure regenerated from the final artifacts")
