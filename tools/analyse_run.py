"""Analyse a completed run: drift behaviour, adaptation tradeoffs, failures.

Prints what the recorded artifacts actually say, with no interpretation added
that the numbers do not support. Used both interactively and as the source for
the report's analysis sections, so a claim in the report and a number printed
here cannot disagree.

    python tools/analyse_run.py <experiment_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

STRATEGY_LABELS = {
    "none": "no adaptation",
    "threshold_recalibration": "threshold recalibration",
    "recent_window_retrain": "recent-window retrain",
    "rolling_window_retrain": "rolling-window retrain",
    "historical_plus_recent_retrain": "historical + recent retrain",
}


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def split_summary(metrics: dict) -> None:
    section("TEMPORAL SPLIT")
    split = metrics.get("split") or {}
    for label in ("train", "validation", "backtest", "forward"):
        period = split.get(f"{label}_period")
        count = (split.get(f"{label}_count") or {})
        if not period:
            continue
        if isinstance(period, dict):
            bounds = f"{period.get('start')} -> {period.get('end')}"
        else:
            bounds = f"{period[0]} -> {period[1]}"
        print(f"  {label:11s} {count.get('rows', 0):>10,} rows  {bounds}")


def model_table(run: Path, metrics: dict) -> None:
    section("MODELS: backtest versus forward")
    rows = []
    for entry in metrics.get("models", []):
        result = entry["result"]
        bt, fw = result.get("backtest", {}), result.get("forward", {})
        rows.append({
            "model": result["model"],
            "bt_f1": bt.get("f1"), "fw_f1": fw.get("f1"),
            "d_f1": result.get("f1_degradation"),
            "bt_recall": bt.get("recall"), "fw_recall": fw.get("recall"),
            "d_recall": result.get("recall_degradation"),
            "bt_fpr": bt.get("false_positive_rate"), "fw_fpr": fw.get("false_positive_rate"),
            "d_fpr": result.get("fpr_increase"),
            "fw_pr_auc": fw.get("pr_auc"),
            "train_s": result.get("train_seconds"),
        })
    if not rows:
        print("  no models recorded")
        return
    frame = pd.DataFrame(rows)
    for column in frame.columns:
        if column != "model":
            frame[column] = pd.to_numeric(frame[column], errors="coerce").round(4)
    print(frame.to_string(index=False))

    # Brier and ECE are written to calibration_<model>.json rather than into the
    # per-period metrics block, so they are merged in from there. Reading them
    # from the metrics block prints NaN and looks like a missing measurement.
    for row in rows:
        path = run / f"calibration_{row['model']}.json"
        if not path.is_file():
            continue
        block = json.loads(path.read_text(encoding="utf-8")).get("forward", {})
        row["fw_brier"] = block.get("brier")
        row["fw_ece"] = block.get("ece")
    if any(r.get("fw_brier") is not None for r in rows):
        print()
        print("  calibration on the forward period:")
        for row in rows:
            if row.get("fw_brier") is not None:
                print(f"    {row['model']:22s} Brier {row['fw_brier']:.5f}   "
                      f"ECE {row['fw_ece']:.5f}")

    degraded = [r for r in rows if (r["d_f1"] or 0) > 0]
    print()
    if not rows:
        return
    worst = max(rows, key=lambda r: r["d_f1"] or 0)
    best = max(rows, key=lambda r: (r["fw_f1"] or 0))
    print(f"  {len(degraded)} of {len(rows)} models lost F1 forward")
    print(f"  largest loss : {worst['model']} {worst['d_f1']:+.4f} "
          f"({worst['bt_f1']:.4f} -> {worst['fw_f1']:.4f})")
    print(f"  best forward : {best['model']} F1 {best['fw_f1']:.4f}")


def adaptation_table(metrics: dict) -> None:
    section("ADAPTATION: what each strategy bought and what it cost")
    for entry in metrics.get("models", []):
        result = entry["result"]
        name = result["model"]
        adaptation = entry.get("adaptation") or {}
        if not adaptation:
            print(f"\n  {name}: no adaptation results recorded")
            continue
        rows = []
        # Strategies are a list of records, each naming itself. Reading them as
        # dict keys silently yields nothing.
        for record in adaptation.get("strategies") or []:
            if not isinstance(record, dict):
                continue
            label = STRATEGY_LABELS.get(record.get("strategy"), record.get("strategy"))
            metrics = record.get("metrics") or {}
            recovery = record.get("recovery") or {}
            rows.append({
                "strategy": label,
                "f1": metrics.get("f1"),
                "recall": metrics.get("recall"),
                "fpr": metrics.get("false_positive_rate"),
                "pr_auc": metrics.get("pr_auc"),
                "f1_gain": recovery.get("f1_gain"),
                "recovery_%": recovery.get("f1_recovery_pct"),
                "refits": record.get("refits"),
                "threshold": record.get("threshold"),
                "seconds": record.get("training_seconds"),
            })
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        for column in frame.columns:
            if column != "strategy":
                frame[column] = pd.to_numeric(frame[column], errors="coerce").round(4)
        print(f"\n  {name}")
        print(frame.to_string(index=False))

        # Rolling and recent are supposed to differ. If they do not, that is
        # either a genuinely flat signal or the strategies are still aliased.
        by_label = {r["strategy"]: r["f1"] for r in rows}
        recent = by_label.get("recent-window retrain")
        rolling = by_label.get("rolling-window retrain")
        if recent is not None and rolling is not None and abs(recent - rolling) < 1e-9:
            print(f"    NOTE: recent and rolling produced identical F1 ({recent:.4f}). "
                  f"Either the forward period is too flat for the difference to show, "
                  f"or the two strategies are still aliased.")


def drift_table(run: Path, metrics: dict) -> None:
    section("DRIFT DETECTION")
    path = run / "drift_events.csv"
    if not path.is_file():
        print("  drift_events.csv is missing")
        return
    events = pd.read_csv(path)
    if events.empty:
        print("  no drift comparisons recorded")
        return

    print(f"  {len(events)} comparisons: "
          f"{events['window_start'].nunique()} windows x {events['feature'].nunique()} features "
          f"x {events['method'].nunique()} methods")
    print()
    grouped = events.groupby("method").agg(
        n=("alert", "size"),
        alerts=("alert", "sum"),
        alert_rate=("alert", "mean"),
        median_effect=("effect_size", lambda s: s.abs().median()),
        max_effect=("effect_size", lambda s: s.abs().max()),
        median_p=("p_value", "median"),
    )
    grouped["alert_rate"] = grouped["alert_rate"].map("{:.1%}".format)
    for column in ("median_effect", "max_effect"):
        grouped[column] = grouped[column].map(lambda v: f"{v:.4f}")
    grouped["median_p"] = grouped["median_p"].map(lambda v: f"{v:.3g}")
    print(grouped.to_string())
    print()

    # Does the effect size carry information, or does the alert fire regardless?
    if "effect_size" in events.columns:
        print("  effect size when alerting vs when quiet (median |d|):")
        for method, group in events.groupby("method"):
            alerting = group.loc[group["alert"], "effect_size"].abs().median()
            quiet = group.loc[~group["alert"], "effect_size"].abs().median()
            def fmt(v):
                return f"{v:.4f}" if pd.notna(v) else "n/a"
            print(f"    {method:12s} alerting {fmt(alerting):>8s}   quiet {fmt(quiet):>8s}")
        print()

    # Detector agreement, per (window, feature).
    if events["method"].nunique() > 1:
        pivot = events.pivot_table(index=["window_start", "feature"],
                                   columns="method", values="alert", aggfunc="first")
        counts = pivot.sum(axis=1)
        print(f"  agreement across {pivot['method'].nunique() if 'method' in pivot else events['method'].nunique()} detectors, "
              f"per window-feature pair ({len(pivot)} pairs):")
        print(f"    all agreed on drift : {int((counts == events['method'].nunique()).sum())}")
        print(f"    split               : "
              f"{int(((counts > 0) & (counts < events['method'].nunique())).sum())}")
        print(f"    none alerted        : {int((counts == 0).sum())}")
        print()


def failure_report(run: Path) -> None:
    section("FAILURE ANALYSIS")
    paths = sorted(run.glob("failure_analysis_*.json"))
    if not paths:
        print("  no per-model failure analysis recorded")
        return
    for path in paths:
        name = path.stem.replace("failure_analysis_", "")
        block = json.loads(path.read_text(encoding="utf-8"))
        fps = block.get("false_positives", {})
        fns = block.get("false_negatives", {})
        print(f"\n  {name}")
        print(f"    false positives : {fps.get('count', 0):>8,}  "
              f"rate {fps.get('rate', 0):.4f}  mean score {fps.get('mean_score', 0):.4f}")
        print(f"    false negatives : {fns.get('count', 0):>8,}  "
              f"rate {fns.get('rate', 0):.6f}  mean score {fns.get('mean_score', 0):.4f}  "
              f"confident misses {fns.get('high_confidence_misses', 0)}")
        confident = block.get("high_confidence_errors", {})
        if confident:
            print(f"    confident (>{confident.get('confidence_threshold', 0.9)}): "
                  f"{confident.get('wrong', 0):,} wrong of {confident.get('confident_predictions', 0):,} "
                  f"({confident.get('error_rate_among_confident', 0):.2%})")
        timing = block.get("drift_vs_degradation", {})
        if timing.get("available"):
            print(f"    first drift alert  : {timing.get('first_alert')}")
            print(f"    first bad window   : {timing.get('first_degraded_window')}")
            print(f"    drift preceded it  : {timing.get('drift_precedes_degradation')} "
                  f"(lag {timing.get('lag')}, tolerance {timing.get('tolerance')})")
        alerts = block.get("alerts_without_degradation", {})
        if alerts:
            print(f"    drift alerts       : {alerts.get('alerts')} "
                  f"({alerts.get('in_degraded_windows')} in degraded windows, "
                  f"{alerts.get('without_degradation')} in quiet ones, "
                  f"{alerts.get('unmatched_alerts', 0)} unplaceable"
                  + ("" if alerts.get("matched", True) else " -- NOT ALIGNED") + ")")
        calibration = block.get("calibration", {})
        if calibration:
            print(f"    Brier  {calibration.get('brier_backtest', 0):.5f} -> "
                  f"{calibration.get('brier_forward', 0):.5f} "
                  f"({calibration.get('brier_change', 0):+.5f})")
            print(f"    ECE    {calibration.get('ece_backtest', 0):.5f} -> "
                  f"{calibration.get('ece_forward', 0):.5f} "
                  f"({calibration.get('ece_change', 0):+.5f})")
        strongest = block.get("strongest_drift_features", []) or []
        if strongest:
            top = ", ".join(f"{e['feature']} (d={e['median_effect_size']:.3f})" for e in strongest[:4])
            print(f"    strongest drift    : {top}")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: analyse_run.py <experiment_dir>", file=sys.stderr)
        return 2
    run = Path(sys.argv[1])
    metrics_path = run / "metrics.json"
    if not metrics_path.is_file():
        print(f"{run} has no metrics.json, so the run never finished.", file=sys.stderr)
        return 1
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    print(f"run     : {run.name}")
    print(f"dataset : {metrics.get('dataset')}")
    split_summary(metrics)
    model_table(run, metrics)
    adaptation_table(metrics)
    drift_table(run, metrics)
    failure_report(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
