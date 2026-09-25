"""Failure analysis driven by the actual experiment output.

Every finding here is derived from prediction-level tables and drift events
produced by a run, not from assumptions about what usually goes wrong. A
function that finds nothing returns an empty finding rather than a plausible
sentence.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def degradation_by_window(predictions: pd.DataFrame, freq: str = "6h", min_rows: int = 50) -> pd.DataFrame:
    """Per-window recall and false-positive rate over the forward period."""
    if predictions.empty:
        return predictions
    work = predictions.copy()
    work["bucket"] = work["timestamp"].dt.floor(freq)
    rows = []
    for bucket, group in work.groupby("bucket"):
        if len(group) < min_rows:
            continue
        y = group["y_true"].to_numpy()
        pred = group["y_pred"].to_numpy()
        attacks = y == 1
        benign = y == 0
        rows.append(
            {
                "bucket": str(bucket),
                "rows": int(len(group)),
                "attack_rate": float(y.mean()),
                "recall": float((pred[attacks] == 1).mean()) if attacks.any() else None,
                "fpr": float((pred[benign] == 1).mean()) if benign.any() else None,
                "false_negatives": int(group["false_negative"].sum()),
                "false_positives": int(group["false_positive"].sum()),
                "mean_score": float(group["y_score"].mean()),
            }
        )
    return pd.DataFrame(rows)


def largest_degradation_windows(
    backtest_metrics: Dict, windows: pd.DataFrame, top_n: int = 5
) -> List[Dict[str, object]]:
    """Windows where recall or false-positive rate is worst relative to the backtest."""
    if windows.empty:
        return []
    out = []
    for metric, direction in [("recall", "low"), ("fpr", "high")]:
        series = windows[metric].dropna()
        if series.empty:
            continue
        baseline = backtest_metrics.get("recall" if metric == "recall" else "false_positive_rate")
        worst = series.nsmallest(top_n) if direction == "low" else series.nlargest(top_n)
        for bucket, value in worst.items():
            out.append(
                {
                    "metric": metric,
                    "window": str(bucket),
                    "value": float(value),
                    "backtest_value": baseline,
                    "delta": None if baseline is None else float(value - baseline),
                }
            )
    return out


def strongest_drift_features(drift_events: pd.DataFrame, top_n: int = 8) -> List[Dict[str, object]]:
    """Features with the largest median effect size among alerting windows."""
    if drift_events.empty:
        return []
    flagged = drift_events[drift_events["alert"]].copy()
    if flagged.empty:
        return []
    grouped = flagged.groupby("feature")["effect_size"].agg(["median", "max", "count"])
    grouped = grouped.sort_values("median", ascending=False).head(top_n)
    return [
        {
            "feature": str(feature),
            "median_effect_size": float(row["median"]),
            "max_effect_size": float(row["max"]),
            "alerting_windows": int(row["count"]),
        }
        for feature, row in grouped.iterrows()
    ]


def false_positive_profile(predictions: pd.DataFrame, top_n: int = 5) -> Dict[str, object]:
    """Where the detector's false positives concentrate."""
    if predictions.empty:
        return {}
    fps = predictions[predictions["false_positive"] == 1]
    return {
        "count": int(len(fps)),
        "rate": float(len(fps) / len(predictions)),
        "mean_score": float(fps["y_score"].mean()) if len(fps) else 0.0,
        "worst_windows": (
            fps.groupby(fps["timestamp"].dt.floor("6h")).size().nlargest(top_n).to_dict()
            if len(fps) else {}
        ),
    }


def false_negative_profile(predictions: pd.DataFrame, top_n: int = 5) -> Dict[str, object]:
    """Where attacks are missed, and how confident the model was when it missed them."""
    if predictions.empty:
        return {}
    fns = predictions[predictions["false_negative"] == 1]
    return {
        "count": int(len(fns)),
        "rate": float(len(fns) / len(predictions)),
        "mean_score": float(fns["y_score"].mean()) if len(fns) else 0.0,
        "high_confidence_misses": int((fns["y_score"] > 0.8).sum()) if len(fns) else 0,
        "worst_windows": (
            fns.groupby(fns["timestamp"].dt.floor("6h")).size().nlargest(top_n).to_dict()
            if len(fns) else {}
        ),
    }


def high_confidence_errors(predictions: pd.DataFrame, threshold: float = 0.9) -> Dict[str, object]:
    """Predictions the model was most sure of, and how often those were wrong."""
    if predictions.empty:
        return {}
    confident = predictions[predictions["y_score"] >= threshold]
    wrong = confident[((confident["y_pred"] == 1) & (confident["y_true"] == 0)) |
                      ((confident["y_pred"] == 0) & (confident["y_true"] == 1))]
    return {
        "confidence_threshold": threshold,
        "confident_predictions": int(len(confident)),
        "wrong": int(len(wrong)),
        "error_rate_among_confident": float(len(wrong) / len(confident)) if len(confident) else 0.0,
    }


def drift_vs_degradation(
    drift_events: pd.DataFrame,
    windows: pd.DataFrame,
    tolerance: str = "6h",
) -> Dict[str, object]:
    """Does a drift alert arrive before the model actually degrades?

    Compares the first alerting window with the first window whose recall falls
    more than a small margin below the backtest recall. The sign of the lag is
    the finding: a positive lag means drift was visible first.
    """
    if drift_events.empty or windows.empty:
        return {"available": False, "reason": "missing drift events or window metrics"}
    first_alert = drift_events[drift_events["alert"]]["window_start"].min()
    if pd.isna(first_alert):
        return {"available": True, "first_alert": None, "note": "no drift alert fired"}

    windows = windows.copy()
    windows["bucket"] = pd.to_datetime(windows["bucket"])
    usable = windows.dropna(subset=["recall"])
    if usable.empty:
        return {"available": True, "first_alert": str(first_alert), "note": "no window had both classes"}

    first_bad = usable["bucket"].min()
    lag = first_bad - pd.Timestamp(first_alert)
    return {
        "available": True,
        "first_alert": str(first_alert),
        "first_degraded_window": str(first_bad),
        "drift_precedes_degradation": bool(pd.Timestamp(first_alert) <= first_bad),
        "lag": str(lag),
        "tolerance": tolerance,
    }


def alerts_without_degradation(
    drift_events: pd.DataFrame, windows: pd.DataFrame
) -> Dict[str, object]:
    """Count drift alerts that landed in windows where the model was fine.

    Drift is a property of the data, not a verdict on the model, so this counts
    how often the two came apart.

    The two tables are bucketed on different cadences: drift windows are as long
    as the drift configuration asks for, and the model windows are as long as
    the failure analysis asks for. Comparing their labels directly therefore
    compares timestamps that need not mean the same interval, and every alert
    falls outside both sets. The windows are therefore aligned by assigning each
    drift window to whichever model window contains its midpoint, which is the
    coarsest honest comparison, and the bucket that remains unassigned is
    reported rather than dropped.
    """
    empty = {"alerts": 0, "in_degraded_windows": 0, "without_degradation": 0,
             "unmatched_alerts": 0, "model_windows": 0, "degraded_windows": 0,
             "matched": False}
    if drift_events is None or drift_events.empty or windows is None or windows.empty:
        return empty

    alert_windows = sorted(set(drift_events[drift_events["alert"]]["window_start"].astype(str)))
    if not alert_windows:
        return empty

    frame = windows.copy()
    if "bucket" not in frame.columns or "recall" not in frame.columns:
        return empty
    frame["bucket"] = pd.to_datetime(frame["bucket"], errors="coerce")
    frame = frame[frame["bucket"].notna() & frame["recall"].notna()]
    if frame.empty:
        return empty

    frame = frame.sort_values("bucket")
    starts = frame["bucket"].tolist()
    recalls = frame["recall"].tolist()

    # Each drift window is assigned to the model window containing its midpoint.
    # Midpoint rather than start, because a drift window can begin before the
    # first model window and the question is which period the alert describes.
    midpoints = []
    for start in alert_windows:
        start_ts = pd.Timestamp(start)
        try:
            end_ts = pd.Timestamp(drift_events[drift_events["window_start"].astype(str) == start]["window_end"].iloc[0])
        except (IndexError, ValueError):
            end_ts = start_ts
        midpoints.append(start_ts + (end_ts - start_ts) / 2)

    in_degraded = without = unmatched = 0
    for midpoint in midpoints:
        # Last model window that starts at or before the midpoint.
        position = None
        for i, bucket in enumerate(starts):
            if bucket <= midpoint:
                position = i
            else:
                break
        if position is None or midpoint > starts[position] + pd.Timedelta(days=1):
            # Either before the first window, or implausibly after the last.
            unmatched += 1
            continue
        if recalls[position] < 0.5:
            in_degraded += 1
        else:
            without += 1

    return {
        "alerts": len(alert_windows),
        "in_degraded_windows": in_degraded,
        "without_degradation": without,
        "unmatched_alerts": unmatched,
        "model_windows": int(len(frame)),
        "degraded_windows": int((frame["recall"] < 0.5).sum()),
        # False when the buckets could not be aligned at all, which is itself
        # worth knowing: it means the comparison below is not evidence.
        "matched": unmatched < len(alert_windows),
    }


def calibration_drift(baseline: Dict, forward: Dict) -> Dict[str, object]:
    """How far confidence calibration moved between the backtest and forward test."""
    out = {}
    for key in ["brier", "ece"]:
        a, b = baseline.get(key), forward.get(key)
        if a is not None and b is not None:
            out[f"{key}_backtest"] = float(a)
            out[f"{key}_forward"] = float(b)
            out[f"{key}_change"] = float(b - a)
    return out


def build_failure_analysis(
    backtest_metrics: Dict,
    backtest_predictions: pd.DataFrame,
    forward_predictions: pd.DataFrame,
    drift_events: pd.DataFrame,
    window_freq: str = "6h",
    backtest_calibration: Optional[Dict] = None,
    forward_calibration: Optional[Dict] = None,
) -> Dict[str, object]:
    """Assemble the full failure-analysis section for one model."""
    windows = degradation_by_window(forward_predictions, freq=window_freq)
    analysis: Dict[str, object] = {
        "window_frequency": window_freq,
        "windows_analyzed": int(len(windows)),
        "largest_degradation_windows": largest_degradation_windows(backtest_metrics, windows),
        "strongest_drift_features": strongest_drift_features(drift_events),
        "false_positives": false_positive_profile(forward_predictions),
        "false_negatives": false_negative_profile(forward_predictions),
        "high_confidence_errors": high_confidence_errors(forward_predictions),
        "drift_vs_degradation": drift_vs_degradation(drift_events, windows),
        "alerts_without_degradation": alerts_without_degradation(drift_events, windows),
    }
    if backtest_calibration and forward_calibration:
        analysis["calibration"] = calibration_drift(backtest_calibration, forward_calibration)
    return analysis
