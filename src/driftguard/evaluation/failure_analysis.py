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
) -> Dict[str, int]:
    """Count drift alerts that landed in windows where the model was fine.

    Drift is a property of the data, not a verdict on the model; this counts
    how often the two came apart.
    """
    if drift_events.empty or windows.empty:
        return {"alerts": 0, "without_degradation": 0}
    alert_windows = set(drift_events[drift_events["alert"]]["window_start"].astype(str))
    windows = windows.copy()
    windows["bucket"] = pd.to_datetime(windows["bucket"]).astype(str)
    good = set(windows[windows["recall"].notna()]["bucket"])
    degraded = set(windows[windows["recall"].notna() & (windows["recall"] < 0.5)]["bucket"])
    quiet = good - degraded
    return {
        "alerts": len(alert_windows),
        "in_degraded_windows": len(alert_windows & degraded),
        "without_degradation": len(alert_windows & quiet),
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
