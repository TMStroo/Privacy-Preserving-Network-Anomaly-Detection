"""Evaluation: classification metrics, a fixed false-positive budget, and
degradation measurements that compare a backtest against a forward test.

The operating threshold is chosen on validation data against a configured
false-positive target and then applied unchanged to the forward test. No
function here ever inspects forward-test labels to pick a threshold.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

DEFAULT_TARGET_FPR = 0.05


def basic_metrics(y_true, y_score, threshold: float = 0.5) -> Dict[str, float]:
    """Standard classification metrics at a fixed threshold."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    out = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "rows": int(len(y_true)),
        "attacks": int(y_true.sum()),
        "threshold": float(threshold),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def threshold_for_target_fpr(y_true, y_score, target_fpr: float = DEFAULT_TARGET_FPR) -> float:
    """Pick the lowest threshold whose false-positive rate stays within budget.

    The score at which the budget is first met is found by sorting the negative
    scores and taking the (1 - target_fpr) quantile, so the threshold is a
    property of the negative class alone.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    negatives = y_score[y_true == 0]
    if negatives.size == 0:
        return 0.5
    quantile = 1.0 - float(target_fpr)
    return float(np.quantile(negatives, quantile))


def metrics_at_target_fpr(y_true, y_score, target_fpr: float = DEFAULT_TARGET_FPR) -> Dict[str, float]:
    """Metrics at the threshold that meets the false-positive budget."""
    threshold = threshold_for_target_fpr(y_true, y_score, target_fpr)
    out = basic_metrics(y_true, y_score, threshold)
    out["target_fpr"] = float(target_fpr)
    out["threshold_source"] = "validation split"
    return out


def degradation(backtest: Dict[str, float], forward: Dict[str, float]) -> Dict[str, float]:
    """Change from backtest to forward test for every comparable metric."""
    out: Dict[str, float] = {}
    for key in ["precision", "recall", "f1", "false_positive_rate", "false_negative_rate", "balanced_accuracy", "pr_auc", "roc_auc"]:
        a, b = backtest.get(key), forward.get(key)
        if a is None or b is None:
            out[f"{key}_change"] = None
            out[f"{key}_relative_change"] = None
            continue
        out[f"{key}_change"] = float(b - a)
        out[f"{key}_relative_change"] = float((b - a) / a) if a else None
    return out


def f1_degradation(backtest: Dict[str, float], forward: Dict[str, float]) -> float:
    return float(backtest["f1"] - forward["f1"])


def recall_degradation(backtest: Dict[str, float], forward: Dict[str, float]) -> float:
    return float(backtest["recall"] - forward["recall"])


def fpr_increase(backtest: Dict[str, float], forward: Dict[str, float]) -> float:
    return float(forward["false_positive_rate"] - backtest["false_positive_rate"])


def calibration(y_true, y_score, n_bins: int = 10) -> Dict[str, object]:
    """Brier score, expected calibration error and a reliability table."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.size == 0:
        return {"brier": None, "ece": None, "curve": []}

    brier = float(brier_score_loss(y_true, y_score))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_score, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    curve = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            curve.append({"bin": b, "n": 0, "mean_pred": None, "observed": None})
            continue
        mean_pred = float(y_score[mask].mean())
        observed = float(y_true[mask].mean())
        ece += (count / y_true.size) * abs(mean_pred - observed)
        curve.append({"bin": b, "n": count, "mean_pred": mean_pred, "observed": observed})
    return {"brier": brier, "ece": float(ece), "curve": curve}


def predictions_frame(y_true, y_score, timestamps, threshold: float) -> pd.DataFrame:
    """Row-level predictions, kept for failure analysis and calibration."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(pd.Series(list(timestamps))).to_numpy(),
            "y_true": np.asarray(y_true),
            "y_score": np.asarray(y_score, dtype=float),
            "y_pred": y_pred,
            "false_positive": ((y_pred == 1) & (np.asarray(y_true) == 0)).astype(int),
            "false_negative": ((y_pred == 0) & (np.asarray(y_true) == 1)).astype(int),
        }
    )


def grouped_predictions(predictions: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate predictions into time buckets for per-window degradation."""
    if predictions.empty:
        return predictions
    work = predictions.copy()
    work["bucket"] = work["timestamp"].dt.floor(freq)
    rows = []
    for bucket, group in work.groupby("bucket"):
        y_true = group["y_true"].to_numpy()
        y_pred = group["y_pred"].to_numpy()
        rows.append(
            {
                "bucket": bucket,
                "rows": int(len(group)),
                "attacks": int(y_true.sum()),
                "attack_rate": float(y_true.mean()) if len(group) else 0.0,
                "recall": float((y_pred[y_true == 1] == 1).mean()) if (y_true == 1).any() else None,
                "fpr": float((y_pred[y_true == 0] == 1).mean()) if (y_true == 0).any() else None,
                "mean_score": float(group["y_score"].mean()),
                "false_positives": int(group["false_positive"].sum()),
                "false_negatives": int(group["false_negative"].sum()),
            }
        )
    return pd.DataFrame(rows)
