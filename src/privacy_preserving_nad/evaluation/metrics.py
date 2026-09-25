"""Evaluation metrics computation and reporting."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    classification_report
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray = None) -> Dict[str, Any]:
    """Compute all evaluation metrics."""
    # Basic metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    # Rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Class counts
    n_normal = int(np.sum(y_true == 0))
    n_anomaly = int(np.sum(y_true == 1))

    results = {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "true_positive_rate": float(tpr),
        "true_negative_rate": float(tnr),
        "confusion_matrix": cm.tolist(),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "n_samples": int(len(y_true)),
        "n_normal": n_normal,
        "n_anomaly": n_anomaly
    }

    # AUC metrics if probabilities available
    if y_proba is not None and y_proba.shape[1] >= 2:
        try:
            results["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            results["pr_auc"] = float(average_precision_score(y_true, y_proba[:, 1]))
        except Exception:
            results["roc_auc"] = None
            results["pr_auc"] = None
    else:
        results["roc_auc"] = None
        results["pr_auc"] = None

    return results


def create_comparison_table(results: Dict[str, Any]) -> pd.DataFrame:
    """Create a comparison table from evaluation results."""
    rows = []

    for feature_set, models in results.items():
        if "error" in models:
            continue
        for model_name, metrics in models.items():
            if "error" in metrics:
                continue

            row = {
                "Feature Set": feature_set,
                "Model": model_name.replace("_", " ").title(),
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1 Score": metrics["f1_score"],
                "False Positive Rate": metrics["false_positive_rate"],
                "False Negative Rate": metrics["false_negative_rate"],
                "ROC AUC": metrics.get("roc_auc"),
                "PR AUC": metrics.get("pr_auc"),
                "Test Samples": metrics["n_samples"],
                "Normal Samples": metrics["n_normal"],
                "Anomaly Samples": metrics["n_anomaly"]
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Sort for readability (an empty result set means every run failed - the
    # caller reports that instead of crashing on missing columns)
    feature_order = ["FULL_METADATA", "RESTRICTED_METADATA"]
    model_order = ["Majority Baseline", "Logistic Regression", "Random Forest"]

    if df.empty:
        return pd.DataFrame(columns=[
            "Feature Set", "Model", "Precision", "Recall", "F1 Score",
            "False Positive Rate", "False Negative Rate", "ROC AUC", "PR AUC",
            "Test Samples", "Normal Samples", "Anomaly Samples",
        ])

    df["Feature Set"] = pd.Categorical(df["Feature Set"], categories=feature_order, ordered=True)
    df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)
    df = df.sort_values(["Feature Set", "Model"]).reset_index(drop=True)

    return df


def format_comparison_table(df: pd.DataFrame) -> str:
    """Format comparison table as markdown."""
    # Select and format columns for display
    display_cols = ["Feature Set", "Model", "Precision", "Recall", "F1 Score",
                    "False Positive Rate", "False Negative Rate"]

    fmt_df = df[display_cols].copy()
    for col in ["Precision", "Recall", "F1 Score", "False Positive Rate", "False Negative Rate"]:
        fmt_df[col] = fmt_df[col].apply(lambda x: f"{x:.4f}")

    return fmt_df.to_markdown(index=False)


def save_metrics_csv(results: Dict[str, Any], output_path: str = "results/metrics/comparison.csv"):
    """Save comparison table as CSV."""
    df = create_comparison_table(results)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Comparison table saved to: {output_path}")


def save_metrics_json(results: Dict[str, Any], output_path: str = "results/metrics/all_metrics.json"):
    """Save all metrics as JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"All metrics saved to: {output_path}")


def load_evaluation_results(path: str = "results/metrics/evaluation_results.json") -> Dict:
    """Load evaluation results from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def print_summary(results: Dict[str, Any]):
    """Print a summary of results."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    df = create_comparison_table(results)
    if df.empty:
        print("No model produced metrics - every evaluation failed.")
        print("See the errors reported by the evaluation step above.")
        return
    print(format_comparison_table(df))

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Compare feature sets for each model
    for model in ["Logistic Regression", "Random Forest"]:
        full_row = df[(df["Feature Set"] == "FULL_METADATA") & (df["Model"] == model)]
        rest_row = df[(df["Feature Set"] == "RESTRICTED_METADATA") & (df["Model"] == model)]

        if not full_row.empty and not rest_row.empty:
            full_f1 = full_row["F1 Score"].values[0]
            rest_f1 = rest_row["F1 Score"].values[0]
            full_fpr = full_row["False Positive Rate"].values[0]
            rest_fpr = rest_row["False Positive Rate"].values[0]

            f1_diff = full_f1 - rest_f1
            fpr_diff = full_fpr - rest_fpr

            print(f"\n{model}:")
            print(f"  F1 Score:  FULL={full_f1:.4f}  RESTRICTED={rest_f1:.4f}  Diff={f1_diff:+.4f}")
            print(f"  FPR:       FULL={full_fpr:.4f}  RESTRICTED={rest_fpr:.4f}  Diff={fpr_diff:+.4f}")

    # Baseline comparison
    for feature_set in ["FULL_METADATA", "RESTRICTED_METADATA"]:
        base_row = df[(df["Feature Set"] == feature_set) & (df["Model"] == "Majority Baseline")]
        lr_row = df[(df["Feature Set"] == feature_set) & (df["Model"] == "Logistic Regression")]
        rf_row = df[(df["Feature Set"] == feature_set) & (df["Model"] == "Random Forest")]

        if not base_row.empty and not lr_row.empty and not rf_row.empty:
            base_f1 = base_row["F1 Score"].values[0]
            lr_f1 = lr_row["F1 Score"].values[0]
            rf_f1 = rf_row["F1 Score"].values[0]

            print(f"\n{feature_set} - Improvement over Majority Baseline:")
            print(f"  Logistic Regression: {lr_f1 - base_f1:+.4f} F1")
            print(f"  Random Forest:       {rf_f1 - base_f1:+.4f} F1")


def main():
    """Load and summarize evaluation results."""
    results = load_evaluation_results()
    print_summary(results)

    # Save formatted comparison
    df = create_comparison_table(results)
    save_metrics_csv(results)
    save_metrics_json(results)


if __name__ == "__main__":
    main()