"""Model prediction module."""

import numpy as np
from typing import Dict, Any
from pathlib import Path
import joblib

from privacy_preserving_nad.data.preprocess import load_processed_data
from privacy_preserving_nad.models.baseline import load_model


def predict(model: Any, X: np.ndarray) -> np.ndarray:
    """Generate predictions."""
    return model.predict(X)


def predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """Generate prediction probabilities."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    else:
        # For models without predict_proba (e.g., DummyClassifier)
        preds = model.predict(X)
        n_classes = len(np.unique(preds))
        proba = np.zeros((len(preds), n_classes))
        for i, p in enumerate(preds):
            proba[i, p] = 1.0
        return proba


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                         y_proba: np.ndarray = None) -> Dict[str, Any]:
    """Compute evaluation metrics from predictions."""
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score, average_precision_score
    )

    # Basic metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
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

    return results


def run_inference(model_name: str, feature_set: str, model_dir: str = "results/models",
                  processed_dir: str = "data/processed") -> Dict[str, Any]:
    """Run inference on test set for a trained model."""
    # Load model
    model = load_model(model_name, feature_set, model_dir)

    # Load test data
    data = load_processed_data(feature_set, processed_dir)
    X_test, y_test = data["X_test"], data["y_test"]

    # Predict
    y_pred = predict(model, X_test)
    y_proba = predict_proba(model, X_test)

    # Evaluate
    metrics = evaluate_predictions(y_test, y_pred, y_proba)
    metrics["model_name"] = model_name
    metrics["feature_set"] = feature_set

    return metrics


def run_all_inference(model_dir: str = "results/models",
                      processed_dir: str = "data/processed",
                      metrics_path: str = "results/metrics/evaluation_results.json") -> Dict[str, Any]:
    """Run inference for all model/feature set combinations."""
    import json

    model_names = ["majority_baseline", "logistic_regression", "random_forest"]
    feature_sets = ["FULL_METADATA", "RESTRICTED_METADATA"]

    all_results = {}

    for feature_set in feature_sets:
        all_results[feature_set] = {}
        for model_name in model_names:
            try:
                print(f"Evaluating {model_name} on {feature_set}...")
                metrics = run_inference(model_name, feature_set, model_dir, processed_dir)
                all_results[feature_set][model_name] = metrics
                print(f"  F1: {metrics['f1_score']:.4f}, FPR: {metrics['false_positive_rate']:.4f}")
            except Exception as e:
                print(f"  Error: {e}")
                all_results[feature_set][model_name] = {"error": str(e)}

    # Save evaluation results
    output_path = Path(metrics_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nEvaluation results saved to: {output_path}")

    return all_results


def main():
    """Run inference on all models."""
    run_all_inference()


if __name__ == "__main__":
    main()