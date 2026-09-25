"""Evaluation plotting module."""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def plot_confusion_matrix(cm: np.ndarray, model_name: str, feature_set: str,
                          output_dir: str = "results/figures") -> Path:
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Normalize for display
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax)

    # Labels
    classes = ["Normal", "Anomaly"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix: {model_name} ({feature_set})")

    # Add text annotations
    thresh = cm_norm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2%})",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=12)

    plt.tight_layout()

    output_path = Path(output_dir) / f"confusion_matrix_{model_name}_{feature_set}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_model_comparison_f1(results: Dict[str, Any],
                              output_dir: str = "results/figures") -> Path:
    """Plot model comparison by F1 score."""
    df_data = []

    for feature_set, models in results.items():
        if "error" in models:
            continue
        for model_name, metrics in models.items():
            if "error" in metrics:
                continue
            df_data.append({
                "Feature Set": feature_set,
                "Model": model_name.replace("_", " ").title(),
                "F1 Score": metrics["f1_score"]
            })

    df = pd.DataFrame(df_data)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Bar plot
    feature_sets = ["FULL_METADATA", "RESTRICTED_METADATA"]
    models = ["Majority Baseline", "Logistic Regression", "Random Forest"]
    x = np.arange(len(models))
    width = 0.35

    for i, fs in enumerate(feature_sets):
        fs_data = df[df["Feature Set"] == fs]
        values = [fs_data[fs_data["Model"] == m]["F1 Score"].values[0] if not fs_data[fs_data["Model"] == m].empty else 0 for m in models]
        bars = ax.bar(x + (i - 0.5) * width, values, width, label=fs, alpha=0.8)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Model")
    ax.set_ylabel("F1 Score")
    ax.set_title("Model Comparison by F1 Score")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(title="Feature Set")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    output_path = Path(output_dir) / "model_comparison_f1.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_model_comparison_fpr(results: Dict[str, Any],
                               output_dir: str = "results/figures") -> Path:
    """Plot model comparison by False Positive Rate."""
    df_data = []

    for feature_set, models in results.items():
        if "error" in models:
            continue
        for model_name, metrics in models.items():
            if "error" in metrics:
                continue
            df_data.append({
                "Feature Set": feature_set,
                "Model": model_name.replace("_", " ").title(),
                "FPR": metrics["false_positive_rate"]
            })

    df = pd.DataFrame(df_data)

    fig, ax = plt.subplots(figsize=(8, 5))

    feature_sets = ["FULL_METADATA", "RESTRICTED_METADATA"]
    models = ["Majority Baseline", "Logistic Regression", "Random Forest"]
    x = np.arange(len(models))
    width = 0.35

    for i, fs in enumerate(feature_sets):
        fs_data = df[df["Feature Set"] == fs]
        values = [fs_data[fs_data["Model"] == m]["FPR"].values[0] if not fs_data[fs_data["Model"] == m].empty else 0 for m in models]
        bars = ax.bar(x + (i - 0.5) * width, values, width, label=fs, alpha=0.8)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Model")
    ax.set_ylabel("False Positive Rate")
    ax.set_title("Model Comparison by False Positive Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(title="Feature Set")

    plt.tight_layout()

    output_path = Path(output_dir) / "model_comparison_fpr.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_feature_importance(model: Any, feature_names: List[str],
                            model_name: str, feature_set: str,
                            top_n: int = 20,
                            output_dir: str = "results/figures") -> Path:
    """Plot feature importance for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        print(f"  Model {model_name} does not have feature_importances_")
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))

    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    bars = ax.barh(range(len(top_features)), top_importances, align="center")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Feature Importances: {model_name} ({feature_set})")

    # Add value labels
    for bar, val in zip(bars, top_importances):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", ha="left", va="center", fontsize=9)

    plt.tight_layout()

    output_path = Path(output_dir) / f"feature_importance_{model_name}_{feature_set}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_class_distribution(y_train: np.ndarray, y_test: np.ndarray,
                            output_dir: str = "results/figures") -> Path:
    """Plot class distribution in train and test sets."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, y, title in [(axes[0], y_train, "Training Set"), (axes[1], y_test, "Test Set")]:
        unique, counts = np.unique(y, return_counts=True)
        labels = ["Normal", "Anomaly"]
        colors = ["#2ecc71", "#e74c3c"]

        bars = ax.bar(labels, counts, color=colors, alpha=0.8)
        ax.set_title(title)
        ax.set_ylabel("Count")

        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f"{count:,} ({count/len(y):.1%})", ha="center", va="bottom")

    plt.tight_layout()

    output_path = Path(output_dir) / "class_distribution.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def generate_all_plots(results: Dict[str, Any], feature_names: Dict[str, List[str]],
                       models_dir: str = "results/models",
                       output_dir: str = "results/figures",
                       processed_dir: str = "data/processed") -> Dict[str, Path]:
    """Generate all evaluation plots."""
    from privacy_preserving_nad.models.baseline import load_model

    output_paths = {}

    print("Generating plots...")

    # 1. Confusion matrices for main models (skip majority baseline)
    for feature_set in ["FULL_METADATA", "RESTRICTED_METADATA"]:
        for model_name in ["logistic_regression", "random_forest"]:
            if feature_set in results and model_name in results[feature_set]:
                metrics = results[feature_set][model_name]
                if "error" not in metrics:
                    cm = np.array(metrics["confusion_matrix"])
                    path = plot_confusion_matrix(cm, model_name, feature_set, output_dir)
                    output_paths[f"confusion_matrix_{model_name}_{feature_set}"] = path
                    print(f"  Created: {path.name}")

    # 2. Model comparison by F1
    path = plot_model_comparison_f1(results, output_dir)
    output_paths["model_comparison_f1"] = path
    print(f"  Created: {path.name}")

    # 3. Model comparison by FPR
    path = plot_model_comparison_fpr(results, output_dir)
    output_paths["model_comparison_fpr"] = path
    print(f"  Created: {path.name}")

    # 4. Feature importance for Random Forest (FULL_METADATA)
    try:
        rf_model = load_model("random_forest", "FULL_METADATA", models_dir)
        fnames = feature_names.get("FULL_METADATA", [])
        if fnames:
            path = plot_feature_importance(
                rf_model, fnames, "random_forest", "FULL_METADATA",
                output_dir=output_dir,
            )
            if path:
                output_paths["feature_importance_rf_full"] = path
                print(f"  Created: {path.name}")
    except Exception as e:
        print(f"  Could not create feature importance plot: {e}")

    # 5. Class distribution (load from one feature set)
    try:
        from privacy_preserving_nad.data.preprocess import load_processed_data
        data = load_processed_data("FULL_METADATA", processed_dir)
        path = plot_class_distribution(data["y_train"], data["y_test"], output_dir)
        output_paths["class_distribution"] = path
        print(f"  Created: {path.name}")
    except Exception as e:
        print(f"  Could not create class distribution plot: {e}")

    print(f"\nAll plots saved to: results/figures/")
    return output_paths


def main():
    """Generate all plots from evaluation results."""
    # Load evaluation results
    with open("results/metrics/evaluation_results.json", "r") as f:
        results = json.load(f)

    # Load feature names
    feature_names = {}
    for fs in ["FULL_METADATA", "RESTRICTED_METADATA"]:
        try:
            with open(f"data/processed/{fs}/feature_names.json", "r") as f:
                feature_names[fs] = json.load(f)
        except Exception:
            feature_names[fs] = []

    generate_all_plots(results, feature_names)


if __name__ == "__main__":
    main()