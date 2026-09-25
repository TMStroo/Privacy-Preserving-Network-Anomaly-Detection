"""Model training module."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import yaml
from pathlib import Path
import time

from privacy_preserving_nad.models.baseline import (
    MajorityBaseline,
    create_logistic_regression,
    create_random_forest,
    get_model_config,
    save_model
)
from privacy_preserving_nad.data.preprocess import load_processed_data


def train_model(model: Any, X_train: np.ndarray, y_train: np.ndarray,
                model_name: str, feature_set: str) -> Dict[str, Any]:
    """Train a single model and return training info."""
    print(f"  Training {model_name} on {feature_set}...")

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Get training accuracy (for reference)
    train_score = model.score(X_train, y_train)

    info = {
        "model_name": model_name,
        "feature_set": feature_set,
        "train_time_seconds": train_time,
        "train_accuracy": train_score,
        "n_train_samples": len(X_train),
        "n_features": X_train.shape[1]
    }

    print(f"    Train time: {train_time:.2f}s, Train accuracy: {train_score:.4f}")

    # Save model
    model_path = save_model(model, model_name, feature_set)
    info["model_path"] = str(model_path)

    return info


def train_all_models(feature_set: str, config: Dict = None) -> Dict[str, Any]:
    """Train all baseline models for a feature set."""
    if config is None:
        config = get_model_config()

    # Load processed data
    data = load_processed_data(feature_set)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    print(f"\nTraining models on {feature_set}")
    print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")

    results = {"feature_set": feature_set, "models": {}}

    # 1. Majority Baseline
    majority = MajorityBaseline(random_state=config["random_seed"])
    results["models"]["majority_baseline"] = train_model(
        majority, X_train, y_train, "majority_baseline", feature_set
    )

    # 2. Logistic Regression
    lr = create_logistic_regression(config)
    results["models"]["logistic_regression"] = train_model(
        lr, X_train, y_train, "logistic_regression", feature_set
    )

    # 3. Random Forest
    rf = create_random_forest(config)
    results["models"]["random_forest"] = train_model(
        rf, X_train, y_train, "random_forest", feature_set
    )

    return results


def main():
    """Train models for all feature sets."""
    import json

    config = get_model_config()
    all_results = {}

    for feature_set in ["FULL_METADATA", "RESTRICTED_METADATA"]:
        try:
            all_results[feature_set] = train_all_models(feature_set, config)
        except Exception as e:
            print(f"Error training on {feature_set}: {e}")
            all_results[feature_set] = {"error": str(e)}

    # Save training report
    output_path = Path("results/metrics/training_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nTraining report saved to: {output_path}")


if __name__ == "__main__":
    main()