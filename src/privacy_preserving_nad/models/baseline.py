"""Baseline models for anomaly detection."""

import numpy as np
from typing import Dict, Any
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
import yaml


class MajorityBaseline:
    """Majority class baseline - always predicts the most frequent class."""

    def __init__(self, random_state: int = 42):
        self.model = DummyClassifier(strategy="most_frequent", random_state=random_state)
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MajorityBaseline":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.model.score(X, y)

    def get_params(self, deep: bool = True) -> Dict:
        return self.model.get_params(deep)

    def set_params(self, **params) -> "MajorityBaseline":
        self.model.set_params(**params)
        return self


def create_logistic_regression(config: Dict[str, Any]) -> LogisticRegression:
    """Create Logistic Regression model with config parameters."""
    lr_config = config.get("logistic_regression", {})
    return LogisticRegression(
        max_iter=lr_config.get("max_iter", 1000),
        C=lr_config.get("C", 1.0),
        solver=lr_config.get("solver", "lbfgs"),
        class_weight=lr_config.get("class_weight", "balanced"),
        random_state=config.get("random_seed", 42)
    )


def create_random_forest(config: Dict[str, Any]) -> RandomForestClassifier:
    """Create Random Forest model with config parameters."""
    rf_config = config.get("random_forest", {})
    return RandomForestClassifier(
        n_estimators=rf_config.get("n_estimators", 200),
        max_depth=rf_config.get("max_depth", 15),
        min_samples_split=rf_config.get("min_samples_split", 5),
        min_samples_leaf=rf_config.get("min_samples_leaf", 2),
        class_weight=rf_config.get("class_weight", "balanced"),
        random_state=config.get("random_seed", 42),
        n_jobs=rf_config.get("n_jobs", -1)
    )


def get_model_config(config_path: str = "configs/config.yaml") -> Dict:
    """Load model configuration from YAML."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["models"]


def save_model(model: Any, model_name: str, feature_set: str, output_dir: str = "results/models") -> Path:
    """Save a trained model to disk."""
    output_path = Path(output_dir) / feature_set
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / f"{model_name}.joblib"
    joblib.dump(model, filepath)
    return filepath


def load_model(model_name: str, feature_set: str, model_dir: str = "results/models") -> Any:
    """Load a trained model from disk."""
    filepath = Path(model_dir) / feature_set / f"{model_name}.joblib"
    return joblib.load(filepath)


def main():
    """Test model creation."""
    config = get_model_config()

    print("Creating models...")
    majority = MajorityBaseline(random_state=config["random_seed"])
    lr = create_logistic_regression(config)
    rf = create_random_forest(config)

    print(f"Majority baseline: {majority}")
    print(f"Logistic Regression: {lr}")
    print(f"Random Forest: {rf}")


if __name__ == "__main__":
    main()