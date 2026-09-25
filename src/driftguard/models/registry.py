"""Model registry for the anomaly-detection experiments.

Four models spanning the usual complexity range, plus a neural network for the
uncertainty study. Every estimator is created through ``build_model`` so that
seeds, thread counts and hyperparameter provenance stay in one place instead of
being repeated at each call site.
"""

import time
from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

MODELS: Dict[str, str] = {
    "majority": "Majority baseline (predicts the training majority class)",
    "logistic_regression": "Logistic regression with balanced class weights",
    "random_forest": "Random forest with balanced class weights",
    "gradient_boosting": "Gradient boosting on the same feature set",
    "neural_network": "Small multilayer perceptron (uncertainty study only)",
}


class MajorityBaseline:
    """Predicts the majority class seen during training.

    It exists to give every recall figure a reference point: a detector that
    cannot beat "always say the majority class" has not learned anything.
    """

    def __init__(self):
        self.constant_: Optional[int] = None

    def fit(self, X, y):
        y = np.asarray(y)
        self.constant_ = int(np.bincount(y.astype(int)).argmax())
        return self

    def predict_proba(self, X) -> np.ndarray:
        n = len(X)
        proba = np.zeros((n, 2), dtype=float)
        proba[:, 1 - self.constant_] = 1.0
        return proba

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.constant_, dtype=int)

    @property
    def params(self) -> dict:
        return {"constant": self.constant_}


def build_model(name: str, config: Dict, seed: int) -> object:
    """Instantiate one estimator from its configuration block."""
    if name == "majority":
        return MajorityBaseline()
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=int(config.get("max_iter", 1000)),
            C=float(config.get("C", 1.0)),
            solver=str(config.get("solver", "lbfgs")),
            class_weight=config.get("class_weight", "balanced"),
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(config.get("n_estimators", 200)),
            max_depth=config.get("max_depth", None),
            min_samples_split=int(config.get("min_samples_split", 2)),
            min_samples_leaf=int(config.get("min_samples_leaf", 1)),
            class_weight=config.get("class_weight", "balanced"),
            n_jobs=int(config.get("n_jobs", -1)),
            random_state=seed,
        )
    if name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=int(config.get("n_estimators", 100)),
            max_depth=int(config.get("max_depth", 3)),
            learning_rate=float(config.get("learning_rate", 0.1)),
            subsample=float(config.get("subsample", 1.0)),
            random_state=seed,
        )
    if name == "neural_network":
        hidden = tuple(config.get("hidden_layer_sizes", (64, 32)))
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            activation=str(config.get("activation", "relu")),
            alpha=float(config.get("alpha", 1e-4)),
            learning_rate_init=float(config.get("learning_rate_init", 1e-3)),
            max_iter=int(config.get("max_iter", 60)),
            early_stopping=bool(config.get("early_stopping", True)),
            n_iter_no_change=int(config.get("n_iter_no_change", 5)),
            random_state=seed,
        )
    raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODELS)}")


def model_config_names(config: Dict) -> List[str]:
    """Model names enabled by a configuration block, in a stable order.

    Two shapes are accepted because both appear in real configs: a mapping with
    an ``enabled`` list plus per-model parameter blocks, and a bare list of
    names for a quick run. The order always follows the registry so a run is
    reproducible regardless of how the config was written.
    """
    block = config.get("models", config) or {}
    if isinstance(block, list):
        enabled = block
    else:
        enabled = block.get("enabled")
    if enabled:
        unknown = sorted(set(enabled) - set(MODELS))
        if unknown:
            raise KeyError(f"unknown models in config: {unknown}. Available: {sorted(MODELS)}")
        return [name for name in MODELS if name in set(enabled)]
    return [name for name in MODELS if name != "neural_network"]


def model_params(config: Dict, model_name: str) -> Dict:
    """Parameter overrides for one model, tolerating both config shapes.

    A list-style ``models: [logistic_regression]`` block carries no parameters,
    so it yields an empty dict rather than failing.
    """
    block = config.get("models", config) or {}
    if not isinstance(block, dict):
        return {}
    return block.get(model_name) or {}


def fit_with_timing(model, X, y) -> Dict[str, object]:
    """Fit a model and record how long it took."""
    start = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - start
    return {"model": model, "train_seconds": round(elapsed, 4)}


def predict_with_timing(model, X) -> Dict[str, object]:
    """Score a model and record how long inference took."""
    start = time.perf_counter()
    proba = model.predict_proba(X)
    elapsed = time.perf_counter() - start
    return {"scores": proba[:, 1], "infer_seconds": round(elapsed, 4)}
