"""Tests for configuration loading and metric calculation."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from privacy_preserving_nad.evaluation.metrics import (
    compute_metrics,
    create_comparison_table,
    format_comparison_table,
    save_metrics_csv,
    save_metrics_json
)
from privacy_preserving_nad.models.baseline import get_model_config, create_logistic_regression, create_random_forest


class TestConfigLoading:
    """Test configuration loading."""

    def test_load_config(self, tmp_path):
        """Test loading config from YAML."""
        config = {
            "models": {
                "random_seed": 42,
                "logistic_regression": {"max_iter": 1000, "C": 1.0},
                "random_forest": {"n_estimators": 100, "max_depth": 10}
            },
            "dataset": {"raw_dir": "data/raw"}
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        loaded = get_model_config(str(config_file))
        assert loaded["random_seed"] == 42
        assert loaded["logistic_regression"]["max_iter"] == 1000
        assert loaded["random_forest"]["n_estimators"] == 100


class TestModelCreation:
    """Test model creation from config."""

    def test_create_logistic_regression(self):
        """Test logistic regression creation."""
        config = {
            "random_seed": 42,
            "logistic_regression": {
                "max_iter": 500,
                "C": 0.5,
                "solver": "liblinear",
                "class_weight": "balanced"
            }
        }
        model = create_logistic_regression(config)
        assert model.max_iter == 500
        assert model.C == 0.5
        assert model.solver == "liblinear"
        assert model.class_weight == "balanced"
        assert model.random_state == 42

    def test_create_random_forest(self):
        """Test random forest creation."""
        config = {
            "random_seed": 123,
            "random_forest": {
                "n_estimators": 50,
                "max_depth": 5,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "class_weight": "balanced",
                "n_jobs": 2
            }
        }
        model = create_random_forest(config)
        assert model.n_estimators == 50
        assert model.max_depth == 5
        assert model.min_samples_split == 10
        assert model.min_samples_leaf == 5
        assert model.class_weight == "balanced"
        assert model.n_jobs == 2
        assert model.random_state == 123


class TestMetricsComputation:
    """Test metric computation."""

    def test_compute_metrics_perfect(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9],
                            [0.7, 0.3], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8]])

        metrics = compute_metrics(y_true, y_pred, y_proba)

        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert metrics["false_positive_rate"] == 0.0
        assert metrics["false_negative_rate"] == 0.0
        assert metrics["roc_auc"] == 1.0
        # y_true has 4 positive samples (indices 2, 3, 5, 7)
        assert metrics["true_positives"] == 4
        assert metrics["true_negatives"] == 4
        assert metrics["false_positives"] == 0
        assert metrics["false_negatives"] == 0

    def test_compute_metrics_all_wrong(self):
        """Test metrics with all wrong predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])

        metrics = compute_metrics(y_true, y_pred)

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1_score"] == 0.0
        assert metrics["false_positive_rate"] == 1.0
        assert metrics["false_negative_rate"] == 1.0
        assert metrics["true_positives"] == 0
        assert metrics["true_negatives"] == 0
        assert metrics["false_positives"] == 2
        assert metrics["false_negatives"] == 2

    def test_compute_metrics_imbalanced(self):
        """Test metrics with imbalanced data."""
        # 90% normal, 10% anomaly
        y_true = np.array([0]*90 + [1]*10)
        # Model predicts all normal
        y_pred = np.array([0]*100)

        metrics = compute_metrics(y_true, y_pred)

        assert metrics["precision"] == 0.0  # No positive predictions
        assert metrics["recall"] == 0.0     # Missed all anomalies
        assert metrics["f1_score"] == 0.0
        assert metrics["false_positive_rate"] == 0.0  # No false positives
        assert metrics["false_negative_rate"] == 1.0  # All anomalies missed
        assert metrics["n_normal"] == 90
        assert metrics["n_anomaly"] == 10

    def test_compute_metrics_without_proba(self):
        """Test metrics without probability scores."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 1])

        metrics = compute_metrics(y_true, y_pred)

        assert "roc_auc" in metrics
        assert metrics["roc_auc"] is None
        assert metrics["pr_auc"] is None

    def test_compute_metrics_edge_cases(self):
        """Test metrics with edge cases."""
        # All same class in true labels
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])

        metrics = compute_metrics(y_true, y_pred)

        assert metrics["precision"] == 0.0  # No positive class
        assert metrics["recall"] == 0.0
        assert metrics["n_anomaly"] == 0
        assert metrics["n_normal"] == 4


class TestComparisonTable:
    """Test comparison table creation."""

    def test_create_comparison_table(self):
        """Test creating comparison table from results."""
        results = {
            "FULL_METADATA": {
                "logistic_regression": {
                    "precision": 0.85, "recall": 0.80, "f1_score": 0.82,
                    "false_positive_rate": 0.10, "false_negative_rate": 0.20,
                    "roc_auc": 0.90, "pr_auc": 0.85,
                    "n_samples": 1000, "n_normal": 800, "n_anomaly": 200
                },
                "random_forest": {
                    "precision": 0.90, "recall": 0.85, "f1_score": 0.87,
                    "false_positive_rate": 0.05, "false_negative_rate": 0.15,
                    "roc_auc": 0.95, "pr_auc": 0.90,
                    "n_samples": 1000, "n_normal": 800, "n_anomaly": 200
                }
            },
            "RESTRICTED_METADATA": {
                "logistic_regression": {
                    "precision": 0.75, "recall": 0.70, "f1_score": 0.72,
                    "false_positive_rate": 0.15, "false_negative_rate": 0.30,
                    "roc_auc": 0.80, "pr_auc": 0.75,
                    "n_samples": 1000, "n_normal": 800, "n_anomaly": 200
                }
            }
        }

        df = create_comparison_table(results)

        assert len(df) == 3  # 3 model/feature-set combinations
        assert "Feature Set" in df.columns
        assert "Model" in df.columns
        assert "F1 Score" in df.columns
        assert "False Positive Rate" in df.columns

        # Check sorting
        assert df.iloc[0]["Feature Set"] == "FULL_METADATA"
        assert df.iloc[0]["Model"] == "Logistic Regression"

    def test_format_comparison_table(self):
        """Test formatting comparison table as markdown."""
        results = {
            "FULL_METADATA": {
                "logistic_regression": {
                    "precision": 0.85, "recall": 0.80, "f1_score": 0.82,
                    "false_positive_rate": 0.10, "false_negative_rate": 0.20,
                    "n_samples": 1000, "n_normal": 800, "n_anomaly": 200
                }
            }
        }

        df = create_comparison_table(results)
        markdown = format_comparison_table(df)

        # Check for expected columns (format uses spaces for alignment)
        assert "Feature Set" in markdown
        assert "Model" in markdown
        assert "FULL_METADATA" in markdown
        assert "Logistic Regression" in markdown
        # Precision formatted as 0.8500 (4 decimal places)
        assert "0.8500" in markdown or "0.85" in markdown


class TestMetricsSaving:
    """Test metrics saving functions."""

    def test_save_metrics_csv(self, tmp_path):
        """Test saving metrics as CSV."""
        results = {
            "FULL_METADATA": {
                "logistic_regression": {
                    "precision": 0.85, "recall": 0.80, "f1_score": 0.82,
                    "false_positive_rate": 0.10, "false_negative_rate": 0.20,
                    "n_samples": 1000, "n_normal": 800, "n_anomaly": 200
                }
            }
        }

        output_path = tmp_path / "comparison.csv"
        save_metrics_csv(results, str(output_path))

        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert len(df) == 1
        assert df.iloc[0]["F1 Score"] == 0.82

    def test_save_metrics_json(self, tmp_path):
        """Test saving metrics as JSON."""
        results = {
            "FULL_METADATA": {
                "logistic_regression": {
                    "precision": 0.85, "recall": 0.80, "f1_score": 0.82
                }
            }
        }

        output_path = tmp_path / "metrics.json"
        save_metrics_json(results, str(output_path))

        assert output_path.exists()
        import json
        with open(output_path) as f:
            loaded = json.load(f)
        assert loaded["FULL_METADATA"]["logistic_regression"]["f1_score"] == 0.82


if __name__ == "__main__":
    pytest.main([__file__, "-v"])