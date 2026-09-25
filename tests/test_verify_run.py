"""Tests for the run verifier.

The verifier is what CI and a reviewer use to decide whether an experiment
directory counts as a result. If it accepts a run that measured nothing, or
rejects a good one, every downstream claim inherits the mistake, so the failure
cases are pinned down here.
"""

import json
from pathlib import Path

import pytest

from tools.verify_run import verify

ROOT = Path(__file__).resolve().parents[1]


def _write_run(directory: Path, *, models, metadata=None, drift=True, calibration=True,
               figures=True, metrics=True):
    directory.mkdir(parents=True, exist_ok=True)
    if metrics:
        (directory / "metrics.json").write_text(
            json.dumps({"models": models, "drift_summary": {"ks": {"alert_rate": 0.1}} if drift else {}}),
            encoding="utf-8",
        )
    if metadata is not False:
        base = {
            "git_commit": "abc123",
            "random_seed": 42,
            "feature_schema_hash": "deadbeef",
            "package_versions": {"python": "3.11.0", "numpy": "2.0", "pandas": "2.0", "scikit-learn": "1.4"},
            "temporal_split": {
                "train_period": ["2015-01-22", "2015-01-22"],
                "validation_period": ["2015-01-23", "2015-01-23"],
                "backtest_period": ["2015-01-23", "2015-01-23"],
                "forward_period": ["2015-02-18", "2015-02-18"],
                "train_count": {"rows": 100},
            },
            "dataset": "synthetic",
        }
        base.update(metadata or {})
        (directory / "metadata.json").write_text(json.dumps(base), encoding="utf-8")
    (directory / "drift_events.csv").write_text("feature,method,alert\nx,ks,True\n", encoding="utf-8")
    if figures:
        figures_dir = directory / "figures"
        figures_dir.mkdir(exist_ok=True)
        (figures_dir / "degradation.png").write_bytes(b"x" * 2000)
    return directory


def _model(name="logistic_regression", strategies=None):
    block = {
        "precision": 0.5, "recall": 0.5, "f1": 0.5, "roc_auc": 0.6, "pr_auc": 0.4,
        "false_positive_rate": 0.05, "balanced_accuracy": 0.5,
    }
    model = {
        "result": {"model": name, "backtest": dict(block), "forward": dict(block),
                   "f1_degradation": 0.0},
        "adaptation": {},
    }
    if strategies:
        for strategy in strategies:
            model["adaptation"][strategy] = {"forward": {"f1": 0.5, "recall": 0.5}}
    return model


ALL_STRATEGIES = [
    "no_adaptation", "threshold_recalibration", "recent_window_retrain",
    "rolling_window_retrain", "historical_plus_recent_retrain",
]


ALL_MODELS = ["majority", "logistic_regression", "random_forest", "gradient_boosting"]


def test_a_complete_run_verifies(tmp_path):
    models = [_model(name, ALL_STRATEGIES) for name in ALL_MODELS]
    run = _write_run(tmp_path / "run", models=models, calibration=True)
    for name in ALL_MODELS:
        (run / f"calibration_{name}.json").write_text(
            json.dumps({"backtest": {"brier": 0.2, "ece": 0.1},
                        "forward": {"brier": 0.3, "ece": 0.15}}),
            encoding="utf-8",
        )
    assert verify(str(run), strict_models=True) == []


def test_a_run_without_metrics_is_not_a_result(tmp_path):
    run = _write_run(tmp_path / "run", models=[], metrics=False)
    problems = verify(str(run))
    assert problems
    assert "never finished" in problems[0]


def test_an_empty_drift_summary_is_rejected(tmp_path):
    """The silent no-op that produced this project's worst bug."""
    run = _write_run(tmp_path / "run", models=[_model()], drift=False)
    (run / "calibration_logistic_regression.json").write_text(
        json.dumps({"backtest": {"brier": 0.2, "ece": 0.1}, "forward": {"brier": 0.3, "ece": 0.1}}),
        encoding="utf-8",
    )
    problems = verify(str(run))
    assert any("drift_summary is empty" in p for p in problems)


def test_a_nan_metric_is_rejected(tmp_path):
    model = _model()
    model["result"]["forward"]["f1"] = float("nan")
    run = _write_run(tmp_path / "run", models=[model])
    (run / "calibration_logistic_regression.json").write_text(
        json.dumps({"backtest": {"brier": 0.2, "ece": 0.1}, "forward": {"brier": 0.3, "ece": 0.1}}),
        encoding="utf-8",
    )
    problems = verify(str(run))
    assert any("nan" in p.lower() for p in problems)


def test_missing_package_versions_are_rejected(tmp_path):
    """A version recorded as 'not installed' is worse than a missing key."""
    run = _write_run(tmp_path / "run", models=[_model()])
    (run / "calibration_logistic_regression.json").write_text(
        json.dumps({"backtest": {"brier": 0.2, "ece": 0.1}, "forward": {"brier": 0.3, "ece": 0.1}}),
        encoding="utf-8",
    )
    meta = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    meta["package_versions"]["scikit-learn"] = "not installed"
    (run / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    problems = verify(str(run))
    assert any("scikit-learn" in p for p in problems)


def test_a_superseded_run_is_rejected(tmp_path):
    run = _write_run(tmp_path / "run", models=[_model()])
    (run / "calibration_logistic_regression.json").write_text(
        json.dumps({"backtest": {"brier": 0.2, "ece": 0.1}, "forward": {"brier": 0.3, "ece": 0.1}}),
        encoding="utf-8",
    )
    meta = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    meta["superseded"] = {"reason": "pre-fix run"}
    (run / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    problems = verify(str(run))
    assert any("superseded" in p for p in problems)


def test_a_missing_adaptation_strategy_is_rejected_in_strict_mode(tmp_path):
    run = _write_run(tmp_path / "run", models=[_model(strategies=["no_adaptation", "threshold_recalibration"])])
    (run / "calibration_logistic_regression.json").write_text(
        json.dumps({"backtest": {"brier": 0.2, "ece": 0.1}, "forward": {"brier": 0.3, "ece": 0.1}}),
        encoding="utf-8",
    )
    problems = verify(str(run), strict_models=True)
    assert any("adaptation strategies missing" in p for p in problems)


def test_an_empty_figures_directory_is_flagged(tmp_path):
    run = _write_run(tmp_path / "run", models=[_model()], figures=False)
    (run / "figures").mkdir(exist_ok=True)
    (run / "calibration_logistic_regression.json").write_text(
        json.dumps({"backtest": {"brier": 0.2, "ece": 0.1}, "forward": {"brier": 0.3, "ece": 0.1}}),
        encoding="utf-8",
    )
    problems = verify(str(run))
    assert any("no png" in p for p in problems)


def test_missing_temporal_boundaries_are_rejected(tmp_path):
    run = _write_run(tmp_path / "run", models=[_model()])
    (run / "calibration_logistic_regression.json").write_text(
        json.dumps({"backtest": {"brier": 0.2, "ece": 0.1}, "forward": {"brier": 0.3, "ece": 0.1}}),
        encoding="utf-8",
    )
    meta = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    del meta["temporal_split"]["forward_period"]
    (run / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    problems = verify(str(run))
    assert any("forward_period" in p for p in problems)
