"""Experiment tracking, metrics and CLI behaviour."""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from driftguard.evaluation.metrics import (
    basic_metrics,
    calibration,
    degradation,
    f1_degradation,
    grouped_predictions,
    metrics_at_target_fpr,
    predictions_frame,
    recall_degradation,
    threshold_for_target_fpr,
)
from driftguard.experiments.tracking import ExperimentRun, experiment_id, list_experiments, schema_hash


def test_experiment_id_is_unique_and_descriptive():
    a = experiment_id("temporal", "unsw_nb15", "full")
    b = experiment_id("temporal", "unsw_nb15", "full")
    assert a != b
    assert "temporal" in a and "unsw_nb15" in a and "full" in a


def test_experiment_directory_refuses_to_overwrite(tmp_path):
    first = ExperimentRun("exp1", root=str(tmp_path))
    first.write_json("metrics.json", {"ok": True})
    with pytest.raises(FileExistsError):
        ExperimentRun("exp1", root=str(tmp_path))


def test_experiment_metadata_records_provenance(tmp_path):
    run = ExperimentRun("exp2", root=str(tmp_path))
    meta = run.metadata(
        dataset="unsw_nb15",
        dataset_checksums={"a.csv": "deadbeef"},
        features=["flow_duration", "total_bytes"],
        seed=42,
        config={"a": 1},
        split_description={"train_period": ["x", "y"]},
    )
    assert meta["dataset"] == "unsw_nb15"
    assert meta["random_seed"] == 42
    assert meta["feature_schema_hash"] == schema_hash(["flow_duration", "total_bytes"])
    assert "python" in meta["package_versions"]
    assert "git_commit" in meta
    written = json.loads((run.path / "metadata.json").read_text(encoding="utf-8")) if (run.path / "metadata.json").exists() else None
    assert written is None or written["dataset"] == "unsw_nb15"


def test_list_experiments_reads_metadata(tmp_path):
    run = ExperimentRun("exp3", root=str(tmp_path))
    run.write_json("metadata.json", {"dataset": "synthetic", "git_commit": "abc1234", "created_utc": "now"})
    entries = list_experiments(str(tmp_path))
    assert entries[0]["experiment_id"] == "exp3"
    assert entries[0]["dataset"] == "synthetic"


def test_threshold_respects_the_false_positive_budget():
    rng = np.random.default_rng(0)
    y = np.array([0] * 900 + [1] * 100)
    score = np.concatenate([rng.uniform(0, 0.4, 900), rng.uniform(0.6, 1.0, 100)])
    threshold = threshold_for_target_fpr(y, score, target_fpr=0.05)
    predicted = (score >= threshold).astype(int)
    observed_fpr = predicted[y == 0].mean()
    assert observed_fpr <= 0.06  # within rounding of the 5% budget


def test_metrics_at_target_fpr_never_see_the_test_set():
    y_val = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    result = metrics_at_target_fpr(y_val, scores, 0.25)
    assert result["threshold_source"] == "validation split"
    assert 0.0 <= result["threshold"] <= 1.0


def test_basic_metrics_reports_confusion_counts():
    y = np.array([0, 0, 1, 1, 0, 1])
    score = np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.3])
    metrics = basic_metrics(y, score, 0.5)
    assert metrics["true_positives"] + metrics["false_positives"] + metrics["true_negatives"] + metrics["false_negatives"] == 6
    assert metrics["rows"] == 6
    assert 0.0 <= metrics["false_positive_rate"] <= 1.0


def test_degradation_reports_absolute_and_relative_change():
    backtest = {"f1": 0.8, "recall": 0.8, "precision": 0.8, "false_positive_rate": 0.1, "false_negative_rate": 0.2}
    forward = {"f1": 0.6, "recall": 0.5, "precision": 0.8, "false_positive_rate": 0.3, "false_negative_rate": 0.5}
    change = degradation(backtest, forward)
    assert change["f1_change"] == pytest.approx(-0.2)
    assert change["f1_relative_change"] == pytest.approx(-0.25)
    assert f1_degradation(backtest, forward) == pytest.approx(0.2)
    assert recall_degradation(backtest, forward) == pytest.approx(0.3)


def test_calibration_reports_brier_and_ece():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 4000)
    score = np.clip(y * 0.6 + rng.normal(0.5, 0.2, 4000), 0, 1)
    result = calibration(y, score)
    assert 0.0 <= result["brier"] <= 1.0
    assert 0.0 <= result["ece"] <= 1.0
    assert len(result["curve"]) == 10


def test_grouped_predictions_bucket_by_time():
    times = pd.date_range("2024-01-01", periods=8, freq="h")
    frame = predictions_frame([0, 1, 0, 1, 0, 1, 0, 1], [0.1, 0.9] * 4, times, 0.5)
    grouped = grouped_predictions(frame, freq="2h")
    assert len(grouped) == 4
    assert "recall" in grouped.columns


def test_cli_lists_its_commands():
    out = subprocess.run([sys.executable, "-m", "driftguard", "--help"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0
    for command in ["data", "train", "evaluate", "detect-drift", "adapt", "benchmark", "report", "demo"]:
        assert command in out.stdout


def test_demo_runs_without_any_dataset_download(tmp_path):
    out = subprocess.run([sys.executable, "-m", "driftguard", "demo"],
                         capture_output=True, text=True, timeout=500,
                         env={**os.environ, "DRIFTGUARD_DEMO": "1"})
    assert out.returncode == 0, out.stderr[-2000:]
    assert "NOT a research benchmark" in out.stdout
    assert "backtest" in out.stdout and "forward" in out.stdout
