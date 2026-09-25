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
    model["adaptation"]["strategies"] = [
        {"strategy": strategy,
         "metrics": {"precision": 0.5, "recall": 0.5, "f1": 0.5,
                     "false_positive_rate": 0.05},
         "refits": 0, "threshold": 0.5, "training_seconds": 0.1,
         "recovery": {"f1_gain": 0.0}}
        for strategy in (strategies or [])
    ]
    return model


ALL_STRATEGIES = [
    "none", "threshold_recalibration", "recent_window_retrain",
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
    run = _write_run(tmp_path / "run", models=[_model(strategies=["none", "threshold_recalibration"])])
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


def test_index_marks_a_finished_but_superseded_run_as_superseded(tmp_path, monkeypatch):
    """A run can look complete and still be wrong; the index must say so."""
    from conftest_helpers import load_repo_tool

    index_runs = load_repo_tool("index_runs")

    experiments = tmp_path / "experiments"
    superseded = tmp_path / "superseded"
    run = _write_run(superseded / "20260101T000000Z_temporal_x_deadbee", models=[_model()])
    (run / "calibration_logistic_regression.json").write_text(
        json.dumps({"backtest": {"brier": 0.2, "ece": 0.1}, "forward": {"brier": 0.3, "ece": 0.1}}),
        encoding="utf-8")
    (run / "metrics.json").write_text(json.dumps({
        "models": [_model()], "drift_summary": {"ks": {"comparisons": 4, "alert_rate": 0.25}}}),
        encoding="utf-8")
    experiments.mkdir(parents=True)

    monkeypatch.setattr(index_runs, "EXPERIMENTS", experiments)
    monkeypatch.setattr(index_runs, "SUPERSEDED", superseded)

    rows = {r["experiment_id"]: r for r in index_runs.build()}
    row = rows["20260101T000000Z_temporal_x_deadbee"]
    assert row["status"] == "superseded", row
    assert row["drift_comparisons"] == "4"


def test_index_reports_a_run_with_no_metrics_as_incomplete(tmp_path, monkeypatch):
    from conftest_helpers import load_repo_tool

    index_runs = load_repo_tool("index_runs")

    experiments = tmp_path / "experiments"
    _write_run(experiments / "20260101T000000Z_temporal_x_cafebabe", models=[], metrics=False)
    (tmp_path / "superseded").mkdir()

    monkeypatch.setattr(index_runs, "EXPERIMENTS", experiments)
    monkeypatch.setattr(index_runs, "SUPERSEDED", tmp_path / "superseded")

    rows = {r["experiment_id"]: r for r in index_runs.build()}
    row = rows["20260101T000000Z_temporal_x_cafebabe"]
    assert row["status"] == "incomplete", row
    assert "did not finish" in row["notes"]


def test_git_commit_prefers_the_build_argument(monkeypatch):
    """A container has no .git, so the commit arrives as an argument."""
    from driftguard.experiments.tracking import git_commit

    monkeypatch.setenv("DRIFTGUARD_GIT_COMMIT", "abc123def456")
    assert git_commit() == "abc123def456"


def test_git_commit_falls_back_to_git_when_no_argument_is_set(monkeypatch):
    from driftguard.experiments.tracking import git_commit

    monkeypatch.delenv("DRIFTGUARD_GIT_COMMIT", raising=False)
    # On a real checkout this reads the actual HEAD, which is never empty.
    assert len(git_commit()) > 0


def test_experiment_directories_are_immutable(tmp_path):
    """A recorded result must not be overwritable by a later run."""
    from driftguard.experiments.tracking import ExperimentRun

    root = tmp_path / "experiments"
    first = ExperimentRun("20260101T000000Z_temporal_x_abc123", root=str(root))
    first.write_json("metrics.json", {"models": ["original"]})

    with pytest.raises(FileExistsError):
        second = ExperimentRun("20260101T000000Z_temporal_x_abc123", root=str(root))
        second.write_json("metrics.json", {"models": ["overwritten"]})

    # The original content is untouched.
    stored = json.loads((first.path / "metrics.json").read_text(encoding="utf-8"))
    assert stored["models"] == ["original"]


def test_git_commit_recovers_from_a_transient_git_failure(monkeypatch):
    """A benchmark can start at the same moment as a commit.

    git rev-parse fails while another process holds the index lock, and
    recording "unknown" would make the run unreproducible for a reason that has
    nothing to do with the run. The commit is retried across candidate roots.
    """
    from driftguard.experiments import tracking

    calls = {"n": 0}

    class Result:
        def __init__(self, text):
            self.stdout = text
            self.stderr = ""
            self.returncode = 0

    def flaky_run(cmd, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return Result("")          # index locked, empty output
        return Result("a" * 40)

    monkeypatch.delenv("DRIFTGUARD_GIT_COMMIT", raising=False)
    monkeypatch.setattr(tracking.subprocess, "run", flaky_run)
    root = str(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(tracking, "_package_root", lambda: root)
    assert tracking.git_commit(root) == "a" * 40
    assert calls["n"] >= 2


def test_index_treats_each_result_kind_by_its_own_artifact(tmp_path, monkeypatch):
    """A finished shift run is not an incomplete run.

    The index demanded metrics.json from every experiment directory. Shift and
    ablation runs write controlled_shift_results.csv and ablation_results.csv
    instead, so every completed one of them was listed as incomplete, which is
    the same class of error as a tool reporting 'no data' when the data exists.
    """
    from conftest_helpers import load_repo_tool

    index_runs = load_repo_tool("index_runs")

    experiments = tmp_path / "experiments"
    meta = json.dumps({"dataset": "unsw_nb15", "git_commit": "a" * 40,
                       "random_seed": 42, "temporal_split": {}})
    shift = experiments / "20260101T000000Z_shift_unsw_nb15_aaaaaa"
    shift.mkdir(parents=True)
    (shift / "metadata.json").write_text(meta, encoding="utf-8")
    (shift / "controlled_shift_results.csv").write_text("model,shift\nlogistic_regression,packet_size\n",
                                                        encoding="utf-8")
    ablation = experiments / "20260101T000001Z_ablation_unsw_nb15_bbbbbb"
    ablation.mkdir(parents=True)
    (ablation / "metadata.json").write_text(meta, encoding="utf-8")
    (ablation / "ablation_results.csv").write_text("ablation,variant\ntarget_fpr,fpr_0.01\n", encoding="utf-8")

    monkeypatch.setattr(index_runs, "EXPERIMENTS", experiments)
    monkeypatch.setattr(index_runs, "SUPERSEDED", tmp_path / "superseded")
    rows = index_runs.build()
    by_id = {r["experiment_id"]: r for r in rows}
    assert by_id[shift.name]["status"] == "complete"
    assert by_id[ablation.name]["status"] == "complete"
