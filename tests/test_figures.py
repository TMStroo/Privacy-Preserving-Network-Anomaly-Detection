"""Figures must be drawable from a recorded experiment.

Every figure reads a CSV or JSON the run itself wrote, so a figure cannot show
a number the experiment did not produce. The test that matters here is that a
run produces figures at all: they used to be built only by the report command,
which left every experiment directory with an empty figures/ folder.
"""

import json
import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

from driftguard.reporting.figures import (
    adaptation_bars,
    backtest_vs_forward,
    build_all_figures,
    calibration_curves,
    class_timeline,
    degradation_by_model,
    drift_timeline,
    forward_windows,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """A real, small, end-to-end run whose output the tests then read."""
    from driftguard.pipeline import run_benchmark

    out = tmp_path_factory.mktemp("figures")
    config = yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
    result = run_benchmark(config, experiment_root=str(out))
    return result, Path(result["run"].path)


def test_a_run_writes_figures(smoke_run):
    _, path = smoke_run
    assert (path / "figures").is_dir(), "the run left an empty figures directory"
    assert list((path / "figures").glob("*.png")), "no figure was rendered"
    assert (path / "figures.json").is_file()


def test_figures_are_non_trivial(smoke_run):
    _, path = smoke_run
    for png in (path / "figures").glob("*.png"):
        assert png.stat().st_size > 5_000, f"{png.name} is suspiciously small"


def test_drift_timeline_reads_the_recorded_events(smoke_run):
    _, path = smoke_run
    assert (path / "drift_events.csv").is_file()
    out = drift_timeline(str(path / "drift_events.csv"), str(path / "f.png"))
    assert out and Path(out).is_file()


def test_forward_windows_reads_the_recorded_table(smoke_run):
    result, path = smoke_run
    model = result["metrics"]["models"][0]["result"]["model"]
    table = path / f"forward_windows_{model}.csv"
    assert table.is_file()
    out = forward_windows(str(table), str(path / "w.png"))
    assert out and Path(out).is_file()


def test_calibration_curve_reads_the_recorded_json(smoke_run):
    result, path = smoke_run
    model = result["metrics"]["models"][0]["result"]["model"]
    out = calibration_curves(str(path), str(path / "c.png"), model)
    assert out and Path(out).is_file()


def test_build_all_figures_reports_what_it_drew(smoke_run):
    result, path = smoke_run
    figures = build_all_figures(str(path), result["metrics"], str(path / "all"))
    assert figures, "build_all_figures returned nothing"
    for key, value in figures.items():
        assert Path(value).is_file(), f"{key} points at a missing file"


def test_figures_handles_an_empty_metric_set(tmp_path):
    """A figure function must return None rather than raise on absent data."""
    assert backtest_vs_forward({"models": []}, str(tmp_path / "x.png")) is None
    assert degradation_by_model({"models": []}, str(tmp_path / "y.png")) is None
    assert drift_timeline(str(tmp_path / "missing.csv"), str(tmp_path / "z.png")) is None


def test_extra_figures_render_from_a_recorded_run(tmp_path):
    """The five added figures must be drawable, not just importable."""
    import json

    from driftguard.reporting.figures_extra import build_extra_figures

    run = tmp_path / "run"
    run.mkdir()
    metrics = {
        "split": {
            "train_period": {"start": "2015-01-22 09:00:00", "end": "2015-01-22 12:00:00"},
            "validation_period": {"start": "2015-01-22 12:00:00", "end": "2015-01-22 14:00:00"},
            "backtest_period": {"start": "2015-01-22 14:00:00", "end": "2015-01-22 16:00:00"},
            "forward_period": {"start": "2015-02-18 09:00:00", "end": "2015-02-18 13:00:00"},
            "train_count": {"rows": 1000}, "validation_count": {"rows": 200},
            "backtest_count": {"rows": 200}, "forward_count": {"rows": 300},
        },
        "models": [
            {"result": {"model": "logistic_regression",
                        "backtest": {"recall": 0.7, "false_positive_rate": 0.05},
                        "forward": {"recall": 0.5, "false_positive_rate": 0.09}}},
            {"result": {"model": "random_forest",
                        "backtest": {"recall": 0.8, "false_positive_rate": 0.03},
                        "forward": {"recall": 0.6, "false_positive_rate": 0.07}}},
        ],
    }
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run / "drift_events.csv").write_text(
        "feature,method,alert,effect_size\nx,ks,True,0.4\nx,psi,False,0.1\n"
        "y,cusum,True,0.3\ny,wasserstein,False,0.2\n", encoding="utf-8")
    (run / "failure_analysis_logistic_regression.json").write_text(json.dumps({
        "false_positives": {"count": 40, "mean_score": 0.61},
        "false_negatives": {"count": 25, "mean_score": 0.38}}), encoding="utf-8")

    shift_csv = tmp_path / "shift.csv"
    shift_csv.write_text(
        "kind,magnitude,realized_verified,f1_degradation\n"
        "packet_size,2.0,True,-0.03\npacket_size,3.0,True,0.04\n"
        "byte_rate,1.5,False,0.001\n", encoding="utf-8")

    out = run / "figures"
    figures = build_extra_figures(str(run), metrics, str(out), str(shift_csv))

    for key in ("split_timeline", "recall_vs_fpr", "drift_detectors",
                "failure_analysis", "shift_degradation"):
        assert key in figures, f"{key} was not produced"
        path = figures[key]
        assert path.endswith(".png")
        assert os.path.exists(path), f"{key} produced no file"
        assert os.path.getsize(path) > 2000, f"{key} is suspiciously small"


def test_extra_figures_degrade_gracefully_when_data_is_absent(tmp_path):
    """A missing optional table must cost one figure, not the whole report."""
    from driftguard.reporting.figures_extra import build_extra_figures

    run = tmp_path / "run"
    run.mkdir()
    figures = build_extra_figures(str(run), {"models": []}, str(run / "figures"), None)
    assert figures == {}


def path_exists(p):
    return os.path.exists(p) and os.path.getsize(p) > 0


def size_of(p):
    return os.path.getsize(p)
