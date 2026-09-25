"""Figures must be drawable from a recorded experiment.

Every figure reads a CSV or JSON the run itself wrote, so a figure cannot show
a number the experiment did not produce. The test that matters here is that a
run produces figures at all: they used to be built only by the report command,
which left every experiment directory with an empty figures/ folder.
"""

import json
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
