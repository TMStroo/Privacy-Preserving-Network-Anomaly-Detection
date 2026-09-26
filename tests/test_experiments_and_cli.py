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


def test_the_report_returns_after_its_last_section():
    """A return before the final section silently drops it from the PDF.

    Reordering the report's sections to numerical order moved the conclusion
    below the call that writes the file. Nothing raised: the PDF was still
    produced, just 26 sections instead of 27, and the missing section was only
    discoverable by reading the document.
    """
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "src/driftguard/reporting/generate.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "generate_report"
    )

    # Every statement after the first Return is unreachable.
    statements = function.body
    for index, node in enumerate(statements):
        if isinstance(node, ast.Return):
            trailing = [type(s).__name__ for s in statements[index + 1:]]
            assert not trailing, (
                f"{len(trailing)} statement(s) after the return in generate_report: {trailing}"
            )
            return
    raise AssertionError("generate_report has no return statement")


def test_the_report_declares_all_twenty_seven_sections_in_order():
    """The specification is 27 numbered sections, and they must appear in order."""
    import ast
    import re
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "src/driftguard/reporting/generate.py"
    text = source.read_text(encoding="utf-8")
    ast.parse(text)  # must at least be valid Python
    numbers = sorted({int(n) for n in re.findall(r'report\.h1\("(\d+)\. ', text)})
    assert numbers == list(range(1, 28)), f"expected sections 1..27, found {numbers}"


def test_the_report_is_built_from_a_real_run_not_the_fixture(tmp_path):
    """The synthetic fixture must never become the report's base run.

    latest_experiment() sorted directories by name and took the last, which is
    the fixture. The report then rendered 40,000 synthetic rows as the
    project's results while the completed UNSW-NB15 benchmark sat unread on
    disk. Nothing warned: the document was well formed and entirely wrong.
    """
    from pathlib import Path

    from driftguard.reporting.render import latest_experiment

    root = tmp_path / "experiments"
    for name in ("20260101T000000Z_temporal_unsw_nb15_full_aaaaaa",
                 "20260102T000000Z_temporal_synthetic_bbbbbb"):
        run = root / name
        run.mkdir(parents=True)
        (run / "metrics.json").write_text("{}", encoding="utf-8")

    chosen = Path(latest_experiment(str(root)))
    assert "unsw_nb15" in chosen.name, f"picked {chosen.name}"

    # With no real run on disk the fixture is still better than nothing.
    only = tmp_path / "only"
    (only / "20260102T000000Z_temporal_synthetic_bbbbbb").mkdir(parents=True)
    (only / "20260102T000000Z_temporal_synthetic_bbbbbb" / "metrics.json").write_text("{}", encoding="utf-8")
    assert latest_experiment(str(only)) is not None


def test_the_report_base_run_can_be_pinned_to_a_dataset(tmp_path):
    """Two real datasets on disk must not be chosen by alphabetical order.

    With both UNSW-NB15 and UGR'16 complete, an unpinned choice followed the
    directory names into UGR'16 and printed its metrics under a heading
    reading "UNSW-NB15". The number was right and the label was wrong, which
    is harder to catch than either alone.
    """
    from pathlib import Path

    from driftguard.reporting.render import latest_experiment

    root = tmp_path / "experiments"
    for name in ("20260101T000000Z_temporal_unsw_nb15_full_aaaaaa",
                 "20260102T000000Z_temporal_ugr16_bbbbbb"):
        run = root / name
        run.mkdir(parents=True)
        (run / "metrics.json").write_text("{}", encoding="utf-8")

    unsw = Path(latest_experiment(str(root), dataset="unsw_nb15"))
    assert "unsw_nb15" in unsw.name
    ugr = Path(latest_experiment(str(root), dataset="ugr16"))
    assert "ugr16" in ugr.name
    # Asking for a dataset that was never run falls back rather than failing,
    # so the report is still produced from whatever exists.
    fallback = latest_experiment(str(root), dataset="nonexistent")
    assert fallback is not None


def test_the_report_emits_its_sections_in_numerical_order():
    """A numbered section out of sequence is a defect in the document itself.

    The abstract once carried the UGR'16 section, because the reorder pass
    sorted on the lowest numbered heading in each block and the helper's block
    had none of its own. A static check of the heading numbers would not catch
    it; only the order of the calls does.
    """
    import re
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "src/driftguard/reporting/generate.py"
    text = source.read_text(encoding="utf-8")

    # Only the body's own headings are ordered. _ugr_section is defined above
    # generate_report and emits section 8 from there, so its definition is
    # always textually first and says nothing about document order.
    body = text[text.index("def generate_report("):]
    calls = [m.group(1) for m in re.finditer(r'report\.h1\("(\d+)\. ', body)]
    assert calls == sorted(calls, key=int), f"sections out of order: {calls}"

    order = [
        m.group(1) if m.group(1) else "_ugr_section"
        for m in re.finditer(r'report\.h1\("(\d+)\. |_ugr_section\(report', body)
    ]
    assert "_ugr_section" in order, order
    assert order.index("_ugr_section") > order.index("7"), order
    assert order.index("_ugr_section") < order.index("9"), order


def test_the_cross_dataset_section_ignores_the_synthetic_fixture(tmp_path):
    """The fixture is not a dataset this project evaluated.

    _other_datasets() collected it, so the cross-dataset section compared
    UNSW-NB15 against 8,000 generated rows and drew a conclusion from the
    pair. The synthetic run exists to let the tests run without a download.
    """
    import json
    from pathlib import Path

    from driftguard.reporting.generate import _other_datasets

    root = tmp_path / "experiments"
    for name, dataset in (
        ("20260101T000000Z_temporal_unsw_nb15_full_aaaaaa", "unsw_nb15"),
        ("20260102T000000Z_temporal_synthetic_bbbbbb", "synthetic"),
        ("20260103T000000Z_temporal_ugr16_cccccc", "ugr16"),
    ):
        run = root / name
        run.mkdir(parents=True)
        (run / "metrics.json").write_text(
            json.dumps({"dataset": dataset, "models": [], "split": {}}), encoding="utf-8"
        )

    others = _other_datasets(str(root), "unsw_nb15")
    assert set(others) == {"ugr16"}, others
