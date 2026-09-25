"""Regression tests for two defects that made the benchmark report clean
results while silently measuring nothing.

1. ``analyse_windows`` returned an empty list whenever the configured window was
   longer than the target period. UNSW-NB15's forward period spans under four
   hours against a 6h window, so the entire drift stage was a no-op and
   ``drift_summary`` was written as ``{}`` - indistinguishable from "no drift".
2. The retraining adaptation strategies called ``sklearn.base.clone`` on the
   incumbent model and immediately discarded the result. That raised on the
   majority baseline, which is not a BaseEstimator, and all three retrain
   strategies were recorded as errors rather than run.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from driftguard.adaptation.strategies import ADAPTATION_STRATEGIES, apply_adaptation
from driftguard.data.schema import FlowFrame
from driftguard.drift.windowing import analyse_windows
from driftguard.models.registry import build_model


def _frame(name, n, start, label_rate=0.1, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start=start, periods=n, freq="1min")
    labels = (rng.random(n) < label_rate).astype(int)
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "label": labels,
            "flow_duration": rng.gamma(2.0, 1.0, n),
            "total_packets": rng.poisson(8, n).astype(float),
            "total_bytes": rng.gamma(3.0, 20.0, n),
        }
    )
    return FlowFrame(name, frame, ["synthetic"], {})


def test_short_target_period_still_produces_drift_rows(caplog):
    """A 90-minute target must not silently yield zero windows for a 6h window."""
    reference = _frame("ref", 900, "2016-05-01 00:00:00")
    target = _frame("tgt", 90, "2016-05-02 00:00:00")

    with caplog.at_level(logging.WARNING):
        results = analyse_windows(
            reference,
            target,
            ["flow_duration", "total_packets"],
            {"window": "6h", "stride": "6h", "min_samples": 20, "methods": ["ks"]},
        )

    assert results, "a target shorter than the window must still be analysed"
    assert any(
        "shorter than" in r.getMessage() for r in caplog.records
    ), "shrinking the window must be reported, not silent"


def test_window_larger_than_data_is_not_a_false_negative():
    """The key property: no exception, no empty result, and drift is measurable."""
    reference = _frame("ref", 600, "2016-05-01 00:00:00")
    target = _frame("tgt", 120, "2016-05-02 00:00:00", label_rate=0.5)
    results = analyse_windows(
        reference, target, ["flow_duration"],
        {"window": "24h", "stride": "24h", "min_samples": 10, "methods": ["ks", "psi"]},
    )
    assert len(results) == 2, "both methods must report, so a reader can compare them"


def test_zero_window_is_rejected_loudly():
    reference = _frame("ref", 200, "2016-05-01 00:00:00")
    target = _frame("tgt", 200, "2016-05-02 00:00:00")
    with pytest.raises(ValueError, match="must be positive"):
        analyse_windows(reference, target, ["flow_duration"], {"window": "0h"})


class _Pre:
    def transform(self, frame):
        if isinstance(frame, FlowFrame):
            return frame.frame[["flow_duration", "total_packets"]].reset_index(drop=True)
        return frame[["flow_duration", "total_packets"]]


@pytest.mark.parametrize("strategy", [s for s in ADAPTATION_STRATEGIES if "retrain" in s])
def test_every_retrain_strategy_runs_for_every_model(strategy):
    """Each retrain strategy must produce a fitted model, not an error entry."""
    for model_name in ["majority", "logistic_regression", "random_forest"]:
        reference = _frame("ref", 400, "2016-05-01 00:00:00", seed=1)
        history = _frame("hist", 200, "2016-05-01 08:00:00", seed=2)
        forward = _frame("fwd", 200, "2016-05-01 12:00:00", seed=3)
        model = build_model(model_name, {}, 42)
        pre = _Pre()
        model.fit(pre.transform(reference).to_numpy(), reference.targets.to_numpy())

        result = apply_adaptation(
            strategy,
            model,
            pre,
            history,
            reference,
            forward,
            {
                "model": {"name": model_name, "params": {}},
                "target_fpr": 0.05,
                "window": "1h",
            },
            42,
        )
        assert result.refits == 1, f"{strategy}/{model_name} did not refit"
        assert result.rows_used > 0
        assert 0.0 <= result.threshold <= 1.0


def test_retrain_of_majority_baseline_is_possible():
    """The original failure: sklearn's clone() rejects a non-BaseEstimator."""
    reference = _frame("ref", 300, "2016-05-01 00:00:00", seed=4)
    history = _frame("hist", 200, "2016-05-01 08:00:00", seed=5)
    forward = _frame("fwd", 100, "2016-05-01 12:00:00", seed=6)
    pre = _Pre()
    model = build_model("majority", {}, 42)
    model.fit(pre.transform(reference).to_numpy(), reference.targets.to_numpy())

    result = apply_adaptation(
        "recent_window_retrain", model, pre, history, reference, forward,
        {"model": {"name": "majority", "params": {}}, "target_fpr": 0.05, "window": "1h"},
        42,
    )
    assert result.refits == 1


def test_pipeline_refuses_to_write_an_empty_drift_result():
    """A drift table with no rows must fail the run, not read as 'no drift'."""
    import driftguard.pipeline as pipeline

    source = open(pipeline.__file__, encoding="utf-8").read()
    assert "the run would otherwise report an empty result" in source, (
        "the empty-drift guard was removed; drift results would silently vanish"
    )


def test_errored_strategies_fail_the_run_instead_of_being_recorded():
    """The first benchmark wrote three errored strategies per model and the
    directory still looked like a completed experiment. A strategy that raises
    must now stop the run rather than be recorded as a result."""
    import driftguard.pipeline as pipeline

    source = open(pipeline.__file__, encoding="utf-8").read()
    assert "adaptation strategy" in source and "failed" in source, (
        "an errored adaptation strategy is no longer reported as a result"
    )
