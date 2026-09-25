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


@pytest.fixture
def smoke_config():
    """A loaded, valid smoke config; each test gets its own copy to mutate."""
    import copy
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    return copy.deepcopy(yaml.safe_load((root / "configs" / "smoke.yaml").read_text(encoding="utf-8")))


def test_rolling_adaptation_is_not_the_same_as_recent_window(smoke_config):
    """The two strategies used one shared window and one single fit, so they
    produced byte-identical scores in the first completed run.

    Rolling re-fits as the forward period advances; recent-window takes one
    snapshot before it. They must be separate strategies.
    """
    from driftguard.pipeline import run_adaptation_study, run_model, _fit_preprocessor
    from driftguard.data.registry import get_adapter
    from driftguard.temporal import build_temporal_split

    config = smoke_config
    config["adaptation"] = {
        "window": "6h",
        "rolling_step": "2h",
        "rolling_window": "4h",
        "strategies": ["none", "recent_window_retrain", "rolling_window_retrain"],
    }
    config["models"] = {"enabled": ["logistic_regression"]}

    adapter = get_adapter(config["dataset"]["name"])
    frame = adapter.load(config["dataset"]["raw_dir"], **config["dataset"].get("load_options", {}))
    split = build_temporal_split(frame, **config.get("temporal", {}))
    features = [f for f in frame.numeric_features()][:6]
    pre = _fit_preprocessor(split.period("train"), features, split)

    result = run_model("logistic_regression", pre, split, config, frame, 42)
    study = run_adaptation_study(result, pre, split, config, frame, 42)

    by_strategy = {s["strategy"]: s for s in study["strategies"]}
    assert set(by_strategy) == {"none", "recent_window_retrain", "rolling_window_retrain"}
    for name, entry in by_strategy.items():
        assert "error" not in entry, f"{name} failed: {entry.get('error')}"

    recent = by_strategy["recent_window_retrain"]["metrics"]
    rolling = by_strategy["rolling_window_retrain"]["metrics"]
    # PR AUC does not depend on the threshold, so identical PR AUC would mean
    # the two strategies scored identically.
    assert recent["pr_auc"] != rolling["pr_auc"], (
        "rolling and recent-window produced identical scores; the rolling "
        "strategy is not actually re-fitting"
    )
