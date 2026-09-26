"""Adaptation tests: every strategy must stay inside the data available at
the moment it runs, and recovery must be measured honestly.
"""

import pandas as pd
import pytest

from driftguard.adaptation.strategies import (
    ADAPTATION_STRATEGIES,
    apply_adaptation,
    recovery_summary,
)
from driftguard.data.schema import FlowFrame
from driftguard.data.synthetic import generate_flows
from driftguard.features.preprocess import FeaturePreprocessor
from driftguard.models.registry import build_model
from driftguard.temporal import TemporalLeakageError, build_temporal_split


@pytest.fixture(scope="module")
def setup():
    frame_data = generate_flows(rows=20000, days=20, shift_at_fraction=0.6,
                                shift_kind="packet_size", shift_magnitude=4.0, seed=13)
    flow = FlowFrame("synthetic", frame_data, (), {})
    split = build_temporal_split(flow)
    features = ["flow_duration", "total_bytes", "total_packets", "byte_rate"]
    pre = FeaturePreprocessor(features).fit(split.period("train"), cutoff=split.cutoff("train"))
    model = build_model("logistic_regression", {"max_iter": 400}, 42)
    model.fit(pre.transform(split.period("train")).to_numpy(), split.period("train").targets.to_numpy())
    return flow, split, pre, model


def _config():
    return {
        "target_fpr": 0.05,
        "base_threshold": 0.5,
        "window": "12h",
        "model": {"name": "logistic_regression", "params": {"max_iter": 400}},
    }


def test_all_required_strategies_are_available():
    for strategy in ["none", "recent_window_retrain", "rolling_window_retrain",
                     "historical_plus_recent_retrain", "threshold_recalibration"]:
        assert strategy in ADAPTATION_STRATEGIES


def test_every_strategy_records_its_provenance(setup):
    flow, split, pre, model = setup
    cutoff = pd.Timestamp(split.time_range("forward")[0])
    history = flow.between(cutoff - pd.Timedelta("12h"), cutoff, include_end=False)
    forward = split.period("forward")

    for strategy in ADAPTATION_STRATEGIES:
        result = apply_adaptation(strategy, model, pre, history, split.period("train"), forward, _config(), 42)
        payload = result.as_dict()
        assert payload["strategy"] == strategy
        assert "rows_used" in payload and "training_seconds" in payload
        if strategy != "none":
            # Nothing a strategy consumed may come from the period it is scored on.
            assert payload["latest_row_used"] <= str(cutoff)


def test_no_adaptation_consumes_nothing(setup):
    flow, split, pre, model = setup
    cutoff = pd.Timestamp(split.time_range("forward")[0])
    history = flow.between(cutoff - pd.Timedelta("12h"), cutoff, include_end=False)
    result = apply_adaptation("none", model, pre, history, split.period("train"),
                              split.period("forward"), _config(), 42)
    assert result.rows_used == 0
    assert result.latest_row_used is None


def test_threshold_recalibration_keeps_parameters(setup):
    flow, split, pre, model = setup
    cutoff = pd.Timestamp(split.time_range("forward")[0])
    history = flow.between(cutoff - pd.Timedelta("12h"), cutoff, include_end=False)
    result = apply_adaptation("threshold_recalibration", model, pre, history, split.period("train"),
                              split.period("forward"), _config(), 42)
    assert result.refits == 0
    assert result.training_seconds == 0.0
    assert 0.0 <= result.threshold <= 1.0


def test_retraining_uses_more_rows_than_recalibration(setup):
    flow, split, pre, model = setup
    cutoff = pd.Timestamp(split.time_range("forward")[0])
    history = flow.between(cutoff - pd.Timedelta("12h"), cutoff, include_end=False)
    recent = apply_adaptation("recent_window_retrain", model, pre, history, split.period("train"),
                              split.period("forward"), _config(), 42)
    combined = apply_adaptation("historical_plus_recent_retrain", model, pre, history,
                                split.period("train"), split.period("forward"), _config(), 42)
    assert combined.rows_used > recent.rows_used
    assert combined.refits == recent.refits == 1


def test_adaptation_rejects_a_history_reaching_into_the_forward_period(setup):
    flow, split, pre, model = setup
    # Hand it forward rows: the guard must refuse rather than silently use them.
    leaky_history = flow.between(pd.Timestamp(split.time_range("forward")[0]) - pd.Timedelta("1h"),
                                 pd.Timestamp(split.time_range("forward")[0]) + pd.Timedelta("1h"),
                                 include_end=True)
    with pytest.raises(TemporalLeakageError):
        apply_adaptation("recent_window_retrain", model, pre, leaky_history,
                         split.period("train"), split.period("forward"), _config(), 42)


def test_unknown_strategy_is_rejected(setup):
    flow, split, pre, model = setup
    with pytest.raises(KeyError):
        apply_adaptation("teleport_weights", model, pre, split.period("train"),
                         split.period("train"), split.period("forward"), _config(), 42)


def test_recovery_percentage_is_relative_to_the_lost_performance():
    backtest = {"f1": 0.90, "recall": 0.95, "precision": 0.86, "false_positive_rate": 0.05}
    before = {"f1": 0.60, "recall": 0.70, "precision": 0.52, "false_positive_rate": 0.20}
    after = {"f1": 0.75, "recall": 0.82, "precision": 0.69, "false_positive_rate": 0.12}
    recovery = recovery_summary(before, after, backtest)
    # (0.75 - 0.60) / (0.90 - 0.60) = 50%
    assert recovery["f1_recovery_pct"] == pytest.approx(50.0)
    assert recovery["f1_gain"] == pytest.approx(0.15)


def test_recovery_is_negative_when_adaptation_hurts():
    backtest = {"f1": 0.90, "recall": 0.95, "precision": 0.86, "false_positive_rate": 0.05}
    before = {"f1": 0.60, "recall": 0.70, "precision": 0.52, "false_positive_rate": 0.20}
    after = {"f1": 0.50, "recall": 0.60, "precision": 0.42, "false_positive_rate": 0.30}
    recovery = recovery_summary(before, after, backtest)
    assert recovery["f1_recovery_pct"] < 0


def test_a_recorded_threshold_reproduces_its_own_metrics():
    """The stored operating point must be the one that was applied.

    as_dict() rounded the threshold to six decimals. On UGR'16's random forest
    the applied threshold was 0.4598784961389474 and 0.459878 was written, and
    the gap moved 8,701 of 1,763,251 forward predictions across the decision
    boundary, so the recorded metrics could not be recomputed from the recorded
    threshold. Nothing about the metrics was wrong; the record simply no longer
    described them.
    """
    from driftguard.adaptation.strategies import AdaptationResult

    exact = 0.4598784961389474
    outcome = AdaptationResult(
        strategy="none",
        threshold=exact,
        rows_used=0,
        latest_row_used=None,
        training_seconds=0.0,
        refits=0,
    )
    assert outcome.as_dict()["threshold"] == exact
    # A value that survives a JSON round trip unchanged is the actual test.
    import json

    assert json.loads(json.dumps(outcome.as_dict()))["threshold"] == exact
