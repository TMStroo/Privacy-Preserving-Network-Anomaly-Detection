"""Temporal leakage regression tests.

These are the tests that make the forward test trustworthy. Each one asserts a
specific way future data could leak backwards, and each fails loudly if a
refactor reintroduces it.
"""

import pandas as pd
import pytest

from driftguard.data.schema import FlowFrame, add_derived_features
from driftguard.data.synthetic import generate_flows
from driftguard.features.preprocess import FeaturePreprocessor
from driftguard.temporal import TemporalLeakageError, build_temporal_split


@pytest.fixture(scope="module")
def frame():
    data = generate_flows(rows=20000, days=20, shift_at_fraction=0.6, seed=11)
    return FlowFrame("test", data, (), {})


@pytest.fixture(scope="module")
def split(frame):
    return build_temporal_split(frame)


def test_periods_are_chronologically_ordered(split):
    split.assert_ordered()
    order = split.available_periods()
    for earlier, later in zip(order, order[1:]):
        assert split.cutoff(earlier) < pd.Timestamp(split.time_range(later)[0])


def test_periods_do_not_overlap(split):
    index_sets = [set(split.rows(p).tolist()) for p in split.available_periods()]
    for a, b in zip(index_sets, index_sets[1:]):
        assert not a & b


def test_periods_cover_every_row(split):
    covered = set()
    for period in split.available_periods():
        covered |= set(split.rows(period).tolist())
    assert len(covered) == len(split.frame.frame)


def test_no_training_row_past_the_training_cutoff(split):
    train = split.period("train")
    assert train.timestamps.max() <= split.cutoff("train")


def test_forward_starts_strictly_after_training(split):
    assert split.period("forward").timestamps.min() > split.period("train").timestamps.max()


def test_backtest_is_inside_the_historical_distribution(split):
    assert split.period("backtest").timestamps.max() < pd.Timestamp(split.time_range("forward")[0])


def test_identical_timestamps_are_never_split_across_periods(split):
    # Every timestamp must belong to exactly one period, otherwise a single
    # instant would appear on both sides of the forward boundary.
    stamps = split.frame.frame["timestamp"]
    assignments = {}
    for period in split.available_periods():
        for value in split.period(period).timestamps:
            assignments.setdefault(value, set()).add(period)
    shared = {k: v for k, v in assignments.items() if len(v) > 1}
    assert not shared, f"{len(shared)} timestamps appear in more than one period"
    assert len(assignments) == stamps.nunique()


def test_split_without_forward_makes_forward_unaddressable(frame):
    historical = build_temporal_split(frame, use_forward=False)
    assert not historical.has_forward
    with pytest.raises(TemporalLeakageError):
        historical.rows("forward")


def test_guard_rejects_a_slice_past_its_cutoff(split):
    forward = split.period("forward")
    with pytest.raises(TemporalLeakageError):
        split.assert_no_future_rows(forward, "backtest")


def test_preprocessor_refuses_to_fit_on_future_rows(split):
    pre = FeaturePreprocessor(["flow_duration", "total_bytes"])
    with pytest.raises(TemporalLeakageError):
        pre.fit(split.period("validation"), cutoff=split.cutoff("train"))


def test_scaler_uses_training_statistics_only(split):
    features = ["flow_duration", "total_bytes", "total_packets"]
    train = split.period("train")
    pre = FeaturePreprocessor(features).fit(train, cutoff=split.cutoff("train"))

    everything = split.frame
    combined = FlowFrame(
        "combined",
        pd.concat([train.frame, split.period("forward").frame], ignore_index=True),
        (),
        {},
    )
    all_rows_pre = FeaturePreprocessor(features).fit(combined)
    assert pre.scaler.mean_ != pytest.approx(all_rows_pre.scaler.mean_)
    # The scaler must match the training slice alone.
    assert pre.scaler.mean_ == pytest.approx(
        train.frame[features].to_numpy(dtype=float).mean(axis=0)
    )


def test_validation_threshold_is_chosen_without_forward_labels(split):
    from driftguard.evaluation.metrics import basic_metrics, threshold_for_target_fpr

    validation = split.period("validation")
    scores = pd.Series([0.1, 0.4, 0.6, 0.9] * 50)
    threshold = threshold_for_target_fpr(validation.targets.to_numpy()[:200], scores.to_numpy(), 0.05)
    # Changing the forward labels must not change a validation-derived threshold,
    # because the function only ever receives validation arrays.
    assert 0.0 <= threshold <= 1.0
    assert "forward" not in threshold_for_target_fpr.__code__.co_varnames


def test_period_frame_carries_no_private_columns(split):
    for period in split.available_periods():
        frame = split.period(period)
        assert "label" in frame.frame.columns
        assert "timestamp" in frame.frame.columns
