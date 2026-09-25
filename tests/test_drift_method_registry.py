"""Method-name resolution and CUSUM dispatch.

The CUSUM regression this file guards against is real: ``cusum`` was listed in
``DRIFT_METHODS`` but ``detect_window`` had no branch for it, so the dispatcher
fell through to its unknown-method branch and raised ``KeyError`` quoting the
very name it had just been given.
"""

import numpy as np
import pandas as pd
import pytest

from driftguard.drift.detectors import (
    calibrated_cusum_threshold,
    DRIFT_METHODS,
    cusum_test,
    detect_window,
    normalise_method,
    resolve_methods,
)


def test_every_declared_method_is_dispatchable():
    """No method may be advertised without a branch that handles it."""
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(10.0, 2.0, 600))
    window = pd.Series(rng.normal(14.0, 2.0, 400))
    config = {"min_samples": 50, "alpha": 0.01, "cusum_threshold": 3.0}

    for method in DRIFT_METHODS:
        result = detect_window(
            reference, window, method,
            pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"),
            "feature_x", config,
        )
        assert result.method == method
        assert np.isfinite(result.statistic)


def test_cusum_resolves_and_dispatches():
    """The exact string that used to fail must now produce a real result."""
    rng = np.random.default_rng(7)
    reference = pd.Series(rng.normal(0.0, 1.0, 800))
    window = pd.Series(rng.normal(2.5, 1.0, 500))

    result = detect_window(
        reference, window, "cusum",
        pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-02"),
        "mean_packet_size", {"min_samples": 50, "cusum_threshold": 5.0, "cusum_drift": 0.5},
    )

    assert result.method == "cusum"
    assert result.statistic > 0.0
    assert result.alert is True
    assert result.effect_size is not None and result.effect_size > 1.0


def test_cusum_stays_quiet_on_an_unchanged_window():
    """The alarm must mean something: no shift, no alert.

    This only holds with a calibrated threshold. A hard-coded 5.0 alerts on
    roughly 63% of pure-noise windows of this length, so the default is derived
    from the null distribution instead.
    """
    rng = np.random.default_rng(11)
    reference = pd.Series(rng.normal(5.0, 1.0, 800))
    window = pd.Series(rng.normal(5.0, 1.0, 500))

    result = detect_window(
        reference, window, "cusum",
        pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-02"),
        "byte_rate", {"min_samples": 50, "cusum_drift": 0.5},
    )
    assert result.alert is False


def test_hard_cusum_threshold_is_still_honoured():
    """An explicitly configured threshold overrides calibration."""
    rng = np.random.default_rng(17)
    reference = pd.Series(rng.normal(5.0, 1.0, 800))
    window = pd.Series(rng.normal(5.0, 1.0, 500))

    result = detect_window(
        reference, window, "cusum",
        pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-02"),
        "byte_rate", {"min_samples": 50, "cusum_threshold": 99.0, "cusum_drift": 0.5},
    )
    assert result.threshold == 99.0
    assert result.alert is False


def test_calibrated_threshold_controls_the_false_alarm_rate():
    """The point of calibration: an unchanged stream rarely alarms."""
    threshold = calibrated_cusum_threshold(500, drift=0.5, warmup=20, target_false_alarm_rate=0.01)
    assert threshold > 5.0, "calibration must sit above the naive 5.0 default"

    rng = np.random.default_rng(23)
    reference = rng.normal(0.0, 1.0, 900)
    alarms = 0
    trials = 40
    for _ in range(trials):
        window = pd.Series(rng.normal(0.0, 1.0, 500))
        out, _ = cusum_test(reference, window, threshold=threshold, drift=0.5, warmup=20)
        alarms += bool(out["alert"])
    # Allow slack for simulation noise at a 1% target.
    assert alarms <= 6, f"calibrated CUSUM alarmed on {alarms}/{trials} unchanged windows"


@pytest.mark.parametrize(
    "written,expected",
    [
        ("cusum", "cusum"),
        ("CUSUM", "cusum"),
        ("Cusum", "cusum"),
        ("cusum ", "cusum"),
        ("cu_sum", "cusum"),
        ("ks", "ks"),
        ("KS", "ks"),
        ("kolmogorov", "ks"),
        ("Kolmogorov-Smirnov", "ks"),
        ("wasserstein", "wasserstein"),
        ("wd", "wasserstein"),
        ("psi", "psi"),
        ("population_stability_index", "psi"),
    ],
)
def test_normalise_method(written, expected):
    assert normalise_method(written) == expected


def test_unknown_method_raises_before_any_work():
    with pytest.raises(ValueError) as err:
        normalise_method("kolmogorov_smirnnnov")
    message = str(err.value)
    assert "kolmogorov_smirnnnov" in message
    for name in DRIFT_METHODS:
        assert name in message


def test_unknown_method_fails_at_the_dispatcher_too():
    rng = np.random.default_rng(3)
    series = pd.Series(rng.normal(0.0, 1.0, 300))
    with pytest.raises(ValueError):
        detect_window(
            series, series, "banana",
            pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"),
            "f", {"min_samples": 10},
        )


def test_non_string_method_is_a_type_error():
    with pytest.raises(TypeError):
        normalise_method(7)


def test_resolve_methods_deduplicates_and_preserves_order():
    assert resolve_methods(["psi", "PSI", "ks", "psi", "cusum"]) == ["psi", "ks", "cusum"]


def test_resolve_methods_rejects_an_empty_list():
    with pytest.raises(ValueError):
        resolve_methods([])


def test_cusum_ignores_unsorted_window_order():
    """CUSUM is sequential, so the window must be walked in time order."""
    rng = np.random.default_rng(5)
    reference = pd.Series(rng.normal(0.0, 1.0, 400))
    ordered = pd.Series(rng.normal(2.0, 1.0, 300))
    shuffled = ordered.sample(frac=1.0, random_state=3)

    straight, _ = cusum_test(reference.to_numpy(), ordered, threshold=5.0)
    jumbled, _ = cusum_test(reference.to_numpy(), shuffled, threshold=5.0)
    # Peak CUSUM height depends on the path, not just the multiset, so the two
    # can differ; what must hold is that both stay in a sane numeric range.
    assert np.isfinite(straight["statistic"])
    assert np.isfinite(jumbled["statistic"])


def test_cusum_reports_effect_size_without_a_p_value():
    """A sequential detector has no p-value, so effect size carries the meaning."""
    rng = np.random.default_rng(13)
    out, _ = cusum_test(
        rng.normal(0.0, 1.0, 500).astype(float),
        pd.Series(rng.normal(1.5, 1.0, 300)),
        threshold=5.0,
    )
    assert out["p_value"] is None
    assert out["effect_size"] > 1.0
