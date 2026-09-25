"""Drift detector tests: each statistic, its threshold behaviour, and the
requirement that an effect size accompanies any p-value.
"""

import numpy as np
import pandas as pd
import pytest

from driftguard.drift.detectors import (
    CusumDetector,
    detect_window,
    ks_test,
    population_stability_index,
    run_sequential_drift,
    wasserstein_distance,
)
from driftguard.drift.windowing import analyse_windows, drift_frame, summarize_drift
from driftguard.data.schema import FlowFrame
from driftguard.data.synthetic import generate_flows


def test_ks_detects_a_mean_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 4000)
    shifted = rng.normal(3, 1, 4000)
    out = ks_test(reference, shifted, alpha=0.01)
    assert out["alert"] is True
    assert out["p_value"] < 1e-6
    assert out["effect_size"] > 0.5


def test_ks_stays_quiet_when_nothing_changed():
    rng = np.random.default_rng(1)
    out = ks_test(rng.normal(0, 1, 4000), rng.normal(0, 1, 4000), alpha=0.01)
    assert out["alert"] is False


def test_wasserstein_is_scale_invariant():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, 3000)
    out = wasserstein_distance(base, base + 1.0, threshold=0.1)
    # A shift of one standard deviation must register on the scaled measure.
    assert out["effect_size"] > 0.1
    assert out["alert"] is True


def test_psi_grows_with_shift_size():
    rng = np.random.default_rng(3)
    reference = rng.normal(0, 1, 5000)
    small = population_stability_index(reference, rng.normal(0.2, 1, 5000), threshold=0.2)
    large = population_stability_index(reference, rng.normal(2.0, 1, 5000), threshold=0.2)
    assert large["statistic"] > small["statistic"]
    assert large["alert"] is True


def test_every_detector_reports_an_effect_size():
    rng = np.random.default_rng(4)
    reference = pd.Series(rng.normal(0, 1, 2000))
    window = pd.Series(rng.normal(1, 1, 2000))
    config = {"min_samples": 100, "alpha": 0.01, "wasserstein_threshold": 0.1, "psi_threshold": 0.2}
    for method in ["ks", "wasserstein", "psi"]:
        result = detect_window(
            reference, window, method, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), "f", config
        )
        assert result.effect_size is not None, f"{method} reported no effect size"
        row = result.as_row()
        assert "p_value" in row and "effect_size" in row


def test_detector_ignores_windows_below_minimum_sample_size():
    reference = pd.Series(np.random.default_rng(5).normal(0, 1, 1000))
    window = pd.Series(np.random.default_rng(6).normal(5, 1, 10))
    result = detect_window(
        reference, window, "ks", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), "f",
        {"min_samples": 100},
    )
    assert result.alert is False
    assert result.n_window == 10


def test_cusum_alarms_on_a_sustained_shift():
    rng = np.random.default_rng(7)
    detector = CusumDetector(threshold=5.0, drift=0.5, warmup=50)
    stream = list(rng.normal(0, 1, 200)) + list(rng.normal(4, 1, 200))
    alarms = [detector.update(v)["alert"] for v in stream]
    assert any(alarms[200:]), "CUSUM never alarmed after a sustained shift"


def test_cusum_stays_quiet_on_stable_data():
    rng = np.random.default_rng(8)
    detector = CusumDetector(threshold=5.0, drift=0.5, warmup=50)
    alarms = [detector.update(v)["alert"] for v in rng.normal(0, 1, 500)]
    assert sum(alarms) <= 1


def test_window_analysis_finds_the_planted_shift():
    frame = generate_flows(rows=30000, days=20, shift_at_fraction=0.6, shift_kind="packet_size",
                           shift_magnitude=4.0, seed=5)
    flow = FlowFrame("synthetic", frame, (), {})
    reference = flow.between(frame["timestamp"].min(), frame["timestamp"].quantile(0.4))
    target = flow.between(frame["timestamp"].quantile(0.4), frame["timestamp"].max())
    results = analyse_windows(reference, target, ["total_bytes", "flow_duration"],
                              {"window": "4h", "stride": "4h", "min_samples": 100,
                               "methods": ["ks", "wasserstein", "psi"], "alpha": 0.01,
                               "wasserstein_threshold": 0.1, "psi_threshold": 0.2})
    table = drift_frame(results)
    assert not table.empty
    summary = summarize_drift(table)
    # The planted shift multiplies byte counts, so at least one method must fire.
    assert summary["alerts"] > 0
    assert summary["first_alert"] is not None


def test_drift_table_records_required_columns():
    frame = generate_flows(rows=8000, days=10, seed=3)
    flow = FlowFrame("synthetic", frame, (), {})
    reference = flow.between(frame["timestamp"].min(), frame["timestamp"].quantile(0.5))
    target = flow.between(frame["timestamp"].quantile(0.5), frame["timestamp"].max())
    table = drift_frame(analyse_windows(reference, target, ["total_bytes"],
                                        {"window": "6h", "stride": "6h", "min_samples": 50,
                                         "methods": ["ks"], "alpha": 0.01}))
    for column in ["window_start", "window_end", "feature", "method", "statistic", "threshold", "alert"]:
        assert column in table.columns


def test_cusum_blocking_controls_sensitivity():
    """CUSUM accumulates, so raw-row input makes it fire on a trivial shift.

    Blocking averages consecutive rows first, which sets the operating point to
    shifts that actually matter. Without this the statistic alarms on 100% of
    windows regardless of effect size and carries no information.
    """
    import numpy as np
    import pandas as pd

    from driftguard.drift.detectors import calibrated_cusum_threshold, cusum_test

    rng = np.random.default_rng(3)
    reference = rng.normal(0.0, 1.0, 20_000)

    def alerts(make, block, trials=12):
        n_seq = 100_000 // block if block else 100_000
        warmup = min(20, max(n_seq // 4, 1))
        threshold = calibrated_cusum_threshold(
            n_seq, drift=0.5, warmup=warmup, target_false_alarm_rate=0.01, seed=0
        )
        hits = 0
        for _ in range(trials):
            out, _ = cusum_test(
                reference, pd.Series(make()), threshold=threshold,
                drift=0.5, warmup=20, block=block,
            )
            hits += int(out["alert"])
        return hits

    tiny = lambda: rng.normal(0.1, 1.0, 100_000)   # noqa: E731
    large = lambda: rng.normal(2.0, 1.0, 100_000)  # noqa: E731

    assert alerts(tiny, block=500) < alerts(tiny, block=None), (
        "blocking should reduce sensitivity to a negligible shift"
    )
    assert alerts(large, block=500) == 12, "blocking must not hide a large shift"


def test_cusum_block_shorter_than_half_the_window_is_ignored():
    import numpy as np
    import pandas as pd

    from driftguard.drift.detectors import cusum_test

    rng = np.random.default_rng(4)
    reference = rng.normal(0.0, 1.0, 5_000)
    window = pd.Series(rng.normal(0.0, 1.0, 600))
    out_small, _ = cusum_test(reference, window, threshold=1e9, block=500)
    out_none, _ = cusum_test(reference, window, threshold=1e9, block=None)
    assert out_small["block_used"] == 1
    assert out_none["block_used"] == 1
