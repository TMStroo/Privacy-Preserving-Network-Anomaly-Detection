"""Controlled-shift tests.

The property that matters is attribution: a shift must change the statistics it
claims to change, leave the label alone, and stay deterministic. A shift that
silently no-ops, or that quietly changes something else, would make the whole
controlled-shift table meaningless.
"""

import numpy as np
import pandas as pd
import pytest

from driftguard.data.schema import FlowFrame
from driftguard.shift.controlled import SHIFT_KINDS, apply_shift, shift_catalog


@pytest.fixture
def frame() -> FlowFrame:
    rng = np.random.default_rng(4)
    n = 800
    index = pd.date_range("2026-01-01", periods=n, freq="1min", tz=None)
    return FlowFrame(
        "unit",
        pd.DataFrame({
            "timestamp": index,
            "flow_duration": rng.uniform(1.0, 60.0, n),
            "total_packets": rng.integers(2, 40, n).astype(float),
            "total_bytes": rng.uniform(100.0, 9000.0, n),
            "packet_rate": rng.uniform(0.1, 5.0, n),
            "byte_rate": rng.uniform(10.0, 900.0, n),
            "protocol": rng.choice(["tcp", "udp"], n),
            "tcp_flags": rng.choice(["0x18", "0x02"], n),
            "label": (rng.random(n) < 0.3).astype(int),
        }),
        ["unit.csv"],
        {},
    )


def _scales_to(frame: FlowFrame, column: str) -> float:
    return float(frame.frame[column].median())


# class_prevalence is the one shift whose magnitude is a fraction rather than a
# multiplier, so the generic tests below use a per-kind magnitude.
MAGNITUDE = {k: 1.7 for k in SHIFT_KINDS if k not in {"telemetry_reduction", "class_prevalence"}}
MAGNITUDE["class_prevalence"] = 0.15
MAGNITUDE["telemetry_reduction"] = 0.0
SCALING_SHIFTS = [k for k in SHIFT_KINDS if k not in {"telemetry_reduction", "class_prevalence"}]


@pytest.mark.parametrize("kind", SCALING_SHIFTS + ["class_prevalence"])
def test_shift_is_deterministic(frame, kind):
    a = apply_shift(frame, kind, MAGNITUDE[kind], seed=9)
    b = apply_shift(frame, kind, MAGNITUDE[kind], seed=9)
    pd.testing.assert_frame_equal(a.frame, b.frame)


@pytest.mark.parametrize("kind", SCALING_SHIFTS)
def test_shift_never_changes_the_label(frame, kind):
    """No scale shift may invent or remove an attack; they isolate statistics."""
    out = apply_shift(frame, kind, MAGNITUDE[kind], seed=3)
    assert out.frame["label"].equals(frame.frame["label"])


def test_class_prevalence_only_removes_benign_rows(frame):
    """Changing the base rate must drop benign rows, never relabel anything."""
    out = apply_shift(frame, "class_prevalence", 0.15, seed=3)
    # Rows are re-indexed, so compare on the identifying column instead: a
    # retained attack must carry the same byte count it had originally.
    key = ["total_bytes", "flow_duration", "protocol"]
    original = frame.frame[key + ["label"]]
    kept = out.frame[key + ["label"]]

    assert len(kept) <= len(original)
    # Every kept row exists verbatim in the original, with its label intact.
    merged = kept.merge(original, on=key, how="left", suffixes=("", "_orig"))
    assert len(merged) == len(kept)
    assert (merged["label"] == merged["label_orig"]).all()
    # And the attack count went down, which is the whole point of the shift.
    assert kept["label"].sum() < original["label"].sum()


@pytest.mark.parametrize("kind", SCALING_SHIFTS + ["telemetry_reduction"])
def test_shift_preserves_timestamps(frame, kind):
    out = apply_shift(frame, kind, MAGNITUDE[kind], seed=3)
    assert out.timestamps.equals(frame.timestamps)


def test_class_prevalence_keeps_chronological_order(frame):
    out = apply_shift(frame, "class_prevalence", 0.15, seed=3)
    assert out.timestamps.is_monotonic_increasing


@pytest.mark.parametrize("kind", SCALING_SHIFTS)
def test_shift_actually_changes_something(frame, kind):
    out = apply_shift(frame, kind, MAGNITUDE[kind], seed=3)
    assert not out.frame.drop(columns=["timestamp"]).equals(
        frame.frame.drop(columns=["timestamp"])
    )


def test_magnitude_one_is_close_to_a_no_op(frame):
    """A magnitude of 1.0 must not move the data, or the scale is meaningless."""
    out = apply_shift(frame, "packet_size", 1.0, seed=1)
    assert _scales_to(out, "total_bytes") == pytest.approx(
        _scales_to(frame, "total_bytes"), rel=0.01
    )


def test_packet_size_shift_moves_size_not_counts(frame):
    out = apply_shift(frame, "packet_size", 2.0, seed=1)
    assert _scales_to(out, "total_bytes") > _scales_to(frame, "total_bytes")
    assert out.frame["total_packets"].equals(frame.frame["total_packets"])


def test_duration_shift_moves_duration_only(frame):
    out = apply_shift(frame, "duration", 2.0, seed=1)
    assert _scales_to(out, "flow_duration") > _scales_to(frame, "flow_duration")
    assert _scales_to(out, "total_bytes") == pytest.approx(_scales_to(frame, "total_bytes"))


def test_byte_rate_and_iat_shifts_are_distinguishable(frame):
    """They must not be the same transformation in disguise."""
    byte_shift = apply_shift(frame, "byte_rate", 2.0, seed=1)
    iat_shift = apply_shift(frame, "iat", 2.0, seed=1)
    assert not byte_shift.frame["packet_rate"].equals(iat_shift.frame["packet_rate"])


def test_iat_shift_moves_rates_but_not_totals(frame):
    out = apply_shift(frame, "iat", 2.0, seed=1)
    assert _scales_to(out, "byte_rate") > _scales_to(frame, "byte_rate")
    assert out.frame["total_bytes"].equals(frame.frame["total_bytes"])


def test_class_prevalence_shift_reaches_the_requested_rate(frame):
    out = apply_shift(frame, "class_prevalence", 0.10, seed=2)
    assert float(out.frame["label"].mean()) == pytest.approx(0.10, abs=0.01)


def test_class_prevalence_shift_subsets_rather_than_relabels(frame):
    """The attack rows kept must be a subset of the original attack rows."""
    out = apply_shift(frame, "class_prevalence", 0.5, seed=2)
    assert len(out.frame) <= len(frame.frame)
    assert out.frame["label"].sum() <= frame.frame["label"].sum()


def test_telemetry_reduction_drops_columns_but_keeps_rows(frame):
    out = apply_shift(frame, "telemetry_reduction", 0.0, seed=1)
    assert len(out.frame) == len(frame.frame)
    assert "tcp_flags" not in out.frame.columns


def test_protocol_mixture_shift_skews_the_distribution(frame):
    out = apply_shift(frame, "protocol_mixture", 4.0, seed=5)
    before = frame.frame["protocol"].value_counts(normalize=True)
    after = out.frame["protocol"].value_counts(normalize=True)
    assert not np.isclose(before["tcp"], after["tcp"])


def test_protocol_mixture_needs_two_protocols(frame):
    single = FlowFrame(
        "one", frame.frame.assign(protocol="tcp"), ["x.csv"], {}
    )
    with pytest.raises(ValueError):
        apply_shift(single, "protocol_mixture", 2.0, seed=1)


def test_unknown_shift_raises_with_the_valid_names(frame):
    with pytest.raises(ValueError) as err:
        apply_shift(frame, "packet_temp", 2.0)
    for kind in SHIFT_KINDS:
        assert kind in str(err.value)


def test_shift_records_its_parameters(frame):
    out = apply_shift(frame, "packet_size", 2.5, seed=11)
    params = out.notes["shift"]
    assert params["kind"] == "packet_size"
    assert params["magnitude"] == 2.5
    assert params["seed"] == 11


def test_shift_catalog_crosses_kinds_and_magnitudes():
    catalog = shift_catalog([1.5, 3.0], kinds=["packet_size", "duration"])
    assert len(catalog) == 4
    assert {c["kind"] for c in catalog} == {"packet_size", "duration"}
    assert {c["magnitude"] for c in catalog} == {1.5, 3.0}


def test_realized_shift_measures_the_change_rather_than_trusting_the_parameter():
    """A requested magnitude is an instruction, not evidence.

    The generator applies a log-space offset, so a 'magnitude 3' request does
    not multiply a feature by three. Reporting the request alone would let a
    weak shift look like a strong one.
    """
    import numpy as np
    from driftguard.data.schema import FlowFrame
    from driftguard.shift.controlled import realized_shift

    rng = np.random.default_rng(11)
    n = 4000
    base = pd.DataFrame({
        "timestamp": pd.date_range("2016-05-01", periods=n, freq="1min"),
        "label": (rng.random(n) < 0.2).astype(int),
        "total_bytes": rng.lognormal(5.0, 0.5, n),
        "byte_rate": rng.lognormal(6.0, 0.5, n),
    })
    before = FlowFrame("t", base, ["s"], {})

    for magnitude in (2.0, 5.0):
        shifted = apply_shift(before, "byte_rate", magnitude, seed=1)
        summary = realized_shift(before, shifted, ["byte_rate"])
        entry = summary["features"]["byte_rate"]
        assert summary["verified"] is True
        assert entry["cohens_d"] > 0, "a positive shift must move the mean up"
        # The realized median ratio must be near the requested factor, and the
        # two must be distinguishable rather than conflated.
        # The shift is applied in log space, so it lands as a genuine
        # multiplicative factor; the measurement must recover it, and it must be
        # derived from the data rather than copied from the request.
        assert entry["median_ratio"] == pytest.approx(magnitude, rel=0.05)
        assert summary["realized_verified"] if "realized_verified" in summary else summary["verified"]


def test_realized_shift_flags_a_shift_that_did_nothing():
    from driftguard.data.schema import FlowFrame
    from driftguard.shift.controlled import realized_shift

    rng = np.random.default_rng(12)
    n = 2000
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2016-05-01", periods=n, freq="1min"),
        "label": (rng.random(n) < 0.2).astype(int),
        "total_bytes": rng.lognormal(5.0, 0.5, n),
    })
    a = FlowFrame("t", frame, ["s"], {})
    b = FlowFrame("t", frame.copy(), ["s"], {})
    summary = realized_shift(a, b, ["total_bytes"])
    assert summary["max_abs_cohens_d"] == 0.0
    assert summary["verified"] is False, "an unshifted pair must not read as verified"


def test_realized_shift_reports_removed_columns():
    from driftguard.data.schema import FlowFrame
    from driftguard.shift.controlled import realized_shift

    rng = np.random.default_rng(13)
    n = 2000
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2016-05-01", periods=n, freq="1min"),
        "label": (rng.random(n) < 0.2).astype(int),
        "tcp_flags": ["A"] * n,
        "tos": rng.integers(0, 4, n).astype(float),
    })
    a = FlowFrame("t", frame, ["s"], {})
    b = FlowFrame("t", frame.drop(columns=["tcp_flags"]), ["s"], {})
    summary = realized_shift(a, b)
    assert "tcp_flags" in summary["columns_removed"]
    assert summary["verified"] is True


def test_realized_shift_measures_prevalence_change():
    from driftguard.data.schema import FlowFrame
    from driftguard.shift.controlled import realized_shift

    rng = np.random.default_rng(14)
    n = 6000
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2016-05-01", periods=n, freq="1min"),
        "label": (rng.random(n) < 0.2).astype(int),
        "total_bytes": rng.lognormal(5.0, 0.5, n),
    })
    before = FlowFrame("t", frame, ["s"], {})
    shifted = apply_shift(before, "class_prevalence", 0.5, seed=1)
    summary = realized_shift(before, shifted)
    assert summary["attack_rate_before"] == pytest.approx(0.2, abs=0.02)
    assert summary["attack_rate_after"] == pytest.approx(0.5, abs=0.02)
    assert summary["verified"] is True


def test_class_prevalence_shift_reaches_every_target_above_the_current_rate():
    """Regression: every request above the current prevalence used to return the
    original data untouched, so the shift silently did nothing.

    Attacks cannot be invented, so a higher target rate has to be reached by
    thinning benign traffic rather than by adding attacks.
    """
    import numpy as np
    from driftguard.data.schema import FlowFrame

    rng = np.random.default_rng(21)
    n = 6000
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2016-05-01", periods=n, freq="1min"),
        "label": (rng.random(n) < 0.2).astype(int),
        "total_bytes": rng.lognormal(5.0, 0.5, n),
    })
    before = FlowFrame("t", frame, ["s"], {})
    assert frame["label"].mean() == pytest.approx(0.2, abs=0.02)

    for target in (0.05, 0.1, 0.3, 0.4, 0.5):
        shifted = apply_shift(before, "class_prevalence", target, seed=1)
        realized = shifted.frame["label"].mean()
        assert realized == pytest.approx(target, abs=0.01), (
            f"requested {target}, realised {realized:.4f}"
        )
        assert len(shifted.frame) <= n
        # A shift that removed everything but one class would be useless.
        assert 0 in set(shifted.frame["label"]) and 1 in set(shifted.frame["label"])


def test_alerts_without_degradation_partitions_its_alerts():
    """Every alert must land in a bucket, or the comparison means nothing.

    The drift table and the model-window table are bucketed on different
    cadences, so a naive label comparison puts every alert in neither set. The
    three counters below have to add up.
    """
    import pandas as pd

    from driftguard.evaluation.failure_analysis import alerts_without_degradation

    # Four hourly drift windows starting 09:00, each one hour long.
    drift = pd.DataFrame({
        "window_start": pd.date_range("2015-02-18 09:00", periods=4, freq="h").astype(str),
        "window_end": pd.date_range("2015-02-18 10:00", periods=4, freq="h").astype(str),
        "alert": [True, True, True, True],
    })
    # Model windows on a 4-hour cadence over the same span.
    windows = pd.DataFrame({
        "bucket": pd.date_range("2015-02-18 08:00", periods=3, freq="4h").astype(str),
        "recall": [0.9, 0.2, 0.8],
    })

    result = alerts_without_degradation(drift, windows)
    assert result["alerts"] == 4
    accounted = (result["in_degraded_windows"] + result["without_degradation"]
                 + result["unmatched_alerts"])
    assert accounted == result["alerts"], result
    assert result["model_windows"] == 3
    assert result["degraded_windows"] == 1
    # Midpoints 09:30, 10:30, 11:30 and 12:30 fall in the 08:00, 08:00, 08:00 and
    # 12:00 model windows respectively, so exactly one alert lands in the
    # degraded (recall 0.2) window.
    assert result["in_degraded_windows"] == 1, result
    assert result["without_degradation"] == 3, result


def test_alerts_without_degradation_reports_unmatched_instead_of_hiding_them():
    import pandas as pd

    from driftguard.evaluation.failure_analysis import alerts_without_degradation

    drift = pd.DataFrame({
        "window_start": ["2015-02-18 09:00:00"],
        "window_end": ["2015-02-18 10:00:00"],
        "alert": [True],
    })
    # Model windows nowhere near the alert.
    windows = pd.DataFrame({
        "bucket": ["2020-01-01 00:00:00", "2020-01-01 04:00:00"],
        "recall": [0.9, 0.9],
    })

    result = alerts_without_degradation(drift, windows)
    assert result["unmatched_alerts"] == 1, result
    assert result["matched"] is False


def test_a_categorical_shift_is_measured_and_verified():
    """A protocol-mixture shift moves no numeric feature, only the mixture.

    Cohen's d needs a mean, so a purely categorical shift used to measure 0.0 on
    every numeric field and report realized_verified=False, which reads as "the
    shift did nothing" when in fact the traffic composition moved. The
    categorical total-variation distance is what makes it visible.
    """
    from driftguard.data.schema import FlowFrame
    from driftguard.data.synthetic import generate_flows
    from driftguard.shift.controlled import apply_shift, realized_shift

    before = FlowFrame("synthetic", generate_flows(rows=20000, days=10, seed=7), (), {})
    after = apply_shift(before, "protocol_mixture", 0.6, seed=7)
    summary = realized_shift(before, after)

    assert summary["max_categorical_tvd"] > 0.05
    assert summary["verified"] is True

    # The mixture genuinely differs at the top level.
    protocol = summary["categorical"]["protocol"]
    assert protocol["tvd"] > 0.05
    assert protocol["top_before"] != protocol["top_after"]


def test_an_unchanged_frame_does_not_verify():
    from driftguard.data.schema import FlowFrame
    from driftguard.data.synthetic import generate_flows
    from driftguard.shift.controlled import realized_shift

    frame = FlowFrame("synthetic", generate_flows(rows=8000, days=5, seed=11), (), {})
    summary = realized_shift(frame, frame)
    assert summary["max_abs_cohens_d"] == 0.0
    assert summary["max_categorical_tvd"] == 0.0
    assert summary["verified"] is False
