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
