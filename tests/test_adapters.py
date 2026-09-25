"""Dataset adapter, schema conversion and validation tests."""

import pandas as pd
import pytest

from driftguard.data.registry import ADAPTERS, get_adapter
from driftguard.data.schema import (
    CATEGORICAL_FIELDS,
    COMMON_SCHEMA,
    CROSS_DATASET_SCHEMA,
    FlowFrame,
    add_derived_features,
)
from driftguard.data.synthetic import generate_flows



def test_registry_exposes_every_dataset():
    for name in ["unsw_nb15", "ugr16", "synthetic"]:
        assert name in ADAPTERS


def test_unknown_dataset_is_rejected():
    with pytest.raises(KeyError):
        get_adapter("not_a_dataset")


def test_derived_features_are_consistent():
    frame = generate_flows(rows=2000, days=5, seed=2)
    out = add_derived_features(frame)
    assert (out["total_packets"] >= 0).all()
    assert (out["total_bytes"] >= 0).all()
    # mean packet size is bytes over packets wherever packets exist
    mask = out["total_packets"] > 0
    expected = (out.loc[mask, "total_bytes"] / out.loc[mask, "total_packets"]).to_numpy()
    assert out.loc[mask, "mean_packet_size"].to_numpy() == pytest.approx(expected, rel=1e-9)


def test_rates_are_finite_for_zero_duration_flows():
    frame = generate_flows(rows=500, days=2, seed=4)
    frame["flow_duration"] = 0.0
    out = add_derived_features(frame)
    for column in ["packet_rate", "byte_rate"]:
        assert out[column].notna().all()
        assert (out[column] < float("inf")).all()


def test_direction_fields_absent_without_direction_data():
    # UGR'16 is a unidirectional export; the schema must not fabricate the split.
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "label": [0, 1],
        "flow_duration": [1.0, 2.0],
        "total_packets": [5, 6],
        "total_bytes": [100, 200],
        "protocol": ["TCP", "UDP"],
    })
    out = add_derived_features(frame)
    assert "forward_bytes" not in out.columns
    assert "bytes_ratio" not in out.columns
    assert "packet_rate" in out.columns


def test_feature_names_exclude_the_target():
    flow = FlowFrame("synthetic", generate_flows(rows=1000, days=3, seed=9), (), {})
    names = flow.feature_names()
    assert "label" not in names
    assert "timestamp" not in names


def test_cross_dataset_features_are_a_subset():
    flow = FlowFrame("synthetic", generate_flows(rows=1000, days=3, seed=9), (), {})
    cross = flow.cross_dataset_features()
    assert set(cross) <= set(flow.feature_names())
    assert set(cross) <= set(CROSS_DATASET_SCHEMA)


def test_tcp_flags_is_treated_as_categorical():
    flow = FlowFrame("synthetic", generate_flows(rows=1000, days=3, seed=9), (), {})
    assert "tcp_flags" in CATEGORICAL_FIELDS
    assert "tcp_flags" in flow.categorical_features()
    assert "tcp_flags" not in flow.numeric_features()


def test_describe_time_reports_the_observed_span():
    flow = FlowFrame("synthetic", generate_flows(rows=3000, days=6, seed=12), (), {})
    described = flow.describe_time()
    assert described["rows"] == 3000
    assert described["time_min"] < described["time_max"]
    assert 0.0 <= described["attack_rate"] <= 1.0


def test_ugr16_adapter_reports_missing_files(tmp_path):
    adapter = get_adapter("ugr16")
    missing = adapter.missing_files(str(tmp_path))
    assert missing
    with pytest.raises(FileNotFoundError):
        adapter.load(str(tmp_path))


def test_unsw_adapter_reports_missing_file(tmp_path):
    adapter = get_adapter("unsw_nb15")
    assert adapter.missing_files(str(tmp_path)) == ["UNSW_NB15_full_raw.parquet"]


def test_synthetic_generation_is_deterministic():
    a = generate_flows(rows=1000, days=4, seed=77)
    b = generate_flows(rows=1000, days=4, seed=77)
    assert a.equals(b)
    c = generate_flows(rows=1000, days=4, seed=78)
    assert not a.equals(c)
