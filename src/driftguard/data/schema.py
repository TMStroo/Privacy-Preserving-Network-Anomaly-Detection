"""The common flow representation shared by every dataset adapter.

Only fields that a netflow/IPFIX export genuinely provides are represented, and
a dataset that does not expose a field leaves it out rather than having a value
invented for it. That asymmetry matters: UNSW-NB15 is a bidirectional capture
with per-direction counters, while UGR'16 is a unidirectional netflow v9 export
with no forward/backward split. The two can therefore only be compared on the
fields they share, and the schema marks which those are.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "label"

# Fields every supported dataset supplies. This intersection is what makes
# cross-dataset and within-dataset drift statistics comparable.
CORE_FIELDS = ["flow_duration", "total_packets", "total_bytes", "protocol"]

# Fields only some datasets expose. Used for within-dataset experiments on that
# dataset; excluded from any cross-dataset comparison.
OPTIONAL_FIELDS = [
    "forward_packets",
    "backward_packets",
    "forward_bytes",
    "backward_bytes",
    "tcp_flags",
    "tos",
]

# tcp_flags is a netflow bit-string, not a measurement, so it is encoded rather
# than scaled.
CATEGORICAL_FIELDS = ["protocol", "tcp_flags"]

DERIVED_FIELDS = [
    "packet_rate",
    "byte_rate",
    "mean_packet_size",
    "bytes_per_packet",
    "bytes_ratio",
    "packets_ratio",
    "duration_log1p",
    "packets_log1p",
    "bytes_log1p",
]

COMMON_SCHEMA = CORE_FIELDS + OPTIONAL_FIELDS + DERIVED_FIELDS
CROSS_DATASET_SCHEMA = CORE_FIELDS + DERIVED_FIELDS

FEATURE_GROUPS: Dict[str, Sequence[str]] = {
    "timing": ["flow_duration", "duration_log1p"],
    "volume": ["total_bytes", "bytes_log1p"],
    "count": ["total_packets", "packets_log1p"],
    "rate": ["packet_rate", "byte_rate", "bytes_per_packet"],
    "size": ["mean_packet_size", "bytes_per_packet"],
    "direction": ["forward_bytes", "backward_bytes", "bytes_ratio", "packets_ratio"],
    "tcp": ["tcp_flags", "forward_packets", "backward_packets"],
    "categorical": ["protocol"],
}

_EPS = 1e-9


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """Coerce a column to a non-negative float series, absent columns as zero."""
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).clip(lower=0.0)


@dataclass
class FlowFrame:
    """A dataset mapped onto the common schema, with its time axis intact."""

    name: str
    frame: pd.DataFrame
    source_files: Sequence[str] = field(default_factory=tuple)
    notes: Dict[str, str] = field(default_factory=dict)

    @property
    def timestamps(self) -> pd.Series:
        return self.frame[TIMESTAMP_COLUMN]

    @property
    def targets(self) -> pd.Series:
        return self.frame[TARGET_COLUMN]

    def has_direction(self) -> bool:
        return "forward_bytes" in self.frame.columns and "backward_bytes" in self.frame.columns

    def numeric_features(self) -> List[str]:
        skip = set(CATEGORICAL_FIELDS)
        return [c for c in COMMON_SCHEMA if c in self.frame.columns and c not in skip]

    def categorical_features(self) -> List[str]:
        return [c for c in CATEGORICAL_FIELDS if c in self.frame.columns]

    def feature_names(self) -> List[str]:
        return self.numeric_features() + self.categorical_features()

    def cross_dataset_features(self) -> List[str]:
        skip = set(CATEGORICAL_FIELDS)
        return [c for c in CROSS_DATASET_SCHEMA if c in self.frame.columns and c not in skip]

    def sorted_by_time(self) -> "FlowFrame":
        return FlowFrame(
            self.name,
            self.frame.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True),
            self.source_files,
            self.notes,
        )

    def subset(self, mask) -> "FlowFrame":
        return FlowFrame(
            self.name,
            self.frame.loc[mask].reset_index(drop=True),
            self.source_files,
            self.notes,
        )

    def between(self, start, end, include_end: bool = False) -> "FlowFrame":
        ts = self.timestamps
        if include_end:
            mask = (ts >= start) & (ts <= end)
        else:
            mask = (ts >= start) & (ts < end)
        return self.subset(mask)

    def describe_time(self) -> Dict[str, object]:
        ts = self.timestamps
        counts = self.targets.value_counts().to_dict()
        return {
            "dataset": self.name,
            "rows": int(len(self.frame)),
            "features": len(self.feature_names()),
            "cross_dataset_features": len(self.cross_dataset_features()),
            "has_direction": self.has_direction(),
            "time_min": None if ts.empty else str(ts.min()),
            "time_max": None if ts.empty else str(ts.max()),
            "label_counts": {str(k): int(v) for k, v in counts.items()},
            "attack_rate": float(self.targets.mean()) if len(self.frame) else 0.0,
            "source_files": list(self.source_files),
        }


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute the derived fields from observed flow statistics.

    Rates divide by flow duration; zero-length flows fall back to zero rather
    than producing an infinite rate. Direction fields stay absent when the
    source did not provide them.
    """
    out = frame.copy()
    dur = _numeric(out, "flow_duration")

    has_direction = "forward_bytes" in out.columns and "backward_bytes" in out.columns
    if has_direction:
        f_pkts = _numeric(out, "forward_packets")
        b_pkts = _numeric(out, "backward_packets")
        total_pkts = f_pkts + b_pkts
        f_bytes = _numeric(out, "forward_bytes")
        b_bytes = _numeric(out, "backward_bytes")
        total_bytes = f_bytes + b_bytes
    else:
        total_pkts = _numeric(out, "total_packets")
        total_bytes = _numeric(out, "total_bytes")
    safe_dur = dur.where(dur > 0)
    safe_pkts = total_pkts.where(total_pkts > 0)

    out["total_packets"] = total_pkts
    out["total_bytes"] = total_bytes
    out["packet_rate"] = (total_pkts / safe_dur).fillna(0.0)
    out["byte_rate"] = (total_bytes / safe_dur).fillna(0.0)
    out["mean_packet_size"] = (total_bytes / safe_pkts).fillna(0.0)
    out["bytes_per_packet"] = (total_bytes / safe_pkts).fillna(0.0)
    out["duration_log1p"] = np.log1p(dur)
    out["packets_log1p"] = np.log1p(total_pkts)
    out["bytes_log1p"] = np.log1p(total_bytes)

    if has_direction:
        out["bytes_ratio"] = (f_bytes / (total_bytes + _EPS)).clip(0.0, 1.0)
        out["packets_ratio"] = (f_pkts / (total_pkts + _EPS)).clip(0.0, 1.0)

    return out


def load_common_frame(name: str, raw_dir: str, **kwargs) -> "FlowFrame":
    """Load a dataset by name and return it in the common flow representation."""
    from driftguard.data.registry import get_adapter

    return get_adapter(name).load(raw_dir, **kwargs)
