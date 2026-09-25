"""Deterministic synthetic flow generation for the demo and the test suite.

The generator plants a known shift at a known point in time, which lets the
demo assert that drift detection fires *before* the model degrades without
relying on a downloaded dataset. It is generated data and carries no claim about
real networks.
"""

from typing import Dict

import numpy as np
import pandas as pd

from driftguard.data.schema import TIMESTAMP_COLUMN, add_derived_features


def generate_flows(
    rows: int = 40000,
    days: int = 20,
    shift_at_fraction: float = 0.6,
    shift_kind: str = "packet_size",
    shift_magnitude: float = 3.0,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate flows whose telemetry shifts partway through the time axis."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01 00:00:00")
    times = start + pd.to_timedelta(np.sort(rng.uniform(0, days * 86400, rows)), unit="s")

    after_shift = times >= (start + pd.Timedelta(seconds=days * 86400 * shift_at_fraction))
    factor = float(shift_magnitude) if shift_kind in {"packet_size", "duration", "byte_rate"} else 1.0

    benign = ~after_shift
    # Attack prevalence is highest after the shift, mirroring the way a new
    # campaign would show up on a network that had been quiet before.
    attack_p = np.where(after_shift, 0.30, 0.10)
    is_attack = rng.random(rows) < attack_p

    duration = np.exp(rng.normal(2.0, 0.9, rows))
    packets = np.exp(rng.normal(1.6, 0.7, rows))
    base_bytes = np.exp(rng.normal(4.0, 1.0, rows)) * np.where(is_attack, 3.0, 1.0)
    base_bytes = base_bytes * np.where(benign, 1.0, factor)

    proto = rng.choice(["TCP", "UDP", "ICMP"], size=rows, p=[0.6, 0.33, 0.07])
    flags = rng.choice(["....S.", ".AP.S.", ".A...."], size=rows, p=[0.3, 0.3, 0.4])

    forward_packets = np.maximum(1, (packets * rng.uniform(0.5, 0.9, rows)).round())
    backward_packets = np.maximum(0, (packets - forward_packets).round())
    forward_bytes = (base_bytes * rng.uniform(0.4, 0.8, rows)).round()

    frame = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: times,
            "label": is_attack.astype(int),
            "flow_duration": duration,
            "forward_packets": forward_packets,
            "backward_packets": backward_packets,
            "forward_bytes": forward_bytes,
            "backward_bytes": np.maximum(base_bytes - forward_bytes, 0.0),
            "protocol": proto,
            "tcp_flags": flags,
            "tos": rng.integers(0, 4, rows).astype(float),
        }
    )
    frame = add_derived_features(frame)
    return frame.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)


def shift_time(shift_at_fraction: float, days: int, start: str = "2024-01-01 00:00:00") -> pd.Timestamp:
    return pd.Timestamp(start) + pd.Timedelta(seconds=days * 86400 * shift_at_fraction)
