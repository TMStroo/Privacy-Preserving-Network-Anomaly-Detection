"""Deterministic, isolated distribution shifts for controlled experiments.

These transformations change the *statistics* of the telemetry while leaving
attack semantics untouched: no new attack type is implied and no label is
changed. That is what makes them useful for attributing a performance drop to
one specific property of the data rather than to a general "the data changed".

Each transformation is a pure function of a magnitude and a random seed, so the
same configuration always produces the same shifted data.
"""

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from driftguard.data.schema import FlowFrame

SHIFT_KINDS = (
    "packet_size",
    "duration",
    "packet_rate",
    "byte_rate",
    "iat",
    "protocol_mixture",
    "class_prevalence",
    "telemetry_reduction",
)


def _log_scale(series: pd.Series, factor: float, rng: np.random.Generator) -> pd.Series:
    if factor == 1.0:
        return series
    return np.expm1(np.log1p(series.clip(lower=0)) * np.log(factor) + rng.normal(0.0, 0.02, len(series)))


def apply_shift(
    frame: FlowFrame,
    kind: str,
    magnitude: float,
    seed: int = 42,
) -> FlowFrame:
    """Return a copy of ``frame`` with one controlled shift applied.

    ``magnitude`` is an interpretation depends on the kind: a multiplicative
    factor for the scale shifts, a target attack fraction for the prevalence
    shift, and a keep-rate for telemetry reduction.
    """
    if kind not in SHIFT_KINDS:
        raise KeyError(f"Unknown shift '{kind}'. Available: {SHIFT_KINDS}")

    rng = np.random.default_rng(seed)
    out = frame.frame.copy()
    params: Dict[str, object] = {"kind": kind, "magnitude": magnitude, "seed": seed}

    if kind == "packet_size":
        out["total_bytes"] = _log_scale(out["total_bytes"], magnitude, rng)
        out["bytes_per_packet"] = out["total_bytes"] / out["total_packets"].replace(0, np.nan)
        out["bytes_per_packet"] = out["bytes_per_packet"].fillna(0.0)
        out["mean_packet_size"] = out["bytes_per_packet"]
        out["bytes_log1p"] = np.log1p(out["total_bytes"].clip(lower=0))
        out["byte_rate"] = out["total_bytes"] / out["flow_duration"].replace(0, np.nan)
        out["byte_rate"] = out["byte_rate"].fillna(0.0)
    elif kind == "duration":
        out["flow_duration"] = _log_scale(out["flow_duration"], magnitude, rng)
        out["duration_log1p"] = np.log1p(out["flow_duration"].clip(lower=0))
    elif kind == "packet_rate":
        out["packet_rate"] = _log_scale(out["packet_rate"], magnitude, rng)
        out["byte_rate"] = _log_scale(out["byte_rate"], magnitude, rng)
    elif kind == "byte_rate":
        out["byte_rate"] = _log_scale(out["byte_rate"], magnitude, rng)
    elif kind == "iat":
        # Inter-arrival behaviour is expressed through rate and jitter proxies;
        # UGR'16 and UNSW-NB15 expose no IAT column, so the shift acts on the
        # rate features that the collector would derive IAT from.
        out["packet_rate"] = _log_scale(out["packet_rate"], magnitude, rng)
    elif kind == "protocol_mixture":
        if "protocol" not in out.columns:
            raise KeyError("protocol_mixture shift requires a protocol column")
        values = out["protocol"].astype(str).to_numpy()
        uniq = np.unique(values)
        if uniq.size < 2:
            raise ValueError("protocol_mixture shift needs at least two protocols")
        weights = np.full(uniq.size, 1.0 / uniq.size)
        weights[0] = weights[0] * magnitude
        weights = weights / weights.sum()
        out["protocol"] = rng.choice(uniq, size=len(out), p=weights)
        params["weights"] = {str(p): float(w) for p, w in zip(uniq, weights)}
    elif kind == "class_prevalence":
        target_rate = float(magnitude)
        labels = out["label"].to_numpy()
        attacks = np.flatnonzero(labels == 1)
        benign = np.flatnonzero(labels == 0)
        if attacks.size == 0 or benign.size == 0:
            raise ValueError("class_prevalence shift needs both classes present")
        total = len(labels)
        wanted_attacks = int(round(target_rate * total))
        wanted_attacks = min(max(wanted_attacks, 1), attacks.size)
        wanted_benign = min(benign.size, total - wanted_attacks)
        keep = np.concatenate([rng.choice(attacks, wanted_attacks, replace=False),
                               rng.choice(benign, wanted_benign, replace=False)])
        keep = np.sort(keep)
        out = out.iloc[keep].reset_index(drop=True)
        params["resulting_attack_rate"] = float(out["label"].mean())
    elif kind == "telemetry_reduction":
        # Drop the features a reduced collector would no longer export, rather
        # than corrupting the values of features it still exports.
        drop = [c for c in ["tcp_flags", "tos", "forward_packets", "backward_packets", "forward_bytes", "backward_bytes"] if c in out.columns]
        out = out.drop(columns=drop)
        params["dropped"] = drop

    return FlowFrame(frame.name, out, frame.source_files, {**frame.notes, "shift": params} )


def shift_catalog(magnitudes: Sequence[float], kinds: Sequence[str] = SHIFT_KINDS) -> List[Dict[str, object]]:
    """Every shift configuration to run, as plain data."""
    return [
        {"kind": kind, "magnitude": float(magnitude)}
        for kind in kinds
        for magnitude in magnitudes
    ]
