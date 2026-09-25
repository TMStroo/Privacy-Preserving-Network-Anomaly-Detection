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
    """Scale a non-negative series by ``factor`` in log space.

    Working in log space keeps a value from flipping sign when the factor
    shrinks it, which matters because several of these columns contain zeros.
    The jitter is multiplicative-in-log too, so the perturbation scales with the
    value instead of swamping small flows.
    """
    if factor == 1.0:
        return series
    jitter = rng.normal(0.0, 0.02, len(series))
    scaled = np.expm1(np.log1p(series.clip(lower=0).to_numpy()) + np.log(factor) + jitter)
    return pd.Series(scaled, index=series.index, name=series.name)


def apply_shift(
    frame: FlowFrame,
    kind: str,
    magnitude: float,
    seed: int = 42,
) -> FlowFrame:
    """Return a copy of ``frame`` with one controlled shift applied.

    ``magnitude`` means something different per kind: a multiplicative factor
    for the scale shifts, a target attack fraction for the prevalence shift,
    and a skew factor for the protocol mixture.
    """
    if kind not in SHIFT_KINDS:
        raise ValueError(f"unknown shift {kind!r}; choose from {', '.join(SHIFT_KINDS)}")

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
        # Neither UNSW-NB15 nor the UGR'16 netflow export carries an
        # inter-arrival-time column, so there is nothing to shift directly. The
        # observable consequence of changed inter-arrival spacing is that the
        # same byte total arrives over a different time base: the rate features
        # move, the packet count does not. So this scales the rate features and
        # leaves total_packets and total_bytes alone, which is what separates it
        # from the byte_rate shift above.
        rate_factor = _log_scale(pd.Series(1.0, index=out.index), magnitude, rng).to_numpy()
        for column in ("packet_rate", "byte_rate"):
            if column in out.columns:
                out[column] = (pd.to_numeric(out[column], errors="coerce").fillna(0.0).to_numpy() * rate_factor)
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
        if not 0.0 < target_rate < 1.0:
            raise ValueError("class_prevalence magnitude must be a fraction in (0, 1)")

        # The attack class cannot be invented, so reaching a rate above the
        # current one means thinning the other class. The earlier version sized
        # the sample from whichever class was larger and then capped it, which
        # meant every request above the current prevalence returned the original
        # data unchanged: the shift silently did nothing.
        current = float(labels.mean())
        if target_rate <= current:
            # Fewer attacks wanted: drop attacks, keep all benign traffic.
            keep_benign = benign.size
            wanted = int(round(target_rate / (1.0 - target_rate) * keep_benign))
            wanted = min(max(wanted, 1), attacks.size)
        else:
            # More attacks wanted: keep all attacks, thin the benign class.
            wanted = attacks.size
            keep_benign = int(round(wanted * (1.0 - target_rate) / target_rate))
            keep_benign = max(1, min(keep_benign, benign.size))

        keep = np.concatenate([
            rng.choice(attacks, wanted, replace=False),
            rng.choice(benign, keep_benign, replace=False),
        ])
        keep = np.sort(keep)
        out = out.iloc[keep].reset_index(drop=True)
        params["resulting_attack_rate"] = float(out["label"].mean())
        params["rows_removed"] = int(len(labels) - len(out))
        params["requested_attack_rate"] = target_rate
        params["previous_attack_rate"] = current
    elif kind == "telemetry_reduction":
        # Drop the features a reduced collector would no longer export, rather
        # than corrupting the values of features it still exports.
        drop = [c for c in ["tcp_flags", "tos", "forward_packets", "backward_packets", "forward_bytes", "backward_bytes"] if c in out.columns]
        out = out.drop(columns=drop)
        params["dropped"] = drop

    return FlowFrame(frame.name, out, frame.source_files, {**frame.notes, "shift": params} )


def realized_shift(
    before: FlowFrame, after: FlowFrame, features: Optional[Sequence[str]] = None
) -> Dict[str, object]:
    """Measure what a shift actually did, rather than trusting its parameter.

    A requested magnitude is an instruction to the generator, not evidence. A
    multiplicative shift on a heavy-tailed feature can move the mean a long way
    while barely touching the median, and a prevalence shift is capped by
    whichever class runs out. Reporting only the requested value would let a
    shift that did nothing look like one that worked.

    Returns the per-feature standardised mean shift (Cohen's d, robust to scale
    because it divides by the pooled standard deviation), the median ratio where
    both medians are positive, and the realised class prevalence change.
    """
    out: Dict[str, object] = {"features": {}}
    shared = [f for f in (features or before.frame.columns) if f in before.frame.columns and f in after.frame.columns]
    numeric = before.frame[shared].select_dtypes(include="number").columns.tolist()

    for feature in numeric:
        a = before.frame[feature].to_numpy(dtype=float)
        b = after.frame[feature].to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size < 2 or b.size < 2:
            continue
        mean_a, mean_b = float(a.mean()), float(b.mean())
        sd = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)) or 0.0
        entry = {
            "mean_before": mean_a,
            "mean_after": mean_b,
            "cohens_d": (mean_b - mean_a) / sd if sd else 0.0,
            "median_before": float(np.median(a)),
            "median_after": float(np.median(b)),
        }
        if entry["median_before"] > 0 and entry["median_after"] > 0:
            entry["median_ratio"] = entry["median_after"] / entry["median_before"]
        out["features"][feature] = entry

    if "label" in before.frame.columns and "label" in after.frame.columns:
        rate_before = float(before.frame["label"].mean())
        rate_after = float(after.frame["label"].mean())
        out["attack_rate_before"] = rate_before
        out["attack_rate_after"] = rate_after
        out["attack_rate_ratio"] = (rate_after / rate_before) if rate_before else None

    out["rows_before"] = int(len(before.frame))
    out["rows_after"] = int(len(after.frame))
    dropped = sorted(set(before.frame.columns) - set(after.frame.columns))
    if dropped:
        out["columns_removed"] = dropped

    effects = [abs(v["cohens_d"]) for v in out["features"].values()]
    out["max_abs_cohens_d"] = float(max(effects)) if effects else 0.0
    out["mean_abs_cohens_d"] = float(np.mean(effects)) if effects else 0.0
    # A shift that moved nothing measurable has to be visible as such rather
    # than being reported under its requested label.
    out["verified"] = bool(out["max_abs_cohens_d"] >= 0.05 or dropped or rate_changed(out))
    return out


def rate_changed(summary: Dict[str, object], floor: float = 1e-6) -> bool:
    before = summary.get("attack_rate_before")
    after = summary.get("attack_rate_after")
    if before is None or after is None:
        return False
    return abs(float(after) - float(before)) > floor


def shift_catalog(magnitudes: Sequence[float], kinds: Sequence[str] = SHIFT_KINDS) -> List[Dict[str, object]]:
    """Every shift configuration to run, as plain data."""
    return [
        {"kind": kind, "magnitude": float(magnitude)}
        for kind in kinds
        for magnitude in magnitudes
    ]
