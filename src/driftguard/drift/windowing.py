"""Sliding-window drift analysis over a reference period and later windows.

Window size, stride and minimum sample count come from configuration. The
default sizes are justified by the datasets used here: UGR'16 resolves flows to
one-second buckets over months of traffic, so a day-long window holds far more
than the minimum sample count while still resolving week-to-week movement, and
UNSW-NB15's capture sessions are days long, where an hour keeps a window inside
a single session.
"""

import logging
from typing import Dict, List, Sequence

import pandas as pd

from driftguard.data.schema import FlowFrame
from driftguard.drift.detectors import (
    DRIFT_METHODS,
    DriftResult,
    detect_window,
    resolve_methods,
    run_sequential_drift,
)
from driftguard.utils import as_timedelta

LOG = logging.getLogger(__name__)

DEFAULT_DRIFT_CONFIG = {
    "reference_period": "train",
    "window": "1h",
    "stride": "1h",
    "min_samples": 200,
    "features": None,
    "methods": ["ks", "wasserstein", "psi"],
    "sequential_method": "cusum",
    "alpha": 0.01,
    "wasserstein_threshold": 0.10,
    "psi_threshold": 0.20,
    "psi_bins": 10,
    "cusum_threshold": 5.0,
    "cusum_drift": 0.5,
}


def resolve_features(frame: FlowFrame, config: Dict) -> List[str]:
    requested = config.get("features")
    if not requested:
        return frame.numeric_features()
    missing = [f for f in requested if f not in frame.frame.columns]
    if missing:
        raise KeyError(f"drift features not present in {frame.name}: {missing}")
    return list(requested)


def analyse_windows(
    reference: FlowFrame,
    target: FlowFrame,
    features: Sequence[str],
    config: Dict,
) -> List[DriftResult]:
    """Compare each window of ``target`` against the whole reference period."""
    window = as_timedelta(config.get("window", DEFAULT_DRIFT_CONFIG["window"]))
    stride = as_timedelta(config.get("stride", config.get("window", DEFAULT_DRIFT_CONFIG["window"])))
    methods = resolve_methods(config.get("methods", DEFAULT_DRIFT_CONFIG["methods"]))
    min_samples = int(config.get("min_samples", DEFAULT_DRIFT_CONFIG["min_samples"]))

    target_ordered = target.sorted_by_time()
    if target_ordered.frame.empty:
        return []

    span = target_ordered.timestamps.max() - target_ordered.timestamps.min()
    if window <= pd.Timedelta(0):
        raise ValueError(f"drift window must be positive, got {window}")

    # A capture can be shorter than the configured window. Rather than silently
    # returning nothing - which reads downstream as "no drift found" - shrink
    # the window to the data and warn. Drift is the whole point of the stage, so
    # a silent no-op here would invalidate the experiment while looking clean.
    if span < window:
        usable = max(int(span.total_seconds()), 1)
        LOG.warning(
            "target period spans %s, shorter than the %s drift window; "
            "falling back to %d s so the forward period is still analysed",
            span, window, usable,
        )
        window = pd.Timedelta(seconds=usable)
        stride = min(stride, window)

    edges = pd.date_range(
        start=target_ordered.timestamps.min(),
        end=target_ordered.timestamps.max() + stride,
        freq=stride,
    )
    if len(edges) < 2:
        # A single window with no stride to advance it is still a valid
        # comparison against the reference period.
        edges = pd.DatetimeIndex([target_ordered.timestamps.min()])

    # A feature the target does not carry cannot drift: a reduced collector
    # simply does not report it. Skipping is the honest answer, and it keeps a
    # telemetry-reduction experiment from failing on its own setup.
    comparable = [f for f in features if f in target_ordered.frame.columns and f in reference.frame.columns]
    if not comparable:
        return []

    results: List[DriftResult] = []
    for start in edges[:-1]:
        end = start + window
        sub = target_ordered.frame[(target_ordered.timestamps >= start) & (target_ordered.timestamps < end)]
        if len(sub) < min_samples:
            continue
        for feature in comparable:
            for method in methods:
                results.append(
                    detect_window(
                        reference.frame[feature],
                        sub[feature],
                        method,
                        start,
                        end,
                        feature,
                        config,
                    )
                )
    return results


def run_cusum_stream(
    reference: FlowFrame,
    target: FlowFrame,
    feature: str,
    config: Dict,
) -> List[DriftResult]:
    """Run the sequential detector over the target period's rows in time order."""
    ordered = target.sorted_by_time()
    if ordered.frame.empty or feature not in ordered.frame.columns:
        return []
    return run_sequential_drift(
        reference.frame[feature],
        ordered.frame[feature],
        method="cusum",
        timestamps=ordered.timestamps,
        feature=feature,
        config=config,
    )


def drift_frame(results: List[DriftResult]) -> pd.DataFrame:
    """Flatten drift results into a table."""
    if not results:
        return pd.DataFrame(
            columns=["window_start", "window_end", "feature", "method", "statistic", "threshold", "alert", "p_value", "effect_size"]
        )
    return pd.DataFrame([r.as_row() for r in results])


def summarize_drift(table: pd.DataFrame) -> Dict[str, object]:
    """Per-method and per-feature alert counts, plus first-alarm timing."""
    if table.empty:
        return {"windows": 0, "alerts": 0, "by_method": {}, "by_feature": {}}
    by_method = table.groupby("method")["alert"].agg(["sum", "count"]).to_dict("index")
    by_feature = table.groupby("feature")["alert"].agg(["sum", "count"]).to_dict("index")
    first_alarm = None
    flagged = table[table["alert"]]
    if not flagged.empty:
        first_alarm = str(flagged["window_start"].min())
    return {
        "windows": int(table["window_start"].nunique()),
        "comparisons": int(len(table)),
        "alerts": int(table["alert"].sum()),
        "first_alert": first_alarm,
        "by_method": {k: {"alerts": int(v["sum"]), "comparisons": int(v["count"])} for k, v in by_method.items()},
        "by_feature": {k: {"alerts": int(v["sum"]), "comparisons": int(v["count"])} for k, v in by_feature.items()},
    }
