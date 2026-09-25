"""Distribution-shift measurement: how far a window has moved from a reference.

Four complementary tests are provided. A two-sample test alone is not enough
to call drift operationally meaningful - with millions of windows a tiny
p-value fires constantly - so every window also reports an effect size or a
distance that can be compared against a configured threshold.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

DRIFT_METHODS = ("ks", "wasserstein", "psi", "cusum")


@dataclass
class DriftResult:
    """One window's verdict for one feature."""

    window_start: pd.Timestamp
    window_end: pd.Timestamp
    feature: str
    method: str
    statistic: float
    threshold: float
    alert: bool
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    n_reference: int = 0
    n_window: int = 0
    detail: Dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict:
        return {
            "window_start": str(self.window_start),
            "window_end": str(self.window_end),
            "feature": self.feature,
            "method": self.method,
            "statistic": round(float(self.statistic), 6),
            "threshold": round(float(self.threshold), 6),
            "alert": bool(self.alert),
            "p_value": None if self.p_value is None else float(f"{self.p_value:.3e}"),
            "effect_size": None if self.effect_size is None else round(float(self.effect_size), 6),
            "n_reference": int(self.n_reference),
            "n_window": int(self.n_window),
        }


def _clean(reference: np.ndarray, window: np.ndarray) -> tuple:
    ref = reference[np.isfinite(reference)]
    cur = window[np.isfinite(window)]
    return ref, cur


def ks_test(reference: np.ndarray, window: np.ndarray, alpha: float = 0.01) -> Dict[str, float]:
    """Two-sample Kolmogorov-Smirnov test plus a normalised effect size.

    The KS statistic D is already a distance in [0, 1] and is compared against
    the critical value for the sample sizes, so large windows do not trigger on
    numerical noise alone.
    """
    ref, cur = _clean(np.asarray(reference, dtype=float), np.asarray(window, dtype=float))
    if ref.size < 2 or cur.size < 2:
        return {"statistic": 0.0, "p_value": None, "effect_size": 0.0, "alert": False}
    result = stats.ks_2samp(ref, cur)
    n_eff = ref.size * cur.size / (ref.size + cur.size)
    critical = stats.kstwo.ppf(1.0 - alpha, max(int(np.floor(n_eff)), 1)) if n_eff > 0 else 1.0
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "effect_size": float(result.statistic),
        "alert": bool(result.statistic > critical),
        "critical_value": float(critical),
    }


def wasserstein_distance(reference: np.ndarray, window: np.ndarray, threshold: float = 0.1) -> Dict[str, float]:
    """Wasserstein-1 distance, scaled by reference spread to be comparable.

    Raw units make this unusable across features with different scales, so the
    distance is divided by the reference interquartile range.
    """
    ref, cur = _clean(np.asarray(reference, dtype=float), np.asarray(window, dtype=float))
    if ref.size < 2 or cur.size < 2:
        return {"statistic": 0.0, "p_value": None, "effect_size": 0.0, "alert": False}
    raw = float(stats.wasserstein_distance(ref, cur))
    spread = float(np.percentile(ref, 75) - np.percentile(ref, 25))
    if spread <= 0:
        spread = float(np.std(ref)) or 1.0
    scaled = raw / spread
    return {
        "statistic": raw,
        "p_value": None,
        "effect_size": scaled,
        "alert": bool(scaled > threshold),
        "scaled_distance": scaled,
    }


def population_stability_index(
    reference: np.ndarray, window: np.ndarray, bins: int = 10, threshold: float = 0.2
) -> Dict[str, float]:
    """Population Stability Index over reference-derived quantile bins.

    Bins come from the reference quantiles so that the reference always fills
    them, which keeps PSI from being inflated by an empty-bin artefact.
    """
    ref, cur = _clean(np.asarray(reference, dtype=float), np.asarray(window, dtype=float))
    if ref.size < 2 or cur.size < 2:
        return {"statistic": 0.0, "p_value": None, "effect_size": 0.0, "alert": False}
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return {"statistic": 0.0, "p_value": None, "effect_size": 0.0, "alert": False}
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_counts / max(ref.size, 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur.size, 1), 1e-6, None)
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return {
        "statistic": psi,
        "p_value": None,
        "effect_size": psi,
        "alert": bool(psi > threshold),
    }


class CusumDetector:
    """One-sided sequential CUSUM over the standardised mean of a feature.

    Sequential rather than windowed: it accumulates evidence of a sustained
    shift instead of requiring each window to look unusual on its own, which is
    the property a monitoring system actually needs.
    """

    def __init__(self, threshold: float = 5.0, drift: float = 0.5, warmup: int = 20):
        self.threshold = float(threshold)
        self.drift = float(drift)
        self.warmup = int(warmup)
        self.reset()

    def reset(self):
        self._pos = 0.0
        self._neg = 0.0
        self._mean = None
        self._std = None
        self._n = 0
        self._running: List[float] = []

    def update(self, value: float):
        """Feed one observation and return the alarm statistic."""
        x = float(value)
        if not np.isfinite(x):
            return {"statistic": 0.0, "alert": False, "alarm": False}
        if self._mean is None or self._n < self.warmup:
            self._running.append(x)
            if len(self._running) >= 2:
                self._mean = float(np.mean(self._running))
                self._std = float(np.std(self._running)) or 1.0
            self._n += 1
            return {"statistic": 0.0, "alert": False, "alarm": False}

        z = (x - self._mean) / self._std
        self._pos = max(0.0, self._pos + z - self.drift)
        self._neg = max(0.0, self._neg - z - self.drift)
        statistic = max(self._pos, self._neg)
        alarm = statistic > self.threshold
        if alarm:
            # Reset after signalling so one sustained shift raises one alarm.
            self._pos = self._neg = 0.0
        return {"statistic": float(statistic), "alert": bool(alarm), "alarm": bool(alarm), "z": float(z)}


def detect_window(
    reference: pd.Series,
    window: pd.Series,
    method: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    feature: str,
    config: Dict,
) -> DriftResult:
    """Run one drift method over one feature in one window."""
    ref_values = reference.to_numpy(dtype=float)
    cur_values = window.to_numpy(dtype=float)
    min_samples = int(config.get("min_samples", 100))

    if ref_values.size < min_samples or cur_values.size < min_samples:
        return DriftResult(
            window_start, window_end, feature, method, 0.0, float("nan"), False,
            n_reference=int(ref_values.size), n_window=int(cur_values.size),
        )

    if method == "ks":
        out = ks_test(ref_values, cur_values, alpha=float(config.get("alpha", 0.01)))
        threshold = out.get("critical_value", 1.0)
    elif method == "wasserstein":
        out = wasserstein_distance(ref_values, cur_values, threshold=float(config.get("wasserstein_threshold", 0.1)))
        threshold = float(config.get("wasserstein_threshold", 0.1))
    elif method == "psi":
        out = population_stability_index(
            ref_values, cur_values, bins=int(config.get("psi_bins", 10)),
            threshold=float(config.get("psi_threshold", 0.2)),
        )
        threshold = float(config.get("psi_threshold", 0.2))
    else:
        raise KeyError(f"Unknown drift method '{method}'. Available: {DRIFT_METHODS}")

    return DriftResult(
        window_start=window_start,
        window_end=window_end,
        feature=feature,
        method=method,
        statistic=out.get("statistic", 0.0),
        threshold=threshold,
        alert=bool(out.get("alert", False)),
        p_value=out.get("p_value"),
        effect_size=out.get("effect_size"),
        n_reference=int(ref_values.size),
        n_window=int(cur_values.size),
        detail={k: v for k, v in out.items() if k not in {"statistic", "p_value", "effect_size", "alert"}},
    )


def run_sequential_drift(
    reference: pd.Series,
    stream: pd.Series,
    method: str,
    timestamps: pd.Series,
    feature: str,
    config: Dict,
) -> List[DriftResult]:
    """Run CUSUM over an ordered stream, emitting one result per observation."""
    detector = CusumDetector(
        threshold=float(config.get("cusum_threshold", 5.0)),
        drift=float(config.get("cusum_drift", 0.5)),
        warmup=int(config.get("min_samples", 100)),
    )
    results = []
    values = stream.to_numpy(dtype=float)
    times = timestamps.to_numpy()
    for i, value in enumerate(values):
        out = detector.update(value)
        results.append(
            DriftResult(
                window_start=pd.Timestamp(times[0]),
                window_end=pd.Timestamp(times[i]),
                feature=feature,
                method=method,
                statistic=out["statistic"],
                threshold=float(config.get("cusum_threshold", 5.0)),
                alert=out["alert"],
                effect_size=out.get("z"),
                n_reference=int(reference.size),
                n_window=i + 1,
            )
        )
    return results
