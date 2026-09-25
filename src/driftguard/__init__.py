"""DriftGuard: reliable network anomaly detection under distribution shift.

The package studies how a machine-learning flow detector behaves once the
traffic it was trained on stops looking the way it did: it trains on a
historical period, backtests inside that distribution, then scores later
unseen traffic, measures the shift, and evaluates drift detection and
adaptation against a fixed false-positive budget.
"""

__version__ = "0.2.0"

from driftguard.temporal import TemporalLeakageError, TemporalSplit, build_temporal_split

__all__ = [
    "__version__",
    "TemporalSplit",
    "TemporalLeakageError",
    "build_temporal_split",
]
