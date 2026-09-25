"""Temporal splitting and the guards that keep later data out of earlier steps.

The forward test only means something if nothing from it can reach a component
that ran before it. That is enforced here instead of being left to convention:
a split stores integer row boundaries over a chronologically sorted frame, so
each period is a contiguous slice and two periods can never overlap by
construction. The recorded timestamp boundaries describe those slices and are
used for reporting and for the ordering assertions, never for re-slicing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from driftguard.data.schema import FlowFrame

PERIODS = ("train", "validation", "backtest", "forward")


class TemporalLeakageError(RuntimeError):
    """Raised when a component would consume data from outside its allowed period."""


@dataclass
class TemporalSplit:
    """Disjoint, chronologically ordered periods over one sorted time axis.

    train       fits model parameters
    validation  picks operating thresholds only
    backtest    held-out evaluation inside the historical distribution
    forward     later traffic the model never saw, for the forward test

    ``forward`` is optional; without it the split serves a purely historical
    experiment and the forward slice is not addressable at all.
    """

    dataset: str
    frame: FlowFrame
    bounds: Dict[str, tuple]
    fractions: Dict[str, float] = field(default_factory=dict)
    _index: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._build_index()
        self._check_order()

    def _build_index(self):
        order = ["train", "validation", "backtest"] + (["forward"] if "forward" in self.bounds else [])
        start = 0
        n = len(self.frame.frame)
        for period in order:
            stop = self.bounds[period][1]
            self._index[period] = np.arange(start, min(stop, n))
            start = self.bounds[period][1]
        self._index["_order"] = np.arange(n)

    def _check_order(self):
        for period in PERIODS:
            if period not in self.bounds:
                continue
            lo, hi = self.bounds[period]
            if lo < 0 or hi > len(self.frame.frame):
                raise ValueError(f"period '{period}' bounds [{lo}, {hi}) fall outside the frame")
            if hi < lo:
                raise ValueError(f"period '{period}' has inverted bounds")
        edges = [self.bounds[p][0] for p in ("train", "validation", "backtest") if p in self.bounds]
        edges.append(len(self.frame.frame))
        if edges != sorted(edges):
            raise TemporalLeakageError(f"period row boundaries are not monotonic: {edges}")

    @property
    def has_forward(self) -> bool:
        return "forward" in self.bounds

    def available_periods(self) -> List[str]:
        return [p for p in PERIODS if p in self.bounds]

    def rows(self, period: str) -> np.ndarray:
        if period not in self._index:
            raise TemporalLeakageError(
                f"period '{period}' is not part of this split (available: {self.available_periods()})"
            )
        return self._index[period]

    def period(self, period: str) -> FlowFrame:
        idx = self.rows(period)
        return FlowFrame(
            self.dataset,
            self.frame.frame.iloc[idx].reset_index(drop=True),
            self.frame.source_files,
            self.frame.notes,
        )

    def time_range(self, period: str) -> tuple:
        idx = self.rows(period)
        ts = self.frame.frame["timestamp"].iloc[idx]
        return (str(ts.min()), str(ts.max()))

    def cutoff(self, period: str) -> pd.Timestamp:
        """Latest timestamp a component for ``period`` is allowed to observe."""
        return pd.Timestamp(self.time_range(period)[1])

    def assert_ordered(self) -> None:
        """Regression guard: every period must sit strictly after the previous one."""
        ordered = self.available_periods()
        for earlier, later in zip(ordered, ordered[1:]):
            a = self.cutoff(earlier)
            b = pd.Timestamp(self.time_range(later)[0])
            if not a < b:
                raise TemporalLeakageError(
                    f"period '{later}' starts at {b} but '{earlier}' already reaches {a}; "
                    "a later period would contain rows an earlier component trained on"
                )

    def assert_no_future_rows(self, frame: FlowFrame, period: str) -> None:
        """Guard for hand-assembled slices: no row may exceed this period's cutoff."""
        limit = self.cutoff(period)
        ts = frame.timestamps
        if not ts.empty and ts.max() > limit:
            raise TemporalLeakageError(
                f"slice for '{period}' contains rows up to {ts.max()}, past its cutoff {limit}"
            )

    def describe(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "dataset": self.dataset,
            "has_forward_test": self.has_forward,
            "fractions": self.fractions,
            "rows_total": int(len(self.frame.frame)),
        }
        for period in self.available_periods():
            idx = self.rows(period)
            sub = self.frame.frame.iloc[idx]
            lo, hi = self.time_range(period)
            out[f"{period}_period"] = [lo, hi]
            out[f"{period}_count"] = {
                "rows": int(len(sub)),
                "attacks": int(sub["label"].sum()),
                "attack_rate": round(float(sub["label"].mean()), 6) if len(sub) else 0.0,
            }
        return out


def build_temporal_split(
    frame: FlowFrame,
    train_fraction: float = 0.50,
    validation_fraction: float = 0.15,
    backtest_fraction: float = 0.15,
    forward_fraction: float = 0.20,
    use_forward: bool = True,
) -> TemporalSplit:
    """Cut one dataset's time axis into train / validation / backtest / forward.

    Periods are sized by row count and the cut points are snapped forward past
    any rows sharing a boundary timestamp, so a single instant is never split
    across two periods. Sizing by rows rather than elapsed time matters because
    several public IDS datasets are assembled from a few discrete capture
    sessions separated by long idle gaps, where a proportional time cut leaves
    whole periods empty.
    """
    ordered = frame.sorted_by_time()
    n = len(ordered.frame)
    if n == 0:
        raise ValueError(f"{frame.name} has no usable rows")

    ts = ordered.timestamps
    if ts.nunique() < 2:
        raise ValueError(f"{frame.name} carries a single distinct timestamp; no temporal split is possible")

    if use_forward:
        fractions = {
            "train": train_fraction,
            "validation": validation_fraction,
            "backtest": backtest_fraction,
            "forward": forward_fraction,
        }
    else:
        fractions = {
            "train": train_fraction,
            "validation": validation_fraction,
            "backtest": 1.0 - train_fraction - validation_fraction,
        }
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"period fractions must sum to 1.0, got {total}")

    values = ts.to_numpy()
    cuts: List[int] = []
    running = 0.0
    for key in ["train", "validation", "backtest", "forward"]:
        if key not in fractions:
            break
        running += fractions[key]
        position = min(int(round(n * running)), n)
        position = max(position, 1)
        # Snap forward so the boundary timestamp belongs entirely to the later
        # period; without this the same instant appears on both sides of the cut.
        while position < n and values[position] == values[position - 1]:
            position += 1
        cuts.append(position)

    # A snap that runs past the end would leave the final period empty, so pull
    # the boundary back to the last timestamp that still leaves usable rows.
    for i in range(len(cuts) - 1, 0, -1):
        if cuts[i] >= n and cuts[i - 1] < n:
            cuts[i] = n
            break
    for i in range(len(cuts) - 1, 0, -1):
        if cuts[i] >= n:
            continue
        if cuts[i] == cuts[i - 1]:
            raise ValueError(
                f"cannot separate period boundaries: two consecutive cuts landed on the same "
                f"row position ({cuts[i]}). Reduce the period fractions or use a larger dataset."
            )
    if cuts[-1] != n:
        cuts[-1] = n

    order = ["train", "validation", "backtest", "forward"]
    present = [k for k in order if k in fractions]
    bounds: Dict[str, tuple] = {}
    previous = 0
    for key, stop in zip(present, cuts):
        bounds[key] = (previous, stop)
        previous = stop

    empty = [k for k in present if bounds[k][1] <= bounds[k][0]]
    if empty:
        raise ValueError(
            f"temporal split for {frame.name} produces empty periods: {empty}. "
            "The dataset's time axis is too coarse for these fractions."
        )

    split = TemporalSplit(dataset=frame.name, frame=ordered, bounds=bounds, fractions=fractions)
    split.assert_ordered()
    return split
