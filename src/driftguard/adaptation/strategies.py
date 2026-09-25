"""Adaptation strategies evaluated under a fixed false-positive budget.

Every strategy obeys the same rule: at the decision time t it may use data
timestamped at or before t and nothing else. ``AdaptationResult`` records which
rows a strategy actually consumed so that a test can assert the rule held
rather than trusting the implementation.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone

from driftguard.data.schema import FlowFrame
from driftguard.models.registry import build_model
from driftguard.temporal import TemporalLeakageError
from driftguard.utils import as_timedelta

ADAPTATION_STRATEGIES = (
    "none",
    "threshold_recalibration",
    "recent_window_retrain",
    "rolling_window_retrain",
    "historical_plus_recent_retrain",
)


@dataclass
class AdaptationResult:
    """Outcome of one adaptation strategy over the forward period."""

    strategy: str
    threshold: float
    rows_used: int
    latest_row_used: Optional[pd.Timestamp]
    training_seconds: float
    refits: int
    details: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "strategy": self.strategy,
            "threshold": round(float(self.threshold), 6),
            "rows_used": int(self.rows_used),
            "latest_row_used": None if self.latest_row_used is None else str(self.latest_row_used),
            "training_seconds": round(float(self.training_seconds), 4),
            "refits": int(self.refits),
            "details": self.details,
        }


def _assert_within(frame: FlowFrame, limit: pd.Timestamp, strategy: str) -> None:
    if frame.frame.empty:
        return
    latest = frame.timestamps.max()
    if latest > limit:
        raise TemporalLeakageError(
            f"adaptation strategy '{strategy}' wanted rows up to {latest}, past its permitted {limit}"
        )


def apply_adaptation(
    strategy: str,
    model,
    preprocessor,
    history: FlowFrame,
    reference: FlowFrame,
    forward: FlowFrame,
    config: Dict,
    seed: int,
) -> AdaptationResult:
    """Run one strategy and return the model and threshold it produced.

    ``reference`` is the training period, used as the immutable historical
    anchor. ``history`` is the recent labelled window that the strategy may
    additionally consume; it must already end at or before the adaptation
    cutoff, which the caller obtains from the temporal split.
    """
    if strategy not in ADAPTATION_STRATEGIES:
        raise KeyError(f"Unknown adaptation strategy '{strategy}'. Available: {ADAPTATION_STRATEGIES}")

    target_fpr = float(config.get("target_fpr", 0.05))
    model_block = config.get("model", {})
    name = model_block.get("name", "random_forest")
    params = model_block.get("params", {})

    if strategy == "none":
        return AdaptationResult(
            strategy="none",
            threshold=float(config.get("base_threshold", 0.5)),
            rows_used=0,
            latest_row_used=None,
            training_seconds=0.0,
            refits=0,
            details={"note": "model is evaluated as trained; threshold unchanged"},
        )

    # The adaptation cutoff is the first instant of the period this strategy is
    # scored on. Validating against the history's own maximum would be a
    # tautology, so the limit always comes from the evaluation period.
    evaluation_start = forward.timestamps.min() if not forward.frame.empty else None
    if evaluation_start is not None:
        _assert_within(history, evaluation_start, strategy)

    if strategy == "threshold_recalibration":
        # Uses recent labelled data only to move the operating point, never to
        # change model parameters.
        from driftguard.evaluation.metrics import threshold_for_target_fpr

        scores = model.predict_proba(preprocessor.transform(history).to_numpy())[:, 1]
        threshold = threshold_for_target_fpr(history.targets.to_numpy(), scores, target_fpr)
        return AdaptationResult(
            strategy=strategy,
            threshold=threshold,
            rows_used=int(len(history.frame)),
            latest_row_used=history.timestamps.max(),
            training_seconds=0.0,
            refits=0,
            details={"target_fpr": target_fpr, "note": "operating point moved; parameters untouched"},
        )

    window = config.get("window", "1h")
    if strategy in {"recent_window_retrain", "rolling_window_retrain"}:
        training = history
    else:
        combined = pd.concat([reference.frame, history.frame], ignore_index=True)
        training = FlowFrame(reference.name, combined, reference.source_files, reference.notes)

    if evaluation_start is not None:
        _assert_within(training, evaluation_start, strategy)

    start = time.perf_counter()
    adapted = clone(model)
    adapted = build_model(name, params, seed)
    X = preprocessor.transform(training).to_numpy()
    adapted.fit(X, training.targets.to_numpy())
    elapsed = time.perf_counter() - start

    scores = adapted.predict_proba(preprocessor.transform(history).to_numpy())[:, 1]
    from driftguard.evaluation.metrics import threshold_for_target_fpr

    threshold = threshold_for_target_fpr(history.targets.to_numpy(), scores, target_fpr)
    return AdaptationResult(
        strategy=strategy,
        threshold=threshold,
        rows_used=int(len(training.frame)),
        latest_row_used=training.timestamps.max(),
        training_seconds=elapsed,
        refits=1,
        details={
            "target_fpr": target_fpr,
            "window": window,
            "history_rows": int(len(history.frame)),
            "reference_rows": int(len(reference.frame)),
        },
        )


def recovery_summary(
    before: Dict[str, float], after: Dict[str, float], backtest: Dict[str, float]
) -> Dict[str, Optional[float]]:
    """How much of the lost performance a strategy recovered.

    Recovery is measured against the backtest score: 100% means the forward
    score returned to its backtest level, negative means the strategy made the
    forward result worse than not adapting at all is judged against ``before``.
    """
    out: Dict[str, Optional[float]] = {}
    for key in ["f1", "recall", "precision", "false_positive_rate"]:
        b, a, back = before.get(key), after.get(key), backtest.get(key)
        if b is None or a is None:
            out[f"{key}_gain"] = None
            out[f"{key}_recovery_pct"] = None
            continue
        out[f"{key}_gain"] = float(a - b)
        if back is not None and b != back:
            out[f"{key}_recovery_pct"] = float(100.0 * (a - b) / (back - b))
        else:
            out[f"{key}_recovery_pct"] = None
    return out
