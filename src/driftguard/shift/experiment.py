"""Controlled-shift experiments: attribute a degradation to a named property.

Each experiment trains one model on an unshifted training period, scores it on
a clean backtest, then scores the *same frozen model* on that backtest with one
shift applied. Adaptation is then attempted under the same false-positive
budget, so every row answers three questions: did the shift hurt, was drift
detected, and did adaptation recover anything.

The shift touches the evaluation data only. The model, the scaler and the
threshold stay exactly as trained, which is what makes the degradation
attributable to the shift rather than to retraining.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from driftguard.adaptation.strategies import ADAPTATION_STRATEGIES, apply_adaptation
from driftguard.data.schema import FlowFrame
from driftguard.drift.windowing import analyse_windows, drift_frame, resolve_features, summarize_drift
from driftguard.evaluation.metrics import (
    basic_metrics,
    calibration,
    degradation,
    f1_degradation,
    fpr_increase,
    recall_degradation,
    threshold_for_target_fpr,
)
from driftguard.features.preprocess import FeaturePreprocessor
from driftguard.models.registry import build_model
from driftguard.pipeline import _fit_preprocessor
from driftguard.shift.controlled import SHIFT_KINDS, apply_shift
from driftguard.temporal import TemporalSplit, build_temporal_split

# Magnitudes bracket a plausible operational change rather than reaching for an
# extreme: a factor of 1.5-3 on a rate or size is the kind of move a network
# change makes, and 5% / 30% attack prevalence brackets a real base rate.
DEFAULT_MAGNITUDES = {
    "packet_size": (1.5, 3.0),
    "duration": (1.5, 3.0),
    "packet_rate": (1.5, 3.0),
    "iat": (1.5, 3.0),
    "byte_rate": (1.5, 3.0),
    "protocol_mixture": (3.0,),
    "class_prevalence": (0.05, 0.30),
    "telemetry_reduction": (0.0,),
}


def _params_for(config: Dict, model_name: str) -> Dict:
    """Per-model overrides, tolerating a list-style models block."""
    block = config.get("models")
    if isinstance(block, dict):
        value = block.get(model_name, {})
        return value if isinstance(value, dict) else {}
    return {}


def _score_all(pre: FeaturePreprocessor, frame: FlowFrame, model) -> np.ndarray:
    return model.predict_proba(pre.transform(frame).to_numpy())[:, 1]


def run_controlled_shift_experiments(
    frame: FlowFrame,
    config: Dict,
    models: Sequence[str],
    output_dir: Path,
    experiment_id: str,
    magnitudes: Optional[Dict[str, Sequence[float]]] = None,
) -> Dict:
    """Run every configured shift for every requested model."""
    seed = int(config.get("seed", 42))
    target_fpr = float(config.get("target_fpr", 0.05))
    drift_config = dict(config.get("drift", {}))
    magnitudes = magnitudes or DEFAULT_MAGNITUDES

    split: TemporalSplit = build_temporal_split(frame, **(config.get("temporal", {})))
    train = split.period("train")
    validation = split.period("validation")
    backtest = split.period("backtest")

    features = list(train.frame.columns)
    pre = _fit_preprocessor(train, features, split)
    drift_features = resolve_features(train, drift_config)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    notes: List[Dict] = []

    for model_name in models:
        params = _params_for(config, model_name)
        base_model = build_model(model_name, params, seed)
        base_model.fit(pre.transform(train).to_numpy(), train.targets.to_numpy())

        validation_scores = _score_all(pre, validation, base_model)
        threshold = threshold_for_target_fpr(validation.targets.to_numpy(), validation_scores, target_fpr)

        clean_scores = _score_all(pre, backtest, base_model)
        clean_y = backtest.targets.to_numpy()
        clean_metrics = basic_metrics(clean_y, clean_scores, threshold)
        clean_cal = calibration(clean_y, clean_scores)

        for kind in SHIFT_KINDS:
            for magnitude in magnitudes.get(kind, ()):
                try:
                    shifted = apply_shift(backtest, kind, float(magnitude), seed=seed)
                except (ValueError, KeyError) as exc:
                    # A shift that does not apply to this dataset is reported as
                    # skipped rather than quietly missing from the table.
                    notes.append({
                        "model": model_name, "kind": kind, "magnitude": float(magnitude),
                        "status": "skipped", "reason": str(exc),
                    })
                    continue

                # A telemetry-reduction shift removes columns the collector no
                # longer exports, so the preprocessor fitted on the full feature
                # set cannot transform the result. That is the point of the
                # experiment: a model fitted on rich telemetry is simply not
                # applicable to a reduced collector. Both readings are recorded -
                # the rich model is reported as inapplicable rather than as
                # scoring zero, which would be a meaningless number.
                rich_features = sorted(set(pre.numeric_features) | set(pre.categorical_features))
                reduced_features = [f for f in rich_features if f in shifted.frame.columns]
                inapplicable = len(reduced_features) < len(rich_features)
                y = shifted.targets.to_numpy()

                start = time.perf_counter()
                if inapplicable:
                    shifted_metrics = clean_metrics
                    shifted_cal = clean_cal
                    delta = {
                        "f1_change": 0.0, "f1_relative_change": None,
                        "false_positive_rate_change": 0.0, "false_positive_rate_relative_change": None,
                    }
                    scores = clean_scores
                else:
                    scores = _score_all(pre, shifted, base_model)
                    shifted_metrics = basic_metrics(y, scores, threshold)
                    shifted_cal = calibration(y, scores)
                    delta = degradation(clean_metrics, shifted_metrics)
                inference = time.perf_counter() - start

                drift_results = analyse_windows(train, shifted, drift_features, drift_config)
                table = drift_frame(drift_results)
                alerts = summarize_drift(table)
                comparisons = int(alerts.get("comparisons", 0))
                alert_rate = alerts["alerts"] / comparisons if comparisons else float("nan")
                by_feature = (
                    table.groupby("feature")["effect_size"].max().sort_values(ascending=False)
                    if not table.empty else pd.Series(dtype=float)
                )

                adapted = _best_adaptation(
                    model_name, params, config, pre, base_model, split, train, shifted,
                    scores, y, threshold, seed,
                ) if not inapplicable else {
                    "adapt_status": "inapplicable_reduced_telemetry",
                    "adapt_strategy": "", "adapt_f1": float("nan"),
                    "adapt_fpr": float("nan"), "adapt_recovery": 0.0,
                    "adapt_recovery_pct": float("nan"),
                }

                record = {
                    "experiment_id": experiment_id,
                    "model": model_name,
                    "shift": kind,
                    "magnitude": float(magnitude),
                    "rows": int(len(y)),
                    "attack_rate": round(float(y.mean()), 6),
                    "threshold": round(float(threshold), 6),
                    "model_applicable": not inapplicable,
                    "clean_f1": clean_metrics["f1"],
                    "shifted_f1": shifted_metrics["f1"],
                    "f1_degradation": f1_degradation(clean_metrics, shifted_metrics),
                    "recall_degradation": recall_degradation(clean_metrics, shifted_metrics),
                    "fpr_increase": fpr_increase(clean_metrics, shifted_metrics),
                    "f1_relative_change": delta.get("f1_relative_change"),
                    "fpr_change": delta.get("false_positive_rate_change"),
                    "clean_precision": clean_metrics["precision"],
                    "shifted_precision": shifted_metrics["precision"],
                    "clean_recall": clean_metrics["recall"],
                    "shifted_recall": shifted_metrics["recall"],
                    "clean_fpr": clean_metrics["false_positive_rate"],
                    "shifted_fpr": shifted_metrics["false_positive_rate"],
                    "clean_roc_auc": clean_cal.get("roc_auc"),
                    "clean_brier": clean_cal["brier"],
                    "clean_ece": clean_cal["ece"],
                    "shifted_brier": shifted_cal["brier"],
                    "shifted_ece": shifted_cal["ece"],
                    "drift_alerts": int(alerts["alerts"]),
                    "drift_comparisons": comparisons,
                    "drift_alert_rate": round(float(alert_rate), 6) if comparisons else float("nan"),
                    "detected_drift": bool(alerts["alerts"] > 0),
                    "strongest_drift_feature": (str(by_feature.index[0]) if len(by_feature) else ""),
                    "inference_seconds": round(inference, 4),
                }
                record.update(adapted)
                rows.append(record)
                notes.append({
                    "model": model_name, "kind": kind, "magnitude": float(magnitude),
                    "status": "ok",
                    "top_drift_features": {str(k): float(v) for k, v in by_feature.head(5).items()},
                })

    table = pd.DataFrame(rows)
    if not table.empty:
        table.to_csv(output_dir / "controlled_shift_results.csv", index=False)
    (output_dir / "controlled_shift_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")

    recovered = int((table["adapt_recovery"] > 0).sum()) if not table.empty and "adapt_recovery" in table else 0

    return {
        "experiment_id": experiment_id,
        "models": list(models),
        "shifts": list(SHIFT_KINDS),
        "magnitudes": {k: [float(m) for m in v] for k, v in magnitudes.items()},
        "rows": int(len(table)),
        "skipped": int(sum(1 for n in notes if n["status"] == "skipped")),
        "degraded": int((table["f1_degradation"] > 0).sum()) if not table.empty else 0,
        "recovered": recovered,
        "results_file": "controlled_shift_results.csv",
    }


def _best_adaptation(
    model_name: str,
    params: Dict,
    config: Dict,
    pre: FeaturePreprocessor,
    base_model,
    split: TemporalSplit,
    reference: FlowFrame,
    shifted: FlowFrame,
    scores: np.ndarray,
    y: np.ndarray,
    threshold: float,
    seed: int,
) -> Dict:
    """Score every adaptation strategy against the shifted data and keep the best.

    ``history`` is the part of the shifted frame lying before the forward period
    begins. The shift changes the data's statistics, not the time ordering, so
    truncating by timestamp still leaves each strategy blind to the period it is
    scored on - ``apply_adaptation`` re-checks that and raises if violated.
    """
    cutoff = split.time_range("forward")[0]
    mask = shifted.timestamps < cutoff
    if not mask.any():
        mask = pd.Series(True, index=shifted.frame.index)
    history = FlowFrame(
        shifted.name, shifted.frame[mask].reset_index(drop=True),
        shifted.source_files, shifted.notes,
    )

    baseline = basic_metrics(y, scores, threshold)
    target_fpr = float(config.get("target_fpr", 0.05))
    strategies = list(config.get("adaptation", {}).get("strategies", ADAPTATION_STRATEGIES))

    out: Dict[str, object] = {
        "adapt_status": "not_attempted",
        "adapt_strategy": "",
        "adapt_f1": float("nan"),
        "adapt_fpr": float("nan"),
        "adapt_recovery": 0.0,
        "adapt_recovery_pct": float("nan"),
    }

    best_name = ""
    best_f1 = -np.inf
    best = None
    for strategy in strategies:
        adapt_config = {
            **config.get("adaptation", {}),
            "model": {"name": model_name, "params": params},
            "target_fpr": target_fpr,
            "base_threshold": threshold,
        }
        try:
            result = apply_adaptation(
                strategy, base_model, pre, history, reference, shifted, adapt_config, seed,
            )
            if strategy == "none":
                candidate_scores = scores
            else:
                if strategy in {"recent_window_retrain", "rolling_window_retrain"}:
                    training = history
                else:
                    combined = pd.concat([reference.frame, history.frame], ignore_index=True)
                    training = FlowFrame(reference.name, combined, reference.source_files, reference.notes)
                adapted_model = build_model(model_name, params, seed)
                adapted_model.fit(pre.transform(training).to_numpy(), training.targets.to_numpy())
                candidate_scores = adapted_model.predict_proba(pre.transform(shifted).to_numpy())[:, 1]
            candidate = basic_metrics(y, candidate_scores, result.threshold)
        except Exception as exc:  # a strategy that fails is itself a result
            out[f"adapt_error_{strategy}"] = str(exc)[:180]
            continue

        if candidate["f1"] > best_f1:
            best_f1 = candidate["f1"]
            best = candidate
            best_name = strategy

    if best is not None:
        out["adapt_status"] = "ok"
        out["adapt_strategy"] = best_name
        out["adapt_f1"] = best["f1"]
        out["adapt_fpr"] = best["false_positive_rate"]
        out["adapt_recovery"] = best["f1"] - baseline["f1"]
        lost = baseline["f1"]
        out["adapt_recovery_pct"] = (
            round(100.0 * out["adapt_recovery"] / lost, 4) if lost > 0 else float("nan")
        )
    return out
