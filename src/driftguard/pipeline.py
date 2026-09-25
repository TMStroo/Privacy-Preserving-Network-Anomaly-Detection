"""The DriftGuard experiment pipeline: train, backtest, forward test, drift,
adaptation, and failure analysis, in that order.

Every stage is handed only the periods it is allowed to see. The forward test is
scored exactly once, at the end, with a threshold that was fixed on validation
data beforehand.
"""

import json
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from driftguard.adaptation.strategies import ADAPTATION_STRATEGIES, apply_adaptation, recovery_summary
from driftguard.data.registry import get_adapter
from driftguard.data.schema import CATEGORICAL_FIELDS, FlowFrame
from driftguard.drift.detectors import DRIFT_METHODS
from driftguard.drift.windowing import analyse_windows, drift_frame, resolve_features, summarize_drift
from driftguard.evaluation.metrics import (
    basic_metrics,
    calibration,
    degradation,
    f1_degradation,
    fpr_increase,
    grouped_predictions,
    metrics_at_target_fpr,
    predictions_frame,
    recall_degradation,
)
from driftguard.features.preprocess import FeaturePreprocessor
from driftguard.evaluation.failure_analysis import (
    build_failure_analysis,
    calibration_drift,
    degradation_by_window,
)
from driftguard.models.registry import MODELS, build_model, model_config_names, model_params
from driftguard.temporal import TemporalSplit, build_temporal_split


class AdaptationStudyError(RuntimeError):
    """Raised when an adaptation strategy cannot run.

    The study is a comparison of all five strategies, so a partial set of
    results is a broken result rather than a partial success.
    """


def _load_dataset(config: Dict) -> FlowFrame:
    name = config["dataset"]["name"]
    adapter = get_adapter(name)
    return adapter.load(config["dataset"]["raw_dir"], **config["dataset"].get("load_options", {}))


def _fit_preprocessor(train: FlowFrame, features: List[str], split: TemporalSplit) -> FeaturePreprocessor:
    categorical_set = set(CATEGORICAL_FIELDS)
    numeric = [f for f in features if f not in categorical_set]
    categorical = [f for f in features if f in categorical_set]
    pre = FeaturePreprocessor(numeric, categorical)
    pre.fit(train, cutoff=split.cutoff("train"))
    return pre


def run_model(
    model_name: str,
    pre: FeaturePreprocessor,
    split: TemporalSplit,
    config: Dict,
    frame: FlowFrame,
    seed: int,
) -> Dict[str, object]:
    """Train one model and score it on the backtest and the forward test.

    The threshold comes from validation data under the configured false-positive
    budget and is then applied unchanged to both later periods.
    """
    target_fpr = float(config.get("metrics", {}).get("target_fpr", 0.05))
    params = model_params(config, model_name)

    train = split.period("train")
    validation = split.period("validation")
    backtest = split.period("backtest")

    X_train = pre.transform(train).to_numpy()
    y_train = train.targets.to_numpy()

    start = time.perf_counter()
    model = build_model(model_name, params, seed)
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start

    # Threshold selection: validation data only.
    val_scores = model.predict_proba(pre.transform(validation).to_numpy())[:, 1]
    from driftguard.evaluation.metrics import threshold_for_target_fpr

    threshold = threshold_for_target_fpr(validation.targets.to_numpy(), val_scores, target_fpr)

    backtest_scores = model.predict_proba(pre.transform(backtest).to_numpy())[:, 1]
    backtest_metrics = basic_metrics(backtest.targets.to_numpy(), backtest_scores, threshold)
    backtest_metrics["target_fpr_metrics"] = metrics_at_target_fpr(
        validation.targets.to_numpy(), val_scores, target_fpr
    )

    result: Dict[str, object] = {
        "model": model_name,
        "threshold": float(threshold),
        "target_fpr": target_fpr,
        "train_seconds": round(train_seconds, 4),
        "backtest": backtest_metrics,
        "params": _serializable_params(model, params),
    }

    if split.has_forward:
        forward = split.period("forward")
        infer_start = time.perf_counter()
        forward_scores = model.predict_proba(pre.transform(forward).to_numpy())[:, 1]
        infer_seconds = time.perf_counter() - infer_start
        forward_metrics = basic_metrics(forward.targets.to_numpy(), forward_scores, threshold)
        result["forward"] = forward_metrics
        result["forward_infer_seconds"] = round(infer_seconds, 4)
        result["degradation"] = degradation(backtest_metrics, forward_metrics)
        result["f1_degradation"] = f1_degradation(backtest_metrics, forward_metrics)
        result["recall_degradation"] = recall_degradation(backtest_metrics, forward_metrics)
        result["fpr_increase"] = fpr_increase(backtest_metrics, forward_metrics)
        result["_predictions"] = {
            "backtest": predictions_frame(backtest.targets, backtest_scores, backtest.timestamps, threshold),
            "forward": predictions_frame(forward.targets, forward_scores, forward.timestamps, threshold),
        }
    result["_model_object"] = model
    return result


def _serializable_params(model, declared: Dict) -> Dict[str, object]:
    if hasattr(model, "get_params"):
        params = model.get_params()
        simple = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool, type(None)))}
        return simple or dict(declared)
    if hasattr(model, "params"):
        return dict(model.params)
    return dict(declared)


def run_adaptation_study(
    model_result: Dict[str, object],
    pre: FeaturePreprocessor,
    split: TemporalSplit,
    config: Dict,
    frame: FlowFrame,
    seed: int,
) -> Dict[str, object]:
    """Score every adaptation strategy on the forward test.

    The adaptation window is the last ``window`` of history *before* the forward
    test begins, so a strategy never sees a row from the period it is scored on.
    """
    model = model_result["_model_object"]
    target_fpr = float(config.get("metrics", {}).get("target_fpr", 0.05))
    forward = split.period("forward")
    strategies = config.get("adaptation", {}).get("strategies", list(ADAPTATION_STRATEGIES))
    window = config.get("adaptation", {}).get("window", "6h")
    model_name = model_result["model"]
    params = model_params(config, model_name)

    # History available to any strategy: everything strictly before the forward
    # test, truncated to the configured recent window.
    cutoff = pd.Timestamp(split.time_range("forward")[0])
    earlier = frame.between(frame.timestamps.min(), cutoff, include_end=False)
    if window:
        history_start = cutoff - pd.Timedelta(window)
        history = earlier.between(history_start, cutoff, include_end=False)
    else:
        history = earlier

    y_forward = forward.targets.to_numpy()
    forward_scores_unadapted = model.predict_proba(pre.transform(forward).to_numpy())[:, 1]
    baseline_metrics = basic_metrics(y_forward, forward_scores_unadapted, model_result["threshold"])

    outcomes = []
    failed: List[str] = []
    for strategy in strategies:
        try:
            adapt_config = {
                "target_fpr": target_fpr,
                "base_threshold": model_result["threshold"],
                "window": window,
                "model": {"name": model_name, "params": params},
            }
            outcome = apply_adaptation(
                strategy, model, pre, history, split.period("train"), forward, adapt_config, seed
            )
            if strategy == "none":
                scores, threshold = forward_scores_unadapted, model_result["threshold"]
            elif strategy == "threshold_recalibration":
                scores, threshold = forward_scores_unadapted, outcome.threshold
            else:
                # Score the estimator the strategy actually refitted.
                training = history
                if strategy == "historical_plus_recent_retrain":
                    training = FlowFrame(
                        history.name,
                        pd.concat([split.period("train").frame, history.frame], ignore_index=True),
                        history.source_files,
                        history.notes,
                    )
                estimator = build_model(model_name, params, seed)
                estimator.fit(pre.transform(training).to_numpy(), training.targets.to_numpy())
                scores = estimator.predict_proba(pre.transform(forward).to_numpy())[:, 1]
                threshold = outcome.threshold

            metrics = basic_metrics(y_forward, scores, threshold)
            outcomes.append(
                {
                    **outcome.as_dict(),
                    "metrics": metrics,
                    "recovery": recovery_summary(baseline_metrics, metrics, model_result["backtest"]),
                }
            )
        except Exception as exc:
            # A strategy that cannot run is a broken experiment, not a result.
            # The first benchmark swallowed these and wrote three errored
            # strategies per model into a directory that looked complete, which
            # is how two of the five strategies went unreported for a whole run.
            failed.append(f"{model_name}/{strategy}: {type(exc).__name__}: {exc}")

    if failed:
        raise AdaptationStudyError(
            f"{len(failed)} adaptation strategies failed, so the study is incomplete "
            "and its results would be misleading:\n  " + "\n  ".join(failed)
        )

    return {
        "model": model_name,
        "target_fpr": target_fpr,
        "history_window": window,
        "history_rows": int(len(history.frame)),
        "history_period": [str(history.timestamps.min()), str(history.timestamps.max())],
        "baseline_forward": baseline_metrics,
        "strategies": outcomes,
    }


def run_benchmark(config: Dict, experiment_root: str = "results/experiments") -> Dict[str, object]:
    """Run the full temporal benchmark for every enabled model."""
    from driftguard.experiments.tracking import ExperimentRun, experiment_id, file_checksums

    seed = int(config.get("seed", 42))
    np.random.seed(seed)

    frame = _load_dataset(config)
    split = build_temporal_split(
        frame,
        train_fraction=float(config.get("temporal", {}).get("train_fraction", 0.50)),
        validation_fraction=float(config.get("temporal", {}).get("validation_fraction", 0.15)),
        backtest_fraction=float(config.get("temporal", {}).get("backtest_fraction", 0.15)),
        forward_fraction=float(config.get("temporal", {}).get("forward_fraction", 0.20)),
        use_forward=bool(config.get("temporal", {}).get("use_forward", True)),
    )
    split.assert_ordered()

    features = frame.feature_names()
    if config.get("features", {}).get("use_cross_dataset_only"):
        features = frame.cross_dataset_features()

    pre = _fit_preprocessor(split.period("train"), features, split)

    run = ExperimentRun(
        experiment_id("temporal", frame.name, config.get("tag"), now=_parse_stamp(config.get("now"))),
        root=experiment_root,
    )
    dataset_files = list(frame.source_files)
    meta = run.metadata(
        dataset=frame.name,
        dataset_checksums=file_checksums([f"{config['dataset']['raw_dir']}/{f}" for f in dataset_files]),
        features=features,
        seed=seed,
        config=config,
        split_description=split.describe(),
    )
    run.write_json("metadata.json", meta)
    run.write_text("config.yaml", _dump_yaml(config))
    run.write_json("dataset_summary.json", frame.describe_time())

    drift_config = config.get("drift", {})
    drift_features = resolve_features(frame, drift_config) if drift_config.get("enabled", True) else []
    drift_results: List = []
    if drift_features:
        reference = split.period(drift_config.get("reference_period", "train"))
        target = split.period("forward") if split.has_forward else split.period("backtest")
        drift_results = analyse_windows(reference, target, drift_features, drift_config)
    drift_table = drift_frame(drift_results)
    if drift_config.get("enabled", True) and drift_features and drift_table.empty:
        # An empty drift table is indistinguishable from "no drift", so it must
        # never be written as a successful result. Either the window/stride
        # cannot fit the period or every window fell under min_samples.
        raise ValueError(
            f"drift analysis produced no rows for {frame.name} "
            f"(window={drift_config.get('window')}, "
            f"stride={drift_config.get('stride')}, "
            f"min_samples={drift_config.get('min_samples')}); "
            "the run would otherwise report an empty result as 'no drift'"
        )
    if not drift_table.empty:
        run.write_table("drift_events.csv", drift_table)

    model_names = model_config_names(config)
    results = []
    for name in model_names:
        result = run_model(name, pre, split, config, frame, seed)
        adaptation = run_adaptation_study(result, pre, split, config, frame, seed)
        predictions = result.pop("_predictions", {})
        result.pop("_model_object", None)
        results.append({"model": name, "result": result, "adaptation": adaptation})
        for period, table in predictions.items():
            if not table.empty:
                run.write_table(f"predictions_{period}_{name}.csv", table)

        if "backtest" in predictions and "forward" in predictions:
            backtest_cal = calibration(
                predictions["backtest"]["y_true"], predictions["backtest"]["y_score"]
            )
            forward_cal = calibration(
                predictions["forward"]["y_true"], predictions["forward"]["y_score"]
            )
            analysis = build_failure_analysis(
                backtest_metrics=result["backtest"],
                backtest_predictions=predictions["backtest"],
                forward_predictions=predictions["forward"],
                drift_events=drift_table,
                window_freq=str(drift_config.get("window", "6h")),
                backtest_calibration=backtest_cal,
                forward_calibration=forward_cal,
            )
            run.write_json(f"failure_analysis_{name}.json", prune_private(analysis))
            run.write_json(f"calibration_{name}.json", {
                "backtest": backtest_cal, "forward": forward_cal,
                "change": calibration_drift(backtest_cal, forward_cal),
            })
            degradation_by_window(
                predictions["forward"], freq=str(drift_config.get("window", "6h"))
            ).to_csv(run.path / f"forward_windows_{name}.csv", index=False)

    payload = {
        "dataset": frame.name,
        "split": split.describe(),
        "features": features,
        "drift_summary": summarize_drift(drift_table) if not drift_table.empty else {},
        "models": results,
    }
    run.write_json("metrics.json", payload)
    return {"experiment": run.summary(), "metrics": payload, "run": run, "split": split}


def _parse_stamp(value):
    if value is None:
        return None
    from datetime import datetime

    return datetime.fromisoformat(value)


def _dump_yaml(config: Dict) -> str:
    import yaml

    return yaml.safe_dump(config, sort_keys=False, default_flow_style=False)


def prune_private(obj):
    """Strip internal estimator references and make the result JSON-safe.

    Grouped failure-analysis tables produce Timestamp keys, so keys are coerced
    as well as values.
    """
    if isinstance(obj, dict):
        return {
            str(k): prune_private(v)
            for k, v in obj.items()
            if not (isinstance(k, str) and (k.startswith("_") or k.endswith("_object")))
        }
    if isinstance(obj, list):
        return [prune_private(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
