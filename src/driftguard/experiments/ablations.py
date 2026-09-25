"""Targeted ablations.

Each ablation answers one methodological question rather than filling a grid:

- temporal vs random split: how optimistic is a randomly shuffled split, and
  therefore how much of a random-split benchmark is leakage?
- full vs restricted metadata: what the metadata-only policy costs, and whether
  the payload-derived features that were removed were load bearing.
- drift thresholds: how much of an "alert" is a property of the threshold
  rather than of the traffic.
- adaptation window size: whether more recent data helps or hurts.
- telemetry families: which features carry the detector's signal.

The expensive models are deliberately absent. Gradient boosting costs about
220 s per fit on 2.5M rows, and this matrix has five axes; logistic regression
and random forest answer every one of these questions at a fraction of the cost,
and the headline benchmark already covers the expensive models.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from driftguard.data.registry import get_adapter
from driftguard.data.schema import CATEGORICAL_FIELDS, FlowFrame
from driftguard.evaluation.metrics import basic_metrics, threshold_for_target_fpr
from driftguard.models.registry import build_model, model_params
from driftguard.pipeline import _fit_preprocessor
from driftguard.temporal import TemporalLeakageError, build_temporal_split

# Payload-derived in the broad sense: anything that would let a detector learn
# from the contents of a packet rather than its metadata. The V1 project removed
# five of these after finding the metadata-only claim was not strictly true; the
# ablation below is what justifies that removal.
RICH_ONLY_FEATURES = (
    "trans_depth", "response_body_len", "is_ftp_login",
    "ct_ftp_cmd", "ct_flw_http_mthd",
)

TELEMETRY_FAMILIES = {
    "timing": ["flow_duration", "duration_log1p"],
    "volume": ["total_packets", "total_bytes", "packets_log1p", "bytes_log1p"],
    "direction": ["forward_packets", "backward_packets", "forward_bytes", "backward_bytes",
                  "bytes_ratio", "packets_ratio"],
    "rates": ["packet_rate", "byte_rate"],
    "size_shape": ["mean_packet_size", "bytes_per_packet"],
}


def _load(config: Dict) -> FlowFrame:
    import os

    raw_dir = os.environ.get("DRIFTGUARD_RAW_DIR") or config["dataset"]["raw_dir"]
    return get_adapter(config["dataset"]["name"]).load(
        raw_dir, **config["dataset"].get("load_options", {})
    )


def _score(pre, frame: FlowFrame, model) -> np.ndarray:
    return model.predict_proba(pre.transform(frame).to_numpy())[:, 1]


def _fit_and_score(
    train: FlowFrame,
    evaluate: FlowFrame,
    features: Sequence[str],
    model_name: str,
    params: Dict,
    seed: int,
    target_fpr: float,
    split,
) -> Dict[str, object]:
    pre = _fit_preprocessor(train, features, split)
    model = build_model(model_name, params, seed)
    started = time.perf_counter()
    model.fit(pre.transform(train).to_numpy(), train.targets.to_numpy())
    fit_seconds = time.perf_counter() - started
    scores = _score(pre, evaluate, model)
    return {
        "metrics": basic_metrics(evaluate.targets.to_numpy(), scores,
                                 threshold_for_target_fpr(evaluate.targets.to_numpy(), scores, target_fpr)),
        "scores": scores,
        "train_seconds": fit_seconds,
    }


def split_mode_ablation(frame: FlowFrame, config: Dict, model_name: str, seed: int) -> List[Dict]:
    """Temporal split against a random split on the same rows.

    The random split is the control every intrusion-detection benchmark
    implicitly runs. The point is not that it scores better - it always does -
    but how much of the difference is contamination rather than generalisation.
    """
    from sklearn.model_selection import train_test_split

    target_fpr = float(config.get("metrics", {}).get("target_fpr", 0.05))
    params = model_params(config, model_name)
    features = list(frame.numeric_features())
    rows: List[Dict] = []

    split = build_temporal_split(frame, **config.get("temporal", {}))
    temporal_train, temporal_eval = split.period("train"), split.period("forward")
    result = _fit_and_score(temporal_train, temporal_eval, features, model_name, params, seed, target_fpr, split)
    rows.append({
        "ablation": "split_mode", "variant": "temporal", "model": model_name,
        "f1": result["metrics"]["f1"], "precision": result["metrics"]["precision"],
        "recall": result["metrics"]["recall"],
        "false_positive_rate": result["metrics"]["false_positive_rate"],
        "roc_auc": result["metrics"]["roc_auc"], "pr_auc": result["metrics"]["pr_auc"],
        "train_rows": int(len(temporal_train.frame)), "eval_rows": int(len(temporal_eval.frame)),
        "note": "trained on the earliest 50%, evaluated on the latest 20%",
    })

    # The control: the same rows, shuffled before splitting. Nothing else moves,
    # so any difference is attributable to the split.
    frame_shuffled = FlowFrame(
        frame.name,
        frame.frame.sample(frac=1.0, random_state=seed).reset_index(drop=True),
        frame.source_files,
        frame.notes,
    )
    cut = int(len(frame_shuffled.frame) * 0.5)
    train_a, train_b = train_test_split(
        frame_shuffled.frame, train_size=cut, random_state=seed, stratify=frame_shuffled.frame["label"]
    )
    random_train = FlowFrame(frame.name, train_a.reset_index(drop=True), frame.source_files, frame.notes)
    random_eval = FlowFrame(frame.name, train_b.tail(len(temporal_eval.frame)).reset_index(drop=True),
                            frame.source_files, frame.notes)

    # The random control deliberately violates the temporal protocol, so it
    # cannot use the temporal split's cutoff: _fit_preprocessor would correctly
    # refuse, because a shuffled training set reaches past the chronological
    # boundary. Passing cutoff=None states the intent explicitly - this arm is
    # the leakage the rest of the project is designed to avoid.
    from driftguard.features.preprocess import FeaturePreprocessor

    try:
        categorical = {c for c in CATEGORICAL_FIELDS if c in frame.frame.columns}
        pre = FeaturePreprocessor(
            [f for f in features if f not in categorical],
            [f for f in features if f in categorical],
        )
        pre.fit(random_train, cutoff=None)
        model = build_model(model_name, params, seed)
        model.fit(pre.transform(random_train).to_numpy(), random_train.targets.to_numpy())
        scores = _score(pre, random_eval, model)
        metrics = basic_metrics(
            random_eval.targets.to_numpy(), scores,
            threshold_for_target_fpr(random_eval.targets.to_numpy(), scores, target_fpr),
        )
        rows.append({
            "ablation": "split_mode", "variant": "random", "model": model_name,
            "f1": metrics["f1"], "precision": metrics["precision"], "recall": metrics["recall"],
            "false_positive_rate": metrics["false_positive_rate"],
            "roc_auc": metrics["roc_auc"], "pr_auc": metrics["pr_auc"],
            "train_rows": int(len(random_train.frame)), "eval_rows": int(len(random_eval.frame)),
            "note": "same rows, shuffled first; evaluates on contemporaneous traffic",
        })
    except Exception as exc:
        rows.append({
            "ablation": "split_mode", "variant": "random", "model": model_name,
            "note": f"failed: {type(exc).__name__}: {exc}",
        })
    return rows


def feature_policy_ablation(frame: FlowFrame, config: Dict, model_name: str, seed: int) -> List[Dict]:
    """All available features against the metadata-only policy."""
    target_fpr = float(config.get("metrics", {}).get("target_fpr", 0.05))
    params = model_params(config, model_name)
    split = build_temporal_split(frame, **config.get("temporal", {}))
    train, forward = split.period("train"), split.period("forward")

    available = list(frame.frame.columns)
    rich = [f for f in RICH_ONLY_FEATURES if f in available]
    metadata_only = [f for f in frame.numeric_features() if f not in rich]

    if not rich:
        # DriftGuard's common schema already excludes the payload-derived fields
        # the V1 project removed, so there is nothing left to ablate. Reporting
        # two identical arms would look like a finding when it is really the
        # absence of one, so the ablation records why it is inapplicable.
        result = _fit_and_score(train, forward, metadata_only, model_name, params, seed, target_fpr, split)
        metrics = result["metrics"]
        return [{
            "ablation": "feature_policy", "variant": "not_applicable", "model": model_name,
            "n_features": len(metadata_only), "dropped": [],
            "f1": metrics["f1"], "precision": metrics["precision"], "recall": metrics["recall"],
            "false_positive_rate": metrics["false_positive_rate"],
            "roc_auc": metrics["roc_auc"], "pr_auc": metrics["pr_auc"],
            "note": (
                "the common schema already excludes every payload-derived field "
                f"({', '.join(RICH_ONLY_FEATURES)}), so the metadata-only policy is "
                "already in force and there is no richer arm to compare against; "
                "the numbers reported here are the metadata-only detector"
            ),
        }]

    rows: List[Dict] = []
    for variant, features, dropped in [
        ("full_metadata", [f for f in frame.numeric_features()], []),
        ("restricted_metadata", metadata_only, rich),
    ]:
        if not features:
            continue
        try:
            result = _fit_and_score(train, forward, features, model_name, params, seed, target_fpr, split)
            metrics = result["metrics"]
            rows.append({
                "ablation": "feature_policy", "variant": variant, "model": model_name,
                "n_features": len(features), "dropped": dropped,
                "f1": metrics["f1"], "precision": metrics["precision"], "recall": metrics["recall"],
                "false_positive_rate": metrics["false_positive_rate"],
                "roc_auc": metrics["roc_auc"], "pr_auc": metrics["pr_auc"],
            })
        except Exception as exc:
            rows.append({"ablation": "feature_policy", "variant": variant, "model": model_name,
                         "note": f"failed: {type(exc).__name__}: {exc}"})
    return rows


def drift_threshold_ablation(frame: FlowFrame, config: Dict, model_name: str, seed: int) -> List[Dict]:
    """How many alerts each drift threshold produces on the same data."""
    from driftguard.drift.windowing import analyse_windows, drift_frame

    drift_config = dict(config.get("drift", {}))
    split = build_temporal_split(frame, **config.get("temporal", {}))
    reference = split.period(drift_config.get("reference_period", "train"))
    target = split.period("forward")
    features = drift_config.get("features") or list(reference.numeric_features())
    features = [f for f in features if f in target.frame.columns]

    rows: List[Dict] = []
    grid = [
        ("ks_alpha_0.001", {"alpha": 0.001}),
        ("ks_alpha_0.01", {"alpha": 0.01}),
        ("ks_alpha_0.05", {"alpha": 0.05}),
        ("psi_0.10", {"psi_threshold": 0.10}),
        ("psi_0.20", {"psi_threshold": 0.20}),
        ("psi_0.50", {"psi_threshold": 0.50}),
        ("wasserstein_0.05", {"wasserstein_threshold": 0.05}),
        ("wasserstein_0.10", {"wasserstein_threshold": 0.10}),
        ("wasserstein_0.25", {"wasserstein_threshold": 0.25}),
        ("cusum_far_0.01", {"cusum_false_alarm_rate": 0.01}),
        ("cusum_far_0.05", {"cusum_false_alarm_rate": 0.05}),
        ("cusum_far_0.20", {"cusum_false_alarm_rate": 0.20}),
    ]
    for name, override in grid:
        cfg = {**drift_config, **override}
        try:
            table = drift_frame(analyse_windows(reference, target, features, cfg))
            comparisons = int(len(table))
            alerts = int(table["alert"].sum()) if not table.empty else 0
            rows.append({
                "ablation": "drift_threshold", "variant": name, "model": model_name,
                "alerts": alerts, "comparisons": comparisons,
                "alert_rate": (alerts / comparisons) if comparisons else float("nan"),
                "mean_effect_size": float(table["effect_size"].abs().mean()) if not table.empty else float("nan"),
            })
        except Exception as exc:
            rows.append({"ablation": "drift_threshold", "variant": name, "model": model_name,
                         "note": f"failed: {type(exc).__name__}: {exc}"})
    return rows


def adaptation_window_ablation(frame: FlowFrame, config: Dict, model_name: str, seed: int) -> List[Dict]:
    """Whether a longer recent window helps or hurts the adapted model."""
    import pandas as pd

    from driftguard.adaptation.strategies import apply_adaptation, recovery_summary

    target_fpr = float(config.get("metrics", {}).get("target_fpr", 0.05))
    params = model_params(config, model_name)
    split = build_temporal_split(frame, **config.get("temporal", {}))
    train, forward = split.period("train"), split.period("forward")
    features = list(frame.numeric_features())
    pre = _fit_preprocessor(train, features, split)

    base = build_model(model_name, params, seed)
    base.fit(pre.transform(train).to_numpy(), train.targets.to_numpy())
    base_threshold = threshold_for_target_fpr(
        split.period("validation").targets.to_numpy(),
        _score(pre, split.period("validation"), base), target_fpr,
    )
    baseline = basic_metrics(forward.targets.to_numpy(), _score(pre, forward, base), base_threshold)
    backtest = basic_metrics(split.period("backtest").targets.to_numpy(),
                             _score(pre, split.period("backtest"), base), base_threshold)

    rows: List[Dict] = [{"ablation": "adaptation_window", "variant": "none", "model": model_name,
                         "f1": baseline["f1"], "recall": baseline["recall"],
                         "false_positive_rate": baseline["false_positive_rate"],
                         "note": "unadapted forward performance"}]
    cutoff = pd.Timestamp(split.time_range("forward")[0])
    for window in ("1h", "3h", "6h", "12h"):
        start = cutoff - pd.Timedelta(window)
        history = frame.between(start, cutoff, include_end=False)
        if history.frame.empty or history.frame["label"].nunique() < 2:
            continue
        try:
            outcome = apply_adaptation(
                "recent_window_retrain", base, pre, history, train, forward,
                {"target_fpr": target_fpr, "base_threshold": base_threshold, "window": window,
                 "model": {"name": model_name, "params": params}},
                seed,
            )
            adapted = build_model(model_name, params, seed)
            adapted.fit(pre.transform(history).to_numpy(), history.targets.to_numpy())
            scores = _score(pre, forward, adapted)
            metrics = basic_metrics(forward.targets.to_numpy(), scores, outcome.threshold)
            recovery = recovery_summary(baseline, metrics, backtest)
            rows.append({
                "ablation": "adaptation_window", "variant": f"recent_{window}", "model": model_name,
                "history_rows": int(len(history.frame)), "threshold": outcome.threshold,
                "f1": metrics["f1"], "recall": metrics["recall"],
                "precision": metrics["precision"],
                "false_positive_rate": metrics["false_positive_rate"],
                "f1_gain": recovery["f1_gain"],
                "f1_recovery_pct": recovery["f1_recovery_pct"],
            })
        except Exception as exc:
            rows.append({"ablation": "adaptation_window", "variant": f"recent_{window}", "model": model_name,
                         "note": f"failed: {type(exc).__name__}: {exc}"})
    return rows


def telemetry_family_ablation(frame: FlowFrame, config: Dict, model_name: str, seed: int) -> List[Dict]:
    """Leave-one-family-out: which feature group the detector actually leans on."""
    target_fpr = float(config.get("metrics", {}).get("target_fpr", 0.05))
    params = model_params(config, model_name)
    split = build_temporal_split(frame, **config.get("temporal", {}))
    train, forward = split.period("train"), split.period("forward")
    available = set(frame.frame.columns)
    everything = list(frame.numeric_features())

    rows: List[Dict] = []
    for family, members in [("all_features", [])] + list(TELEMETRY_FAMILIES.items()):
        dropped = [f for f in members if f in available and f in everything]
        features = [f for f in everything if f not in dropped]
        if not features or len(features) == len(everything):
            continue
        try:
            result = _fit_and_score(train, forward, features, model_name, params, seed, target_fpr, split)
            metrics = result["metrics"]
            rows.append({
                "ablation": "telemetry_family", "variant": f"without_{family}", "model": model_name,
                "dropped": dropped, "n_features": len(features),
                "f1": metrics["f1"], "roc_auc": metrics["roc_auc"], "pr_auc": metrics["pr_auc"],
                "recall": metrics["recall"],
                "false_positive_rate": metrics["false_positive_rate"],
            })
        except Exception as exc:
            rows.append({"ablation": "telemetry_family", "variant": f"without_{family}",
                         "model": model_name, "note": f"failed: {type(exc).__name__}: {exc}"})
    return rows


def target_fpr_ablation(frame: FlowFrame, config: Dict, model_name: str, seed: int) -> List[Dict]:
    """What the false-positive budget costs in recall.

    A detector is only comparable to another under a stated operating point, so
    this records the full trade rather than a single chosen FPR.
    """
    from driftguard.evaluation.metrics import metrics_at_target_fpr

    params = model_params(config, model_name)
    split = build_temporal_split(frame, **config.get("temporal", {}))
    train, forward = split.period("train"), split.period("forward")
    features = list(frame.numeric_features())
    pre = _fit_preprocessor(train, features, split)
    model = build_model(model_name, params, seed)
    model.fit(pre.transform(train).to_numpy(), train.targets.to_numpy())
    scores = _score(pre, forward, model)
    y = forward.targets.to_numpy()

    rows: List[Dict] = []
    for fpr in (0.01, 0.02, 0.05, 0.10, 0.20):
        try:
            metrics = metrics_at_target_fpr(y, scores, fpr)
            rows.append({
                "ablation": "target_fpr", "variant": f"fpr_{fpr}", "model": model_name,
                "target_fpr": fpr, "threshold": metrics.get("threshold"),
                "recall": metrics["recall"], "precision": metrics["precision"],
                "f1": metrics["f1"], "achieved_fpr": metrics["false_positive_rate"],
            })
        except Exception as exc:
            rows.append({"ablation": "target_fpr", "variant": f"fpr_{fpr}", "model": model_name,
                         "note": f"failed: {type(exc).__name__}: {exc}"})
    return rows


ABLATIONS = {
    "split_mode": split_mode_ablation,
    "feature_policy": feature_policy_ablation,
    "drift_threshold": drift_threshold_ablation,
    "adaptation_window": adaptation_window_ablation,
    "telemetry_family": telemetry_family_ablation,
    "target_fpr": target_fpr_ablation,
}


def run_ablations(
    config: Dict,
    out_dir: Path,
    experiment_id: str,
    ablations: Optional[Sequence[str]] = None,
    models: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Run the requested ablations and write one table plus the notes."""
    frame = _load(config)
    seed = int(config.get("seed", 42))
    models = list(models or config.get("ablation", {}).get("models", ["logistic_regression", "random_forest"]))
    names = list(ablations or ABLATIONS)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    for model_name in models:
        for name in names:
            fn = ABLATIONS.get(name)
            if fn is None:
                raise KeyError(f"unknown ablation {name!r}; available: {sorted(ABLATIONS)}")
            started = time.perf_counter()
            print(f"  ablation {name} / {model_name} ...", flush=True)
            try:
                produced = fn(frame, config, model_name, seed)
            except Exception as exc:
                produced = [{"ablation": name, "variant": "all", "model": model_name,
                             "note": f"failed: {type(exc).__name__}: {exc}"}]
            for row in produced:
                row.setdefault("experiment_id", experiment_id)
                row.setdefault("seconds", round(time.perf_counter() - started, 2))
            rows.extend(produced)
            print(f"    {len(produced)} rows in {time.perf_counter() - started:.1f}s", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "ablation_results.csv", index=False)
    (out_dir / "ablation_results.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8"
    )
    return {
        "experiment_id": experiment_id,
        "ablations": names,
        "models": models,
        "rows": len(rows),
        "failed": sum(1 for r in rows if r.get("note", "").startswith("failed")),
    }
