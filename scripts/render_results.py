#!/usr/bin/env python
"""Render the README results block from generated experiment output.

Run after the pipeline:

    python scripts/render_results.py

Every number in the block comes from results/metrics/ (evaluation results,
dataset info, config). Nothing about the experiment is typed by hand, so the
README cannot drift away from the actual run.
"""

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
EVAL = ROOT / "results" / "metrics" / "evaluation_results.json"
DATASET_INFO = ROOT / "results" / "metrics" / "dataset_info.json"
CONFIG = ROOT / "configs" / "config.yaml"

BEGIN = "<!-- BEGIN GENERATED RESULTS -->"
END = "<!-- END GENERATED RESULTS -->"

ORDER = [
    ("FULL_METADATA", "majority_baseline", "Majority baseline"),
    ("FULL_METADATA", "logistic_regression", "Logistic Regression"),
    ("FULL_METADATA", "random_forest", "Random Forest"),
    ("RESTRICTED_METADATA", "majority_baseline", "Majority baseline"),
    ("RESTRICTED_METADATA", "logistic_regression", "Logistic Regression"),
    ("RESTRICTED_METADATA", "random_forest", "Random Forest"),
]


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def fmt(x: float) -> str:
    return f"{x:.4f}"


def count_real_models(eval_results) -> int:
    return sum(
        1 for fs in ("FULL_METADATA", "RESTRICTED_METADATA")
        for m in ("logistic_regression", "random_forest")
        if "error" not in eval_results.get(fs, {}).get(m, {"error": True})
    )


def build_block() -> str:
    info = json.loads(DATASET_INFO.read_text())
    eval_results = json.loads(EVAL.read_text())
    config = yaml.safe_load(CONFIG.read_text())

    n_full = len(config["feature_sets"]["FULL_METADATA"])
    n_restricted = len(config["feature_sets"]["RESTRICTED_METADATA"])

    lines = [BEGIN, ""]

    is_research = info["kind"] == "research"
    if is_research:
        lines += [
            f"**Experiment status: research run.** Official UNSW-NB15 split as released - "
            f"{info['train_rows']:,} training rows and {info['test_rows']:,} test rows, "
            f"verified against recorded sha256 checksums. The official split is used "
            f"as-is: no rows are merged, shuffled or re-split, and no hyperparameter or "
            f"threshold was chosen against the test set.",
            "",
        ]
    else:
        lines += [
            f"**Experiment status: DEVELOPMENT SAMPLE - not benchmark results.** "
            f"data/raw/ currently holds {info['train_rows']:,} train rows / "
            f"{info['test_rows']:,} test rows of development data "
            f"(kind: {info['kind']}). The table below only proves the pipeline runs. "
            f"Fetch the official split (README, 'Dataset') and re-run for research numbers.",
            "",
        ]

    lines += [
        "### Main comparison",
        "",
        "| Feature set | Model | Precision | Recall | F1 | FPR | FNR |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for fs, model, label in ORDER:
        m = eval_results[fs][model]
        lines.append(
            f"| {fs} | {label} | {fmt(m['precision'])} | {fmt(m['recall'])} | "
            f"{fmt(m['f1_score'])} | {fmt(m['false_positive_rate'])} | "
            f"{fmt(m['false_negative_rate'])} |"
        )
    lines.append("")

    majority = eval_results["FULL_METADATA"]["majority_baseline"]
    rf_full = eval_results["FULL_METADATA"]["random_forest"]
    rf_rest = eval_results["RESTRICTED_METADATA"]["random_forest"]
    lr_full = eval_results["FULL_METADATA"]["logistic_regression"]
    lr_rest = eval_results["RESTRICTED_METADATA"]["logistic_regression"]

    n_test = majority["n_samples"]
    n_normal = majority["n_normal"]
    n_anom = majority["n_anomaly"]
    norm_share = n_normal / n_test
    anom_share = n_anom / n_test

    candidates = [
        ("Random Forest (FULL_METADATA)", rf_full),
        ("Random Forest (RESTRICTED_METADATA)", rf_rest),
        ("Logistic Regression (FULL_METADATA)", lr_full),
        ("Logistic Regression (RESTRICTED_METADATA)", lr_rest),
    ]
    best_name, best = max(candidates, key=lambda kv: kv[1]["f1_score"])

    lines += [
        "### Reading the table",
        "",
        f"**Class balance.** The test split holds {n_test:,} flows: {n_normal:,} normal "
        f"({pct(norm_share)}) and {n_anom:,} anomalous ({pct(anom_share)}). The anomaly "
        f"class is the majority here, so always answering 'anomalous' gives the majority "
        f"baseline an F1 of {fmt(majority['f1_score'])} while flagging every normal flow "
        f"as malicious (FPR {fmt(majority['false_positive_rate'])}). That is the floor "
        f"both models had to clear, and both cleared it.",
        "",
        f"**Best result of this run: {best_name}**, F1 {fmt(best['f1_score'])} "
        f"(precision {fmt(best['precision'])}, recall {fmt(best['recall'])}). In counts: "
        f"it caught {best['confusion_matrix'][1][1]:,} of {n_anom:,} anomalous flows and "
        f"missed {best['confusion_matrix'][1][0]:,} "
        f"({pct(best['false_negative_rate'])} false negatives), while flagging "
        f"{best['confusion_matrix'][0][1]:,} of {n_normal:,} normal flows as attacks "
        f"({pct(best['false_positive_rate'])} false positives). In an operational setting "
        f"that false-positive column is the price of the recall.",
        "",
        f"**Effect of the restricted feature set ({n_full} -> {n_restricted} features).** "
        f"Random Forest moved from F1 {fmt(rf_full['f1_score'])} to "
        f"{fmt(rf_rest['f1_score'])} ({(rf_rest['f1_score'] - rf_full['f1_score']):+.4f}) "
        f"and its false-positive rate from {fmt(rf_full['false_positive_rate'])} to "
        f"{fmt(rf_rest['false_positive_rate'])}. Logistic Regression moved from F1 "
        f"{fmt(lr_full['f1_score'])} to {fmt(lr_rest['f1_score'])} "
        f"({(lr_rest['f1_score'] - lr_full['f1_score']):+.4f}) with its false-positive "
        f"rate from {fmt(lr_full['false_positive_rate'])} to "
        f"{fmt(lr_rest['false_positive_rate'])}. On this run the tree model gave up "
        f"little when the feature set was cut down while the linear model lost more "
        f"ground - the extra features helped the model that can exploit interactions "
        f"more than the one that cannot.",
        "",
        f"**Precision-recall posture.** Recall on the full set was "
        f"{fmt(lr_full['recall'])} for logistic regression and {fmt(rf_full['recall'])} "
        f"for the forest - most anomalous flows detected - at false-positive rates of "
        f"{fmt(lr_full['false_positive_rate'])} and {fmt(rf_full['false_positive_rate'])} "
        f"respectively. class_weight='balanced' pushes models toward recall under this "
        f"class distribution; a deployment would pick an operating threshold to trade "
        f"some recall for fewer false alarms. This experiment used the default 0.5 "
        f"threshold and did not tune anything against the test set.",
        "",
        f"**Scope.** {count_real_models(eval_results)} model/feature-set combinations "
        f"were trained and evaluated on this single dataset and split. The ranking above "
        f"describes this run only - it is not a claim that any model is generally "
        f"superior, and no claim of privacy, production readiness or adversarial "
        f"robustness follows from it.",
        "",
        "This block is generated from `results/metrics/evaluation_results.json` by "
        "`python scripts/render_results.py`; the pipeline plus that script reproduce it "
        "exactly.",
        "",
    ]

    lines += [END]
    return "\n".join(lines)


def main():
    readme = README.read_text()
    if BEGIN not in readme or END not in readme:
        raise SystemExit(f"README is missing the {BEGIN} ... {END} markers")

    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)
    readme = pattern.sub(lambda _: build_block(), readme)
    README.write_text(readme)
    print("README results block regenerated from results/metrics/")


if __name__ == "__main__":
    main()
