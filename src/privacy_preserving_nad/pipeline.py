"""Main pipeline for Privacy-Preserving Network Anomaly Detection.

This module orchestrates the complete experimental pipeline:
1. Data validation
2. Preprocessing (for both feature sets)
3. Model training
4. Evaluation
5. Plotting
"""

import sys
import yaml
from pathlib import Path
from typing import Dict, Any

from privacy_preserving_nad.data.validate import DataValidator
from privacy_preserving_nad.data.preprocess import DataPreprocessor
from privacy_preserving_nad.data.dataset import (
    detect_dataset,
    is_research_dataset,
    print_dataset_banner,
    save_dataset_info,
)
from privacy_preserving_nad.models.train import train_all_models
from privacy_preserving_nad.models.predict import run_all_inference
from privacy_preserving_nad.evaluation.metrics import load_evaluation_results, print_summary, save_metrics_csv, save_metrics_json
from privacy_preserving_nad.evaluation.plots import generate_all_plots


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """Load configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_validation(config_path: str) -> Dict[str, Any]:
    """Step 1: Validate dataset."""
    print("\n" + "=" * 60)
    print("STEP 1: DATA VALIDATION")
    print("=" * 60)

    validator = DataValidator(config_path)
    results = validator.run_full_validation()
    return results


def run_preprocessing(config_path: str) -> Dict[str, Any]:
    """Step 2: Preprocess data for both feature sets."""
    print("\n" + "=" * 60)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 60)

    preprocessor = DataPreprocessor(config_path)
    results = {}

    for feature_set in ["FULL_METADATA", "RESTRICTED_METADATA"]:
        try:
            results[feature_set] = preprocessor.process_feature_set(feature_set)
        except Exception as e:
            print(f"Error preprocessing {feature_set}: {e}")
            results[feature_set] = {"error": str(e)}

    return results


def run_training(config: Dict) -> Dict[str, Any]:
    """Step 3: Train all models."""
    print("\n" + "=" * 60)
    print("STEP 3: MODEL TRAINING")
    print("=" * 60)

    all_results = {}
    processed_dir = config["dataset"]["processed_dir"]
    models_dir = config.get("output", {}).get("models_dir", "results/models")

    for feature_set in ["FULL_METADATA", "RESTRICTED_METADATA"]:
        try:
            all_results[feature_set] = train_all_models(
                feature_set, config["models"], processed_dir, models_dir
            )
        except Exception as e:
            print(f"Error training on {feature_set}: {e}")
            all_results[feature_set] = {"error": str(e)}

    return all_results


def run_evaluation(config: Dict) -> Dict[str, Any]:
    """Step 4: Evaluate all models."""
    print("\n" + "=" * 60)
    print("STEP 4: MODEL EVALUATION")
    print("=" * 60)

    processed_dir = config["dataset"]["processed_dir"]
    metrics_dir = config.get("output", {}).get("metrics_dir", "results/metrics")
    models_dir = config.get("output", {}).get("models_dir", "results/models")
    results = run_all_inference(
        model_dir=models_dir,
        processed_dir=processed_dir,
        metrics_path=str(Path(metrics_dir) / "evaluation_results.json"),
    )
    return results


def run_plotting(config: Dict, eval_results: Dict) -> Dict[str, Any]:
    """Step 5: Generate plots."""
    print("\n" + "=" * 60)
    print("STEP 5: GENERATING PLOTS")
    print("=" * 60)

    # Load feature names
    import json

    processed_dir = config["dataset"]["processed_dir"]
    metrics_dir = config.get("output", {}).get("metrics_dir", "results/metrics")
    models_dir = config.get("output", {}).get("models_dir", "results/models")
    figures_dir = config.get("output", {}).get("figures_dir", "results/figures")

    feature_names = {}
    for fs in ["FULL_METADATA", "RESTRICTED_METADATA"]:
        try:
            with open(Path(processed_dir) / fs / "feature_names.json", "r") as f:
                feature_names[fs] = json.load(f)
        except Exception:
            feature_names[fs] = []

    plot_paths = generate_all_plots(
        eval_results, feature_names,
        models_dir=models_dir,
        output_dir=figures_dir,
        processed_dir=processed_dir,
    )
    return plot_paths


def run_full_pipeline(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Run the complete experimental pipeline."""
    print("=" * 60)
    print("PRIVACY-PRESERVING NETWORK ANOMALY DETECTION PIPELINE")
    print("=" * 60)

    config = load_config(config_path)

    # Set random seeds for reproducibility
    import numpy as np
    import random
    seed = config["models"]["random_seed"]
    np.random.seed(seed)
    random.seed(seed)

    pipeline_results = {}
    metrics_dir = config.get("output", {}).get("metrics_dir", "results/metrics")

    # Step 0: identify which dataset is present before touching it
    dataset_info = detect_dataset(config)
    print_dataset_banner(dataset_info)
    dataset_info_path = save_dataset_info(
        dataset_info, str(Path(metrics_dir) / "dataset_info.json")
    )
    pipeline_results["dataset"] = dataset_info
    if dataset_info["kind"] == "missing":
        print("\nNo dataset present - stopping before validation.")
        print("Fetch the research dataset (python -m privacy_preserving_nad.data.download)")
        print("or write a development sample (python scripts/create_sample_data.py).")
        return pipeline_results

    # Step 1: Validation
    try:
        pipeline_results["validation"] = run_validation(config_path)
        # class counts ride along in dataset_info.json so the README block and
        # the PDF report can show the split without re-reading the raw CSVs
        validator = DataValidator(config_path)
        for key, filename in (
            ("train_class_counts", config["dataset"]["train_file"]),
            ("test_class_counts", config["dataset"]["test_file"]),
        ):
            counts = validator.validate_target(validator.load_dataset(filename))
            dataset_info[key] = {
                str(int(k)): int(v) for k, v in counts["value_counts"].items()
            }
        dataset_info_path = save_dataset_info(
            dataset_info, str(Path(metrics_dir) / "dataset_info.json")
        )
    except Exception as e:
        print(f"Validation failed: {e}")
        pipeline_results["validation"] = {"error": str(e)}

    # Step 2: Preprocessing
    try:
        pipeline_results["preprocessing"] = run_preprocessing(config_path)
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        pipeline_results["preprocessing"] = {"error": str(e)}

    # Step 3: Training
    try:
        pipeline_results["training"] = run_training(config)
    except Exception as e:
        print(f"Training failed: {e}")
        pipeline_results["training"] = {"error": str(e)}

    # Step 4: Evaluation
    try:
        pipeline_results["evaluation"] = run_evaluation(config)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        pipeline_results["evaluation"] = {"error": str(e)}

    # Step 5: Plotting
    try:
        pipeline_results["plots"] = run_plotting(config, pipeline_results.get("evaluation", {}))
    except Exception as e:
        print(f"Plotting failed: {e}")
        pipeline_results["plots"] = {"error": str(e)}

    # Print final summary
    if "evaluation" in pipeline_results and "error" not in pipeline_results["evaluation"]:
        metrics_dir = config.get("output", {}).get("metrics_dir", "results/metrics")
        print_summary(pipeline_results["evaluation"])
        save_metrics_csv(
            pipeline_results["evaluation"],
            str(Path(metrics_dir) / "comparison.csv"),
        )
        save_metrics_json(
            pipeline_results["evaluation"],
            str(Path(metrics_dir) / "all_metrics.json"),
        )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    if is_research_dataset(dataset_info):
        print("Dataset: official UNSW-NB15 research split - results are benchmark results.")
    else:
        print("Dataset: DEVELOPMENT SAMPLE - results are a smoke test, not benchmark results.")
    print(f"Dataset info: {dataset_info_path}")
    print("Results saved to: results/")
    print("  - Metrics: results/metrics/")
    print("  - Figures: results/figures/")
    print("  - Models: results/models/")
    print("  - Processed data: data/processed/")

    return pipeline_results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Privacy-Preserving Network Anomaly Detection Pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--step", choices=["validate", "preprocess", "train", "evaluate", "plot", "all"],
                        default="all", help="Pipeline step to run")
    args = parser.parse_args()

    if args.step == "all":
        run_full_pipeline(args.config)
        return

    config = load_config(args.config)

    # Set random seeds
    import numpy as np
    import random
    seed = config["models"]["random_seed"]
    np.random.seed(seed)
    random.seed(seed)

    metrics_dir = config.get("output", {}).get("metrics_dir", "results/metrics")
    dataset_info = detect_dataset(config)
    print_dataset_banner(dataset_info)
    save_dataset_info(dataset_info, str(Path(metrics_dir) / "dataset_info.json"))

    if args.step == "validate":
        run_validation(args.config)
    elif args.step == "preprocess":
        run_preprocessing(args.config)
    elif args.step == "train":
        run_training(config)
    elif args.step == "evaluate":
        results = run_evaluation(config)
        metrics_dir = config.get("output", {}).get("metrics_dir", "results/metrics")
        print_summary(results)
        save_metrics_csv(results, str(Path(metrics_dir) / "comparison.csv"))
        save_metrics_json(results, str(Path(metrics_dir) / "all_metrics.json"))
    elif args.step == "plot":
        run_plotting(config, load_evaluation_results())


if __name__ == "__main__":
    main()