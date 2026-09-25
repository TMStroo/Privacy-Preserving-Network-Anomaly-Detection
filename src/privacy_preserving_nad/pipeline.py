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
from privacy_preserving_nad.models.train import train_all_models
from privacy_preserving_nad.models.predict import run_all_inference
from privacy_preserving_nad.evaluation.metrics import load_evaluation_results, print_summary, save_metrics_csv, save_metrics_json
from privacy_preserving_nad.evaluation.plots import generate_all_plots


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """Load configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_validation(config: Dict) -> Dict[str, Any]:
    """Step 1: Validate dataset."""
    print("\n" + "=" * 60)
    print("STEP 1: DATA VALIDATION")
    print("=" * 60)

    validator = DataValidator()
    results = validator.run_full_validation()
    return results


def run_preprocessing(config: Dict) -> Dict[str, Any]:
    """Step 2: Preprocess data for both feature sets."""
    print("\n" + "=" * 60)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 60)

    preprocessor = DataPreprocessor()
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

    for feature_set in ["FULL_METADATA", "RESTRICTED_METADATA"]:
        try:
            all_results[feature_set] = train_all_models(feature_set, config["models"])
        except Exception as e:
            print(f"Error training on {feature_set}: {e}")
            all_results[feature_set] = {"error": str(e)}

    return all_results


def run_evaluation(config: Dict) -> Dict[str, Any]:
    """Step 4: Evaluate all models."""
    print("\n" + "=" * 60)
    print("STEP 4: MODEL EVALUATION")
    print("=" * 60)

    results = run_all_inference()
    return results


def run_plotting(config: Dict, eval_results: Dict) -> Dict[str, Any]:
    """Step 5: Generate plots."""
    print("\n" + "=" * 60)
    print("STEP 5: GENERATING PLOTS")
    print("=" * 60)

    # Load feature names
    feature_names = {}
    for fs in ["FULL_METADATA", "RESTRICTED_METADATA"]:
        try:
            import json
            with open(f"data/processed/{fs}/feature_names.json", "r") as f:
                feature_names[fs] = json.load(f)
        except Exception:
            feature_names[fs] = []

    plot_paths = generate_all_plots(eval_results, feature_names)
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

    # Step 1: Validation
    try:
        pipeline_results["validation"] = run_validation(config)
    except Exception as e:
        print(f"Validation failed: {e}")
        pipeline_results["validation"] = {"error": str(e)}

    # Step 2: Preprocessing
    try:
        pipeline_results["preprocessing"] = run_preprocessing(config)
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
        print_summary(pipeline_results["evaluation"])
        save_metrics_csv(pipeline_results["evaluation"])
        save_metrics_json(pipeline_results["evaluation"])

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
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

    config = load_config(args.config)

    # Set random seeds
    import numpy as np
    import random
    seed = config["models"]["random_seed"]
    np.random.seed(seed)
    random.seed(seed)

    if args.step == "validate" or args.step == "all":
        run_validation(config)

    if args.step == "preprocess" or args.step == "all":
        run_preprocessing(config)

    if args.step == "train" or args.step == "all":
        run_training(config)

    if args.step == "evaluate" or args.step == "all":
        results = run_evaluation(config)
        print_summary(results)
        save_metrics_csv(results)
        save_metrics_json(results)

    if args.step == "plot" or args.step == "all":
        if args.step != "all":
            # Load evaluation results if running plot alone
            from src.evaluation.metrics import load_evaluation_results
            results = load_evaluation_results()
        else:
            results = pipeline_results.get("evaluation", {})
        run_plotting(config, results)


if __name__ == "__main__":
    main()