"""Data validation module for UNSW-NB15 dataset."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml


class DataValidator:
    """Validates UNSW-NB15 dataset structure and content."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize validator with configuration."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.raw_dir = Path(self.config["dataset"]["raw_dir"])
        self.target_column = self.config["target"]["column"]
        self.excluded_features = self.config["excluded_features"]
        self.categorical_features = self.config["categorical_features"]

        # Expected columns from UNSW-NB15 features description
        self.expected_columns = [
            "srcip", "dstip", "sport", "dsport", "proto", "state", "dur",
            "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
            "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb",
            "tcprtt", "synack", "ackdat", "smean", "dmean", "trans_depth",
            "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
            "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
            "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm",
            "ct_srv_dst", "is_sm_ips_ports", "attack_cat", "label"
        ]

    def load_dataset(self, filename: str) -> pd.DataFrame:
        """Load a dataset file."""
        filepath = self.raw_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        df = pd.read_csv(filepath, low_memory=False)
        return df

    def validate_structure(self, df: pd.DataFrame, name: str) -> Dict:
        """Validate dataset structure."""
        results = {
            "name": name,
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "issues": []
        }

        # Check expected columns
        missing_cols = set(self.expected_columns) - set(df.columns)
        extra_cols = set(df.columns) - set(self.expected_columns)

        if missing_cols:
            results["issues"].append(f"Missing expected columns: {missing_cols}")
        if extra_cols:
            results["issues"].append(f"Extra columns not in schema: {extra_cols}")

        # Check target column
        if self.target_column not in df.columns:
            results["issues"].append(f"Target column '{self.target_column}' not found")

        # Check for completely empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            results["issues"].append(f"Completely empty columns: {empty_cols}")

        return results

    def validate_target(self, df: pd.DataFrame) -> Dict:
        """Validate target column values."""
        results = {
            "unique_values": df[self.target_column].unique().tolist(),
            "value_counts": df[self.target_column].value_counts().to_dict(),
            "issues": []
        }

        # Check for unexpected values
        expected_values = {self.config["target"]["normal_value"], self.config["target"]["anomaly_value"]}
        actual_values = set(df[self.target_column].unique())

        unexpected = actual_values - expected_values
        if unexpected:
            results["issues"].append(f"Unexpected target values: {unexpected}")

        missing = expected_values - actual_values
        if missing:
            results["issues"].append(f"Missing expected target values: {missing}")

        return results

    def validate_features(self, df: pd.DataFrame) -> Dict:
        """Validate feature columns."""
        results = {
            "numeric_features": [],
            "categorical_features": [],
            "excluded_features_present": [],
            "issues": []
        }

        for col in df.columns:
            if col in self.excluded_features:
                results["excluded_features_present"].append(col)
            elif col == self.target_column:
                continue
            elif col in self.categorical_features:
                results["categorical_features"].append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                results["numeric_features"].append(col)
            else:
                results["categorical_features"].append(col)

        return results

    def run_full_validation(self) -> Dict:
        """Run complete validation on both train and test sets."""
        results = {}

        for split in ["train", "test"]:
            filename = self.config["dataset"][f"{split}_file"]
            print(f"\nValidating {split} set: {filename}")

            try:
                df = self.load_dataset(filename)

                results[split] = {
                    "structure": self.validate_structure(df, f"{split}_set"),
                    "target": self.validate_target(df),
                    "features": self.validate_features(df),
                }

                # Print summary
                struct = results[split]["structure"]
                print(f"  Rows: {struct['shape'][0]:,}, Columns: {struct['shape'][1]}")
                print(f"  Duplicates: {struct['duplicate_rows']}")
                print(f"  Target distribution: {results[split]['target']['value_counts']}")

                if struct["issues"]:
                    print(f"  Issues: {struct['issues']}")

            except Exception as e:
                results[split] = {"error": str(e)}
                print(f"  Error: {e}")

        return results


def main():
    """Main entry point for data validation."""
    validator = DataValidator()
    results = validator.run_full_validation()

    # Save validation report
    import json
    output_path = Path("results/metrics/validation_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        # Convert numpy types to native Python types
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        json.dump(convert(results), f, indent=2)

    print(f"\nValidation report saved to: {output_path}")


if __name__ == "__main__":
    main()