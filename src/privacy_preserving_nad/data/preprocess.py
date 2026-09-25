"""Data preprocessing module for UNSW-NB15 dataset."""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """Handles preprocessing of UNSW-NB15 dataset."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize preprocessor with configuration."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.raw_dir = Path(self.config["dataset"]["raw_dir"])
        self.processed_dir = Path(self.config["dataset"]["processed_dir"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.target_column = self.config["target"]["column"]
        self.excluded_features = self.config["excluded_features"]
        self.categorical_features = self.config["categorical_features"]
        self.feature_sets = self.config["feature_sets"]
        self.random_seed = self.config["models"]["random_seed"]

        self.preprocessing_config = self.config["preprocessing"]
        self.scaler = None
        self.encoder = None
        self.column_transformer = None
        self.feature_names_out = None

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw training and testing data."""
        train_file = self.config["dataset"]["train_file"]
        test_file = self.config["dataset"]["test_file"]

        train_df = pd.read_csv(self.raw_dir / train_file, low_memory=False)
        test_df = pd.read_csv(self.raw_dir / test_file, low_memory=False)

        return train_df, test_df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataset: handle missing values, remove duplicates."""
        df = df.copy()

        # Remove duplicate rows
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            print(f"  Removed {initial_rows - len(df)} duplicate rows")

        # Handle missing values
        missing_before = df.isnull().sum().sum()
        if missing_before > 0:
            print(f"  Found {missing_before} missing values")

            # For numeric columns, fill with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)

            # For categorical columns, fill with mode
            categorical_cols = df.select_dtypes(include=["object"]).columns
            for col in categorical_cols:
                if df[col].isnull().any():
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else "unknown"
                    df[col] = df[col].fillna(mode_val)

            missing_after = df.isnull().sum().sum()
            print(f"  Missing values after cleaning: {missing_after}")

        return df

    def prepare_target(self, df: pd.DataFrame) -> pd.Series:
        """Convert target to binary classification (0=normal, 1=anomaly)."""
        # The label column should already be 0/1, but ensure it's clean
        y = df[self.target_column].copy()
        y = y.astype(int)
        return y

    def select_features(self, df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
        """Select features based on the specified feature set."""
        if feature_set not in self.feature_sets:
            raise ValueError(f"Unknown feature set: {feature_set}")

        selected_features = self.feature_sets[feature_set]

        # Check which features are actually available
        available_features = [f for f in selected_features if f in df.columns]
        missing_features = [f for f in selected_features if f not in df.columns]

        if missing_features:
            print(f"  Warning: Missing features in dataset: {missing_features}")

        X = df[available_features].copy()
        return X

    def identify_feature_types(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identify numeric and categorical feature columns."""
        numeric_features = []
        categorical_features = []

        for col in X.columns:
            if col in self.categorical_features:
                categorical_features.append(col)
            elif pd.api.types.is_numeric_dtype(X[col]):
                numeric_features.append(col)
            else:
                categorical_features.append(col)

        return numeric_features, categorical_features

    def build_preprocessing_pipeline(self, X: pd.DataFrame) -> ColumnTransformer:
        """Build the preprocessing pipeline for features."""
        numeric_features, categorical_features = self.identify_feature_types(X)

        print(f"  Numeric features ({len(numeric_features)}): {numeric_features}")
        print(f"  Categorical features ({len(categorical_features)}): {categorical_features}")

        # Numeric preprocessing: impute + scale
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()) if self.preprocessing_config.get("scale_numeric", True) else ("passthrough", "passthrough")
        ])

        # Categorical preprocessing: impute + one-hot encode
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features)
            ],
            remainder="drop"
        )

        return preprocessor, numeric_features, categorical_features

    def fit_transform(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit preprocessor on training data and transform both sets."""
        self.column_transformer, numeric_features, categorical_features = self.build_preprocessing_pipeline(X_train)

        print("  Fitting preprocessor on training data...")
        X_train_transformed = self.column_transformer.fit_transform(X_train)
        X_test_transformed = self.column_transformer.transform(X_test)

        # Get feature names after transformation
        self.feature_names_out = self._get_feature_names_out(numeric_features, categorical_features)

        print(f"  Transformed shape - Train: {X_train_transformed.shape}, Test: {X_test_transformed.shape}")
        print(f"  Output features: {len(self.feature_names_out)}")

        return X_train_transformed, X_test_transformed

    def _get_feature_names_out(self, numeric_features: List[str], categorical_features: List[str]) -> List[str]:
        """Get feature names after transformation."""
        feature_names = []

        # Numeric features keep their names
        feature_names.extend(numeric_features)

        # Categorical features get one-hot encoded names
        if categorical_features:
            cat_encoder = self.column_transformer.named_transformers_["cat"].named_steps["encoder"]
            cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
            feature_names.extend(cat_feature_names)

        return feature_names

    def save_preprocessor(self, feature_set: str) -> Path:
        """Save the fitted preprocessor."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.processed_dir / f"preprocessor_{feature_set}.joblib"
        joblib.dump(self.column_transformer, filepath)
        print(f"  Saved preprocessor to: {filepath}")
        return filepath

    def load_preprocessor(self, feature_set: str) -> ColumnTransformer:
        """Load a fitted preprocessor."""
        filepath = self.processed_dir / f"preprocessor_{feature_set}.joblib"
        self.column_transformer = joblib.load(filepath)

        # Rebuild feature names (need to know original feature types)
        # This is a limitation - in practice, we'd save feature names too
        return self.column_transformer

    def process_feature_set(self, feature_set: str) -> Dict:
        """Process a complete feature set: load, clean, split, transform."""
        print(f"\nProcessing feature set: {feature_set}")

        # Load raw data
        train_df, test_df = self.load_raw_data()

        # Clean data
        print("  Cleaning training data...")
        train_df = self.clean_data(train_df)
        print("  Cleaning test data...")
        test_df = self.clean_data(test_df)

        # Select features
        print(f"  Selecting features for {feature_set}...")
        X_train = self.select_features(train_df, feature_set)
        X_test = self.select_features(test_df, feature_set)

        # Prepare targets
        y_train = self.prepare_target(train_df)
        y_test = self.prepare_target(test_df)

        print(f"  Train class distribution: {y_train.value_counts().to_dict()}")
        print(f"  Test class distribution: {y_test.value_counts().to_dict()}")

        # Transform features
        X_train_transformed, X_test_transformed = self.fit_transform(X_train, X_test)

        # Save processed data
        self._save_processed_data(feature_set, X_train_transformed, X_test_transformed,
                                  y_train, y_test, self.feature_names_out)

        # Save preprocessor
        self.save_preprocessor(feature_set)

        return {
            "feature_set": feature_set,
            "n_train": len(X_train_transformed),
            "n_test": len(X_test_transformed),
            "n_features": len(self.feature_names_out),
            "train_class_dist": y_train.value_counts().to_dict(),
            "test_class_dist": y_test.value_counts().to_dict(),
            "feature_names": self.feature_names_out
        }

    def _save_processed_data(self, feature_set: str, X_train: np.ndarray, X_test: np.ndarray,
                             y_train: pd.Series, y_test: pd.Series, feature_names: List[str]):
        """Save processed data to disk."""
        feature_dir = self.processed_dir / feature_set
        feature_dir.mkdir(parents=True, exist_ok=True)

        # Save as numpy arrays for efficiency
        np.save(feature_dir / "X_train.npy", X_train)
        np.save(feature_dir / "X_test.npy", X_test)
        np.save(feature_dir / "y_train.npy", y_train.values)
        np.save(feature_dir / "y_test.npy", y_test.values)

        # Save feature names
        import json
        with open(feature_dir / "feature_names.json", "w") as f:
            json.dump(feature_names, f)

        print(f"  Saved processed data to: {feature_dir}")


def load_processed_data(feature_set: str, processed_dir: str = "data/processed") -> Dict:
    """Load processed data for a feature set."""
    feature_dir = Path(processed_dir) / feature_set

    X_train = np.load(feature_dir / "X_train.npy")
    X_test = np.load(feature_dir / "X_test.npy")
    y_train = np.load(feature_dir / "y_train.npy")
    y_test = np.load(feature_dir / "y_test.npy")

    with open(feature_dir / "feature_names.json", "r") as f:
        feature_names = json.load(f)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names
    }


def main():
    """Main entry point for preprocessing."""
    import json

    preprocessor = DataPreprocessor()

    results = {}
    for feature_set in ["FULL_METADATA", "RESTRICTED_METADATA"]:
        try:
            results[feature_set] = preprocessor.process_feature_set(feature_set)
        except Exception as e:
            print(f"Error processing {feature_set}: {e}")
            results[feature_set] = {"error": str(e)}

    # Save preprocessing report
    output_path = Path("results/metrics/preprocessing_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nPreprocessing report saved to: {output_path}")


if __name__ == "__main__":
    main()