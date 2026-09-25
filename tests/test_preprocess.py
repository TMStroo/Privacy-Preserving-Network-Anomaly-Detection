"""Tests for data preprocessing module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import yaml
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from privacy_preserving_nad.data.preprocess import DataPreprocessor, load_processed_data


@pytest.fixture
def sample_config():
    """Create a sample config for testing."""
    return {
        "dataset": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "train_file": "train.csv",
            "test_file": "test.csv",
            "features_file": "features.csv"
        },
        "target": {
            "column": "label",
            "normal_value": 0,
            "anomaly_value": 1
        },
        "excluded_features": ["srcip", "dstip", "sport", "dsport", "srcpt", "dstpt", "Ltime", "Stime", "attack_cat", "label", "id"],
        "payload_derived_features": ["trans_depth", "response_body_len", "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd"],
        "categorical_features": ["proto", "service", "state"],
        "feature_sets": {
            "FULL_METADATA": [
                "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
                "sttl", "dttl", "sload", "dload", "sloss", "dloss",
                "sinpkt", "dinpkt", "sjit", "djit", "swin", "dwin",
                "stcpb", "dtcpb", "tcprtt", "synack", "ackdat",
                "smean", "dmean",
                "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
                "ct_dst_sport_ltm", "ct_dst_src_ltm", "ct_src_ltm", "ct_srv_dst",
                "is_sm_ips_ports", "proto", "service", "state"
            ],
            "RESTRICTED_METADATA": [
                "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
                "sttl", "dttl", "sload", "dload", "sinpkt", "dinpkt",
                "sjit", "djit", "tcprtt", "synack", "ackdat"
            ]
        },
        "preprocessing": {
            "test_size": 0.2,
            "handle_missing": "median",
            "scale_numeric": True
        },
        "models": {
            "random_seed": 42
        }
    }


@pytest.fixture
def sample_raw_data(sample_config):
    """Create sample raw data files."""
    np.random.seed(42)
    n_train = 500
    n_test = 150

    # Create training data
    train_data = pd.DataFrame({
        "srcip": [f"192.168.1.{i%255}" for i in range(n_train)],
        "dstip": [f"10.0.0.{i%255}" for i in range(n_train)],
        "sport": np.random.randint(1024, 65535, n_train),
        "dsport": np.random.randint(1, 1024, n_train),
        "proto": np.random.choice(["tcp", "udp", "icmp"], n_train),
        "state": np.random.choice(["FIN", "CON", "REQ", "RST"], n_train),
        "dur": np.random.exponential(10, n_train),
        "sbytes": np.random.randint(100, 10000, n_train),
        "dbytes": np.random.randint(100, 10000, n_train),
        "sttl": np.random.randint(32, 255, n_train),
        "dttl": np.random.randint(32, 255, n_train),
        "sloss": np.random.randint(0, 5, n_train),
        "dloss": np.random.randint(0, 5, n_train),
        "service": np.random.choice(["http", "ftp", "dns", "smtp", "-"], n_train),
        "sload": np.random.exponential(1000, n_train),
        "dload": np.random.exponential(1000, n_train),
        "spkts": np.random.randint(1, 100, n_train),
        "dpkts": np.random.randint(1, 100, n_train),
        "swin": np.random.randint(1024, 65535, n_train),
        "dwin": np.random.randint(1024, 65535, n_train),
        "stcpb": np.random.randint(0, 1000, n_train),
        "dtcpb": np.random.randint(0, 1000, n_train),
        "tcprtt": np.random.exponential(0.1, n_train),
        "synack": np.random.exponential(0.05, n_train),
        "ackdat": np.random.exponential(0.05, n_train),
        "smean": np.random.randint(100, 1500, n_train),
        "dmean": np.random.randint(100, 1500, n_train),
        "trans_depth": np.random.randint(0, 10, n_train),
        "response_body_len": np.random.randint(0, 5000, n_train),
        "ct_srv_src": np.random.randint(1, 50, n_train),
        "ct_state_ttl": np.random.randint(1, 50, n_train),
        "ct_dst_ltm": np.random.randint(1, 50, n_train),
        "ct_src_dport_ltm": np.random.randint(1, 50, n_train),
        "ct_dst_sport_ltm": np.random.randint(1, 50, n_train),
        "ct_dst_src_ltm": np.random.randint(1, 50, n_train),
        "is_ftp_login": np.random.randint(0, 2, n_train),
        "ct_ftp_cmd": np.random.randint(0, 10, n_train),
        "ct_flw_http_mthd": np.random.randint(0, 10, n_train),
        "ct_src_ltm": np.random.randint(1, 50, n_train),
        "ct_srv_dst": np.random.randint(1, 50, n_train),
        "is_sm_ips_ports": np.random.randint(0, 2, n_train),
        "attack_cat": np.random.choice(["Normal", "DoS", "Probe", "Privilege"], n_train),
        "label": np.random.randint(0, 2, n_train)
    })

    # Create test data
    test_data = pd.DataFrame({
        "srcip": [f"192.168.1.{i%255}" for i in range(n_test)],
        "dstip": [f"10.0.0.{i%255}" for i in range(n_test)],
        "sport": np.random.randint(1024, 65535, n_test),
        "dsport": np.random.randint(1, 1024, n_test),
        "proto": np.random.choice(["tcp", "udp", "icmp"], n_test),
        "state": np.random.choice(["FIN", "CON", "REQ", "RST"], n_test),
        "dur": np.random.exponential(10, n_test),
        "sbytes": np.random.randint(100, 10000, n_test),
        "dbytes": np.random.randint(100, 10000, n_test),
        "sttl": np.random.randint(32, 255, n_test),
        "dttl": np.random.randint(32, 255, n_test),
        "sloss": np.random.randint(0, 5, n_test),
        "dloss": np.random.randint(0, 5, n_test),
        "service": np.random.choice(["http", "ftp", "dns", "smtp", "-"], n_test),
        "sload": np.random.exponential(1000, n_test),
        "dload": np.random.exponential(1000, n_test),
        "spkts": np.random.randint(1, 100, n_test),
        "dpkts": np.random.randint(1, 100, n_test),
        "swin": np.random.randint(1024, 65535, n_test),
        "dwin": np.random.randint(1024, 65535, n_test),
        "stcpb": np.random.randint(0, 1000, n_test),
        "dtcpb": np.random.randint(0, 1000, n_test),
        "tcprtt": np.random.exponential(0.1, n_test),
        "synack": np.random.exponential(0.05, n_test),
        "ackdat": np.random.exponential(0.05, n_test),
        "smean": np.random.randint(100, 1500, n_test),
        "dmean": np.random.randint(100, 1500, n_test),
        "trans_depth": np.random.randint(0, 10, n_test),
        "response_body_len": np.random.randint(0, 5000, n_test),
        "ct_srv_src": np.random.randint(1, 50, n_test),
        "ct_state_ttl": np.random.randint(1, 50, n_test),
        "ct_dst_ltm": np.random.randint(1, 50, n_test),
        "ct_src_dport_ltm": np.random.randint(1, 50, n_test),
        "ct_dst_sport_ltm": np.random.randint(1, 50, n_test),
        "ct_dst_src_ltm": np.random.randint(1, 50, n_test),
        "is_ftp_login": np.random.randint(0, 2, n_test),
        "ct_ftp_cmd": np.random.randint(0, 10, n_test),
        "ct_flw_http_mthd": np.random.randint(0, 10, n_test),
        "ct_src_ltm": np.random.randint(1, 50, n_test),
        "ct_srv_dst": np.random.randint(1, 50, n_test),
        "is_sm_ips_ports": np.random.randint(0, 2, n_test),
        "attack_cat": np.random.choice(["Normal", "DoS", "Probe", "Privilege"], n_test),
        "label": np.random.randint(0, 2, n_test)
    })

    return train_data, test_data


class TestDataPreprocessor:
    """Test DataPreprocessor class."""

    def test_preprocessor_init(self, sample_config, tmp_path):
        """Test preprocessor initialization."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        preprocessor = DataPreprocessor(str(config_file))
        assert preprocessor.target_column == "label"
        assert "FULL_METADATA" in preprocessor.feature_sets
        assert "RESTRICTED_METADATA" in preprocessor.feature_sets

    def test_clean_data(self, sample_config, sample_raw_data, tmp_path):
        """clean_data tidies rows but never drops rows or fills values.

        Imputation belongs to the sklearn pipeline so that statistics are
        fitted on the training split only.
        """
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        preprocessor = DataPreprocessor(str(config_file))
        train_data, _ = sample_raw_data

        dirty_data = train_data.copy()
        dirty_data.loc[0:5, "dur"] = np.nan
        dirty_data.loc[10:12, "proto"] = "  tcp  "
        dirty_data = pd.concat([dirty_data, dirty_data.iloc[:3]], ignore_index=True)

        clean_data = preprocessor.clean_data(dirty_data)

        assert len(clean_data) == len(dirty_data)
        # 6 injected NaNs plus their 3 duplicated copies are all preserved
        assert clean_data["dur"].isna().sum() == 9
        assert (clean_data["proto"].iloc[10:13] == "tcp").all()

    def test_prepare_target(self, sample_config, sample_raw_data, tmp_path):
        """Test target preparation."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        preprocessor = DataPreprocessor(str(config_file))
        train_data, _ = sample_raw_data

        y = preprocessor.prepare_target(train_data)

        assert y.dtype == np.int64
        assert set(y.unique()).issubset({0, 1})
        assert len(y) == len(train_data)

    def test_select_features(self, sample_config, sample_raw_data, tmp_path):
        """Test feature selection."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        preprocessor = DataPreprocessor(str(config_file))
        train_data, _ = sample_raw_data

        # Test FULL_METADATA
        X_full = preprocessor.select_features(train_data, "FULL_METADATA")
        assert "dur" in X_full.columns
        assert "proto" in X_full.columns
        assert "srcip" not in X_full.columns  # Excluded
        assert "attack_cat" not in X_full.columns  # Excluded

        # Test RESTRICTED_METADATA
        X_restricted = preprocessor.select_features(train_data, "RESTRICTED_METADATA")
        assert "dur" in X_restricted.columns
        assert "proto" not in X_restricted.columns  # Not in restricted set
        assert len(X_restricted.columns) < len(X_full.columns)

    def test_identify_feature_types(self, sample_config, sample_raw_data, tmp_path):
        """Test feature type identification."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        preprocessor = DataPreprocessor(str(config_file))
        train_data, _ = sample_raw_data

        X_full = preprocessor.select_features(train_data, "FULL_METADATA")
        numeric, categorical = preprocessor.identify_feature_types(X_full)

        assert "dur" in numeric
        assert "sbytes" in numeric
        assert "proto" in categorical
        assert "service" in categorical
        assert "state" in categorical

    def test_build_preprocessing_pipeline(self, sample_config, sample_raw_data, tmp_path):
        """Test preprocessing pipeline building."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        preprocessor = DataPreprocessor(str(config_file))
        train_data, _ = sample_raw_data

        X_full = preprocessor.select_features(train_data, "FULL_METADATA")
        pipeline, numeric_features, categorical_features = preprocessor.build_preprocessing_pipeline(X_full)

        assert pipeline is not None
        assert len(numeric_features) > 0
        assert len(categorical_features) > 0
        assert "proto" in categorical_features

    def test_fit_transform(self, sample_config, sample_raw_data, tmp_path):
        """Test fit and transform."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        preprocessor = DataPreprocessor(str(config_file))
        train_data, test_data = sample_raw_data

        X_train = preprocessor.select_features(train_data, "RESTRICTED_METADATA")
        X_test = preprocessor.select_features(test_data, "RESTRICTED_METADATA")

        X_train_trans, X_test_trans = preprocessor.fit_transform(X_train, X_test)

        assert X_train_trans.shape[0] == len(X_train)
        assert X_test_trans.shape[0] == len(X_test)
        assert X_train_trans.shape[1] == X_test_trans.shape[1]
        assert not np.isnan(X_train_trans).any()
        assert not np.isnan(X_test_trans).any()

    def test_save_load_preprocessor(self, sample_config, sample_raw_data, tmp_path):
        """Test saving and loading preprocessor."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        preprocessor = DataPreprocessor(str(config_file))
        preprocessor.processed_dir = tmp_path / "processed"
        train_data, test_data = sample_raw_data

        X_train = preprocessor.select_features(train_data, "RESTRICTED_METADATA")
        X_test = preprocessor.select_features(test_data, "RESTRICTED_METADATA")

        preprocessor.fit_transform(X_train, X_test)
        saved_path = preprocessor.save_preprocessor("RESTRICTED_METADATA")

        assert saved_path.exists()

        # Load and verify
        loaded = preprocessor.load_preprocessor("RESTRICTED_METADATA")
        assert loaded is not None


class TestLoadProcessedData:
    """Test loading processed data."""

    def test_load_processed_data(self, sample_config, sample_raw_data, tmp_path):
        """Test loading processed data from disk."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        # Save sample data to raw_dir for load_raw_data to find
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        train_data, test_data = sample_raw_data
        train_data.to_csv(raw_dir / "train.csv", index=False)
        test_data.to_csv(raw_dir / "test.csv", index=False)

        preprocessor = DataPreprocessor(str(config_file))
        preprocessor.raw_dir = raw_dir
        preprocessor.processed_dir = tmp_path / "processed"

        # Process and save
        result = preprocessor.process_feature_set("RESTRICTED_METADATA")

        # Load using utility function
        loaded = load_processed_data("RESTRICTED_METADATA", str(preprocessor.processed_dir))

        assert "X_train" in loaded
        assert "X_test" in loaded
        assert "y_train" in loaded
        assert "y_test" in loaded
        assert "feature_names" in loaded
        assert loaded["X_train"].shape[0] == result["n_train"]
        assert loaded["X_test"].shape[0] == result["n_test"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])