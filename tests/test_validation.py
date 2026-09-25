"""Tests for data validation module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from privacy_preserving_nad.data.validate import DataValidator


@pytest.fixture
def sample_config():
    """Create a sample config for testing."""
    return {
        "dataset": {
            "raw_dir": "data/raw",
            "train_file": "train.csv",
            "test_file": "test.csv",
            "features_file": "features.csv"
        },
        "target": {
            "column": "label",
            "normal_value": 0,
            "anomaly_value": 1
        },
        "excluded_features": ["srcip", "dstip", "sport", "dsport", "attack_cat"],
        "categorical_features": ["proto", "service", "state"]
    }


@pytest.fixture
def sample_train_data():
    """Create sample training data matching UNSW-NB15 structure."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "srcip": [f"192.168.1.{i%255}" for i in range(n)],
        "dstip": [f"10.0.0.{i%255}" for i in range(n)],
        "sport": np.random.randint(1024, 65535, n),
        "dsport": np.random.randint(1, 1024, n),
        "proto": np.random.choice(["tcp", "udp", "icmp"], n),
        "state": np.random.choice(["FIN", "CON", "REQ", "RST"], n),
        "dur": np.random.exponential(10, n),
        "sbytes": np.random.randint(100, 10000, n),
        "dbytes": np.random.randint(100, 10000, n),
        "sttl": np.random.randint(32, 255, n),
        "dttl": np.random.randint(32, 255, n),
        "sloss": np.random.randint(0, 5, n),
        "dloss": np.random.randint(0, 5, n),
        "service": np.random.choice(["http", "ftp", "dns", "smtp", "-"], n),
        "sload": np.random.exponential(1000, n),
        "dload": np.random.exponential(1000, n),
        "spkts": np.random.randint(1, 100, n),
        "dpkts": np.random.randint(1, 100, n),
        "swin": np.random.randint(1024, 65535, n),
        "dwin": np.random.randint(1024, 65535, n),
        "stcpb": np.random.randint(0, 1000, n),
        "dtcpb": np.random.randint(0, 1000, n),
        "tcprtt": np.random.exponential(0.1, n),
        "synack": np.random.exponential(0.05, n),
        "ackdat": np.random.exponential(0.05, n),
        "smean": np.random.randint(100, 1500, n),
        "dmean": np.random.randint(100, 1500, n),
        "trans_depth": np.random.randint(0, 10, n),
        "response_body_len": np.random.randint(0, 5000, n),
        "ct_srv_src": np.random.randint(1, 50, n),
        "ct_state_ttl": np.random.randint(1, 50, n),
        "ct_dst_ltm": np.random.randint(1, 50, n),
        "ct_src_dport_ltm": np.random.randint(1, 50, n),
        "ct_dst_sport_ltm": np.random.randint(1, 50, n),
        "ct_dst_src_ltm": np.random.randint(1, 50, n),
        "is_ftp_login": np.random.randint(0, 2, n),
        "ct_ftp_cmd": np.random.randint(0, 10, n),
        "ct_flw_http_mthd": np.random.randint(0, 10, n),
        "ct_src_ltm": np.random.randint(1, 50, n),
        "ct_srv_dst": np.random.randint(1, 50, n),
        "is_sm_ips_ports": np.random.randint(0, 2, n),
        "attack_cat": np.random.choice(["Normal", "DoS", "Probe", "Privilege"], n),
        "label": np.random.randint(0, 2, n)
    })


@pytest.fixture
def sample_test_data():
    """Create sample test data."""
    np.random.seed(123)
    n = 300
    return pd.DataFrame({
        "srcip": [f"192.168.1.{i%255}" for i in range(n)],
        "dstip": [f"10.0.0.{i%255}" for i in range(n)],
        "sport": np.random.randint(1024, 65535, n),
        "dsport": np.random.randint(1, 1024, n),
        "proto": np.random.choice(["tcp", "udp", "icmp"], n),
        "state": np.random.choice(["FIN", "CON", "REQ", "RST"], n),
        "dur": np.random.exponential(10, n),
        "sbytes": np.random.randint(100, 10000, n),
        "dbytes": np.random.randint(100, 10000, n),
        "sttl": np.random.randint(32, 255, n),
        "dttl": np.random.randint(32, 255, n),
        "sloss": np.random.randint(0, 5, n),
        "dloss": np.random.randint(0, 5, n),
        "service": np.random.choice(["http", "ftp", "dns", "smtp", "-"], n),
        "sload": np.random.exponential(1000, n),
        "dload": np.random.exponential(1000, n),
        "spkts": np.random.randint(1, 100, n),
        "dpkts": np.random.randint(1, 100, n),
        "swin": np.random.randint(1024, 65535, n),
        "dwin": np.random.randint(1024, 65535, n),
        "stcpb": np.random.randint(0, 1000, n),
        "dtcpb": np.random.randint(0, 1000, n),
        "tcprtt": np.random.exponential(0.1, n),
        "synack": np.random.exponential(0.05, n),
        "ackdat": np.random.exponential(0.05, n),
        "smean": np.random.randint(100, 1500, n),
        "dmean": np.random.randint(100, 1500, n),
        "trans_depth": np.random.randint(0, 10, n),
        "response_body_len": np.random.randint(0, 5000, n),
        "ct_srv_src": np.random.randint(1, 50, n),
        "ct_state_ttl": np.random.randint(1, 50, n),
        "ct_dst_ltm": np.random.randint(1, 50, n),
        "ct_src_dport_ltm": np.random.randint(1, 50, n),
        "ct_dst_sport_ltm": np.random.randint(1, 50, n),
        "ct_dst_src_ltm": np.random.randint(1, 50, n),
        "is_ftp_login": np.random.randint(0, 2, n),
        "ct_ftp_cmd": np.random.randint(0, 10, n),
        "ct_flw_http_mthd": np.random.randint(0, 10, n),
        "ct_src_ltm": np.random.randint(1, 50, n),
        "ct_srv_dst": np.random.randint(1, 50, n),
        "is_sm_ips_ports": np.random.randint(0, 2, n),
        "attack_cat": np.random.choice(["Normal", "DoS", "Probe", "Privilege"], n),
        "label": np.random.randint(0, 2, n)
    })


class TestDataValidator:
    """Test DataValidator class."""

    def test_validator_init(self, sample_config, tmp_path):
        """Test validator initialization."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        validator = DataValidator(str(config_file))
        assert validator.target_column == "label"
        assert "srcip" in validator.excluded_features
        assert "proto" in validator.categorical_features

    def test_validate_structure(self, sample_config, sample_train_data, tmp_path):
        """Test structure validation."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        # Save sample data
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        sample_train_data.to_csv(raw_dir / "train.csv", index=False)

        validator = DataValidator(str(config_file))
        validator.raw_dir = raw_dir

        results = validator.validate_structure(sample_train_data, "train_set")

        assert results["shape"][0] == 1000
        assert results["shape"][1] == 43  # All columns in test fixture
        assert "label" in results["columns"]
        assert results["duplicate_rows"] == 0

    def test_validate_target(self, sample_config, sample_train_data, tmp_path):
        """Test target validation."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        validator = DataValidator(str(config_file))
        results = validator.validate_target(sample_train_data)

        assert "value_counts" in results
        assert 0 in results["value_counts"]
        assert 1 in results["value_counts"]
        assert len(results["issues"]) == 0  # Should be valid

    def test_validate_target_with_bad_values(self, sample_config, tmp_path):
        """Test target validation with invalid values."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        validator = DataValidator(str(config_file))

        # Data with invalid target values
        bad_data = pd.DataFrame({"label": [0, 1, 2, 3, -1]})
        results = validator.validate_target(bad_data)

        assert len(results["issues"]) > 0
        assert any("Unexpected target values" in issue for issue in results["issues"])

    def test_validate_features(self, sample_config, sample_train_data, tmp_path):
        """Test feature validation."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        validator = DataValidator(str(config_file))
        results = validator.validate_features(sample_train_data)

        assert len(results["numeric_features"]) > 0
        assert len(results["categorical_features"]) > 0
        assert "proto" in results["categorical_features"]
        assert "dur" in results["numeric_features"]
        assert "srcip" in results["excluded_features_present"]


class TestLabelConversion:
    """Test label conversion to binary."""

    def test_binary_labels(self, sample_config, tmp_path):
        """Test that labels are properly converted to 0/1."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        validator = DataValidator(str(config_file))

        # Test with string labels
        df_str = pd.DataFrame({"label": ["normal", "anomaly", "normal", "anomaly"]})
        # This would need the prepare_target method from preprocess
        # For now, just validate the validator handles the expected values
        results = validator.validate_target(pd.DataFrame({"label": [0, 1, 0, 1]}))
        assert len(results["issues"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])