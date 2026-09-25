"""Tests for feature building and feature policy."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from privacy_preserving_nad.features.build_features import (
    FULL_METADATA_FEATURES,
    RESTRICTED_METADATA_FEATURES,
    CATEGORICAL_FEATURES,
    EXCLUDED_FEATURES,
    FEATURE_DOCUMENTATION,
    get_feature_set,
    get_feature_documentation,
    validate_feature_policy,
    generate_feature_report
)


class TestFeatureSets:
    """Test feature set definitions."""

    def test_full_metadata_features(self):
        """Test FULL_METADATA feature list."""
        features = get_feature_set("FULL_METADATA")
        assert len(features) == len(FULL_METADATA_FEATURES)
        assert "dur" in features
        assert "proto" in features
        assert "service" in features
        assert "state" in features
        assert "srcip" not in features  # Excluded
        assert "attack_cat" not in features  # Excluded

    def test_restricted_metadata_features(self):
        """Test RESTRICTED_METADATA feature list."""
        features = get_feature_set("RESTRICTED_METADATA")
        assert len(features) == len(RESTRICTED_METADATA_FEATURES)
        assert "dur" in features
        assert "spkts" in features
        assert "proto" not in features  # Not in restricted
        assert "service" not in features  # Not in restricted

    def test_restricted_is_subset_of_full(self):
        """Test that restricted features are a subset of full features."""
        full = set(get_feature_set("FULL_METADATA"))
        restricted = set(get_feature_set("RESTRICTED_METADATA"))
        assert restricted.issubset(full)

    def test_categorical_features(self):
        """Test categorical feature list."""
        assert "proto" in CATEGORICAL_FEATURES
        assert "service" in CATEGORICAL_FEATURES
        assert "state" in CATEGORICAL_FEATURES
        assert len(CATEGORICAL_FEATURES) == 3

    def test_excluded_features(self):
        """Test excluded feature list."""
        assert "srcip" in EXCLUDED_FEATURES
        assert "dstip" in EXCLUDED_FEATURES
        assert "attack_cat" in EXCLUDED_FEATURES
        assert "sport" in EXCLUDED_FEATURES

    def test_feature_documentation(self):
        """Test feature documentation exists for all features."""
        all_features = set(FULL_METADATA_FEATURES)
        documented = set(FEATURE_DOCUMENTATION.keys())
        assert all_features.issubset(documented), f"Missing docs for: {all_features - documented}"

    def test_get_feature_documentation(self):
        """Test getting documentation for a feature."""
        doc = get_feature_documentation("dur")
        assert "Flow duration" in doc
        assert "Observable" in doc

        doc = get_feature_documentation("proto")
        assert "protocol" in doc.lower()

    def test_unknown_feature_documentation(self):
        """Test documentation for unknown feature."""
        doc = get_feature_documentation("nonexistent_feature")
        assert "No documentation available" in doc


class TestFeaturePolicyValidation:
    """Test feature policy validation."""

    def test_validate_compliant_dataset(self):
        """Test validation with compliant dataset columns."""
        # Dataset has all allowed features, no excluded features
        df_columns = FULL_METADATA_FEATURES + ["label"]
        result = validate_feature_policy(df_columns)

        assert result["policy_compliant"] is True
        assert len(result["excluded_features_present"]) == 0
        assert len(result["full_metadata_available"]) == len(FULL_METADATA_FEATURES)

    def test_validate_with_excluded_features(self):
        """Test validation catches excluded features."""
        df_columns = FULL_METADATA_FEATURES + ["srcip", "dstip", "label"]
        result = validate_feature_policy(df_columns)

        assert result["policy_compliant"] is False
        assert "srcip" in result["excluded_features_present"]
        assert "dstip" in result["excluded_features_present"]

    def test_validate_missing_features(self):
        """Test validation reports missing features."""
        df_columns = ["dur", "spkts", "label"]  # Minimal columns
        result = validate_feature_policy(df_columns)

        assert len(result["full_metadata_missing"]) > 0
        assert "proto" in result["full_metadata_missing"]

    def test_restricted_features_available(self):
        """Test restricted feature availability check."""
        df_columns = RESTRICTED_METADATA_FEATURES + ["label"]
        result = validate_feature_policy(df_columns)

        assert len(result["restricted_metadata_available"]) == len(RESTRICTED_METADATA_FEATURES)
        assert len(result["restricted_metadata_missing"]) == 0


class TestFeatureReport:
    """Test feature report generation."""

    def test_generate_feature_report(self, tmp_path):
        """Test generating feature policy report."""
        # Create a minimal config
        config = {
            "feature_sets": {
                "FULL_METADATA": FULL_METADATA_FEATURES,
                "RESTRICTED_METADATA": RESTRICTED_METADATA_FEATURES
            }
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        report = generate_feature_report(str(config_file))

        assert "# Feature Policy Documentation" in report
        assert "FULL_METADATA" in report
        assert "RESTRICTED_METADATA" in report
        assert "Excluded Features" in report
        assert "srcip" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])