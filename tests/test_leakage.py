"""Tests for target-leakage protections and dataset classification.

These guard the research claim: labels, payload-derived fields and test-set
statistics must never reach model training, and development data must never be
mistaken for the research dataset.
"""

import yaml
import numpy as np
import pandas as pd
import pytest

from privacy_preserving_nad.data.dataset import detect_dataset, is_research_dataset
from privacy_preserving_nad.data.preprocess import DataPreprocessor


def write_config(tmp_path, feature_sets, payload=None):
    config = {
        "dataset": {
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"),
            "train_file": "train.csv",
            "test_file": "test.csv",
            "research_dataset": {
                "train_rows": 175341,
                "test_rows": 82332,
                "train_sha256": "0" * 64,
                "test_sha256": "0" * 64,
            },
        },
        "target": {"column": "label", "normal_value": 0, "anomaly_value": 1},
        "excluded_features": [
            "srcip", "dstip", "sport", "dsport", "srcpt", "dstpt",
            "Ltime", "Stime", "attack_cat", "label", "id",
        ],
        "payload_derived_features": payload or [
            "trans_depth", "response_body_len", "is_ftp_login",
            "ct_ftp_cmd", "ct_flw_http_mthd",
        ],
        "categorical_features": ["proto", "service", "state"],
        "feature_sets": feature_sets,
        "preprocessing": {"handle_missing": "median", "scale_numeric": True},
        "models": {"random_seed": 42},
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


class TestFeaturePolicyGuards:

    def test_config_feature_sets_have_no_forbidden_names(self, tmp_path):
        """The shipped config must never select labels, ids or payload fields."""
        with open("configs/config.yaml") as f:
            config = yaml.safe_load(f)

        preprocessor = DataPreprocessor("configs/config.yaml")
        forbidden = preprocessor.forbidden_feature_names()
        for name, features in config["feature_sets"].items():
            violations = set(features) & forbidden
            assert violations == set(), f"{name} violates policy: {violations}"

    def test_select_features_rejects_label(self, tmp_path):
        path = write_config(tmp_path, {"BAD": ["dur", "label"]})
        preprocessor = DataPreprocessor(str(path))
        df = pd.DataFrame({"dur": [1.0], "label": [0]})

        with pytest.raises(ValueError, match="violates the metadata-only policy"):
            preprocessor.select_features(df, "BAD")

    def test_select_features_rejects_attack_category(self, tmp_path):
        path = write_config(tmp_path, {"BAD": ["dur", "attack_cat"]})
        preprocessor = DataPreprocessor(str(path))
        df = pd.DataFrame({"dur": [1.0], "attack_cat": ["Normal"]})

        with pytest.raises(ValueError, match="violates the metadata-only policy"):
            preprocessor.select_features(df, "BAD")

    def test_select_features_rejects_payload_derived(self, tmp_path):
        path = write_config(tmp_path, {"BAD": ["dur", "response_body_len"]})
        preprocessor = DataPreprocessor(str(path))
        df = pd.DataFrame({"dur": [1.0], "response_body_len": [100]})

        with pytest.raises(ValueError, match="violates the metadata-only policy"):
            preprocessor.select_features(df, "BAD")

    def test_select_features_rejects_row_id(self, tmp_path):
        path = write_config(tmp_path, {"BAD": ["dur", "id"]})
        preprocessor = DataPreprocessor(str(path))
        df = pd.DataFrame({"dur": [1.0], "id": [7]})

        with pytest.raises(ValueError, match="violates the metadata-only policy"):
            preprocessor.select_features(df, "BAD")


class TestTrainOnlyPreprocessing:

    def test_scaler_is_fitted_on_training_split_only(self, tmp_path):
        """Test-split statistics must not influence the fitted scaler."""
        path = write_config(tmp_path, {"FS": ["dur", "spkts"]})
        preprocessor = DataPreprocessor(str(path))

        X_train = pd.DataFrame({"dur": [1.0, 2.0, 3.0, 4.0], "spkts": [10.0, 20.0, 30.0, 40.0]})
        X_test = pd.DataFrame({"dur": [1000.0, 2000.0], "spkts": [10000.0, 20000.0]})

        preprocessor.fit_transform(X_train, X_test)

        scaler = preprocessor.column_transformer.named_transformers_["num"].named_steps["scaler"]
        assert scaler.mean_[0] == pytest.approx(X_train["dur"].mean())
        assert scaler.mean_[1] == pytest.approx(X_train["spkts"].mean())

        combined_dur_mean = pd.concat([X_train["dur"], X_test["dur"]]).mean()
        assert scaler.mean_[0] != pytest.approx(combined_dur_mean)

    def test_imputer_is_fitted_on_training_split_only(self, tmp_path):
        path = write_config(tmp_path, {"FS": ["dur", "spkts"]})
        preprocessor = DataPreprocessor(str(path))

        X_train = pd.DataFrame({"dur": [1.0, 2.0, np.nan, 4.0], "spkts": [10.0, 20.0, 30.0, 40.0]})
        X_test = pd.DataFrame({"dur": [999.0, 1001.0], "spkts": [5000.0, 6000.0]})

        X_train_t, X_test_t = preprocessor.fit_transform(X_train, X_test)

        imputer = preprocessor.column_transformer.named_transformers_["num"].named_steps["imputer"]
        train_median = X_train["dur"].median()
        assert imputer.statistics_[0] == pytest.approx(train_median)
        assert not np.isnan(X_train_t).any()
        assert not np.isnan(X_test_t).any()


class TestTargetGuards:

    def test_prepare_target_rejects_non_binary(self, tmp_path):
        path = write_config(tmp_path, {"FS": ["dur"]})
        preprocessor = DataPreprocessor(str(path))
        df = pd.DataFrame({"label": [0, 1, 2]})

        with pytest.raises(ValueError, match="binary 0/1"):
            preprocessor.prepare_target(df)

    def test_prepare_target_rejects_missing(self, tmp_path):
        path = write_config(tmp_path, {"FS": ["dur"]})
        preprocessor = DataPreprocessor(str(path))
        df = pd.DataFrame({"label": [0, 1, np.nan]})

        with pytest.raises(ValueError, match="missing"):
            preprocessor.prepare_target(df)

    def test_prepare_target_accepts_binary(self, tmp_path):
        path = write_config(tmp_path, {"FS": ["dur"]})
        preprocessor = DataPreprocessor(str(path))
        df = pd.DataFrame({"label": [0, 1, 1, 0]})

        y = preprocessor.prepare_target(df)
        assert set(y.unique()) == {0, 1}
        assert len(y) == 4


class TestDatasetDetection:

    def test_missing_dataset(self, tmp_path):
        config = {
            "dataset": {
                "raw_dir": str(tmp_path / "raw"),
                "train_file": "UNSW_NB15_training-set.csv",
                "test_file": "UNSW_NB15_testing-set.csv",
                "research_dataset": {"train_rows": 175341, "test_rows": 82332,
                                     "train_sha256": "a", "test_sha256": "b"},
            }
        }
        info = detect_dataset(config)
        assert info["kind"] == "missing"
        assert not is_research_dataset(info)

    def test_development_sample_is_not_research(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        pd.DataFrame({"dur": [1.0, 2.0], "label": [0, 1]}).to_csv(raw / "UNSW_NB15_training-set.csv", index=False)
        pd.DataFrame({"dur": [3.0], "label": [1]}).to_csv(raw / "UNSW_NB15_testing-set.csv", index=False)

        config = {
            "dataset": {
                "raw_dir": str(raw),
                "train_file": "UNSW_NB15_training-set.csv",
                "test_file": "UNSW_NB15_testing-set.csv",
                "research_dataset": {"train_rows": 175341, "test_rows": 82332,
                                     "train_sha256": "a", "test_sha256": "b"},
            }
        }
        info = detect_dataset(config)
        assert info["kind"] == "development-sample"
        assert info["train_rows"] == 2
        assert not is_research_dataset(info)

    def test_row_count_and_checksum_recognition(self, tmp_path):
        """A file with the expected rows but the wrong checksum is flagged."""
        from privacy_preserving_nad.data.dataset import sha256_of

        raw = tmp_path / "raw"
        raw.mkdir()
        train = raw / "UNSW_NB15_training-set.csv"
        test = raw / "UNSW_NB15_testing-set.csv"
        pd.DataFrame({"dur": [1.0, 2.0, 3.0], "label": [0, 1, 1]}).to_csv(train, index=False)
        pd.DataFrame({"dur": [1.0, 2.0], "label": [0, 1]}).to_csv(test, index=False)

        config = {
            "dataset": {
                "raw_dir": str(raw),
                "train_file": train.name,
                "test_file": test.name,
                "research_dataset": {
                    "train_rows": 3, "test_rows": 2,
                    "train_sha256": sha256_of(train),
                    "test_sha256": sha256_of(test),
                },
            }
        }
        info = detect_dataset(config)
        assert info["kind"] == "research"
        assert info["checksums_match"] is True

        config["dataset"]["research_dataset"]["test_sha256"] = "f" * 64
        info = detect_dataset(config)
        assert info["kind"] == "unexpected-content"
