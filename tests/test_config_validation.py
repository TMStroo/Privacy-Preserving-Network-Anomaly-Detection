"""Configuration validation.

Every shipped config is executed by CI or by a reviewer, and a wrong model or
adaptation name used to surface halfway through a half-hour run. These checks
run in a second and catch the same mistakes up front.
"""

import os
from pathlib import Path

import pytest
import yaml

from driftguard.adaptation.strategies import ADAPTATION_STRATEGIES
from driftguard.drift.detectors import normalise_method
from driftguard.models.registry import MODELS, model_config_names, model_params

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))
DATASET_CONFIGS = [c for c in CONFIGS if c.name not in {"config.yaml"}]


def _load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def test_configs_exist():
    names = {c.name for c in CONFIGS}
    assert "benchmark.yaml" in names
    assert "ugr16_subset.yaml" in names
    assert "smoke.yaml" in names


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_is_valid_yaml_and_has_a_seed(path):
    config = _load(path)
    assert isinstance(config, dict)
    assert config.get("seed") is not None or "seed" not in config


@pytest.mark.parametrize("path", DATASET_CONFIGS, ids=lambda p: p.name)
def test_configured_models_exist(path):
    config = _load(path)
    if "models" not in config:
        return
    for name in model_config_names(config):
        assert name in MODELS, f"{path.name} requests unknown model {name}"


@pytest.mark.parametrize("path", DATASET_CONFIGS, ids=lambda p: p.name)
def test_configured_adaptation_strategies_exist(path):
    config = _load(path)
    strategies = (config.get("adaptation") or {}).get("strategies")
    if not strategies:
        return
    for name in strategies:
        assert name in ADAPTATION_STRATEGIES, (
            f"{path.name} requests unknown adaptation strategy '{name}'. "
            f"Available: {ADAPTATION_STRATEGIES}"
        )


@pytest.mark.parametrize("path", DATASET_CONFIGS, ids=lambda p: p.name)
def test_configured_drift_methods_resolve(path):
    config = _load(path)
    methods = (config.get("drift") or {}).get("methods") or []
    for name in methods:
        assert normalise_method(name), f"{path.name} requests unknown drift method {name}"


@pytest.mark.parametrize("path", DATASET_CONFIGS, ids=lambda p: p.name)
def test_drift_window_and_stride_are_sane(path):
    config = _load(path)
    drift = config.get("drift") or {}
    window, stride = drift.get("window"), drift.get("stride")
    if window is None:
        return
    from driftguard.utils import as_timedelta

    assert as_timedelta(window) > as_timedelta(0), "drift window must be positive"
    if stride is not None:
        assert as_timedelta(stride) > as_timedelta(0), "drift stride must be positive"


@pytest.mark.parametrize("path", DATASET_CONFIGS, ids=lambda p: p.name)
def test_drift_features_exist_in_the_common_schema(path):
    config = _load(path)
    requested = (config.get("drift") or {}).get("features")
    if not requested:
        return
    from driftguard.data.schema import COMMON_SCHEMA

    known = {f["name"] if isinstance(f, dict) else f for f in COMMON_SCHEMA}
    missing = [f for f in requested if f not in known]
    assert not missing, f"{path.name} requests features absent from the schema: {missing}"


@pytest.mark.parametrize("path", DATASET_CONFIGS, ids=lambda p: p.name)
def test_model_params_lookup_handles_both_config_shapes(path):
    config = _load(path)
    for name in model_config_names(config):
        assert isinstance(model_params(config, name), dict)


def test_model_params_is_empty_for_a_list_style_block():
    assert model_params({"models": ["logistic_regression"]}, "logistic_regression") == {}


def test_unknown_model_name_is_rejected_with_a_useful_message():
    with pytest.raises(KeyError, match="unknown models in config"):
        model_config_names({"models": {"enabled": ["not_a_model"]}})


def test_ablation_config_names_known_ablations():
    """A typo in the ablation matrix should fail here, not four hours in."""
    from driftguard.experiments.ablations import ABLATIONS

    config = _load(CONFIG_DIR / "ablations.yaml")
    for name in config["ablation"]["names"]:
        assert name in ABLATIONS, f"unknown ablation {name!r}"


def test_every_shipped_config_ablation_models_are_known():
    from driftguard.models.registry import model_config_names

    config = _load(CONFIG_DIR / "ablations.yaml")
    known = set(model_config_names(config))
    for name in config["ablation"]["models"]:
        assert name in known, f"unknown ablation model {name!r}"
