import pytest

from sofa2.config import ConfigError, load_config, parse_bands


def test_defaults_load_and_validate():
    cfg = load_config()
    assert cfg.sofa2["score"] == "sofa2"
    assert cfg.sofa1["score"] == "sofa1"
    assert cfg.pipeline["missing"]["strategy"] == "locf"


def test_override_merges_nested():
    cfg = load_config(overrides={"pipeline": {"missing": {"strategy": "normal"}}})
    assert cfg.pipeline["missing"]["strategy"] == "normal"
    assert cfg.pipeline["missing"]["locf_max_hours"] == 24


def test_unknown_strategy_rejected():
    with pytest.raises(ConfigError):
        load_config(overrides={"pipeline": {"missing": {"strategy": "mean"}}})


def test_unknown_section_rejected():
    with pytest.raises(ConfigError):
        load_config(overrides={"apache": {}})


@pytest.mark.parametrize(
    "raw",
    [
        [{"points": 2, "op": "<", "value": 1}, {"points": 1, "op": "<", "value": 2}],
        [{"points": 1, "op": "<", "value": 1}, {"points": 2, "op": "<", "value": 2}],
        [{"points": 1, "op": "<", "value": 2}, {"points": 2, "op": ">", "value": 1}],
        [{"points": 5, "op": "<", "value": 1}],
        [{"points": 1, "op": "=", "value": 1}],
        [],
    ],
)
def test_bad_bands_rejected(raw):
    with pytest.raises(ConfigError):
        parse_bands(raw)


def test_bad_band_in_yaml_override_rejected():
    bad = {"liver": {"bilirubin_bands": [{"points": 1, "op": ">", "value": 3}, {"points": 2, "op": ">", "value": 1}]}}
    with pytest.raises(ConfigError):
        load_config(overrides={"sofa2": bad})
