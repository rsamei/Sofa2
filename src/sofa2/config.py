"""Load and validate the YAML scoring rules and pipeline settings.

All thresholds live in ``sofa2/config/*.yaml``. A threshold list ("bands") is a list of
``{points, op, value}`` entries; a value scores the highest ``points`` whose condition
``value <op> threshold`` holds, or 0 if none holds (see :func:`sofa2.scoring.bands.score`).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import yaml

OPS = ("<", "<=", ">", ">=")
_DEFAULT_FILES = {"sofa2": "sofa2.yaml", "sofa1": "sofa1.yaml", "pipeline": "pipeline.yaml"}


class ConfigError(ValueError):
    """Raised when a configuration file is inconsistent."""


@dataclass(frozen=True)
class Band:
    """One threshold: a value scores ``points`` if ``value <op> threshold``."""

    points: int
    op: str
    value: float
    requires_support: bool = False


def parse_bands(raw: list[Mapping[str, Any]], name: str = "bands") -> list[Band]:
    """Parse and validate a band list.

    Checks that points are 1-4 and strictly increasing, that all ops point the same way, and
    that thresholds move monotonically towards worse values as points increase.
    """
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{name}: expected a non-empty list")
    bands = []
    for item in raw:
        try:
            band = Band(
                points=int(item["points"]),
                op=str(item["op"]),
                value=float(item["value"]),
                requires_support=bool(item.get("requires_support", False)),
            )
        except (KeyError, TypeError) as exc:
            raise ConfigError(f"{name}: malformed entry {item!r}") from exc
        if band.op not in OPS:
            raise ConfigError(f"{name}: unknown op {band.op!r}")
        if not 1 <= band.points <= 4:
            raise ConfigError(f"{name}: points must be 1-4, got {band.points}")
        bands.append(band)
    points = [b.points for b in bands]
    if points != sorted(points) or len(set(points)) != len(points):
        raise ConfigError(f"{name}: points must be strictly increasing, got {points}")
    lower = {b.op in ("<", "<=") for b in bands}
    if len(lower) != 1:
        raise ConfigError(f"{name}: mixes '<' and '>' operators")
    values = [b.value for b in bands]
    if lower.pop():
        ok = all(a > b for a, b in zip(values, values[1:]))
    else:
        ok = all(a < b for a, b in zip(values, values[1:]))
    if not ok:
        raise ConfigError(f"{name}: thresholds are not monotonic: {values}")
    return bands


def _deep_merge(base: dict, override: Mapping) -> dict:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _validate_tree(node: Any, path: str) -> None:
    """Validate every list whose key ends in ``bands`` anywhere in the tree."""
    if isinstance(node, dict):
        for key, val in node.items():
            sub = f"{path}.{key}"
            if str(key).endswith("bands") and isinstance(val, list):
                parse_bands(val, sub)
            elif str(key) == "drug_bands" and isinstance(val, dict):
                for drug, bands in val.items():
                    parse_bands(bands, f"{sub}.{drug}")
            else:
                _validate_tree(val, sub)


def _read_yaml(source: str | Path) -> dict:
    with open(source, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: top level must be a mapping")
    return data


def _default(name: str) -> dict:
    ref = resources.files("sofa2").joinpath("config", _DEFAULT_FILES[name])
    with resources.as_file(ref) as path:
        return _read_yaml(path)


@dataclass
class Config:
    """The three rule sets used by the pipeline.

    Attributes:
        sofa2: SOFA-2 thresholds (``sofa2.yaml``).
        sofa1: original SOFA thresholds (``sofa1.yaml``).
        pipeline: windows, missing-data strategy and data-handling settings (``pipeline.yaml``).
    """

    sofa2: dict = field(default_factory=lambda: _default("sofa2"))
    sofa1: dict = field(default_factory=lambda: _default("sofa1"))
    pipeline: dict = field(default_factory=lambda: _default("pipeline"))

    def validate(self) -> "Config":
        """Check all band lists and the pipeline options; return self."""
        _validate_tree(self.sofa2, "sofa2")
        _validate_tree(self.sofa1, "sofa1")
        strategy = self.pipeline["missing"]["strategy"]
        if strategy not in ("locf", "normal", "none"):
            raise ConfigError(f"pipeline.missing.strategy: unknown strategy {strategy!r}")
        return self


def load_config(
    sofa2: str | Path | None = None,
    sofa1: str | Path | None = None,
    pipeline: str | Path | None = None,
    overrides: Mapping[str, Mapping] | None = None,
) -> Config:
    """Load the default rules, optionally replaced or patched.

    Args:
        sofa2, sofa1, pipeline: paths to YAML files that replace the packaged defaults.
        overrides: nested mapping merged on top, eg
            ``{"pipeline": {"missing": {"strategy": "normal"}}}``.

    Returns:
        A validated :class:`Config`.
    """
    cfg = Config(
        sofa2=_read_yaml(sofa2) if sofa2 else _default("sofa2"),
        sofa1=_read_yaml(sofa1) if sofa1 else _default("sofa1"),
        pipeline=_read_yaml(pipeline) if pipeline else _default("pipeline"),
    )
    for key, val in (overrides or {}).items():
        if key not in _DEFAULT_FILES:
            raise ConfigError(f"unknown config section {key!r}")
        setattr(cfg, key, _deep_merge(getattr(cfg, key), val))
    return cfg.validate()
