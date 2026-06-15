"""Configuration stubs for MMAI."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from importlib import resources


@dataclass
class MMAIConfig:
    """Minimal configuration container."""

    preset_name: str
    debug_mode: bool
    trial: dict[str, Any]
    patient: dict[str, Any]
    local: dict[str, Any]
    remote: dict[str, Any]
    embedding: dict[str, Any]
    model_metadata_cache_dir: str | None
    raw: dict[str, Any]


def config_snapshot(config: MMAIConfig) -> dict[str, Any]:
    """Build a metadata snapshot from the live config object."""
    snapshot = deepcopy(config.raw)
    snapshot.update(
        {
            "preset_name": config.preset_name,
            "debug_mode": config.debug_mode,
            "trial": deepcopy(config.trial),
            "patient": deepcopy(config.patient),
            "local": deepcopy(config.local),
            "remote": deepcopy(config.remote),
            "embedding": deepcopy(config.embedding),
            "model_metadata_cache_dir": config.model_metadata_cache_dir,
        }
    )
    return snapshot


def _load_preset_data(name: str) -> dict[str, Any]:
    preset_path = resources.files("matchminer_ai.presets").joinpath(f"{name}.yaml")
    with preset_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Preset {name} did not parse into a mapping.")
    return data


def _config_from_data(data: dict[str, Any], preset_name: str) -> MMAIConfig:
    return MMAIConfig(
        preset_name=preset_name,
        debug_mode=bool(data["debug_mode"]),
        trial=dict(data["trial"]),
        patient=dict(data["patient"]),
        local=dict(data.get("local", {})),
        remote=dict(data.get("remote", {})),
        embedding=dict(data["embedding"]),
        model_metadata_cache_dir=data["model_metadata_cache_dir"],
        raw=deepcopy(data),
    )


def load_preset(name: str) -> MMAIConfig:
    """Load a named configuration preset."""
    data = _load_preset_data(name)
    return _config_from_data(data, preset_name=name)


def load_config(path: str | Path) -> MMAIConfig:
    """Load a configuration YAML file from a user-provided path."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {config_path} did not parse into a mapping.")
    return _config_from_data(data, preset_name=str(config_path))


def load_default_preset() -> MMAIConfig:
    """Load the default configuration preset."""
    return load_preset("default")
