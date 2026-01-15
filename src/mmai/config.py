"""Configuration stubs for MMAI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml

from importlib import resources


@dataclass
class MMAIConfig:
    """Minimal configuration container."""

    preset_name: str
    debug_mode: bool
    backend: str
    trial: dict[str, Any]


def _load_preset_data(name: str) -> dict[str, Any]:
    preset_path = resources.files("mmai.presets").joinpath(f"{name}.yaml")
    with preset_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Preset {name} did not parse into a mapping.")
    return data


def load_preset(name: str) -> MMAIConfig:
    """Load a named configuration preset."""
    data = _load_preset_data(name)
    return MMAIConfig(
        preset_name=name,
        debug_mode=bool(data.get("debug_mode", False)),
        backend=str(data.get("backend", "local")),
        trial=dict(data.get("trial", {})),
    )


def load_default_preset() -> MMAIConfig:
    """Load the default configuration preset."""
    return load_preset("default")
