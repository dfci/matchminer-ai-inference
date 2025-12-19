"""Configuration stubs for MMAI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MMAIConfig:
    """Minimal configuration container (stub)."""

    preset_name: str = "default"
    debug_mode: bool = False
    backend: str = "local"


def load_preset(name: str) -> MMAIConfig:
    """Load a named configuration preset (stub)."""
    return MMAIConfig(preset_name=name)


def load_default_preset() -> MMAIConfig:
    """Load the default configuration preset (stub)."""
    return MMAIConfig()
