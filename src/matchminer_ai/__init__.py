"""Public package exports for MatchMiner-AI."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

from .config import load_config, load_preset
from .pipeline import MMAIPipeline

_LAZY_SUBMODULES = {
    "embedding",
    "llm",
    "matching",
    "patients",
    "trials",
}

__all__ = ["MMAIPipeline", "load_config", "load_preset"]


def __getattr__(name: str) -> ModuleType:
    """Lazily expose public subpackages for documentation and introspection."""
    if name in _LAZY_SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
