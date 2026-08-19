"""Package-level run metadata helpers."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

PACKAGE_NAME = "matchminer-ai"


def package_version() -> str:
    """Return the installed package version."""
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return "0+unknown"


def package_metadata() -> dict[str, str]:
    """Identify the installed package that produced a run."""
    return {
        "name": PACKAGE_NAME,
        "version": package_version(),
    }
