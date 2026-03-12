"""Matching helpers."""

from __future__ import annotations

from .match import generate_candidate_matches
from .reasonable_check import reasonable_match_check


def exclusion_criteria_check(*args: object, **kwargs: object) -> None:
    """Phase-level exclusion criteria check (stub)."""
    raise NotImplementedError(
        "Exclusion criteria check phase not implemented in skeleton."
    )


__all__ = [
    "exclusion_criteria_check",
    "generate_candidate_matches",
    "reasonable_match_check",
]
