"""Matching stubs."""

from __future__ import annotations


def generate_candidate_matches(*args: object, **kwargs: object) -> None:
    """Phase-level candidate generation (stub)."""
    raise NotImplementedError("Candidate generation phase not implemented in skeleton.")


def reasonable_match_check(*args: object, **kwargs: object) -> None:
    """Phase-level reasonable match check (stub)."""
    raise NotImplementedError(
        "Reasonable match check phase not implemented in skeleton."
    )


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
