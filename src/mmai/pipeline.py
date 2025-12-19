"""High-level pipeline wrapper stubs."""

from __future__ import annotations


class MMAIPipeline:
    """Pipeline orchestrator (stub)."""

    def __init__(self, config: object | None = None) -> None:
        self.config = config

    def run_patient_centric_matching_pipeline(self, *args: object, **kwargs: object) -> None:
        """Run the full pipeline (stub)."""
        raise NotImplementedError("Pipeline logic not implemented in skeleton.")


def run_patient_centric_matching_pipeline(*args: object, **kwargs: object) -> None:
    """Module-level pipeline wrapper (stub)."""
    raise NotImplementedError("Pipeline logic not implemented in skeleton.")
