"""High-level pipeline wrapper stubs."""

from __future__ import annotations

from .config import MMAIConfig, load_default_preset


class MMAIPipeline:
    """Pipeline orchestrator (stub)."""

    def __init__(self, config: MMAIConfig | None = None) -> None:
        self.config = config or load_default_preset()
        self.debug_mode = getattr(self.config, "debug_mode", False)

    def run_patient_centric_matching_pipeline(
        self, *args: object, **kwargs: object
    ) -> None:
        """Run the full pipeline (stub)."""
        raise NotImplementedError("Pipeline logic not implemented in skeleton.")


def run_patient_centric_matching_pipeline(*args: object, **kwargs: object) -> None:
    """Module-level pipeline wrapper (stub)."""
    raise NotImplementedError("Pipeline logic not implemented in skeleton.")
