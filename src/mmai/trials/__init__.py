"""Trial-related stubs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def summarize_trials(
    trials: pd.DataFrame, *, return_metadata: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Summarize clinical trials into clinical spaces and general exclusion criteria.

    Parameters
    ----------
    trials : pd.DataFrame
        Trial-level input. One row per trial.

        Expected columns
        ----------------
        trial_id : str
            Unique trial identifier.
        trial_title : str
            Human-readable trial title.
        brief_summary : str
            Brief textual summary of the trial.
        eligibility_criteria : str
            Full eligibility criteria text for the trial.

    Returns
    -------
    pd.DataFrame
        Clinical-space-level DataFrame. One row per clinical space per trial.

        Columns
        -------
        space_trial_id : str
            Unique identifier for a specific trial + clinical space combination.
        trial_id : str
            Original trial identifier (copied through from input).
        clinical_space_number : int
            Integer index of the clinical space within the trial.
        clinical_space_summary : str
            Summary of the clinical space (disease context, line of therapy, etc).
        general_exclusion_criteria : str
            General trial-level exclusion criteria text extracted for this space.

        Debug Columns
        (Available only if pipeline initialized with debug_mode=True)
        -------------------------------------------------------------
        trial_text : str
            Concatenation of trial_title + brief_summary + eligibility_criteria.
            This is the raw input text fed into the LLM.
        space_reasoning_and_output : str
            Raw LLM response text.
    """
    from mmai.config import load_default_preset

    from .postprocess import postprocess_trial_summaries
    from .summarize import run_llm_summarization

    config = load_default_preset()
    trials_with_summaries, metadata = run_llm_summarization(trials, config)
    result = postprocess_trial_summaries(trials_with_summaries, config)
    if return_metadata:
        return result, metadata
    return result


__all__ = ["summarize_trials"]
