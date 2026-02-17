"""Trial-related stubs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mmai.config import MMAIConfig

if TYPE_CHECKING:
    import pandas as pd
else:
    import pandas as pd


def summarize_trials(
    trials: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
    return_metadata: bool = False,
    return_qc: bool = False,
) -> (
    pd.DataFrame
    | tuple[pd.DataFrame, dict]
    | tuple[pd.DataFrame, pd.DataFrame]
    | tuple[pd.DataFrame, dict, pd.DataFrame]
):
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
    return_metadata : bool, optional
        When True, also return a metadata dict containing the config snapshot
        and model metadata for this run.
    return_qc : bool, optional
        When True, also return a QC report DataFrame for this run.

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
    tuple[pd.DataFrame, dict]
        When return_metadata is True, returns the DataFrame plus a metadata dict.
    tuple[pd.DataFrame, pd.DataFrame]
        When return_qc is True, returns the DataFrame plus a QC report DataFrame.
    tuple[pd.DataFrame, dict, pd.DataFrame]
        When return_metadata and return_qc are True, returns the DataFrame,
        metadata dict, and QC report DataFrame.
    """
    from mmai.config import MMAIConfig, load_default_preset

    from .postprocess import postprocess_trial_summaries
    from .summarize import run_llm_summarization

    logger = logging.getLogger(__name__)
    resolved_config = config or load_default_preset()
    if not isinstance(resolved_config, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")

    required_columns = {
        "trial_id",
        "trial_title",
        "brief_summary",
        "eligibility_criteria",
    }
    missing = required_columns.difference(trials.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(
            f"summarize_trials requires columns: {', '.join(sorted(required_columns))}. "
            f"Missing: {missing_list}."
        )

    logger.info("Starting trial summarization for %d trials.", len(trials))
    trials_with_summaries, metadata, finish_reasons = run_llm_summarization(
        trials, resolved_config
    )
    logger.info("Completed LLM summarization. Beginning postprocessing.")
    if return_qc:
        # Capture unfiltered spaces for QC before keyword filtering.
        result, unfiltered_spaces = postprocess_trial_summaries(
            trials_with_summaries,
            resolved_config,
            return_qc_data=True,
        )
    else:
        unfiltered_spaces = None
        result = postprocess_trial_summaries(trials_with_summaries, resolved_config)
    logger.info("Postprocessing complete. Produced %d rows.", len(result))
    # Build QC report only when requested.
    if return_qc:
        from mmai._qc.trials import trial_qc_report

        qc_report = trial_qc_report(
            result,
            trial_source=trials,
            unfiltered_spaces=unfiltered_spaces,
            finish_reasons=finish_reasons,
            config=resolved_config,
        )
    else:
        qc_report = None
    if return_metadata:
        # Optionally return metadata, and append QC when requested.
        metadata_payload = {
            "config_snapshot": resolved_config.raw,
            "model_metadata": {
                "trial_summarizer": metadata["model_metadata"],
            },
        }
        if return_qc:
            return result, metadata_payload, qc_report
        return result, metadata_payload
    if return_qc:
        return result, qc_report
    return result


__all__ = ["summarize_trials"]
