"""Patient summarization workflows."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from matchminer_ai.config import MMAIConfig, config_snapshot, load_default_preset

from .summarize import summarize_patient_notes

if TYPE_CHECKING:
    import pandas as pd
else:
    import pandas as pd


def summarize_patients(
    notes: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
    existing_summaries: pd.DataFrame | None = None,
    return_metadata: bool = False,
    return_qc: bool = False,
) -> (
    pd.DataFrame
    | tuple[pd.DataFrame, dict]
    | tuple[pd.DataFrame, pd.DataFrame]
    | tuple[pd.DataFrame, dict, pd.DataFrame]
):
    """
    Summarize longitudinal patient notes into a cancer history summary and
    evidence related to general clinical trial exclusion criteria.

    Parameters
    ----------
    notes : pd.DataFrame
        Note-level input. One row per note.

        Expected columns
        ----------------
        patient_id : str
            Unique patient identifier.
        note_text : str
            Full note text.
        note_date : str or datetime
            Date of the note.
    existing_summaries : pd.DataFrame, optional
        Optional patient-level prior summaries used as the starting state for
        serial updates.

        Expected columns
        ----------------
        patient_id : str
            Unique patient identifier.
        patient_summary : str
            Existing full patient summary text to update.
    return_metadata : bool, optional
        When True, also return a metadata dict containing the config snapshot
        and model metadata for this run.
    return_qc : bool, optional
        When True, also return a QC report DataFrame for this run.

    Returns
    -------
    pd.DataFrame
        Patient-level DataFrame. One row per patient.

        Columns
        -------
        patient_id : str
            Original patient identifier.
        cancer_history_summary : str
            Summary of the patient's cancer history.
        general_exclusion_criteria_evidence : str
            Summary of conditions / findings that correspond to common
            clinical trial exclusion criteria.

        Debug Columns
        (Available only if pipeline initialized with debug_mode=True)
        -------------------------------------------------------------
        original_patient_summary : str
            Final serial summary before boilerplate splitting.
        cleaned_patient_summary : str
            Final serial summary after whitespace cleanup.
        final_round_patient_summary_raw_output : str
            Raw generated text from the final serial summarization round when
            available.
        final_round_patient_summary_reasoning : str
            Parsed reasoning text from the final serial summarization round
            when available.
    tuple[pd.DataFrame, dict]
        When return_metadata is True, returns the DataFrame plus a metadata dict.
    tuple[pd.DataFrame, pd.DataFrame]
        When return_qc is True, returns the DataFrame plus a QC report DataFrame.
    tuple[pd.DataFrame, dict, pd.DataFrame]
        When return_metadata and return_qc are True, returns the DataFrame,
        metadata dict, and QC report DataFrame.

    """
    logger = logging.getLogger(__name__)
    resolved_config = config or load_default_preset()
    if not isinstance(resolved_config, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")

    required_columns = [
        "patient_id",
        "note_text",
        "note_date",
    ]
    missing = [col for col in required_columns if col not in notes.columns]
    if missing:
        raise ValueError(
            "summarize_patients requires columns "
            f"{', '.join(missing)} in the input DataFrame."
        )

    logger.info("Preparing serial patient summarization for %d notes.", len(notes))
    summaries, metadata, qc_report = cast(
        tuple[pd.DataFrame, dict, pd.DataFrame],
        summarize_patient_notes(
            notes,
            config=resolved_config,
            existing_summaries=existing_summaries,
            return_qc=True,
        ),
    )
    logger.info("Patient summarization complete. Produced %d rows.", len(summaries))

    if return_metadata:
        metadata_payload = {
            "config_snapshot": config_snapshot(resolved_config),
            "model_metadata": {
                "patient_summarizer": metadata["model_metadata"],
            },
        }
        if return_qc:
            return summaries, metadata_payload, qc_report
        return summaries, metadata_payload
    if return_qc:
        return summaries, qc_report
    return summaries


__all__ = [
    "summarize_patients",
    "summarize_patient_notes",
]
