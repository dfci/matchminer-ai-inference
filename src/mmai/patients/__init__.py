"""Patient summarization workflows."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from mmai.config import MMAIConfig, load_default_preset

from .summarize import summarize_from_relevant_sentences
from .tagging import extract_relevant_sentences

if TYPE_CHECKING:
    import pandas as pd
else:
    import pandas as pd


def summarize_patients(
    notes: pd.DataFrame,
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
        note_type: str
            Type of note (clinical_note, pathology_report, etc.).
        note_date : str or datetime
            Date of the note.
    return_metadata : bool, optional
        When True, also return a metadata dict containing the config snapshot
        and model metadata for this run.
    return_qc : bool, optional
        When True, also return a QC report DataFrame for this run.

    Returns
    -------
    pd.DataFrame
        Patient-level summary DataFrame. One row per patient.

        Columns
        -------
        patient_id : str
            Unique patient identifier.
        cancer_history_summary : str
            Summary of the patient's cancer history.
        general_exclusion_criteria_evidence : str
            Summary of conditions / findings that correspond to common
            clinical trial exclusion criteria.

        Debug Columns
        (Available only if pipeline initialized with debug_mode=True)
        -------------------------------------------------------------
        patient_long_text : str
            Concatenated set of tagger-selected (relevant) patient note text
            used as the input context for the LLM.
        original_patient_summary : str
            Raw LLM response generated from the patient long text.
        cleaned_patient_summary : str
            LLM response with reasoning text removed.
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
        "note_type",
        "note_date",
    ]
    missing = [col for col in required_columns if col not in notes.columns]
    if missing:
        raise ValueError(
            "summarize_patients requires columns "
            f"{', '.join(missing)} in the input DataFrame."
        )

    logger.info("Extracting relevant patient sentences from %d notes.", len(notes))
    relevant_sentences, tagger_metadata, tagger_qc = cast(
        tuple[pd.DataFrame, dict, pd.DataFrame],
        extract_relevant_sentences(
            notes,
            config=resolved_config,
            return_qc=True,
        ),
    )
    logger.info("Extracted relevant text for %d patients.", len(relevant_sentences))

    summaries, metadata, summary_qc = cast(
        tuple[pd.DataFrame, dict, pd.DataFrame],
        summarize_from_relevant_sentences(
            relevant_sentences,
            config=resolved_config,
            return_qc=True,
        ),
    )
    logger.info("Patient summarization complete. Produced %d rows.", len(summaries))
    # Build the full QC report using original notes, tagged notes, and summary QC.
    qc_report = None
    if return_qc:
        from mmai._qc.patients import patient_qc_report

        qc_report = patient_qc_report(
            summaries,
            patient_note_source=notes,
            summary_qc_report=summary_qc,
            tagger_qc_report=tagger_qc,
        )
    if return_metadata:
        metadata_payload = {
            "config_snapshot": resolved_config.raw,
            "model_metadata": {
                "patient_tagger": tagger_metadata["model_metadata"],
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
    "extract_relevant_sentences",
    "summarize_from_relevant_sentences",
]
