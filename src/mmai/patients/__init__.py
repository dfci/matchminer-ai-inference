"""Patient summarization workflows."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mmai.config import MMAIConfig, load_default_preset

from .summarize import summarize_from_relevant_sentences
from .tagging import extract_relevant_sentences

if TYPE_CHECKING:
    import pandas as pd


def summarize_patients(
    notes: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Summarize longitudinal patient notes into a cancer history summary and
    evidence related to general exclusion criteria.

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
    relevant_sentences, tagger_metadata = extract_relevant_sentences(
        notes,
        config=resolved_config,
    )
    logger.info("Extracted relevant text for %d patients.", len(relevant_sentences))

    summaries, metadata = summarize_from_relevant_sentences(
        relevant_sentences, resolved_config
    )
    logger.info("Patient summarization complete. Produced %d rows.", len(summaries))
    if return_metadata:
        return summaries, {
            "config_snapshot": resolved_config.raw,
            "model_metadata": {
                "patient_tagger": tagger_metadata["model_metadata"],
                "patient_summarizer": metadata["model_metadata"],
            },
        }
    return summaries


__all__ = [
    "summarize_patients",
    "extract_relevant_sentences",
    "summarize_from_relevant_sentences",
]
