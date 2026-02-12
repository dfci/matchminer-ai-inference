"""Patient summarization logic."""

from __future__ import annotations

from typing import Any

import pandas as pd

from mmai.backends import get_backend
from mmai.config import MMAIConfig, load_default_preset

from .postprocess import postprocess_patient_summaries
from .prompt_builder import get_filled_patient_prompt


def summarize_from_relevant_sentences(
    df: pd.DataFrame,
    config: MMAIConfig | None = None,
    *,
    return_qc: bool = False,
) -> (
    tuple[pd.DataFrame, dict[str, Any]]
    | tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]
):
    """
    Summarize patient text extracted from relevant sentences.

    Parameters
    ----------
    df : pd.DataFrame
        Patient-level input with `patient_long_text`.

        Expected columns
        ----------------
        patient_id : str
            Unique patient identifier.
    patient_long_text : str
        Concatenated relevant note text for the patient.
    return_qc : bool, optional
        When True, also return a QC report DataFrame for this summarization step.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        Patient-level summaries and metadata about the run.
    tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]
        When return_qc is True, also returns a QC report DataFrame.
    """
    resolved_config = config or load_default_preset()
    if not isinstance(resolved_config, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")

    patient_config = dict(resolved_config.patient)
    prompt_files = dict(patient_config["prompt_files"])
    primer_filename = prompt_files["primer"]
    question_filename = prompt_files["question"]

    patient_long_text_col = "patient_long_text"

    df = df.copy()
    df = df[df[patient_long_text_col] != ""]
    df = df.dropna(subset=[patient_long_text_col])

    backend = get_backend(resolved_config.backend)
    patient_texts = df[patient_long_text_col].astype(str).tolist()
    truncated_texts = backend.truncate_texts(
        patient_texts,
        patient_config=patient_config,
    )

    messages_list = [
        get_filled_patient_prompt(patient_text, primer_filename, question_filename)
        for patient_text in truncated_texts
    ]

    summaries, model_metadata, finish_reasons = backend.generate_llm_outputs(
        messages_list=messages_list,
        llm_config=patient_config,
        model_metadata_cache_dir=resolved_config.model_metadata_cache_dir,
    )
    finish_reason_by_patient = pd.Series(
        finish_reasons,
        index=df["patient_id"].astype(str).tolist(),
    )

    df["original_patient_summary"] = summaries
    if return_qc:
        df, dropped_ids = postprocess_patient_summaries(
            df, resolved_config, return_qc_data=True
        )
    else:
        df = postprocess_patient_summaries(df, resolved_config)

    metadata = {
        "config_snapshot": resolved_config.raw,
        "model_metadata": model_metadata,
    }
    if return_qc:
        from mmai._qc.patients import patient_summary_qc_report

        qc_report = patient_summary_qc_report(
            df,
            noninformative_summary_drop_ids=dropped_ids or [],
            finish_reasons=finish_reason_by_patient,
        )
        return df, metadata, qc_report
    return df, metadata


__all__ = ["summarize_from_relevant_sentences"]
