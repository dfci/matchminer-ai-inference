"""Patient summarization logic."""

from __future__ import annotations

import os
from typing import Any, cast

import pandas as pd
from transformers import AutoTokenizer

from mmai.llm.backends import (
    build_summarization_runtime_config,
    get_summarization_backend,
)
from mmai.config import MMAIConfig, load_default_preset
from mmai.llm.prompt_rendering import Prompt

from .postprocess import postprocess_patient_summaries
from .prepare import prepare_patient_notes
from .prompt_builder import (
    PromptWorkItem,
    build_prompt_worker,
    prep_prompt_pool,
    shutdown_prompt_pool,
)


def validate_existing_summaries(
    existing_summaries: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and normalize existing patient summary state."""
    required_columns = ["patient_id", "patient_summary"]
    missing = [
        column
        for column in required_columns
        if column not in existing_summaries.columns
    ]
    if missing:
        raise ValueError(
            "existing summaries input must include columns "
            "'patient_id' and 'patient_summary'. Missing: "
            f"{', '.join(missing)}"
        )

    normalized = existing_summaries.copy()
    normalized["patient_id"] = normalized["patient_id"].astype(str)
    normalized["patient_summary"] = normalized["patient_summary"].where(
        normalized["patient_summary"].notna(),
        None,
    )
    return normalized.drop_duplicates(subset=["patient_id"], keep="last")


def _build_existing_summary_lookup(
    existing_summaries: pd.DataFrame | None,
) -> dict[str, str | None]:
    if existing_summaries is None:
        return {}
    normalized = validate_existing_summaries(existing_summaries)
    return cast(
        dict[str, str | None],
        normalized.set_index("patient_id")["patient_summary"].to_dict(),
    )


def _build_rounds(prepared_chunks: pd.DataFrame) -> list[pd.DataFrame]:
    """Organize patient chunks into rounds by chunk index."""
    if prepared_chunks.empty:
        return []
    rounds: list[pd.DataFrame] = []
    ordered = prepared_chunks.sort_values(["chunk_index", "patient_id"]).reset_index(
        drop=True
    )
    for _, group in ordered.groupby("chunk_index", sort=True):
        rounds.append(group.reset_index(drop=True))
    return rounds


def _build_prompt_list(
    round_df: pd.DataFrame,
    *,
    current_summaries: dict[str, str | None],
    prompt_pool: Any,
    n_prompt_workers: int,
) -> tuple[list[Prompt], list[str]]:
    work_items: list[PromptWorkItem] = []
    round_patient_ids: list[str] = []
    for row_idx, (_, row) in enumerate(round_df.iterrows()):
        patient_id = str(row["patient_id"])
        round_patient_ids.append(patient_id)
        work_items.append(
            PromptWorkItem(
                row_idx=row_idx,
                prior_summary_text=current_summaries.get(patient_id),
                first_date=str(row["first_date"]),
                last_date=str(row["last_date"]),
                chunk_text=str(row["chunk_text"]),
            )
        )

    chunksize = max(1, len(work_items) // (n_prompt_workers * 4)) if work_items else 1
    return (
        list(prompt_pool.map(build_prompt_worker, work_items, chunksize=chunksize)),
        round_patient_ids,
    )


def summarize_patient_notes(
    notes: pd.DataFrame,
    config: MMAIConfig | None = None,
    *,
    existing_summaries: pd.DataFrame | None = None,
    return_qc: bool = False,
) -> (
    tuple[pd.DataFrame, dict[str, Any]]
    | tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]
):
    """
    Summarize longitudinal patient notes using serial chunk-based updates.

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
    return_qc : bool, optional
        When True, also return a QC report DataFrame for this summarization step.
    """
    resolved_config = config or load_default_preset()
    if not isinstance(resolved_config, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")

    patient_config = dict(resolved_config.patient)
    runtime_patient_config = build_summarization_runtime_config(
        "patient",
        patient_config,
        config=resolved_config,
    )

    # Convert note-level input into patient-level metadata plus chunk-level
    # rows. The chunk rows are what drive the serial summarization loop.
    tokenizer = AutoTokenizer.from_pretrained(patient_config["model_name"])
    prepared_patients, prepared_chunks = prepare_patient_notes(
        notes,
        tokenizer,
        chunk_size=int(patient_config["chunk_size"]),
        chunk_overlap=int(patient_config["chunk_overlap"]),
    )
    existing_summary_lookup = _build_existing_summary_lookup(existing_summaries)
    rounds = _build_rounds(prepared_chunks)

    backend = get_summarization_backend(resolved_config)
    # This dict holds the latest available summary for each patient. If the
    # caller provided an existing summary, that is used for round 1; after each
    # round, the newly generated summary overwrites the prior one.
    current_summaries = {
        patient_id: summary for patient_id, summary in existing_summary_lookup.items()
    }
    model_metadata: dict[str, Any] = {}
    prompt_pool = None

    # Round N contains the Nth chunk for every patient that still has one.
    # Processing by rounds ensures each patient's next chunk sees the most
    # recent summary generated from prior chunks.
    n_prompt_workers = max(
        1,
        int(patient_config.get("prompt_build_workers", min(os.cpu_count() or 4, 32))),
    )
    try:
        if rounds:
            prompt_pool = prep_prompt_pool(
                patient_config=patient_config,
                n_workers=n_prompt_workers,
            )

        for round_df in rounds:
            prompt_list, round_patient_ids = _build_prompt_list(
                round_df,
                current_summaries=current_summaries,
                prompt_pool=prompt_pool,
                n_prompt_workers=n_prompt_workers,
            )
            summaries, round_model_metadata, _finish_reasons = (
                backend.generate_llm_outputs(
                    prompt_list=prompt_list,
                    llm_config=runtime_patient_config,
                    model_metadata_cache_dir=resolved_config.model_metadata_cache_dir,
                )
            )
            if not model_metadata:
                model_metadata = round_model_metadata
            # Persist each round's output so it becomes the prior summary for the
            # next chunk from that same patient.
            for patient_id, summary in zip(round_patient_ids, summaries, strict=False):
                current_summaries[patient_id] = summary
    finally:
        if prompt_pool is not None:
            shutdown_prompt_pool(prompt_pool)

    # Collapse the running patient state back to one final row per patient,
    # then do postprocessing and QC report generation.
    final_rows = prepared_patients.copy()
    final_rows["original_patient_summary"] = final_rows["patient_id"].map(
        current_summaries
    )
    final_rows = final_rows.dropna(subset=["original_patient_summary"]).copy()

    final_rows, noninformative_summary_qc_artifact = postprocess_patient_summaries(
        final_rows, resolved_config
    )

    metadata = {
        "config_snapshot": resolved_config.raw,
        "model_metadata": model_metadata,
    }

    from mmai._qc.patients import patient_summary_qc_report

    qc_report = patient_summary_qc_report(
        final_rows,
        noninformative_summary_qc_artifact=noninformative_summary_qc_artifact,
        config=resolved_config,
    )
    if return_qc:
        return final_rows, metadata, qc_report
    return final_rows, metadata


__all__ = [
    "summarize_patient_notes",
    "validate_existing_summaries",
]
