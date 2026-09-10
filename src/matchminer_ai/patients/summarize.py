"""Patient summarization logic."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import pandas as pd
from transformers import AutoTokenizer

from matchminer_ai.llm.backends import (
    build_llm_runtime_config,
    get_llm_backend,
)
from matchminer_ai._metadata import package_metadata
from matchminer_ai.config import MMAIConfig, config_snapshot, load_default_preset
from matchminer_ai.llm.prompt_rendering import Prompt

from .checkpoints import (
    load_prepared_chunks,
    load_round_checkpoints,
    prepare_checkpoint_dir,
    save_prepared_chunks,
    save_round_checkpoint,
)
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
    checkpoint_dir: str | Path | None = None,
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
    checkpoint_dir : str or pathlib.Path, optional
        Directory used to save and reuse patient summarization checkpoints.
    return_qc : bool, optional
        When True, also return a QC report DataFrame for this summarization step.
    """
    resolved_config = config or load_default_preset()
    if not isinstance(resolved_config, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")

    patient_config = dict(resolved_config.patient)
    runtime_patient_config = build_llm_runtime_config(
        "patient",
        patient_config,
        config=resolved_config,
    )

    prepared_chunks = None
    if checkpoint_dir is not None:
        prepare_checkpoint_dir(checkpoint_dir)
        prepared_chunks = load_prepared_chunks(checkpoint_dir)

    if prepared_chunks is None:
        # Tokenization and chunking are only repeated when no reusable checkpoint exists.
        tokenizer_name = runtime_patient_config.get(
            "tokenizer_name",
            runtime_patient_config["model_name"],
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
        )
        prepared_chunks = prepare_patient_notes(
            notes,
            tokenizer,
            chunk_size=int(patient_config["chunk_size"]),
            chunk_overlap=int(patient_config["chunk_overlap"]),
        )
        if checkpoint_dir is not None:
            save_prepared_chunks(checkpoint_dir, prepared_chunks)

    existing_summary_lookup = _build_existing_summary_lookup(existing_summaries)
    rounds = _build_rounds(prepared_chunks)
    completed_rounds = (
        load_round_checkpoints(checkpoint_dir) if checkpoint_dir is not None else {}
    )

    backend = get_llm_backend(resolved_config)
    # This dict holds the latest available summary for each patient. If the
    # caller provided an existing summary, that is used for round 1; after each
    # round, the newly generated summary overwrites the prior one.
    current_summaries = {
        patient_id: summary for patient_id, summary in existing_summary_lookup.items()
    }
    current_reasoning_outputs: dict[str, str] = {}
    model_metadata: dict[str, Any] = {}
    prompt_pool = None

    # Rebuild the latest patient state before continuing with the next round.
    for _, round_checkpoint in sorted(completed_rounds.items()):
        patient_ids = round_checkpoint["patient_id"].astype(str)
        current_summaries.update(
            dict(zip(patient_ids, round_checkpoint["summary"], strict=False))
        )
        current_reasoning_outputs.update(
            dict(zip(patient_ids, round_checkpoint["reasoning"], strict=False))
        )

    # Round N contains the Nth chunk for every patient that still has one.
    # Processing by rounds ensures each patient's next chunk sees the most
    # recent summary generated from prior chunks.
    n_prompt_workers = max(
        1,
        int(patient_config.get("prompt_build_workers", min(os.cpu_count() or 4, 32))),
    )
    try:
        if len(completed_rounds) < len(rounds):
            prompt_pool = prep_prompt_pool(
                patient_config=runtime_patient_config,
                n_workers=n_prompt_workers,
            )

        for round_idx in range(len(completed_rounds), len(rounds)):
            round_df = rounds[round_idx]
            prompt_list, round_patient_ids = _build_prompt_list(
                round_df,
                current_summaries=current_summaries,
                prompt_pool=prompt_pool,
                n_prompt_workers=n_prompt_workers,
            )
            generation = backend.generate_llm_outputs(
                prompt_list=prompt_list,
                llm_config=runtime_patient_config,
                model_metadata_cache_dir=resolved_config.model_metadata_cache_dir,
            )
            if not model_metadata:
                model_metadata = generation.model_metadata
            summaries = generation.final_outputs
            reasoning_outputs = generation.reasoning_outputs
            finish_reasons = generation.finish_reasons
            round_results: list[dict[str, str | None]] = []
            # Persist each round's final summary, not the reasoning trace, so
            # it becomes the prior summary for the next patient chunk.
            for patient_id, summary, reasoning, finish_reason in zip(
                round_patient_ids,
                summaries,
                reasoning_outputs,
                finish_reasons,
                strict=False,
            ):
                prior_summary = current_summaries.get(patient_id)
                round_results.append(
                    {
                        "patient_id": patient_id,
                        "prior_summary": prior_summary,
                        "reasoning": str(reasoning),
                        "summary": str(summary),
                        "finish_reason": str(finish_reason),
                    }
                )
                current_summaries[patient_id] = str(summary)
                current_reasoning_outputs[patient_id] = str(reasoning)

            # A round becomes resumable only after inference finishes for the batch.
            if checkpoint_dir is not None:
                save_round_checkpoint(
                    checkpoint_dir,
                    round_idx,
                    pd.DataFrame(round_results).sort_values("patient_id"),
                )
    finally:
        if prompt_pool is not None:
            shutdown_prompt_pool(prompt_pool)

    # Collapse chunk-level work to one final row for each summarized patient.
    final_rows = prepared_chunks[["patient_id"]].drop_duplicates().copy()
    final_rows["patient_answer_text"] = final_rows["patient_id"].map(current_summaries)
    if resolved_config.debug_mode:
        # These columns preserve final-round debug traces without feeding them
        # back into the serial patient summary state.
        final_rows["patient_reasoning_text"] = final_rows["patient_id"].map(
            current_reasoning_outputs
        )
    final_rows = final_rows.dropna(subset=["patient_answer_text"]).copy()

    final_rows = postprocess_patient_summaries(final_rows, resolved_config)

    metadata = {
        "package": package_metadata(),
        "config_snapshot": config_snapshot(resolved_config),
        "model_metadata": model_metadata,
    }

    if return_qc:
        from matchminer_ai._qc.patients import patient_summary_qc_report

        qc_report = patient_summary_qc_report(
            final_rows,
            config=resolved_config,
        )
        return final_rows, metadata, qc_report
    return final_rows, metadata


__all__ = [
    "summarize_patient_notes",
    "validate_existing_summaries",
]
