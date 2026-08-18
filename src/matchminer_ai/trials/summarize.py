"""Trial summarization logic."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import pandas as pd

from matchminer_ai._qc.trials import build_qc_artifact
from matchminer_ai.llm.backends import (
    LLMGenerationResult,
    LocalBackend,
    RemoteBackend,
    build_llm_runtime_config,
    get_llm_backend,
)
from matchminer_ai.llm.prompt_rendering import build_prompt_list

from .prompt_builder import build_trial_text, get_filled_trial_prompt

if TYPE_CHECKING:
    from matchminer_ai.config import MMAIConfig


logger = logging.getLogger(__name__)


def summarize_trials_multi_cohort(
    trial_texts: list[str],
    backend: LocalBackend | RemoteBackend,
    *,
    trial_config: dict[str, Any],
    primer_filename: str,
    question_filename: str,
    model_metadata_cache_dir: str | None = None,
) -> LLMGenerationResult:
    """Summarize trials using the configured backend."""
    messages_list = [
        get_filled_trial_prompt(text, primer_filename, question_filename)
        for text in trial_texts
    ]
    prompt_list = build_prompt_list(messages_list, llm_config=trial_config)
    return backend.generate_llm_outputs(
        prompt_list=prompt_list,
        llm_config=trial_config,
        model_metadata_cache_dir=model_metadata_cache_dir,
    )


def _filter_failed_trial_inference(
    trials_with_summaries: pd.DataFrame,
    generation: LLMGenerationResult,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove failed inference results before trial postprocessing."""
    trial_ids = trials_with_summaries["trial_id"].astype(str).tolist()
    generation_rows = list(
        zip(
            trial_ids,
            generation.final_outputs,
            generation.reasoning_outputs,
            generation.finish_reasons,
            strict=True,
        )
    )
    successful_positions: list[int] = []
    failed_trial_ids: list[str] = []
    for position, (trial_id, _output, _reasoning, finish_reason) in enumerate(
        generation_rows
    ):
        if str(finish_reason) == "error":
            failed_trial_ids.append(trial_id)
            logger.warning("Filtering trial %s after inference failed.", trial_id)
        else:
            successful_positions.append(position)

    # Get DF of trials that passed inference step
    successful_trials = trials_with_summaries.iloc[successful_positions].copy()
    successful_trials["trial_answer_text"] = [
        generation_rows[position][1] for position in successful_positions
    ]
    successful_trials["trial_reasoning_text"] = [
        generation_rows[position][2] for position in successful_positions
    ]
    return successful_trials, failed_trial_ids


def run_llm_summarization(
    trials_to_process: pd.DataFrame, config: MMAIConfig
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
    dict[str, object],
    dict[str, object],
]:
    """Run LLM-based trial summarization."""
    trial_config = dict(config.trial)
    runtime_trial_config = build_llm_runtime_config(
        "trial",
        trial_config,
        config=config,
    )
    prompt_files = dict(trial_config["prompt_files"])
    primer_filename = prompt_files["primer"]
    question_filename = prompt_files["question"]

    backend = get_llm_backend(config)

    trials_with_summaries = trials_to_process.copy()
    trials_with_summaries["trial_text"] = build_trial_text(trials_to_process)
    generation = summarize_trials_multi_cohort(
        trials_with_summaries["trial_text"].tolist(),
        backend,
        trial_config=runtime_trial_config,
        primer_filename=primer_filename,
        question_filename=question_filename,
        model_metadata_cache_dir=config.model_metadata_cache_dir,
    )

    # Filter out trials that errored on the LLM inference step and record the trial IDs
    trials_with_summaries, failed_trial_ids = _filter_failed_trial_inference(
        trials_with_summaries,
        generation,
    )
    trial_ids = trials_to_process["trial_id"].astype(str).tolist()
    failed_llm_qc_artifact = build_qc_artifact(
        metric="trials_failed_inference",
        ids=failed_trial_ids,
        denominator=len(trial_ids),
    )
    truncated_llm_qc_artifact = build_qc_artifact(
        metric="trials_truncated_llm_response",
        ids=[
            trial_id
            for trial_id, reason in zip(
                trial_ids, generation.finish_reasons, strict=False
            )
            if str(reason) == "length"
        ],
        denominator=len(trial_ids),
    )
    metadata = {
        "config_snapshot": {"trial": trial_config},
        "model_metadata": generation.model_metadata,
    }
    return (
        trials_with_summaries,
        metadata,
        truncated_llm_qc_artifact,
        failed_llm_qc_artifact,
    )
