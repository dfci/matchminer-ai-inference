"""Trial summarization logic."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING, cast

import pandas as pd

from mmai.backends import get_backend

from .prompt_builder import build_trial_text, get_filled_trial_prompt

if TYPE_CHECKING:
    from mmai.config import MMAIConfig


def summarize_trials_multi_cohort(
    trial_texts: list[str],
    backend: Any,
    *,
    trial_config: dict[str, Any],
    primer_filename: str,
    question_filename: str,
    model_metadata_cache_dir: str | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Summarize trials using the configured backend."""
    messages_list = [
        get_filled_trial_prompt(text, primer_filename, question_filename)
        for text in trial_texts
    ]
    return cast(
        tuple[list[str], dict[str, Any]],
        backend.generate_llm_outputs(
            messages_list=messages_list,
            trial_config=trial_config,
            model_metadata_cache_dir=model_metadata_cache_dir,
        ),
    )


def run_llm_summarization(
    trials_to_process: pd.DataFrame, config: MMAIConfig
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run LLM-based trial summarization."""
    trial_config = dict(config.trial)
    prompt_files = dict(trial_config.get("prompt_files", {}))
    primer_filename = prompt_files.get("primer", "trial.user.primer.txt")
    question_filename = prompt_files.get("question", "trial.user.question.txt")

    backend = get_backend(config.backend)

    trials_with_summaries = trials_to_process.copy()
    trials_with_summaries["trial_text"] = build_trial_text(trials_to_process)
    trial_summaries, model_metadata = summarize_trials_multi_cohort(
        trials_with_summaries["trial_text"].tolist(),
        backend,
        trial_config=trial_config,
        primer_filename=primer_filename,
        question_filename=question_filename,
        model_metadata_cache_dir=config.model_metadata_cache_dir,
    )

    trials_with_summaries["space_reasoning_and_output"] = trial_summaries
    metadata = {
        "config_snapshot": {"trial": trial_config},
        "model_metadata": model_metadata,
    }
    return trials_with_summaries, metadata
