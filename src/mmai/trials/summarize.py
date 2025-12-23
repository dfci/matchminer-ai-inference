"""Trial summarization logic."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import pandas as pd

from mmai.backends import get_backend

from .prompt_builder import build_trial_text, get_filled_trial_prompt

if TYPE_CHECKING:
    from mmai.config import MMAIConfig


def summarize_trials_multi_cohort(
    trial_texts: list[str],
    llama_model: Any,
    sampling_params: dict[str, Any],
    primer_filename: str,
    question_filename: str,
) -> list[str]:
    """Summarize trials using the configured LLM."""
    from vllm import SamplingParams
    from vllm.outputs import RequestOutput
    from vllm.transformers_utils.tokenizer import AnyTokenizer

    tokenizer: AnyTokenizer = llama_model.get_tokenizer()
    prompts: list[str] = []
    for trial in trial_texts:
        messages = get_filled_trial_prompt(trial, primer_filename, question_filename)
        prompts.append(
            tokenizer.apply_chat_template(
                conversation=messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        )

    responses: list[RequestOutput] = llama_model.generate(
        prompts=prompts,
        sampling_params=SamplingParams(
            temperature=sampling_params["temperature"],
            top_k=sampling_params["top_k"],
            max_tokens=sampling_params["max_tokens"],
            repetition_penalty=sampling_params["repetition_penalty"],
        ),
    )

    return [response.outputs[0].text for response in responses]


def run_llm_summarization(
    trials_to_process: pd.DataFrame, config: MMAIConfig
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run LLM-based trial summarization."""
    trial_config = dict(config.trial)
    sampling_params = dict(trial_config.get("sampling_params", {}))
    prompt_files = dict(trial_config.get("prompt_files", {}))
    primer_filename = prompt_files.get("primer", "trial.user.primer.txt")
    question_filename = prompt_files.get("question", "trial.user.question.txt")

    backend = get_backend(config.backend)
    llama, model_metadata = backend.get_llm(
        model_name=trial_config["model_name"],
        max_model_len=trial_config["max_model_len"],
        tensor_parallel_size=trial_config["tensor_parallel_size"],
        gpu_memory_utilization=trial_config["gpu_memory_utilization"],
        sampling_params=sampling_params,
        model_metadata_cache_dir=trial_config.get("model_metadata_cache_dir"),
    )

    trials_with_summaries = trials_to_process.copy()
    trials_with_summaries["trial_text"] = build_trial_text(trials_to_process)
    trial_summaries = summarize_trials_multi_cohort(
        trials_with_summaries["trial_text"].tolist(),
        llama,
        sampling_params,
        primer_filename,
        question_filename,
    )

    trials_with_summaries["space_reasoning_and_output"] = trial_summaries
    metadata = {
        "config_snapshot": {"trial": trial_config},
        "model_metadata": model_metadata,
    }
    return trials_with_summaries, metadata
