"""Patient prompt builders."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from importlib import resources
from multiprocessing import Pool
from multiprocessing.pool import Pool as PoolType
from typing import Any
from typing import cast

from transformers import AutoTokenizer

from mmai.prompt_rendering import Prompt


_worker_tokenizer: Any = None
_worker_config: dict[str, Any] = {}


def load_prompt_text(filename: str) -> str:
    """Load a prompt text file from the package."""
    prompt_path = resources.files("mmai.prompts").joinpath(filename)
    with prompt_path.open("r", encoding="utf-8") as handle:
        return handle.read()


def truncate_chunk_text(
    chunk_text: str,
    tokenizer: Any,
    *,
    max_model_len: int,
    margin_tokens: int = 5000,
) -> str:
    """
    Truncate chunk text if needed, keeping the beginning and end of the chunk.
    """
    threshold = max(1024, max_model_len - margin_tokens)
    chunk_tokens = tokenizer(chunk_text, add_special_tokens=False).input_ids
    if len(chunk_tokens) <= threshold:
        return chunk_text

    half = threshold // 2
    first_part = chunk_tokens[:half]
    last_part = chunk_tokens[-half:]
    return (
        cast(str, tokenizer.decode(first_part))
        + " ... "
        + cast(str, tokenizer.decode(last_part))
    )


def format_serial_summary_text(
    *,
    prior_summary: str | None,
    first_date: str,
    last_date: str,
    chunk_text: str,
    primer_filename: str,
    question_filename: str,
) -> str:
    """Wrap a serial patient-summary update in the configured prompt template."""
    prior_summary_text = (
        prior_summary
        if prior_summary
        else "None - this is the first segment for this patient"
    )
    serial_input = (
        "The following are the patient's data.\n"
        "---\n"
        "PRIOR SUMMARY:\n"
        f"{prior_summary_text}\n\n"
        f"NEXT CLINICAL RECORD SEGMENT (covering {first_date} to {last_date}):\n"
        f"{chunk_text}\n"
        "---\n"
    )
    return (
        load_prompt_text(primer_filename)
        + serial_input
        + load_prompt_text(question_filename)
    )


def get_serial_patient_prompt(
    *,
    prior_summary: str | None,
    first_date: str,
    last_date: str,
    chunk_text: str,
    tokenizer: Any,
    max_model_len: int,
    primer_filename: str,
    question_filename: str,
    margin_tokens: int = 5000,
    model_name: str = "",
) -> list[dict[str, str]]:
    """Prepare chat-style messages for one serial patient-summary update."""
    safe_chunk_text = truncate_chunk_text(
        chunk_text,
        tokenizer,
        max_model_len=max_model_len,
        margin_tokens=margin_tokens,
    )
    system_content = "Reasoning: high" if "gpt-oss" in model_name.lower() else ""
    return [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": format_serial_summary_text(
                prior_summary=prior_summary,
                first_date=first_date,
                last_date=last_date,
                chunk_text=safe_chunk_text,
                primer_filename=primer_filename,
                question_filename=question_filename,
            ),
        },
    ]


@dataclass
class PromptWorkItem:
    """A prompt that needs building."""

    row_idx: int
    prior_summary_text: str | None
    first_date: str
    last_date: str
    chunk_text: str


def _init_prompt_worker(patient_config: dict[str, Any]) -> None:
    """Initialize tokenizer and config in each prompt-building worker."""
    global _worker_tokenizer, _worker_config
    _worker_config = dict(patient_config)
    _worker_tokenizer = AutoTokenizer.from_pretrained(_worker_config["model_name"])


def prep_prompt_pool(
    patient_config: dict[str, Any],
    n_workers: int = min(os.cpu_count() or 4, 32),
) -> PoolType:
    """Create a multiprocessing pool for parallel prompt building."""
    n_workers = max(1, int(n_workers))
    logging.info("Creating prompt-building pool with %d workers.", n_workers)
    return Pool(
        processes=n_workers,
        initializer=_init_prompt_worker,
        initargs=(patient_config,),
    )


def build_prompt_worker(item: PromptWorkItem) -> Prompt:
    """Build a single patient prompt in a worker process."""
    prompt_files = dict(_worker_config["prompt_files"])
    messages = get_serial_patient_prompt(
        prior_summary=item.prior_summary_text,
        first_date=item.first_date,
        last_date=item.last_date,
        chunk_text=item.chunk_text,
        tokenizer=_worker_tokenizer,
        max_model_len=int(_worker_config["max_model_len"]),
        primer_filename=str(prompt_files["primer"]),
        question_filename=str(prompt_files["question"]),
        margin_tokens=int(_worker_config["prompt_margin_tokens"]),
        model_name=str(_worker_config["model_name"]),
    )
    prompt_text = cast(
        str,
        _worker_tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            tokenize=False,
        ),
    )
    prompt_token_count = len(
        _worker_tokenizer(prompt_text, add_special_tokens=False).input_ids
    )
    max_tokens = int(_worker_config["sampling_params"]["max_tokens"])
    gen_tokens = max(
        1,
        min(int(_worker_config["max_model_len"]) - prompt_token_count, max_tokens),
    )
    return Prompt(
        row_idx=item.row_idx,
        prompt_text=prompt_text,
        max_tokens=gen_tokens,
    )


def shutdown_prompt_pool(prompt_pool: PoolType) -> None:
    """Shutdown the workers used for building prompts."""
    prompt_pool.close()
    prompt_pool.join()


__all__ = [
    "PromptWorkItem",
    "build_prompt_worker",
    "format_serial_summary_text",
    "get_serial_patient_prompt",
    "load_prompt_text",
    "prep_prompt_pool",
    "shutdown_prompt_pool",
    "truncate_chunk_text",
]
