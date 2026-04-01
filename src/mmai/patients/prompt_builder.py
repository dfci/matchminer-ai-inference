"""Patient prompt builders."""

from __future__ import annotations

from importlib import resources
from typing import Any
from typing import cast


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


__all__ = [
    "format_serial_summary_text",
    "get_serial_patient_prompt",
    "load_prompt_text",
    "truncate_chunk_text",
]
