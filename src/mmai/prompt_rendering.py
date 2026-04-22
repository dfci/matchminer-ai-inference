"""Shared LLM prompt rendering helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast


@dataclass
class Prompt:
    """
    A rendered prompt ready for inference.

    Attributes
    ----------
    row_idx
        Original input position. Backends use it to restore output order after
        concurrent or distributed execution.
    prompt_text
        Final text prompt after applying the model chat template.
    max_tokens
        Generation token limit for this specific prompt.
    """

    row_idx: int
    prompt_text: str
    max_tokens: int


@lru_cache(maxsize=4)
def _get_chat_template_tokenizer(model_name: str):
    """Load and cache the tokenizer used to apply a model chat template."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )


def build_chat_prompts(
    messages_list: list[list[dict[str, str]]],
    *,
    model_name: str,
    prompt_build_workers: int | None = None,
) -> list[str]:
    """
    Render chat-style messages into model prompt strings.

    Parameters
    ----------
    messages_list
        One chat conversation per model request.
    model_name
        Hugging Face model name used to load the tokenizer/chat template.
    prompt_build_workers
        Optional number of threads for applying chat templates. Output order
        matches input order.
    """
    tokenizer = _get_chat_template_tokenizer(model_name)

    def apply_template(messages: list[dict[str, str]]) -> str:
        return cast(
            str,
            tokenizer.apply_chat_template(
                conversation=messages,
                add_generation_prompt=True,
                tokenize=False,
            ),
        )

    if len(messages_list) <= 1:
        return [apply_template(messages) for messages in messages_list]

    max_workers = prompt_build_workers or min(32, len(messages_list))
    max_workers = max(1, int(max_workers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(apply_template, messages_list))


def build_prompt_list(
    messages_list: list[list[dict[str, str]]],
    *,
    llm_config: dict[str, Any],
) -> list[Prompt]:
    """
    Build indexed ``Prompt`` objects from chat-style message lists.

    The rendered prompts keep input ordering through ``Prompt.row_idx`` and use
    ``llm_config["sampling_params"]["max_tokens"]`` as the per-prompt generation
    limit.
    """
    prompt_texts = build_chat_prompts(
        messages_list,
        model_name=str(llm_config["model_name"]),
        prompt_build_workers=llm_config.get("prompt_build_workers"),
    )
    return [
        Prompt(
            row_idx=row_idx,
            prompt_text=prompt_text,
            max_tokens=int(llm_config["sampling_params"]["max_tokens"]),
        )
        for row_idx, prompt_text in enumerate(prompt_texts)
    ]


__all__ = [
    "Prompt",
    "build_chat_prompts",
    "build_prompt_list",
]
