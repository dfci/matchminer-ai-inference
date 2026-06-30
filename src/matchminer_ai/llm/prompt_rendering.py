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
    messages
        Original chat-style messages. Remote OpenAI-compatible backends use
        these directly so vLLM can return structured reasoning fields.
    """

    row_idx: int
    prompt_text: str
    max_tokens: int
    messages: list[dict[str, str]] | None = None


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
    chat_template_kwargs: dict[str, Any] | None = None,
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
    chat_template_kwargs
        Extra keyword arguments passed to ``tokenizer.apply_chat_template``.
    """
    tokenizer = _get_chat_template_tokenizer(model_name)
    template_kwargs = dict(chat_template_kwargs or {})

    def apply_template(messages: list[dict[str, str]]) -> str:
        return cast(
            str,
            tokenizer.apply_chat_template(
                conversation=messages,
                add_generation_prompt=True,
                tokenize=False,
                **template_kwargs,
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
    the configured generation token limit for each prompt.
    """
    prompt_texts = build_chat_prompts(
        messages_list,
        model_name=str(llm_config["model_name"]),
        prompt_build_workers=llm_config.get("prompt_build_workers"),
        chat_template_kwargs=llm_config.get("chat_template_kwargs"),
    )
    sampling_params = dict(llm_config.get("sampling_params", {}))
    configured_max_tokens = sampling_params.get(
        "max_tokens",
        sampling_params.get("max_completion_tokens"),
    )
    if configured_max_tokens is None:
        raise ValueError(
            "LLM config must include sampling_params['max_tokens'] or "
            "sampling_params['max_completion_tokens']."
        )
    max_tokens = int(configured_max_tokens)
    return [
        Prompt(
            row_idx=row_idx,
            prompt_text=prompt_text,
            max_tokens=max_tokens,
            messages=messages,
        )
        for row_idx, (messages, prompt_text) in enumerate(
            zip(messages_list, prompt_texts, strict=False)
        )
    ]


__all__ = [
    "Prompt",
    "build_chat_prompts",
    "build_prompt_list",
]
