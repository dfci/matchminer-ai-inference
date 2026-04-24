"""Checker-model inference helpers."""

from __future__ import annotations

from typing import Any, Dict, cast

from mmai.llm.backends import get_model_metadata


def run_checker(
    prompts: list[str],
    *,
    checker_config: Dict[str, Any],
    model_metadata_cache_dir: str | None = None,
) -> tuple[list[dict[str, Any]], Dict[str, Any]]:
    """Run a text-classification checker model on prompts."""
    from transformers import AutoTokenizer, pipeline

    model_name = checker_config["model_name"]
    device = checker_config["device"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=model_metadata_cache_dir,
        trust_remote_code=True,
    )
    checker_pipeline = pipeline(
        "text-classification",
        model_name,
        tokenizer=tokenizer,
        truncation=True,
        padding="max_length",
        max_length=4096,
        device=device,
    )
    model_metadata = get_model_metadata(
        model_name,
        cache_dir=model_metadata_cache_dir,
    )
    outputs = cast(
        list[dict[str, Any]],
        checker_pipeline(prompts),
    )
    return outputs, model_metadata


__all__ = ["run_checker"]
