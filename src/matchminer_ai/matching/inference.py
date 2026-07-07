"""Checker-model inference helpers."""

from __future__ import annotations

import gc
from functools import lru_cache
from typing import Any, Dict, cast

from matchminer_ai.llm.backends import get_model_metadata


@lru_cache(maxsize=2)
def _get_checker_pipeline(
    model_name: str,
    device: str,
    max_length: int,
    model_metadata_cache_dir: str | None,
):
    """Load and cache a Transformers text-classification checker pipeline."""
    from transformers import AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=model_metadata_cache_dir,
        trust_remote_code=True,
    )
    return pipeline(
        "text-classification",
        model_name,
        tokenizer=tokenizer,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        device=device,
    )


def clear_checker_pipeline_cache() -> None:
    """Release cached checker pipeline handles and clear Python/GPU caches."""
    _get_checker_pipeline.cache_clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_checker(
    prompts: list[str],
    *,
    checker_config: Dict[str, Any],
    model_metadata_cache_dir: str | None = None,
) -> tuple[list[dict[str, Any]], Dict[str, Any]]:
    """Run a text-classification checker model on prompts."""
    model_name = checker_config["model_name"]
    device = checker_config["device"]
    max_length = int(checker_config.get("max_length", 4096))
    checker_pipeline = _get_checker_pipeline(
        model_name,
        device,
        max_length,
        model_metadata_cache_dir,
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


__all__ = ["clear_checker_pipeline_cache", "run_checker"]
